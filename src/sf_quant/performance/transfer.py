import polars as pl

from sf_quant.data.assets import load_assets


def generate_transfer_coefficients(weights: pl.DataFrame, alphas: pl.DataFrame) -> pl.DataFrame:
    """
    Transfer coefficient per formation date (Clarke, de Silva & Thorley, 2002).

    The Silver Fund Manual (Ch. 1–4) does not define a transfer coefficient, so
    this follows the source: Clarke, R., de Silva, H. & Thorley, S. (2002),
    "Portfolio Constraints and the Fundamental Law of Active Management",
    *Financial Analysts Journal* 58(5). Under their diagonal-covariance form,
    the TC is the cross-sectional correlation between risk-adjusted active
    weights and risk-adjusted forecasts:

    .. math::

        TC_t = \\text{corr}_i\\left(h_{i,t}\\,\\sigma_{i,t},\\;
        \\frac{\\alpha_{i,t}}{\\sigma_{i,t}}\\right)

    where :math:`\\sigma_{i,t}` is Barra specific risk. A TC of 1 means the
    constraints did not change the bet; the realized IR is roughly
    :math:`TC \\cdot IC \\sqrt{BR}`.

    Parameters
    ----------
    weights : pl.DataFrame
        Active weights with ``date``, ``barrid`` and ``active_weight``, as
        returned by :func:`sf_quant.backtester.backtest`.
    alphas : pl.DataFrame
        The alphas the portfolios were formed on, with ``date``, ``barrid`` and
        ``alpha``, keyed by formation date. For a lagged run that is the
        lagged alpha, i.e. the output of :func:`sf_quant.backtester.collapse`.

    Returns
    -------
    pl.DataFrame
        One row per formation date: ``date`` and ``transfer_coefficient``.
        Average it over dates to summarize a run.

    Examples
    --------
    >>> import sf_quant.performance as sfp
    >>> tc = sfp.generate_transfer_coefficients(weights, grid)   # doctest: +SKIP
    >>> tc["transfer_coefficient"].mean()   # doctest: +SKIP
    """
    start, end = weights["date"].min(), weights["date"].max()
    risk = load_assets(
        start=start, end=end, columns=["date", "barrid", "specific_risk"], in_universe=False
    ).with_columns(pl.col("specific_risk").truediv(100))

    return (
        weights.select("date", "barrid", "active_weight")
        .join(alphas.select("date", "barrid", "alpha"), on=["date", "barrid"], how="inner")
        .join(risk, on=["date", "barrid"], how="inner")
        .group_by("date")
        .agg(
            pl.corr(
                pl.col("active_weight") * pl.col("specific_risk"),
                pl.col("alpha") / pl.col("specific_risk"),
            ).alias("transfer_coefficient")
        )
        .sort("date")
    )
