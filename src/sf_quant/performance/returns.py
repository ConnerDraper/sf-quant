import polars as pl
from sf_quant.data.assets import load_assets, load_trading_days
from sf_quant.data.benchmark import load_benchmark
from sf_quant.schema.active_portfolio_schema import ActivePortfolioSchema
from sf_quant.schema.portfolio_schema import PortfolioSchema
from sf_quant.schema.returns_schema import PortfolioRetSchema, MultiPortfolioRetSchema


def generate_active_returns_from_weights(
    weights: ActivePortfolioSchema,
) -> PortfolioRetSchema:
    """
    Generate active returns from active weights on a rebalance grid.

    Each row of the output is the active return earned over one holding period,
    keyed by the date the portfolio was formed. For a portfolio set at
    :math:`t_k` and held untouched until :math:`t_{k+1}`, that return is

    .. math::

        r_{active} = \\sum_i h_i \\, r_{i,(t_k, t_{k+1}]}

    where :math:`r_{i,(t_k, t_{k+1}]}` is asset *i*'s compounded return over the
    period. This is an identity for a buy-and-hold position, so weight drift is
    already accounted for — there is no need to track it separately.

    Works for any rebalance frequency, including daily: with weights on every
    trading day each period is one day long.

    Parameters
    ----------
    weights : ActivePortfolioSchema
        Active weights, as returned by :func:`sf_quant.backtester.backtest`.
        Must include ``date``, ``barrid`` and ``active_weight``.

    Returns
    -------
    PortfolioRetSchema
        One row per completed holding period:

        - ``date`` (date): the formation date that opened the period.
        - ``return`` (float): active return earned over the period.

    Notes
    -----
    - The formation date's own return belongs to the *previous* period, so a
      portfolio never earns the return of the day it was built. This is what
      keeps the alignment free of look-ahead.
    - The final formation date has no closing date and so yields no row; a
      K-date grid produces K-1 period returns.
    - A name that delists or leaves the universe mid-period is treated as
      liquidated at its last observation and flat thereafter: its return is
      compounded over the days that exist. A name with no observations in the
      period contributes nothing.

    Examples
    --------
    >>> import sf_quant.backtester as sfb
    >>> import sf_quant.performance as sfp
    >>> weights = sfb.backtest(grid, target_active_risk=0.05)   # doctest: +SKIP
    >>> returns = sfp.generate_active_returns_from_weights(weights)   # doctest: +SKIP
    """
    period_returns = _period_returns(weights)
    result = _earn(weights, period_returns, "active_weight")
    return PortfolioRetSchema.validate(result)


def _period_returns(weights: pl.DataFrame) -> pl.DataFrame:
    """Each asset's compounded return over every holding period on the grid.

    Keyed by ``form_date``, the formation date that opened the period. A period
    runs from the day after its formation date through the next formation date.
    """
    formation_dates = weights["date"].unique().sort()
    if formation_dates.len() < 2:
        raise ValueError(
            "need at least two rebalance dates to measure one holding period, got "
            f"{formation_dates.len()}"
        )

    daily = (
        load_assets(
            start=formation_dates.min(),
            end=formation_dates.max(),
            columns=["date", "barrid", "return"],
            in_universe=True,
        )
        .with_columns(pl.col("return").truediv(100))
        .drop_nulls("return")
    )

    calendar = pl.DataFrame({"date": daily["date"].unique().sort()}).with_row_index("t")

    # Each trading day belongs to the period opened by the most recent formation
    # date *strictly* before it. Matching against the day after each formation
    # date is what makes the backward join strict: a formation date's own return
    # closes the previous period rather than opening its own.
    opens = calendar.filter(pl.col("date").is_in(formation_dates.implode())).select(
        (pl.col("t") + 1).alias("t_open"), pl.col("date").alias("form_date")
    )

    buckets = (
        calendar.join_asof(opens, left_on="t", right_on="t_open", strategy="backward")
        .drop_nulls("form_date")
        .select("date", "form_date")
    )

    return (
        daily.join(buckets, on="date", how="inner")
        .group_by("form_date", "barrid")
        .agg(pl.col("return").add(1).product().sub(1).alias("return"))
    )


def _earn(
    weights: pl.DataFrame, period_returns: pl.DataFrame, weight: str, by: tuple[str, ...] = ()
) -> pl.DataFrame:
    """Sum of weight times period return per formation date (and per ``by``)."""
    return (
        weights.filter(
            pl.col("date").is_in(period_returns["form_date"].unique().implode())
        )
        .join(
            period_returns,
            left_on=["date", "barrid"],
            right_on=["form_date", "barrid"],
            how="left",
        )
        .group_by("date", *by)
        .agg(
            pl.col(weight)
            .mul(pl.col("return").fill_null(0.0))
            .sum()
            .alias("return")
        )
        .sort("date", *by)
    )


def generate_multi_returns_from_weights(weights: PortfolioSchema) -> MultiPortfolioRetSchema:
    """
    Generate portfolio, benchmark, and active returns from given portfolio weights.

    This function calculates returns by joining provided portfolio weights with
    asset returns and benchmark weights. It derives total, benchmark, and
    active portfolio weights, then computes weighted returns for each
    portfolio type over time.

    Parameters
    ----------
    weights : PortfolioSchema
        Portfolio weights validated against PortfolioSchema.
        Must include the following columns:

        - ``date`` (date): The date for each weight.
        - ``barrid`` (str): The unique asset identifier.
        - ``weight`` (float): The portfolio weight assigned to the asset.

    Returns
    -------
    MultiPortfolioRetSchema
        A validated DataFrame containing portfolio returns with the
        following columns:

        - ``date`` (date): The formation date that opened the period.
        - ``portfolio`` (str): Portfolio type; one of
          ``"total"``, ``"benchmark"``, or ``"active"``.
        - ``return`` (float): The portfolio's return over the period.

    Notes
    -----
    - Asset returns are sourced via ``load_assets``.
    - Benchmark weights are sourced via ``load_benchmark``, on each formation
      date.
    - Holding periods and timing follow
      :func:`generate_active_returns_from_weights`: each row is the return
      earned after its formation date, never on it, and a K-date grid gives
      K-1 rows per portfolio.

    Examples
    --------
    >>> import sf_quant.performance as sfp
    >>> returns = sfp.generate_multi_returns_from_weights(weights)   # doctest: +SKIP
    """
    start = weights["date"].min()
    end = weights["date"].max()

    benchmark = load_benchmark(start=start, end=end)

    stacked = (
        weights.join(benchmark, on=["date", "barrid"], how="left", suffix="_bmk")
        .with_columns(pl.col("weight").sub("weight_bmk").alias("weight_act"))
        .rename({"weight": "total", "weight_bmk": "benchmark", "weight_act": "active"})
        .unpivot(
            index=["date", "barrid"],
            variable_name="portfolio",
            value_name="weight",
        )
    )

    result = _earn(stacked, _period_returns(weights), "weight", by=("portfolio",))
    return MultiPortfolioRetSchema.validate(result)


def generate_returns_from_weights(weights: PortfolioSchema) -> PortfolioRetSchema:
    """
    Generate portfolio returns from given portfolio weights.

    This function calculates returns by joining provided portfolio weights with
    asset returns, computing weighted returns over time.

    Parameters
    ----------
    weights : PortfolioSchema
        Portfolio weights validated against PortfolioSchema.
        Must include the following columns:

        - ``date`` (date): The date for each weight.
        - ``barrid`` (str): The unique asset identifier.
        - ``weight`` (float): The portfolio weight assigned to the asset.

    Returns
    -------
    PortfolioRetSchema
        A validated DataFrame containing portfolio returns with the
        following columns:

        - ``date`` (date): The formation date that opened the period.
        - ``return`` (float): The portfolio's return over the period.

    Notes
    -----
    - Asset returns are sourced via ``load_assets``.
    - Holding periods and timing follow
      :func:`generate_active_returns_from_weights`: each row is the return
      earned after its formation date, never on it, and a K-date grid gives
      K-1 rows.

    Examples
    --------
    >>> import sf_quant.performance as sfp
    >>> returns = sfp.generate_returns_from_weights(weights)   # doctest: +SKIP
    """
    result = _earn(weights, _period_returns(weights), "weight")
    return PortfolioRetSchema.validate(result)

def generate_cumulative_log_returns(
    weights: ActivePortfolioSchema, dates: pl.Series | None = None
) -> pl.DataFrame:
    """
    Cumulative log active return of a weight history, marked on a daily axis.

    A holding-period return is keyed by its formation date :math:`t_k` but
    earned over :math:`(t_k, t_{k+1}]`, so it is booked at :math:`t_{k+1}`;
    between rebalances the path holds its last mark. Putting every run on the
    same daily axis is what lets runs with different rebalance periods or
    starting days be drawn together or averaged.

    Parameters
    ----------
    weights : ActivePortfolioSchema
        Active weights on any rebalance grid, with ``date``, ``barrid`` and
        ``active_weight``.
    dates : pl.Series, optional
        The daily axis. Defaults to the trading calendar from the first to the
        last formation date.

    Returns
    -------
    pl.DataFrame
        ``date`` and ``cumulative_log_return`` (0 at the first date), one row
        per date of the axis.

    Examples
    --------
    >>> import sf_quant.performance as sfp
    >>> path = sfp.generate_cumulative_log_returns(weights)   # doctest: +SKIP
    """
    returns = generate_active_returns_from_weights(weights)
    formation = weights["date"].unique().sort()
    if dates is None:
        dates = load_trading_days(formation.min(), formation.max())

    ends = pl.DataFrame({"date": formation, "end": formation.shift(-1)})
    marks = (
        returns.sort("date")
        .join(ends, on="date")
        .select(
            pl.col("end").alias("date"),
            pl.col("return").log1p().cum_sum().alias("cumulative_log_return"),
        )
    )
    return (
        pl.DataFrame({"date": dates.unique().sort()})
        .join(marks, on="date", how="left")
        .with_columns(pl.col("cumulative_log_return").forward_fill().fill_null(0.0))
    )
