import polars as pl
from sf_quant.data.assets import load_assets
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

    period_returns = (
        daily.join(buckets, on="date", how="inner")
        .group_by("form_date", "barrid")
        .agg(pl.col("return").add(1).product().sub(1).alias("return"))
    )

    result = (
        weights.filter(
            pl.col("date").is_in(buckets["form_date"].unique().implode())
        )
        .join(
            period_returns,
            left_on=["date", "barrid"],
            right_on=["form_date", "barrid"],
            how="left",
        )
        .group_by("date")
        .agg(
            pl.col("active_weight")
            .mul(pl.col("return").fill_null(0.0))
            .sum()
            .alias("return")
        )
        .sort("date")
    )

    return PortfolioRetSchema.validate(result)


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

        - ``date`` (date): The observation date.
        - ``portfolio`` (str): Portfolio type; one of
          ``"total"``, ``"benchmark"``, or ``"active"``.
        - ``return`` (float): The weighted return for the portfolio
          on the given date.

    Notes
    -----
    - Asset returns are sourced via ``load_assets``.
    - Benchmark weights are sourced via ``load_benchmark``.
    - Returns are computed as the weighted sum of returns by portfolio.

    Examples
    --------
    >>> import polars as pl
    >>> import sf_quant.performance as sfp
    >>> import datetime as dt
    >>> weights = pl.DataFrame(
    ...     {
    ...         'date': [dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 3)],
    ...         'barrid': ['USA06Z1', 'USA0771', 'USA06Z1', 'USA0771'],
    ...         'weight': [0.5, 0.5, 0.3, 0.7]
    ...     }
    ... )
    >>> returns = sfp.generate_multi_returns_from_weights(weights)
    >>> returns
    shape: (6, 3)
    ┌────────────┬───────────┬────────────┐
    │ date       ┆ portfolio ┆ return     │
    │ ---        ┆ ---       ┆ ---        │
    │ date       ┆ str       ┆ f64        │
    ╞════════════╪═══════════╪════════════╡
    │ 2024-01-02 ┆ active    ┆ 0.015432   │
    │ 2024-01-02 ┆ benchmark ┆ 1.2047e-7  │
    │ 2024-01-02 ┆ total     ┆ 0.015432   │
    │ 2024-01-03 ┆ active    ┆ 0.020741   │
    │ 2024-01-03 ┆ benchmark ┆ 2.4094e-7  │
    │ 2024-01-03 ┆ total     ┆ 0.020741   │
    └────────────┴───────────┴────────────┘
    """
    start = weights["date"].min()
    end = weights["date"].max()

    columns = ["date", "barrid", "return"]

    returns = load_assets(start=start, end=end, in_universe=True, columns=columns)

    benchmark = load_benchmark(start=start, end=end)

    returns = returns.sort("barrid", "date")

    joined = (
        weights.join(returns, on=["date", "barrid"], how="left")
        .join(benchmark, on=["date", "barrid"], how="left", suffix="_bmk")
        .with_columns(
            pl.col("return").truediv(100),
        )
        .with_columns(pl.col("weight").sub("weight_bmk").alias("weight_act"))
        .rename({"weight": "total", "weight_bmk": "benchmark", "weight_act": "active"})
        .unpivot(
            index=["date", "barrid", "return"],
            variable_name="portfolio",
            value_name="weight",
        )
    )

    # Build aggregation expressions
    agg_exprs = []
    agg_exprs.append(pl.col("return").mul("weight").sum().alias("return"))

    result = (
        joined.group_by("date", "portfolio")
        .agg(agg_exprs)
        .sort("date", "portfolio")
    )
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

        - ``date`` (date): The observation date.
        - ``return`` (float): The weighted return for the portfolio on the given date.

    Notes
    -----
    - Asset returns are sourced via ``load_assets``.
    - Returns are computed as the weighted sum of returns by portfolio.

    Examples
    --------
    >>> import polars as pl
    >>> import sf_quant.performance as sfp
    >>> import datetime as dt
    >>> weights = pl.DataFrame(
    ...     {
    ...         'date': [dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 3)],
    ...         'barrid': ['USA06Z1', 'USA0771', 'USA06Z1', 'USA0771'],
    ...         'weight': [0.5, 0.5, 0.3, 0.7]
    ...     }
    ... )
    >>> returns = sfp.generate_returns_from_weights(weights)
    >>> returns
    shape: (2, 2)
    ┌────────────┬────────────┐
    │ date       ┆ return     │
    │ ---        ┆ ---        │
    │ date       ┆ f64        │
    ╞════════════╪════════════╡
    │ 2024-01-02 ┆ 0.015432   │
    │ 2024-01-03 ┆ 0.020741   │
    └────────────┴────────────┘
    """
    start = weights["date"].min()
    end = weights["date"].max()

    columns = ["date", "barrid", "return"]

    returns = load_assets(start=start, end=end, in_universe=True, columns=columns)

    returns = returns.sort("barrid", "date")

    joined = (
        weights.join(returns, on=["date", "barrid"], how="left")
        .with_columns(
            pl.col("return").truediv(100),
        )
    )

    result = (
        joined.group_by("date")
        .agg([
            pl.col("return").mul("weight").sum().alias("return"),
        ])
        .sort("date")
    )
    return PortfolioRetSchema.validate(result)