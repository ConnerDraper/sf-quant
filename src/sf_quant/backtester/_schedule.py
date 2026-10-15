"""Reduce a daily panel to the rebalance grid.

The backtester optimizes exactly the rows it is handed, one portfolio per
distinct date. ``collapse`` is what turns a daily panel into a coarser grid, so
that a backtest with ``rebalance_period=21`` holds each portfolio for 21
trading days rather than re-solving daily.

Collapsing, rather than solving on the grid and then expanding weights back out
over every daily date, is deliberate. For a portfolio set at :math:`t_k` and
left alone, the active return over :math:`(t_k, t_{k+1}]` is
:math:`\\sum_i h_i r_{i,period}` with :math:`r_{i,period}` the asset's
compounded return — an identity, with drift already inside the compounded
return. Marking a constant ``h`` daily instead measures a portfolio rebalanced
daily *back* to ``h``, which has the turnover of ``rebalance_period=1`` with the
signal staleness of ``rebalance_period=21`` — the exact confound a rebalance
study is meant to isolate. Collapsing also means the long-only bound ``h >= -b``
cannot be breached between rebalances, because there are no dates between them.
"""

import polars as pl


def rebalance_dates(dates: pl.Series, rebalance_period: int = 1) -> pl.Series:
    """
    Take every ``rebalance_period``-th trading day.

    The grid is anchored at ``dates[0]``, so it is a pure function of the
    calendar it is given. Pass the same full-run calendar everywhere and every
    shard of a sharded run agrees on the grid.

    Parameters
    ----------
    dates : pl.Series
        Trading-day calendar. Does not need to be sorted or unique.
    rebalance_period : int, default 1
        Trading days between rebalances. 1 means rebalance every trading day.

    Returns
    -------
    pl.Series
        The rebalance dates, sorted.

    Examples
    --------
    >>> import polars as pl, datetime as dt
    >>> from sf_quant.backtester import rebalance_dates
    >>> dates = pl.Series("date", [dt.date(2024, 1, d) for d in (2, 3, 4, 5, 8)])
    >>> rebalance_dates(dates, 2).to_list()
    [datetime.date(2024, 1, 2), datetime.date(2024, 1, 4), datetime.date(2024, 1, 8)]
    """
    if rebalance_period < 1:
        raise ValueError(f"rebalance_period must be >= 1, got {rebalance_period}")
    return dates.unique().sort().gather_every(rebalance_period)


def collapse(
    data: pl.LazyFrame,
    *,
    dates: pl.Series,
    rebalance_period: int = 1,
    lag: int = 0,
) -> pl.LazyFrame:
    """
    Reduce a daily panel to one row per (rebalance date, barrid).

    ``alpha`` is taken from ``lag`` trading days before each rebalance date;
    every other column — ``benchmark_weight``, ``predicted_beta``, and whatever
    else is present — is observed at the rebalance date itself. That is what a
    signal lag means: stale signal, current risk model and current benchmark.

    The lag is measured in trading days off ``dates``, not in rows the asset
    happens to have. An asset that drops out of the universe and returns gets
    ``null`` for the gap rather than a pre-gap value, and is then dropped —
    a positional ``shift`` would hand it a silently-stale alpha instead.

    Parameters
    ----------
    data : pl.LazyFrame
        Daily panel with at least ``date``, ``barrid`` and ``alpha``.
    dates : pl.Series
        The full trading-day calendar for the whole run. Required and
        defaultless: shard a run by filtering the *output* of this function, not
        by slicing ``data``. Slicing ``data`` first would re-anchor the grid at
        each shard boundary and drop the lag's warm-up period.
    rebalance_period : int, default 1
        Trading days between rebalances.
    lag : int, default 0
        Trading days to lag ``alpha`` by.

    Returns
    -------
    pl.LazyFrame
        One row per (rebalance date, barrid), with ``alpha`` non-null and
        ``benchmark_weight`` null-filled to 0 where present.

    Notes
    -----
    Rows are dropped when a name has no alpha ``lag`` trading days back. Those
    names get an implicit ``h = 0``, i.e. they are held at benchmark weight, so
    ``sum(h) == 0`` and ``w = b >= 0`` still hold across the full universe. The
    aggregate short capacity does shrink with ``lag``, so the per-date sum of
    ``benchmark_weight`` is worth watching to tell signal decay apart from a
    shrinking risk budget.

    Examples
    --------
    >>> import polars as pl, datetime as dt
    >>> import sf_quant.data as sfd
    >>> from sf_quant.backtester import collapse
    >>> dates = sfd.load_trading_days(dt.date(2024, 1, 1), dt.date(2024, 12, 31))
    >>> grid = collapse(
    ...     pl.scan_parquet("alphas.parquet"),
    ...     dates=dates,
    ...     rebalance_period=21,
    ...     lag=5,
    ... )
    """
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")

    calendar = pl.DataFrame({"date": dates.unique().sort()}).with_row_index("t")
    grid = calendar.filter(
        pl.col("date").is_in(rebalance_dates(dates, rebalance_period).implode())
    )

    # Ordinal of the date each rebalance date sources its alpha from. Rebalance
    # dates inside the first `lag` trading days of the calendar have no source
    # and drop out.
    sources = grid.select(
        (pl.col("t") - lag).alias("t"), pl.col("date").alias("rebalance_date")
    ).filter(pl.col("t") >= 0)

    keyed = data.join(calendar.lazy(), on="date", how="inner")

    alphas = (
        keyed.select("t", "barrid", "alpha")
        .drop_nulls("alpha")
        .join(sources.lazy(), on="t", how="inner")
        .select(pl.col("rebalance_date").alias("date"), "barrid", "alpha")
    )

    collapsed = (
        keyed.filter(pl.col("t").is_in(grid["t"].implode()))
        .drop("t", "alpha")
        .join(alphas, on=["date", "barrid"], how="inner")
    )

    if "benchmark_weight" in data.collect_schema().names():
        # A name outside the benchmark has b = 0. Left-joined benchmark weights
        # arrive as null, which would reach the optimizer as NaN.
        collapsed = collapsed.with_columns(pl.col("benchmark_weight").fill_null(0.0))

    return collapsed
