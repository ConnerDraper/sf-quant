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

    grid_t = grid["t"].implode()
    keyed = data.join(calendar.lazy(), on="date", how="inner")

    # The alpha observed on ordinal t is used on rebalance ordinal t + lag, if
    # that is a rebalance date. Rebalance dates inside the first `lag` trading
    # days of the calendar have no source and drop out. Mapping by arithmetic
    # rather than through a join on a date table matters: given that extra
    # join, polars' optimizer reordered the plan into a join on barrid alone,
    # pairing every date of a name with every other, which needed ~15 GB per
    # million rows.
    alphas = (
        keyed.select((pl.col("t") + lag).alias("t"), "barrid", "alpha")
        .drop_nulls("alpha")
        .filter(pl.col("t").is_in(grid_t))
    )

    collapsed = (
        keyed.filter(pl.col("t").is_in(grid_t))
        .drop("alpha")
        .join(alphas, on=["t", "barrid"], how="inner")
        .drop("t")
    )

    if "benchmark_weight" in data.collect_schema().names():
        # A name outside the benchmark has b = 0. Left-joined benchmark weights
        # arrive as null, which would reach the optimizer as NaN.
        collapsed = collapsed.with_columns(pl.col("benchmark_weight").fill_null(0.0))

    return collapsed


def rebalance_phases(
    weights: pl.DataFrame, *, dates: pl.Series, rebalance_period: int
) -> list[pl.DataFrame]:
    """
    Every starting day of a rebalance period, derived from daily weights.

    Each date's solve depends only on that date's alpha, risk model, benchmark
    and beta, never on what was held the day before. So the portfolio a
    ``rebalance_period=N`` run holds on a rebalance date *is* the daily
    portfolio on that date, and an N-day run needs no solves of its own: keep
    the daily weights every N days.

    Which day the grid starts on matters (different trades, different
    Sharpe), and one start gives few holding periods at large N. So this
    returns all N phases. Phase p keeps ``rebalance_dates(dates[p:], N)``,
    exactly the grid a direct run started at ``dates[p]`` would solve, as
    anchored by :func:`collapse`.

    Parameters
    ----------
    weights : pl.DataFrame
        Daily (``rebalance_period=1``) weights from
        :func:`sf_quant.backtester.backtest`.
    dates : pl.Series
        The full trading-day calendar the daily run was built on.
    rebalance_period : int
        Trading days between rebalances.

    Returns
    -------
    list[pl.DataFrame]
        ``rebalance_period`` frames, phase 0 first, each with the columns of
        ``weights``. Average their statistics to summarize the period.

    Examples
    --------
    >>> import sf_quant.backtester as sfb
    >>> import sf_quant.performance as sfp
    >>> phases = sfb.rebalance_phases(daily, dates=dates, rebalance_period=21)   # doctest: +SKIP
    >>> sharpes = [sfp.generate_returns_summary_table(  # doctest: +SKIP
    ...     sfp.generate_active_returns_from_weights(w))["Sharpe"].item() for w in phases]
    """
    calendar = dates.unique().sort()
    return [
        weights.filter(pl.col("date").is_in(rebalance_dates(calendar[phase:], rebalance_period).implode()))
        for phase in range(rebalance_period)
    ]
