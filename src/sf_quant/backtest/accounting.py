"""Sequential daily return accounting.

Timing: holdings traded at the close of an execution date earn returns from
the next trading day through the next execution date, inclusive, drifting
with prices in between. The benchmark return on day d uses benchmark weights
from the previous close. Costs are charged on realized turnover, never in the
optimizer; the initial build from cash is free.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from sf_backtester.data import MarketData


def net_column(bps: float) -> str:
    return f"net_active_{bps:g}bps"


def account(
    periods: list,
    days: list[dt.date],
    benchmark: pl.DataFrame,
    md: MarketData,
    bps_grid: list[float],
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """
    Account for the executed rebalances in ``periods`` (successful solves,
    in execution order).

    Returns
    -------
    returns : pl.DataFrame
        Daily ``portfolio``, ``benchmark``, ``active``, ``active_factor``,
        ``active_specific``, and ``net_active_<bps>bps`` columns.
    turnover : pl.DataFrame
        ``execution_date``, ``turnover`` (sum |w_target - w_drifted|), ``initial``.
    n_missing : int
        Held name-days without a return (the position is frozen that day).
    """
    if not periods:
        raise RuntimeError("No rebalance solved successfully; nothing to account for.")

    first = periods[0].rebalance.execution
    names = set(benchmark.filter(pl.col("date") >= first)["barrid"].unique())
    names = sorted(names.union(*(p.holdings for p in periods)))
    returns = md.asset_returns(first, days[-1], names)
    factor_returns = md.factor_returns(first, days[-1])

    position = {day: i for i, day in enumerate(days)}
    rows, turnover_rows, n_missing, drifted = [], [], 0, None

    for k, period in enumerate(periods):
        execution = period.rebalance.execution
        stop = periods[k + 1].rebalance.execution if k + 1 < len(periods) else days[-1]
        earn_days = days[position[execution] + 1 : position[stop] + 1]
        prev_days = days[position[execution] : position[stop]]  # each earn day's previous close

        target = period.holdings
        turnover_rows.append(
            {"execution_date": execution, "turnover": _turnover(drifted, target), "initial": drifted is None}
        )

        # Columns: everything held or in the benchmark during this period.
        bench = benchmark.filter(pl.col("date").is_between(prev_days[0], prev_days[-1]))
        col = {name: i for i, name in enumerate(sorted(set(target) | set(bench["barrid"])))}
        R, present = _matrix(returns, earn_days, col, "return")
        W_bench, _ = _matrix(bench, prev_days, col, "weight")
        exposures = np.zeros((len(col), len(period.factor_names)))
        for i, name in enumerate(period.risk_ids):
            exposures[col[name]] = period.factor_exposures[i]
        F_returns = (
            pl.DataFrame({"date": earn_days})
            .join(factor_returns, on="date", how="left")
            .select(period.factor_names)
            .fill_null(0.0)
            .to_numpy()
        )

        value = np.zeros(len(col))
        for name, weight in target.items():
            value[col[name]] = weight

        for j, day in enumerate(earn_days):
            w = value / value.sum()
            n_missing += int(((w > 0) & ~present[j]).sum())
            portfolio, bench_return = w @ R[j], W_bench[j] @ R[j]
            active_factor = (exposures.T @ (w - W_bench[j])) @ F_returns[j]
            rows.append(
                {
                    "date": day,
                    "portfolio": portfolio,
                    "benchmark": bench_return,
                    "active": portfolio - bench_return,
                    "active_factor": active_factor,
                    "active_specific": portfolio - bench_return - active_factor,
                }
            )
            value = value * (1 + R[j])

        final = value / value.sum()
        drifted = {name: final[i] for name, i in col.items() if final[i] > 0}

    turnover = pl.DataFrame(turnover_rows)
    traded = turnover.filter(~pl.col("initial")).select(
        pl.col("execution_date").alias("date"), pl.col("turnover").alias("_traded")
    )
    daily = (
        pl.DataFrame(rows)
        .join(traded, on="date", how="left")
        .with_columns(pl.col("_traded").fill_null(0.0))
        .with_columns((pl.col("active") - bps / 1e4 * pl.col("_traded")).alias(net_column(bps)) for bps in bps_grid)
        .drop("_traded")
    )
    return daily, turnover, n_missing


def _turnover(drifted: dict[str, float] | None, target: dict[str, float]) -> float:
    if drifted is None:
        return float(sum(target.values()))
    return float(sum(abs(target.get(n, 0.0) - drifted.get(n, 0.0)) for n in drifted.keys() | target.keys()))


def _matrix(
    df: pl.DataFrame, days: list[dt.date], col: dict[str, int], value: str
) -> tuple[np.ndarray, np.ndarray]:
    """Dense (days x names) matrix of ``value`` (zero where missing) and a presence mask."""
    df = df.filter(
        pl.col("date").is_in(days), pl.col("barrid").is_in(list(col)), pl.col(value).is_not_null()
    )
    row = {day: i for i, day in enumerate(days)}
    r = np.array([row[d] for d in df["date"]], dtype=int)
    c = np.array([col[b] for b in df["barrid"]], dtype=int)
    out = np.zeros((len(days), len(col)))
    present = np.zeros((len(days), len(col)), dtype=bool)
    out[r, c] = df[value].to_numpy()
    present[r, c] = True
    return out, present
