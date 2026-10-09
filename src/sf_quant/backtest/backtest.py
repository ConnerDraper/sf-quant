"""Walk-forward backtest of the Silver Fund mandate.

1. Pick formation dates from the trading calendar. Each trades
   ``execution_lag_days`` later at the close.
2. Solve the mandate problem on each formation date. Dates are
   independent, so they can be solved in parallel.
3. Account for daily returns sequentially (:mod:`sf_backtester.accounting`).

The run refuses to produce numbers when an invariant breaks.
"""

from __future__ import annotations

import datetime as dt
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from typing import NamedTuple

import numpy as np
import polars as pl
from sf_quant.optimizer import active_variance, benchmark_betas, mandate_optimizer

from sf_backtester.accounting import account
from sf_backtester.config import BacktestConfig, ConfigError
from sf_backtester.data import MarketData, SfQuantMarketData
from sf_backtester.results import BacktestResult, build_result

BENCHMARK_VOL_RANGE = (0.08, 0.40)  # sqrt(b'Vb) outside this: risk model isn't annualized decimal
TOLERANCE = 1e-6
DUST = 1e-9  # fund weights below this are not held


class InvariantError(RuntimeError):
    """An input or solution violates a backtester invariant."""


class Rebalance(NamedTuple):
    formation: dt.date
    execution: dt.date


@dataclass
class Solved:
    """Outcome of one formation date's solve."""

    rebalance: Rebalance
    ok: bool
    weights: pl.DataFrame  # barrid, alpha, benchmark_weight, active_weight, weight
    diagnostics: dict
    factor_names: list[str]
    active_factor_exposures: np.ndarray
    risk_ids: list[str]  # names in the risk model, rows of factor_exposures
    factor_exposures: np.ndarray

    @property
    def holdings(self) -> dict[str, float]:
        """Target fund weights, dust removed, summing to one."""
        held = self.weights.filter(pl.col("weight") > DUST)
        return dict(zip(held["barrid"], held["weight"] / held["weight"].sum()))


def run_backtest(
    alphas: pl.DataFrame, config: BacktestConfig, market_data: MarketData | None = None
) -> BacktestResult:
    """
    Backtest an alpha panel under the mandate.

    Parameters
    ----------
    alphas : pl.DataFrame
        ``date`` (formation date), ``barrid``, ``alpha`` (annualized decimal).
        Only rows on formation dates are used; universe members without an
        alpha get zero.
    config : BacktestConfig
    market_data : MarketData, optional
        Defaults to :class:`~sf_backtester.data.SfQuantMarketData`.
    """
    md = market_data if market_data is not None else SfQuantMarketData()
    alphas = _validate_alphas(alphas)
    alphas, end, holdout_used = _apply_holdout(config, alphas)

    benchmark = md.benchmark_weights(config.start, end).sort("date")
    days = benchmark["date"].unique().sort().to_list()
    rebalances = schedule(days, config.rebalance, config.execution_lag_days)
    if not rebalances:
        raise ConfigError(f"No rebalance between {config.start} and {end} has a following trading day.")

    formations = [r.formation for r in rebalances]
    benchmarks = _by_date(benchmark, formations, "weight")
    alpha_maps = _by_date(alphas, formations, "alpha")
    price_maps = _by_date(md.prices(formations[0], formations[-1]), formations, "price")
    _check_coverage(benchmarks, alpha_maps, config.min_alpha_coverage)

    args = (
        rebalances,
        [benchmarks[f] for f in formations],
        [alpha_maps.get(f, {}) for f in formations],
        [price_maps.get(f, {}) for f in formations],
        repeat(md),
        repeat(config),
    )
    if config.n_workers == 1:
        solved = list(map(solve, *args))
    else:
        with ProcessPoolExecutor(max_workers=config.n_workers) as pool:
            solved = list(pool.map(solve, *args))

    failed = [s for s in solved if not s.ok]
    if failed and config.on_failure == "raise":
        raise InvariantError(
            f"{len(failed)} rebalance(s) failed to solve, e.g. {failed[0].rebalance.formation} "
            f"({failed[0].diagnostics['solver_status']})."
        )
    # hold_previous: a failed date doesn't trade, so the previous holdings keep drifting.
    returns, turnover, n_missing = account([s for s in solved if s.ok], days, benchmark, md, config.bps_grid)
    return build_result(solved, returns, turnover, n_missing, config, days[-1], holdout_used)


def schedule(days: list[dt.date], frequency: str, lag: int) -> list[Rebalance]:
    """
    Formation dates are the last trading day of each month or ISO week, or
    every day. Each executes ``lag`` trading days later; formations with no
    trading day left to earn a return are dropped.
    """
    period = {
        "daily": lambda d: d,
        "weekly": lambda d: d.isocalendar()[:2],
        "monthly": lambda d: (d.year, d.month),
    }[frequency]
    rebalances = []
    for i, day in enumerate(days):
        last_of_period = i + 1 == len(days) or period(days[i + 1]) != period(day)
        if last_of_period and i + lag + 1 < len(days):
            rebalances.append(Rebalance(day, days[i + lag]))
    return rebalances


def solve(
    rebalance: Rebalance,
    benchmark: dict[str, float],
    alphas: dict[str, float],
    prices: dict[str, float],
    md: MarketData,
    config: BacktestConfig,
) -> Solved:
    """Solve the mandate problem for one formation date over its benchmark universe."""
    date_ = rebalance.formation
    universe = sorted(benchmark)
    b_all = np.array([benchmark[x] for x in universe])
    b_all /= b_all.sum()
    alpha_all = np.array([alphas.get(x, 0.0) for x in universe])

    # Names without risk data are left out of the problem, which pins them at benchmark weight.
    rm = md.risk_model(date_, universe)
    rows = np.searchsorted(universe, rm.ids)
    B, F, delta = rm.factor_exposures, rm.factor_covariance, rm.specific_variance
    b, alpha = b_all[rows], alpha_all[rows]

    benchmark_vol = float(np.sqrt(active_variance(b / b.sum(), B, F, delta)))
    low, high = BENCHMARK_VOL_RANGE
    if not low <= benchmark_vol <= high:
        raise InvariantError(
            f"{date_}: benchmark volatility sqrt(b'Vb) = {benchmark_vol:.4f} is outside [{low}, {high}]. "
            "The risk model is probably not in annualized decimal units."
        )

    unknown = set(config.factor_bounds) - set(rm.factor_names)
    if unknown:
        raise ConfigError(f"factor_bounds names unknown factor(s): {sorted(unknown)}.")
    bounds = {rm.factor_names.index(name): cap for name, cap in config.factor_bounds.items()}

    result = mandate_optimizer(rm.ids, alpha, B, F, delta, b, config.active_risk_target, bounds)
    h_all = np.zeros(len(universe))
    if result.ok:
        _check_solution(date_, result, b, benchmark_betas(B, F, delta, b), config.active_risk_target)
        h_all[rows] = result.active_weights
    w_all = np.where(b_all + h_all > DUST, b_all + h_all, 0.0)

    held = w_all > 0
    dollars = w_all * config.aum
    price = np.array([prices.get(x, np.nan) for x in universe])
    sigma = np.sqrt(delta)
    diagnostics = {
        "formation_date": date_,
        "execution_date": rebalance.execution,
        "solver_status": result.status,
        "solve_time": result.solve_time,
        "ex_ante_te": result.ex_ante_te,
        "te_shortfall": result.te_shortfall,
        "active_beta": result.active_beta,
        "expected_active_return": result.expected_active_return,
        "ex_ante_ir": result.expected_active_return / result.ex_ante_te if result.ex_ante_te > 0 else np.nan,
        # Clarke–de Silva–Thorley transfer coefficient: corr(alpha / sigma, h * sigma).
        "transfer_coefficient": _corr(alpha / sigma, result.active_weights * sigma),
        "n_at_long_only_bound": result.n_at_long_only_bound,
        "n_holdings": int(held.sum()),
        "alpha_coverage": len(alphas.keys() & benchmark.keys()) / len(universe),
        "n_universe": len(universe),
        "n_missing_risk": len(rm.missing_ids),
        "benchmark_vol": benchmark_vol,
        "n_positions_below_min": int((held & (dollars < config.min_position_dollars)).sum()),
        "n_positions_under_one_share": int((held & (dollars < price)).sum()),
    }

    return Solved(
        rebalance=rebalance,
        ok=result.ok,
        weights=pl.DataFrame(
            {
                "barrid": universe,
                "alpha": alpha_all,
                "benchmark_weight": b_all,
                "active_weight": h_all,
                "weight": w_all,
            }
        ),
        diagnostics=diagnostics,
        factor_names=list(rm.factor_names),
        active_factor_exposures=result.active_factor_exposures,
        risk_ids=list(rm.ids),
        factor_exposures=B.astype(np.float32),
    )


def _validate_alphas(alphas: pl.DataFrame) -> pl.DataFrame:
    missing = {"date", "barrid", "alpha"} - set(alphas.columns)
    if missing:
        raise InvariantError(f"Alpha panel is missing column(s): {sorted(missing)}.")
    alphas = alphas.select(
        pl.col("date").cast(pl.Date), pl.col("barrid").cast(pl.String), pl.col("alpha").cast(pl.Float64)
    )
    if alphas["alpha"].is_null().any() or not alphas["alpha"].is_finite().all():
        raise InvariantError("Alpha panel contains null or non-finite alphas.")
    if alphas.select("date", "barrid").is_duplicated().any():
        raise InvariantError("Alpha panel contains duplicate (date, barrid) rows.")
    return alphas


def _apply_holdout(config: BacktestConfig, alphas: pl.DataFrame) -> tuple[pl.DataFrame, dt.date, bool]:
    """Stop the run before the holdout unless it is explicitly unlocked."""
    holdout, end = config.holdout_start, config.end
    if holdout is None:
        return alphas, end, False
    if config.unlock_holdout:
        return alphas, end, end >= holdout

    n_dropped = alphas.filter(pl.col("date") >= holdout).height
    if end >= holdout or n_dropped:
        warnings.warn(
            f"Stopping at the holdout ({holdout}): dropped {n_dropped} alpha rows. "
            "Set unlock_holdout only for the vote-time run.",
            stacklevel=3,
        )
    return alphas.filter(pl.col("date") < holdout), min(end, holdout - dt.timedelta(days=1)), False


def _by_date(df: pl.DataFrame, dates: list[dt.date], value: str) -> dict[dt.date, dict[str, float]]:
    """{date: {barrid: value}} for the given dates."""
    parts = df.filter(pl.col("date").is_in(dates)).partition_by("date", as_dict=True)
    return {date_: dict(zip(part["barrid"], part[value])) for (date_,), part in parts.items()}


def _check_coverage(
    benchmarks: dict[dt.date, dict[str, float]], alphas: dict[dt.date, dict[str, float]], minimum: float
) -> None:
    coverage = {d: len(alphas.get(d, {}).keys() & b.keys()) / len(b) for d, b in benchmarks.items()}
    low = sorted((c, d) for d, c in coverage.items() if c < minimum)
    if low:
        examples = ", ".join(f"{d} ({c:.0%})" for c, d in low[:5])
        raise InvariantError(
            f"Alpha coverage is below {minimum:.0%} of the universe on {len(low)} formation date(s), e.g. {examples}."
        )


def _check_solution(date_: dt.date, result, b: np.ndarray, beta: np.ndarray, target: float) -> None:
    h = result.active_weights
    problems = [
        f"sum(h) = {h.sum():.2e}" if abs(h.sum()) > TOLERANCE else "",
        f"beta'h = {beta @ h:.2e}" if abs(beta @ h) > TOLERANCE else "",
        f"min(w_b + h) = {(b + h).min():.2e}" if (b + h).min() < -TOLERANCE else "",
        f"ex-ante TE {result.ex_ante_te:.6f} > {target}" if result.ex_ante_te > target + TOLERANCE else "",
    ]
    problems = [p for p in problems if p]
    if problems:
        raise InvariantError(f"{date_}: solution violates the mandate: " + "; ".join(problems) + ".")


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
