"""From raw signal scores to Grinold-Kahn alphas, and what orthogonalization costs.

For a signal (or composite) allowed to keep the factor set A:

- Its score is standardized, then residualized against every Barra factor
  outside A, giving Z.
- Its target return is the Barra specific return plus the returns of the
  factors in A: X_i = u_i + sum_{j in A} B_ij f_j, with risk
  sigma_i^2 = delta_i + (B_A F_AA B_A')_ii. With A empty this is exactly the
  Barra specific return and specific risk.
- IC is the cross-sectional correlation of Z with X / sigma over the forecast
  window, and alpha = IC * sigma * Z * sqrt(252 / H), annualized.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from sf_quant.data.covariance_matrix import RiskModel
from sf_quant.data.market import MarketData, rebalance_dates, trading_days
from sf_quant.optimizer.mandate import active_variance, benchmark_betas, equality_direction

from .base import ALL_FACTORS, NO_FACTORS, Constraints, Signal

HORIZON = {"monthly": 21, "weekly": 5, "daily": 1}


@dataclass(frozen=True)
class AlphaPanel:
    """Alphas plus the factor allowances and constraints the optimizer must honor."""

    alphas: pl.DataFrame  # date, barrid, alpha (annualized, decimal)
    allowed_factors: frozenset[str] = NO_FACTORS
    constraints: Constraints = field(default_factory=Constraints)
    diagnostics: pl.DataFrame | None = None  # per date (and signal): IC, weight, ...


@dataclass(frozen=True)
class Context:
    """Everything the pipeline needs about one formation date."""

    date: dt.date
    index: int  # position in the trading calendar
    rm: RiskModel  # benchmark members that have risk data
    benchmark: np.ndarray  # benchmark weights aligned to rm.ids

    def sigma(self, allowed: frozenset[str]) -> np.ndarray:
        """Risk of the target return: specific risk plus the allowed factors' risk."""
        cols = [k for k, name in enumerate(self.rm.factor_names) if name in allowed]
        B = self.rm.factor_exposures[:, cols]
        F = self.rm.factor_covariance[np.ix_(cols, cols)]
        return np.sqrt(self.rm.specific_variance + np.einsum("ij,jk,ik->i", B, F, B))

    def against(self, allowed: frozenset[str]) -> list[int]:
        return [k for k, name in enumerate(self.rm.factor_names) if name not in allowed]


def contexts(data: MarketData, start: dt.date, end: dt.date, rebalance: str) -> tuple[list[dt.date], list[Context]]:
    """The trading calendar and one Context per formation date in [start, end]."""
    days = trading_days(data, start, end)
    dates = rebalance_dates(days, rebalance)
    position = {d: i for i, d in enumerate(days)}
    bench = data.benchmark_weights(dates[0], dates[-1]).filter(pl.col("date").is_in(dates))
    out = []
    for (date_,), members in sorted(bench.partition_by("date", as_dict=True).items()):
        members = members.sort("barrid")
        rm = data.risk_model(date_, members["barrid"].to_list())
        weights = dict(zip(members["barrid"], members["weight"]))
        out.append(Context(date_, position[date_], rm, np.array([weights[x] for x in rm.ids])))
    return days, out


def universe_frame(ctxs: list[Context]) -> pl.DataFrame:
    return pl.concat(pl.DataFrame({"date": c.date, "barrid": c.rm.ids}) for c in ctxs)


def standardize(scores: pl.DataFrame) -> pl.DataFrame:
    """Per date: winsorize at the 1st/99th percentiles, then z-score."""
    s = pl.col("score")
    clipped = s.clip(s.quantile(0.01).over("date"), s.quantile(0.99).over("date"))
    return scores.with_columns(clipped.alias("score")).with_columns(
        ((s - s.mean().over("date")) / s.std().over("date")).alias("z")
    ).select("date", "barrid", "z")


def residualize(z: np.ndarray, B: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Weighted least-squares residual of z on the columns of B, rescaled to
    unit standard deviation (not recentered).
    """
    scale = z.std()
    if B.shape[1]:
        root = np.sqrt(weights)
        coef = np.linalg.lstsq(B * root[:, None], z * root, rcond=None)[0]
        z = z - B @ coef
    std = z.std()
    # Nothing left: the score was (almost) entirely the removed factors.
    return z / std if std > 1e-8 * scale else np.zeros_like(z)


def residual_scores(
    scores: pl.DataFrame, ctxs: list[Context], allowed: frozenset[str], weight_allowed: frozenset[str] | None = None
) -> pl.DataFrame:
    """
    Standardized scores residualized against every factor outside ``allowed``.

    WLS weights are sigma_i / delta_i, with sigma for ``weight_allowed``
    (default ``allowed``): the weighting under which the unconstrained
    optimum V^-1 alpha has zero exposure to the removed factors.
    """
    weight_allowed = allowed if weight_allowed is None else weight_allowed
    z = standardize(scores)
    out = []
    for ctx in ctxs:
        index = {x: i for i, x in enumerate(ctx.rm.ids)}
        day = z.filter(pl.col("date") == ctx.date, pl.col("barrid").is_in(ctx.rm.ids)).drop_nulls("z")
        if day.height < 3:
            continue
        rows = np.array([index[x] for x in day["barrid"]])
        weights = ctx.sigma(weight_allowed)[rows] / ctx.rm.specific_variance[rows]
        B = ctx.rm.factor_exposures[np.ix_(rows, ctx.against(allowed))]
        out.append(day.with_columns(z=pl.Series(residualize(day["z"].to_numpy(), B, weights))))
    return pl.concat(out)


def target_returns(
    data: MarketData, days: list[dt.date], ctxs: list[Context], horizon: int, lag: int, allowed: frozenset[str]
) -> pl.DataFrame:
    """
    Risk-adjusted target X / sigma over each formation date's forecast window,
    the ``horizon`` trading days after execution (formation + ``lag``).
    Dates whose window runs past the calendar are omitted.
    """
    windows = {c.date: days[c.index + lag + 1 : c.index + lag + 1 + horizon] for c in ctxs}
    windows = {d: w for d, w in windows.items() if len(w) == horizon}
    if not windows:
        return pl.DataFrame(schema={"date": pl.Date, "barrid": pl.String, "target": pl.Float64})

    first, last = min(w[0] for w in windows.values()), max(w[-1] for w in windows.values())
    ids = sorted({x for c in ctxs for x in c.rm.ids})
    u = data.specific_returns(first, last, ids).sort("date")
    f = data.factor_returns(first, last).sort("date")

    out = []
    for ctx in ctxs:
        window = windows.get(ctx.date)
        if window is None:
            continue
        u_sum = (
            _slice(u, window[0], window[-1])
            .group_by("barrid")
            .agg(pl.col("specific_return").sum(), pl.len().alias("n"))
            .filter(pl.col("n") == horizon)
        )
        index = {x: i for i, x in enumerate(ctx.rm.ids)}
        u_sum = u_sum.filter(pl.col("barrid").is_in(ctx.rm.ids))
        rows = np.array([index[x] for x in u_sum["barrid"]], dtype=int)
        cols = [k for k, name in enumerate(ctx.rm.factor_names) if name in allowed]
        x = u_sum["specific_return"].to_numpy()
        if cols:
            f_sum = _slice(f, window[0], window[-1]).select(ctx.rm.factor_names[k] for k in cols).sum().to_numpy().ravel()
            x = x + ctx.rm.factor_exposures[np.ix_(rows, cols)] @ f_sum
        out.append(pl.DataFrame({"date": ctx.date, "barrid": u_sum["barrid"], "target": x / ctx.sigma(allowed)[rows]}))
    return pl.concat(out)


def estimate_ic(z: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Per-date cross-sectional correlation of ``z`` with ``target``: ``date``, ``ic``, ``n``."""
    return (
        z.join(targets, on=["date", "barrid"], how="inner")
        .group_by("date")
        .agg(pl.corr("z", "target").alias("ic"), pl.len().alias("n"))
        .sort("date")
    )


def build_alpha(
    signal: Signal,
    data: MarketData,
    start: dt.date,
    end: dt.date,
    rebalance: str = "monthly",
    ic: float = 0.05,
) -> AlphaPanel:
    """
    Single-signal alphas with a fixed IC: alpha = IC * sigma * Z * sqrt(252 / H).
    Under a fixed risk target the IC scales reported expected return, not weights.
    """
    horizon = HORIZON[rebalance]
    _, ctxs = contexts(data, start, end, rebalance)
    z = residual_scores(signal.compute(data, universe_frame(ctxs)), ctxs, signal.allowed_factors)
    sigmas = pl.concat(
        pl.DataFrame({"date": c.date, "barrid": c.rm.ids, "sigma": c.sigma(signal.allowed_factors)}) for c in ctxs
    )
    alphas = z.join(sigmas, on=["date", "barrid"]).select(
        "date", "barrid", (ic * pl.col("sigma") * pl.col("z") * np.sqrt(252 / horizon)).alias("alpha")
    )
    return AlphaPanel(alphas, signal.allowed_factors, signal.constraints)


@dataclass(frozen=True)
class FactorLoss:
    """What a signal's raw score earns from each Barra factor, and its IC before and after residualizing."""

    table: pl.DataFrame  # factor, allowed, annual_return, t_stat, share
    raw_ic: float  # raw score vs. specific + all factor returns
    residual_ic: float  # residualized score vs. specific + allowed factor returns


def factor_loss(
    signal: Signal,
    data: MarketData,
    start: dt.date,
    end: dt.date,
    rebalance: str = "monthly",
    lag: int = 1,
    target_active_risk: float = 0.05,
) -> FactorLoss:
    """
    Split the raw signal's return into each Barra factor's part plus the specific part.

    Each formation date, the raw (not residualized) score defines alpha =
    specific risk * Z and the mandate's closed-form direction (sum(h) = 0,
    beta'h = 0, no long-only), scaled to ``target_active_risk``. Over the
    forecast window its return is sum_k (B'h)_k f_k plus h'u. The factor
    rows show what orthogonalizing against each factor gives up; ``share``
    is relative to the total.
    """
    horizon = HORIZON[rebalance]
    days, ctxs = contexts(data, start, end, rebalance)
    scores = signal.compute(data, universe_frame(ctxs))
    raw = standardize(scores)
    names = ctxs[0].rm.factor_names

    windows = {c.date: days[c.index + lag + 1 : c.index + lag + 1 + horizon] for c in ctxs}
    ctxs_with_window = [c for c in ctxs if len(windows[c.date]) == horizon]
    first, last = windows[ctxs_with_window[0].date][0], windows[ctxs_with_window[-1].date][-1]
    u_all = data.specific_returns(first, last, sorted({x for c in ctxs for x in c.rm.ids})).sort("date")
    f_all = data.factor_returns(first, last).sort("date")

    rows = []
    for ctx in ctxs_with_window:
        window = windows[ctx.date]
        rm = ctx.rm
        z = dict(zip(*raw.filter(pl.col("date") == ctx.date).select("barrid", "z").to_dict(as_series=False).values()))
        alpha = np.sqrt(rm.specific_variance) * np.array([z.get(x, 0.0) for x in rm.ids])
        if not np.any(alpha):
            continue
        beta = benchmark_betas(rm.factor_exposures, rm.factor_covariance, rm.specific_variance, ctx.benchmark / ctx.benchmark.sum())
        d = equality_direction(alpha, rm.factor_exposures, rm.factor_covariance, rm.specific_variance, np.vstack([np.ones_like(beta), beta]))
        h = d * target_active_risk / np.sqrt(active_variance(d, rm.factor_exposures, rm.factor_covariance, rm.specific_variance))

        u = _slice(u_all, window[0], window[-1]).group_by("barrid").agg(pl.col("specific_return").sum())
        u = dict(zip(u["barrid"], u["specific_return"]))
        f = _slice(f_all, window[0], window[-1]).select(names).sum().to_numpy().ravel()
        contributions = (rm.factor_exposures.T @ h) * f
        rows.append([*contributions, h @ np.array([u.get(x, 0.0) for x in rm.ids])])

    m = np.array(rows) * (252 / horizon)
    mean, std = m.mean(axis=0), m.std(axis=0, ddof=1)
    total = mean.sum()
    table = pl.DataFrame(
        {
            "factor": [*names, "specific"],
            "allowed": [n in signal.allowed_factors for n in names] + [True],
            "annual_return": mean,
            "t_stat": mean / std * np.sqrt(len(m)),
            "share": mean / total if total else np.full(len(mean), np.nan),
        }
    ).sort(pl.col("annual_return").abs(), descending=True)

    all_factors = frozenset(names) | ALL_FACTORS
    raw_target = target_returns(data, days, ctxs, horizon, lag, all_factors)
    residual = residual_scores(scores, ctxs, signal.allowed_factors)
    own_target = target_returns(data, days, ctxs, horizon, lag, signal.allowed_factors)
    return FactorLoss(
        table=table,
        raw_ic=float(estimate_ic(raw, raw_target)["ic"].mean()),
        residual_ic=float(estimate_ic(residual, own_target)["ic"].mean()),
    )


def _slice(df: pl.DataFrame, start: dt.date, end: dt.date) -> pl.DataFrame:
    """Rows of a date-sorted frame with start <= date <= end."""
    lo = df["date"].search_sorted(start, side="left")
    hi = df["date"].search_sorted(end, side="right")
    return df.slice(lo, hi - lo)
