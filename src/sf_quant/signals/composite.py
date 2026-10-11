"""Combine signals into one alpha.

Each formation date t, using only information available by t:

1. IC_k = the mean of signal k's past per-date ICs whose forecast windows
   ended by t, shrunk toward zero by n / (n + n0).
2. R = the mean of past cross-sectional correlation matrices of the Z
   scores, shrunk toward the identity.
3. v = R^-1 IC (the signal analog of max-Sharpe weights), composite score
   Z_c = v'Z / sqrt(v'Rv), composite IC_c = sqrt(IC' R^-1 IC), and
   alpha = IC_c * sigma * Z_c * sqrt(252 / H) = sigma * v'Z * sqrt(252 / H).

Signals are not orthogonalized against each other: R^-1 removes their
overlap, so the result doesn't depend on order. All ICs and sigma use one
target, specific returns plus the union of the signals' allowed factors.
"""

from __future__ import annotations

import datetime as dt
from functools import reduce

import numpy as np
import polars as pl

from sf_quant.data.market import MarketData, cached

from .alpha import HORIZON, AlphaPanel, contexts, estimate_ic, residual_scores, target_returns, universe_frame
from .base import Constraints, Signal


def build_composite(
    signals: list[Signal],
    data: MarketData,
    start: dt.date,
    end: dt.date,
    rebalance: str = "monthly",
    lag: int = 1,
    ic_shrink_obs: float = 24,
    corr_shrink: float = 0.5,
) -> AlphaPanel:
    """
    Composite alphas from estimated, shrunk ICs and signal correlations.

    Parameters
    ----------
    signals : list[Signal]
        The signals to combine, in any order.
    lag : int
        Execution lag in trading days; the forecast window starts after it.
    ic_shrink_obs : float
        n0 in the IC shrinkage n / (n + n0). With no history, IC is zero and
        so is alpha; it reaches half the sample IC after n0 formation dates.
    corr_shrink : float
        Weight on the identity in R = (1 - c) R_hat + c I.

    Returns
    -------
    AlphaPanel
        Alphas, the union of allowed factors, combined constraints, and
        per-date diagnostics: each signal's IC, weight, and marginal IC
        (IC_k minus what the other signals already predict; negative means
        the signal hurts the composite).
    """
    horizon = HORIZON[rebalance]
    labels = _labels(signals)
    allowed = frozenset().union(*(s.allowed_factors for s in signals))
    constraints = reduce(Constraints.combine, (s.constraints for s in signals), Constraints())

    data = cached(data)
    days, ctxs = contexts(data, start, end, rebalance)
    universe = universe_frame(ctxs)
    z = {
        label: residual_scores(s.compute(data, universe), ctxs, s.allowed_factors, weight_allowed=allowed)
        for label, s in zip(labels, signals)
    }
    targets = target_returns(data, days, ctxs, horizon, lag, allowed)
    ic_by_date = {label: dict(estimate_ic(z[label], targets).select("date", "ic").iter_rows()) for label in labels}

    K = len(signals)
    corr_sum, n_corr = np.zeros((K, K)), 0
    alphas, diagnostics = [], []
    for t, ctx in enumerate(ctxs):
        # IC from windows that ended by this formation date.
        done = [c.date for c in ctxs[:t] if c.index + lag + horizon <= ctx.index]
        ic = np.zeros(K)
        for k, label in enumerate(labels):
            past = [ic_by_date[label][d] for d in done if ic_by_date[label].get(d) is not None]
            if past:
                ic[k] = np.mean(past) * len(past) / (len(past) + ic_shrink_obs)

        Z = np.column_stack([_aligned(z[label], ctx.date, ctx.rm.ids) for label in labels])
        complete = ~np.isnan(Z).any(axis=1)
        if complete.sum() > K + 2:
            corr_sum += np.corrcoef(Z[complete], rowvar=False).reshape(K, K)
            n_corr += 1
        R = (1 - corr_shrink) * (corr_sum / n_corr if n_corr else np.eye(K)) + corr_shrink * np.eye(K)

        v, composite_ic = combination_weights(ic, R)
        alpha = np.sqrt(252 / horizon) * ctx.sigma(allowed) * (np.nan_to_num(Z) @ v)
        alphas.append(pl.DataFrame({"date": ctx.date, "barrid": ctx.rm.ids, "alpha": alpha}))
        for k, label in enumerate(labels):
            diagnostics.append(
                {
                    "date": ctx.date,
                    "signal": label,
                    "ic": ic[k],
                    "weight": v[k],
                    "marginal_ic": _marginal_ic(R, ic, k),
                    "composite_ic": composite_ic,
                }
            )

    return AlphaPanel(pl.concat(alphas), allowed, constraints, pl.DataFrame(diagnostics))


def combination_weights(ic: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, float]:
    """Weights v = R^-1 IC and composite IC_c = sqrt(IC' R^-1 IC)."""
    v = np.linalg.solve(R, ic)
    return v, float(np.sqrt(max(ic @ v, 0.0)))


def _marginal_ic(R: np.ndarray, ic: np.ndarray, k: int) -> float:
    """IC_k - rho_{k,c'} IC_c', where c' is the composite of the other signals."""
    others = [j for j in range(len(ic)) if j != k]
    if not others:
        return float(ic[k])
    v, ic_c = combination_weights(ic[others], R[np.ix_(others, others)])
    if ic_c == 0:
        return float(ic[k])
    # rho(k, c') = R[k, others] v / IC_c', so rho * IC_c' = R[k, others] v.
    return float(ic[k] - R[k, others] @ v)


def _aligned(z: pl.DataFrame, date_: dt.date, ids: list[str]) -> np.ndarray:
    day = dict(z.filter(pl.col("date") == date_).select("barrid", "z").iter_rows())
    return np.array([day.get(x, np.nan) for x in ids], dtype=float)


def _labels(signals: list[Signal]) -> list[str]:
    labels, seen = [], {}
    for s in signals:
        seen[s.name] = seen.get(s.name, 0) + 1
        labels.append(s.name if seen[s.name] == 1 else f"{s.name}_{seen[s.name]}")
    return labels
