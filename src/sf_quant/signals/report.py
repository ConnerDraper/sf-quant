"""What a signal earns and risks from each Barra factor, before and after orthogonalization.

Each formation date, two portfolios are built from the signal with the
mandate's closed-form direction (sum(h) = 0, beta'h = 0, no long-only bound),
scaled to the tracking-error target:

- before: the raw standardized score, alpha = specific risk * Z.
- after: the score residualized against every factor the signal may not keep,
  alpha = sigma * Z, held neutral to those factors (the backtester's
  ``neutral`` factor control).

Both skip the long-only bound, so they differ only by the orthogonalization.
With x = B'h, a portfolio's return over the forecast window splits into
sum_k x_k f_k plus the specific part h'u. Its risk splits two ways, each
summing to 1 over the factors and specific:

- ex ante (Euler, from the risk model): x_k (F x)_k / s^2 and h' Delta h / s^2,
  with s^2 = x'Fx + h' Delta h, averaged over dates.
- realized: cov(contribution_k, total) / var(total) across formation dates.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs

from sf_quant.data.market import MarketData
from sf_quant.optimizer.mandate import _independent, active_variance, benchmark_betas, equality_direction

from .alpha import HORIZON, _slice, contexts, estimate_ic, residual_scores, standardize, target_returns, universe_frame
from .base import ALL_FACTORS, COUNTRY_FACTOR, INDUSTRY_FACTORS, STYLE_FACTORS, Signal

PORTFOLIOS = ("before", "after")


@dataclass(frozen=True)
class FactorReport:
    """A signal's return and risk by Barra factor, before and after orthogonalization."""

    signal: str
    # One row per factor plus "specific": factor, group, removed (orthogonalized
    # against), then for each portfolio p in before/after: annual_return_p,
    # t_stat_p, return_share_p, ex_ante_risk_share_p, realized_risk_share_p.
    table: pl.DataFrame
    contributions: pl.DataFrame  # date, portfolio, factor, contribution (return over the forecast window)
    raw_ic: float  # raw score vs. specific + all factor returns
    residual_ic: float  # residualized score vs. specific + allowed factor returns

    def plot(self, file_name: str | None = None) -> None:
        """
        Four panels: annualized return contribution and risk share by factor
        (before vs. after), and cumulative return by factor for each portfolio.
        Industries are summed into one bar; ``table`` has each one.
        """
        labeled = self.table.with_columns(label=_label())
        rows = labeled.group_by("label", maintain_order=True).agg(
            pl.col("removed").all().alias("all_removed"),
            pl.col("removed").any().alias("any_removed"),
            cs.matches("^(annual_return|ex_ante_risk_share|realized_risk_share)_").sum(),
        )
        ticks = [
            label + (" (removed)" if all_ else " (partly removed)" if any_ else "")
            for label, all_, any_ in rows.select("label", "all_removed", "any_removed").iter_rows()
        ]
        y = np.arange(rows.height)

        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        axes[1, 1].sharey(axes[1, 0])  # before and after on one scale
        fig.suptitle(f"{self.signal}: raw IC {self.raw_ic:.3f}, residual IC {self.residual_ic:.3f}")

        ax = axes[0, 0]
        for i, p in enumerate(PORTFOLIOS):
            ax.barh(y + (i - 0.5) * 0.4, rows[f"annual_return_{p}"] * 100, height=0.4, label=p)
        _bar_axes(ax, y, ticks, "Return contribution (% per year)")

        ax = axes[0, 1]
        series = [(f"{kind}_risk_share_{p}", f"{p}, {kind.replace('_', '-')}") for p in PORTFOLIOS for kind in ("ex_ante", "realized")]
        for i, (column, name) in enumerate(series):
            ax.barh(y + (i - 1.5) * 0.2, rows[column] * 100, height=0.2, label=name)
        _bar_axes(ax, y, ticks, "Share of variance (%)")

        cumulative = (
            self.contributions.join(labeled.select("factor", "label"), on="factor")
            .group_by("portfolio", "label", "date")
            .agg(pl.col("contribution").sum())
            .sort("date")
            .with_columns(pl.col("contribution").cum_sum().over("portfolio", "label").mul(100))
        )
        colors = dict(zip(rows["label"], plt.cm.tab20(np.linspace(0, 1, rows.height))))
        for ax, p in zip(axes[1], PORTFOLIOS):
            for label in rows["label"]:
                line = cumulative.filter(pl.col("portfolio") == p, pl.col("label") == label)
                specific = label == "Specific"
                ax.plot(line["date"], line["contribution"], label=label,
                        color="black" if specific else colors[label], linewidth=2.5 if specific else 1.2)
            ax.axhline(0, color="grey", linewidth=0.8)
            ax.set_title(f"Cumulative return by factor, {p}")
            ax.set_ylabel("Cumulative return (%)")
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            ax.grid(alpha=0.3)
        axes[1, 1].legend(fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1))

        fig.tight_layout()
        if file_name is not None:
            fig.savefig(file_name)
        else:
            plt.show()


def factor_report(
    signal: Signal,
    data: MarketData,
    start: dt.date,
    end: dt.date,
    rebalance: str = "monthly",
    lag: int = 1,
    target_active_risk: float = 0.05,
) -> FactorReport:
    """Break a signal's return and risk into Barra factors plus specific, before and after orthogonalization."""
    horizon = HORIZON[rebalance]
    allowed = signal.allowed_factors
    days, ctxs = contexts(data, start, end, rebalance)
    scores = signal.compute(data, universe_frame(ctxs))
    raw, residual = standardize(scores), residual_scores(scores, ctxs, allowed)
    z = {"before": _by_date(raw), "after": _by_date(residual)}
    names = ctxs[0].rm.factor_names

    windows = {c.date: days[c.index + lag + 1 : c.index + lag + 1 + horizon] for c in ctxs}
    with_window = [c for c in ctxs if len(windows[c.date]) == horizon]
    first, last = windows[with_window[0].date][0], windows[with_window[-1].date][-1]
    u_all = data.specific_returns(first, last, sorted({x for c in ctxs for x in c.rm.ids})).sort("date")
    f_all = data.factor_returns(first, last).sort("date")

    dates, returns, ex_ante = [], {p: [] for p in PORTFOLIOS}, {p: [] for p in PORTFOLIOS}
    for ctx in with_window:
        rm, window = ctx.rm, windows[ctx.date]
        B, F, delta = rm.factor_exposures, rm.factor_covariance, rm.specific_variance
        beta = benchmark_betas(B, F, delta, ctx.benchmark / ctx.benchmark.sum())
        mandate = [np.ones_like(beta), beta]
        removed = [B[:, k] for k in ctx.against(allowed)]
        rows = {
            "before": mandate,
            "after": mandate + [c for c, keep in zip(removed, _independent(mandate, removed)) if keep],
        }
        sigma = {"before": np.sqrt(delta), "after": ctx.sigma(allowed)}
        alpha = {p: sigma[p] * np.array([z[p].get(ctx.date, {}).get(x, 0.0) for x in rm.ids]) for p in PORTFOLIOS}
        if not np.any(alpha["before"]):
            continue

        u = _slice(u_all, window[0], window[-1]).group_by("barrid").agg(pl.col("specific_return").sum())
        u = dict(zip(u["barrid"], u["specific_return"]))
        u = np.array([u.get(x, 0.0) for x in rm.ids])
        f = _slice(f_all, window[0], window[-1]).select(names).sum().to_numpy().ravel()

        dates.append(ctx.date)
        for p in PORTFOLIOS:
            h = _portfolio(alpha[p], B, F, delta, rows[p], target_active_risk)
            x = B.T @ h
            parts = np.append(x * (F @ x), h**2 @ delta)
            returns[p].append(np.append(x * f, h @ u))
            ex_ante[p].append(parts / parts.sum() if parts.sum() > 0 else np.full(len(parts), np.nan))

    labels = [*names, "specific"]
    columns = {
        "factor": labels,
        "group": [_group(n) for n in names] + ["specific"],
        "removed": [n not in allowed for n in names] + [False],
    }
    for p in PORTFOLIOS:
        m = np.array(returns[p])
        annual = m * (252 / horizon)
        mean, std = annual.mean(axis=0), annual.std(axis=0, ddof=1)
        total = m.sum(axis=1)
        covariance = ((m - m.mean(axis=0)) * (total - total.mean())[:, None]).sum(axis=0) / (len(m) - 1)
        columns |= {
            f"annual_return_{p}": mean,
            f"t_stat_{p}": _ratio(mean * np.sqrt(len(m)), std),
            f"return_share_{p}": _ratio(mean, np.full(len(mean), mean.sum())),
            f"ex_ante_risk_share_{p}": np.nanmean(np.array(ex_ante[p]), axis=0),
            f"realized_risk_share_{p}": _ratio(covariance, np.full(len(mean), total.var(ddof=1))),
        }

    n = len(labels)
    contributions = pl.DataFrame(
        {
            "date": [d for p in PORTFOLIOS for d in dates for _ in range(n)],
            "portfolio": [p for p in PORTFOLIOS for _ in range(len(dates) * n)],
            "factor": labels * (len(dates) * len(PORTFOLIOS)),
            "contribution": np.concatenate([np.array(returns[p]).ravel() for p in PORTFOLIOS]),
        }
    )

    raw_target = target_returns(data, days, ctxs, horizon, lag, frozenset(names) | ALL_FACTORS)
    own_target = target_returns(data, days, ctxs, horizon, lag, allowed)
    return FactorReport(
        signal=signal.name,
        table=pl.DataFrame(columns),
        contributions=contributions,
        raw_ic=float(estimate_ic(raw, raw_target)["ic"].mean()),
        residual_ic=float(estimate_ic(residual, own_target)["ic"].mean()),
    )


def _portfolio(alpha, B, F, delta, rows, target_active_risk) -> np.ndarray:
    """The closed-form direction under ``rows``, scaled to the risk target (zero if nothing is left)."""
    if not np.any(alpha):
        return np.zeros_like(alpha)
    d = equality_direction(alpha, B, F, delta, np.vstack(rows))
    variance = active_variance(d, B, F, delta)
    return d * target_active_risk / np.sqrt(variance) if variance > 1e-20 else np.zeros_like(alpha)


def _by_date(z: pl.DataFrame) -> dict[dt.date, dict[str, float]]:
    z = z.drop_nulls("z")
    return {d: dict(zip(g["barrid"], g["z"])) for (d,), g in z.partition_by("date", as_dict=True).items()}


def _group(name: str) -> str:
    if name in COUNTRY_FACTOR:
        return "country"
    if name in INDUSTRY_FACTORS:
        return "industry"
    return "style" if name in STYLE_FACTORS else "other"


def _label() -> pl.Expr:
    group = pl.col("group")
    return (
        pl.when(group == "industry").then(pl.lit("Industries"))
        .when(group == "country").then(pl.lit("Country"))
        .when(group == "specific").then(pl.lit("Specific"))
        .otherwise(pl.col("factor").str.replace("USSLOWL_", "").str.to_titlecase())
    )


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a / b, NaN where b is (numerically) zero."""
    return np.divide(a, b, out=np.full(len(a), np.nan), where=np.abs(b) > 1e-12)


def _bar_axes(ax, y, ticks, xlabel) -> None:
    ax.set_yticks(y, ticks)
    ax.invert_yaxis()
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8)
