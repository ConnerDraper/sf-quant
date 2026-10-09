"""Backtest outputs: tables, headline metrics, and the files written by ``save``."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from sf_backtester.accounting import net_column
from sf_backtester.config import BacktestConfig

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    weights: pl.DataFrame  # per rebalance and asset: alpha, benchmark, active, and fund weight
    returns: pl.DataFrame  # daily portfolio, benchmark, active (factor/specific), net of each cost level
    diagnostics: pl.DataFrame  # per rebalance: solver status, ex-ante TE, TC, turnover, ...
    factor_exposures: pl.DataFrame  # formation_date, factor, active_exposure
    summary: dict  # headline metrics plus reproducibility metadata
    config: dict  # resolved config

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.weights.write_parquet(out / "weights.parquet")
        self.returns.write_parquet(out / "returns.parquet")
        self.diagnostics.write_parquet(out / "diagnostics.parquet")
        self.factor_exposures.write_parquet(out / "factor_exposures.parquet")
        (out / "summary.json").write_text(json.dumps(_finite(self.summary), indent=2))
        (out / "config.resolved.yml").write_text(yaml.safe_dump(self.config, sort_keys=False))
        return out


def build_result(
    solved: list,
    returns: pl.DataFrame,
    turnover: pl.DataFrame,
    n_missing: int,
    config: BacktestConfig,
    data_snapshot: dt.date,
    holdout_used: bool,
) -> BacktestResult:
    diagnostics = (
        pl.DataFrame([s.diagnostics for s in solved])
        .join(turnover, on="execution_date", how="left")
        .with_columns(pl.col("turnover").fill_null(0.0), pl.col("initial").fill_null(False))
    )
    summary = summarize(returns, diagnostics, config) | {
        "n_missing_return_days": n_missing,
        "metadata": {
            "config_hash": config.hash(),
            "profile": config.profile,
            "holdout_used": holdout_used,
            "data_snapshot": data_snapshot.isoformat(),
            "git_sha": _git_sha(),
            "packages": {p: _version(p) for p in ("sf-backtester", "sf-quant", "cvxpy", "clarabel", "polars", "numpy")},
            "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        },
    }
    return BacktestResult(
        weights=pl.concat(
            s.weights.select(
                pl.lit(s.rebalance.formation).alias("formation_date"),
                pl.lit(s.rebalance.execution).alias("execution_date"),
                pl.all(),
            )
            for s in solved
        ),
        returns=returns,
        diagnostics=diagnostics,
        factor_exposures=pl.concat(
            pl.DataFrame(
                {
                    "formation_date": [s.rebalance.formation] * len(s.factor_names),
                    "factor": s.factor_names,
                    "active_exposure": s.active_factor_exposures,
                }
            )
            for s in solved
        ),
        summary=summary,
        config=config.resolved(),
    )


def summarize(returns: pl.DataFrame, diagnostics: pl.DataFrame, config: BacktestConfig) -> dict:
    """Headline metrics. Turnover excludes the initial build from cash."""
    active = returns["active"].to_numpy()
    benchmark = returns["benchmark"].to_numpy()
    years = len(active) / TRADING_DAYS
    ann_active = active.mean() * TRADING_DAYS
    te = active.std(ddof=1) * np.sqrt(TRADING_DAYS)

    ok = diagnostics.filter(pl.col("solver_status").is_in(["optimal", "optimal_inaccurate"]))
    mean_ex_ante_te = _mean(ok["ex_ante_te"])
    annual_turnover = diagnostics.filter(~pl.col("initial"))["turnover"].sum() / years
    # Cost per unit traded at which net IR falls to the passing bar.
    breakeven = (ann_active - config.pass_ir * te) / annual_turnover * 1e4 if annual_turnover > 0 else math.inf

    return {
        "start": returns["date"].min().isoformat(),
        "end": returns["date"].max().isoformat(),
        "n_days": len(active),
        "realized_active_return": float(ann_active),
        "realized_te": float(te),
        "ir_gross": _ratio(ann_active, te),
        "ir_net": {f"{bps:g}": _ir(returns[net_column(bps)].to_numpy()) for bps in config.bps_grid},
        "breakeven_bps": float(breakeven),
        "realized_to_ex_ante_te": _ratio(te, mean_ex_ante_te),
        "mean_ex_ante_te": mean_ex_ante_te,
        "mean_te_shortfall": _mean(ok["te_shortfall"]),
        "mean_transfer_coefficient": _mean(ok["transfer_coefficient"].drop_nans()),
        "realized_active_beta": float(np.cov(active, benchmark)[0, 1] / np.var(benchmark, ddof=1)),
        "factor_share_of_active_variance": _ratio(np.var(returns["active_factor"].to_numpy(), ddof=1), np.var(active, ddof=1)),
        "annual_turnover": float(annual_turnover),
        "n_rebalances": diagnostics.height,
        "failed_rebalances": diagnostics.height - ok.height,
    }


def _ir(daily: np.ndarray) -> float:
    return _ratio(daily.mean() * TRADING_DAYS, daily.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator and math.isfinite(denominator) else math.nan


def _mean(series: pl.Series) -> float:
    value = series.mean()
    return math.nan if value is None else float(value)


def _finite(value):
    """JSON can't hold NaN or infinity; write them as null."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_sha() -> str | None:
    """Commit of the research project being run from, if it's a git repo."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None
