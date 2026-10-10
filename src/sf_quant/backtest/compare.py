"""Should a candidate signal join the fund? (manual 2.4)

A tilt toward the candidate helps only if it adds information the current
signals don't already have. That is measured directly: backtest the current
composite, backtest it again with the candidate added, and compare. Under
the mandate, fund Sharpe moves one-for-one with IR (manual 3.5), so the IR
change is the answer. The candidate's standalone run is reported for context.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from sf_quant.data import MarketData, SfQuantMarketData
from sf_quant.signals import FactorReport, Signal, build_composite, factor_report

from .backtest import run_backtest
from .config import BacktestConfig
from .results import BacktestResult


@dataclass
class Comparison:
    summary: pl.DataFrame  # one row per run: IR gross and net, realized TE, TC, turnover
    ir_gain: float  # IR(baseline + candidate) - IR(baseline)
    # IR_with^2 - IR_baseline^2 - IR_candidate^2 (manual 2.4.4). Near zero: the candidate's
    # IR adds in full. Well below zero: it overlaps the baseline or doesn't survive the
    # constraints. Only meaningful when the baseline and candidate IRs are positive.
    additivity_gap: float
    candidate_in_composite: pl.DataFrame  # the candidate's IC, weight, and marginal IC each date
    factor_report: FactorReport  # the candidate's return and risk by Barra factor
    results: dict[str, BacktestResult]


def compare(
    baseline: list[Signal],
    candidate: Signal,
    config: BacktestConfig,
    market_data: MarketData | None = None,
    **composite_options,
) -> Comparison:
    """
    Backtest the baseline composite, baseline + candidate, and the candidate
    alone under ``config``. ``composite_options`` go to
    :func:`sf_quant.signals.build_composite` (e.g. ``ic_shrink_obs``).
    """
    md = market_data if market_data is not None else SfQuantMarketData()
    window = (md, config.start, config.end, config.rebalance, config.execution_lag_days)

    runs = {"with_candidate": [*baseline, candidate], "candidate": [candidate]}
    if baseline:
        runs = {"baseline": baseline, **runs}
    panels = {name: build_composite(signals, *window, **composite_options) for name, signals in runs.items()}
    results = {name: run_backtest(panel, config, md) for name, panel in panels.items()}

    ir = {name: r.summary["ir_gross"] for name, r in results.items()}
    ir_baseline = ir.get("baseline", 0.0)
    label = panels["with_candidate"].diagnostics["signal"].unique(maintain_order=True)[-1]

    return Comparison(
        summary=pl.DataFrame(
            [
                {
                    "run": name,
                    "ir_gross": r.summary["ir_gross"],
                    **{f"ir_net_{bps}bps": v for bps, v in r.summary["ir_net"].items()},
                    "realized_te": r.summary["realized_te"],
                    "mean_transfer_coefficient": r.summary["mean_transfer_coefficient"],
                    "annual_turnover": r.summary["annual_turnover"],
                }
                for name, r in results.items()
            ]
        ),
        ir_gain=ir["with_candidate"] - ir_baseline,
        additivity_gap=ir["with_candidate"] ** 2 - ir_baseline**2 - ir["candidate"] ** 2,
        candidate_in_composite=panels["with_candidate"].diagnostics.filter(pl.col("signal") == label),
        factor_report=factor_report(candidate, *window),
        results=results,
    )
