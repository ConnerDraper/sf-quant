"""Does adding a new signal to the current ones improve the fund?

A signal can look good alone and still add nothing, if it mostly repeats what
the current signals already know. So this runs three backtests: the current
signals combined, the current signals plus the candidate, and the candidate
alone. The answer is ir_gain, IR with the candidate minus IR without it.

Writes to out/add_signal_to_composite/:
  ir_comparison.csv      IR gross and net, realized TE, transfer coefficient, and
                         turnover for each run; ir_gain and additivity_gap
  candidate_weight.csv   per rebalance: the candidate's IC, its weight in the
                         composite, and its marginal IC (negative: it hurts)
  candidate_factors.png  the candidate's return and risk by Barra factor
  <run>/                 each run's full backtest
"""

from pathlib import Path

import polars as pl
from my_signals import LowSpecificVolatility

from sf_quant.backtest import BacktestConfig, compare
from sf_quant.data import SfQuantMarketData
from sf_quant.signals import IdiosyncraticMomentum, ShortTermReversal

cfg = BacktestConfig.from_yaml("config.yml")
out = Path("out/add_signal_to_composite")
out.mkdir(parents=True, exist_ok=True)

result = compare(
    baseline=[IdiosyncraticMomentum(), ShortTermReversal()],
    candidate=LowSpecificVolatility(),
    config=cfg,
    market_data=SfQuantMarketData(),
)

result.summary.with_columns(
    ir_gain=pl.lit(result.ir_gain),  # the answer
    # IR_with^2 - IR_baseline^2 - IR_candidate^2 (manual 2.4.4). Near zero: the candidate's
    # IR adds in full. Well below zero: it overlaps the current signals.
    additivity_gap=pl.lit(result.additivity_gap),
).write_csv(out / "ir_comparison.csv")
result.candidate_in_composite.write_csv(out / "candidate_weight.csv")
result.factor_report.plot(file_name=str(out / "candidate_factors.png"))
for name, run in result.results.items():
    run.save(out / name)
