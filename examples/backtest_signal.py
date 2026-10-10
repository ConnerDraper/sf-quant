"""Backtest a signal under the fund's mandate.

Builds alphas (alpha = IC * sigma * Z, annualized) and runs the walk-forward
backtest. Writes to out/backtest_signal/: alphas.parquet, then the full run
(weights, daily returns, diagnostics, factor exposures and returns,
summary.json, and the resolved config). Chart it with plot_backtest.py.

The same backtest from the command line, using the saved alphas:
  sf_backtest run config.yml --alphas out/backtest_signal/alphas.parquet --out out/backtest_signal
  sf_backtest submit config.yml --alphas out/backtest_signal/alphas.parquet   # as a SLURM job
"""

from pathlib import Path

from my_signals import LowSpecificVolatility

import sf_quant.signals as sg
from sf_quant.backtest import BacktestConfig, run_backtest
from sf_quant.data import SfQuantMarketData

cfg = BacktestConfig.from_yaml("config.yml")
data = SfQuantMarketData()
out = Path("out/backtest_signal")
out.mkdir(parents=True, exist_ok=True)

panel = sg.build_alpha(LowSpecificVolatility(), data, cfg.start, cfg.end, cfg.rebalance)
panel.alphas.write_parquet(out / "alphas.parquet")
run_backtest(panel, cfg, data).save(out)
