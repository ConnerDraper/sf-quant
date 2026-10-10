"""Research a signal: what it earns and risks from each Barra factor.

Writes to out/research_signal/:
  factors.csv  per factor and specific, before and after orthogonalizing:
               annual return, t-stat, return share, ex-ante and realized risk share
  ic.csv       IC of the raw score vs. the orthogonalized score
  factors.png  the same as charts
"""

from pathlib import Path

import polars as pl
from my_signals import LowSpecificVolatility

import sf_quant.signals as sg
from sf_quant.backtest import BacktestConfig
from sf_quant.data import SfQuantMarketData

cfg = BacktestConfig.from_yaml("config.yml")
out = Path("out/research_signal")
out.mkdir(parents=True, exist_ok=True)

report = sg.factor_report(
    LowSpecificVolatility(), SfQuantMarketData(), cfg.start, cfg.end, cfg.rebalance, cfg.execution_lag_days
)

report.table.write_csv(out / "factors.csv")
pl.DataFrame({"raw_ic": [report.raw_ic], "residual_ic": [report.residual_ic]}).write_csv(out / "ic.csv")
report.plot(file_name=str(out / "factors.png"))
