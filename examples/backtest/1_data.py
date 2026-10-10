"""Build an idiosyncratic momentum alpha panel with the sf-quant signal library.

The score is residualized against every Barra factor, and alpha =
IC * specific risk * Z (annualized). factor_report shows the signal's return
and risk by Barra factor, before and after orthogonalizing.
"""

import datetime as dt

import sf_quant.signals as sg
from sf_quant.data import SfQuantMarketData

start, end = dt.date(2010, 1, 1), dt.date(2024, 12, 31)
data = SfQuantMarketData()
signal = sg.IdiosyncraticMomentum()

panel = sg.build_alpha(signal, data, start, end, rebalance="monthly")
panel.alphas.write_parquet("momentum_alphas.parquet")

report = sg.factor_report(signal, data, start, end)
print(report.table)
report.plot(file_name="momentum_factors.png")
