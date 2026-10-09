"""Build an idiosyncratic momentum alpha panel with the sf-quant signal library.

The score is residualized against every Barra factor, and alpha =
IC * specific risk * Z (annualized). factor_loss shows what removing each
factor gave up.
"""

import datetime as dt

import sf_quant.signals as sg
from sf_quant.data import SfQuantMarketData

start, end = dt.date(2010, 1, 1), dt.date(2024, 12, 31)
data = SfQuantMarketData()
signal = sg.IdiosyncraticMomentum()

panel = sg.build_alpha(signal, data, start, end, rebalance="monthly")
panel.alphas.write_parquet("momentum_alphas.parquet")

loss = sg.factor_loss(signal, data, start, end)
print(loss.table)
print(f"IC raw {loss.raw_ic:.3f}, residual {loss.residual_ic:.3f}")
