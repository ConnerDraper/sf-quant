"""
The ``backtest`` module runs walk-forward backtests of the Silver Fund
mandate: :func:`run_backtest` solves the mandate problem at each rebalance and
accounts for drift, execution lag, turnover, and costs. :func:`compare` asks
whether a candidate signal improves the current composite.
"""

from sf_quant.data import MarketData, SfQuantMarketData
from sf_quant.signals import AlphaPanel

from .backtest import InvariantError, run_backtest
from .compare import Comparison, compare
from .config import BacktestConfig, ConfigError
from .results import BacktestResult

__all__ = [
    "AlphaPanel",
    "BacktestConfig",
    "BacktestResult",
    "Comparison",
    "ConfigError",
    "InvariantError",
    "MarketData",
    "SfQuantMarketData",
    "compare",
    "run_backtest",
]
