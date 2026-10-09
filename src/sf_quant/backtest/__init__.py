"""sf-backtester: walk-forward backtests of the Silver Fund mandate."""

from sf_backtester.backtest import InvariantError, run_backtest
from sf_backtester.config import BacktestConfig, ConfigError
from sf_backtester.data import MarketData, SfQuantMarketData
from sf_backtester.results import BacktestResult

__version__ = "0.3.0"
__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "ConfigError",
    "InvariantError",
    "MarketData",
    "SfQuantMarketData",
    "run_backtest",
]
