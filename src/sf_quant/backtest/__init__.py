"""sf-backtester: SLURM-based parallel backtesting for quantitative finance."""

from sf_backtester.config import BacktestConfig, BacktestDynamicConfig, SlurmConfig
from sf_backtester.runner import BacktestRunner, BacktestDynamicRunner

__version__ = "0.1.0"
__all__ = ["BacktestConfig", "BacktestDynamicConfig", "BacktestRunner", "SlurmConfig"]
