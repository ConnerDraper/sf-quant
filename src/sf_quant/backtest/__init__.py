"""sf-backtester: SLURM-based parallel backtesting for quantitative finance."""

from sf_backtester.config import (
    BacktestConfig,
    BacktestCostAwareConfig,
    BacktestDynamicConfig,
    CostModelConfig,
    SlurmConfig,
)
from sf_backtester.runner import BacktestCostAwareRunner, BacktestRunner, BacktestDynamicRunner

__version__ = "0.2.0"
__all__ = [
    "BacktestConfig",
    "BacktestCostAwareConfig",
    "BacktestDynamicConfig",
    "BacktestCostAwareRunner",
    "BacktestRunner",
    "BacktestDynamicRunner",
    "CostModelConfig",
    "SlurmConfig",
]
