"""
The ``slurm`` module orchestrates backtests across a SLURM cluster, sharding a
run into one array task per calendar year.
"""

from .config import FUND_CONSTRAINTS, BacktestConfig, SlurmConfig
from .runner import BacktestRunner

__all__ = ["BacktestConfig", "BacktestRunner", "SlurmConfig", "FUND_CONSTRAINTS"]
