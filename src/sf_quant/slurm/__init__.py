"""
The ``slurm`` module orchestrates backtests across a SLURM cluster, sharding a
run into one array task per calendar year, chaining jobs with ``afterok``
dependencies, and reading the sharded results back.
"""

from .config import FUND_CONSTRAINTS, BacktestConfig, SlurmConfig
from .results import MissingYearsError, expected_years, load_weights
from .runner import BacktestRunner, submit_sweep
from .slurm import submit_command, submit_job

__all__ = [
    "BacktestConfig",
    "BacktestRunner",
    "SlurmConfig",
    "FUND_CONSTRAINTS",
    "MissingYearsError",
    "expected_years",
    "load_weights",
    "submit_command",
    "submit_job",
    "submit_sweep",
]
