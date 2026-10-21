"""Backtest runner."""

import os
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from sf_quant.slurm.config import BacktestConfig
from sf_quant.slurm.slurm import generate_sbatch_script, submit_job


class BacktestRunner:
    """Orchestrates a year-sharded backtest via SLURM."""

    def __init__(self, config: BacktestConfig) -> None:
        """Initialize the runner with configuration.

        Args:
            config: Backtest configuration.
        """
        self.config = config

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "BacktestRunner":
        """Create a runner from a YAML config file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Configured BacktestRunner instance.
        """
        return cls(BacktestConfig.from_yaml(config_path))

    def prepare(self, data: pl.DataFrame | None = None) -> None:
        """Create output directories and optionally write the alpha panel.

        Args:
            data: Optional DataFrame to write to ``config.data_path``.
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.logs_dir, exist_ok=True)

        temp_dir = Path(self.config.project_root) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        if data is not None:
            data.write_parquet(self.config.data_path)

    def submit(
        self,
        data: pl.DataFrame | None = None,
        dry_run: bool = False,
        after: Sequence[str | None] | None = None,
    ) -> str | None:
        """Submit the backtest job to SLURM; return its job id.

        Years come from ``config.start``/``config.end`` rather than from the
        alpha file, so the array shape does not shift when the data vintage
        changes.

        Args:
            data: Optional DataFrame to write to ``config.data_path`` first.
            dry_run: If True, print the script instead of submitting.
            after: Job ids to wait for (``--dependency=afterok``).

        Returns:
            The array job's id, or None on a dry run.
        """
        self.prepare(data)

        years = self.config.years

        print(f"Preparing backtest for {len(years)} years: {years[0]}-{years[-1]}")
        print(f"Signal: {self.config.signal_name}")
        print(f"Window: {self.config.start} to {self.config.end}")
        print(f"Target active risk: {self.config.target_active_risk}")
        print(f"Constraints: {self.config.constraints}")
        print(
            f"Lag: {self.config.lag}, rebalance period: {self.config.rebalance_period}"
        )
        print(f"Output directory: {self.config.output_dir}")

        script = generate_sbatch_script(self.config, years)
        job_id = submit_job(script, dry_run=dry_run, after=after)
        if job_id is not None:
            print(f"Submitted job {job_id}")
        return job_id


def submit_sweep(
    configs: Sequence[BacktestConfig],
    dry_run: bool = False,
    after: Sequence[str | None] | None = None,
) -> list[str | None]:
    """Queue one backtest array job per config; return their job ids.

    Every job waits for ``after``. On a dry run only the first script is
    printed in full, and the ids are None.
    """
    ids = []
    for i, config in enumerate(configs):
        runner = BacktestRunner(config)
        if dry_run and i > 0:
            runner.prepare()
            ids.append(None)
            continue
        ids.append(runner.submit(dry_run=dry_run, after=after))
    print(f"{'Would queue' if dry_run else 'Queued'} {len(configs)} backtest jobs")
    return ids
