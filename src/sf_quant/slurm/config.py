"""Configuration for SLURM-orchestrated backtests."""

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Self

import yaml

#: The fund active mandate, by name. Resolved to constraint objects in the worker.
FUND_CONSTRAINTS = ["ZeroInvestment", "ZeroBeta", "LongOnly"]


@dataclass
class SlurmConfig:
    """SLURM job configuration."""

    n_cpus: int = 8
    mem: str = "32G"
    time: str = "06:00:00"
    mail_type: str = "BEGIN,END,FAIL"
    max_concurrent_jobs: int = 31
    # One array task per physical node. Default off so existing callers are
    # unaffected; set True for Ray-based workers, where SLURM co-scheduling
    # multiple tasks (each booting its own Ray cluster) on one node can
    # saturate that node's GCS/ports/shared memory and cause the whole task
    # to fail at Ray startup, not just run slower.
    exclusive: bool = False


def _as_date(value: str | dt.date) -> dt.date:
    return value if isinstance(value, dt.date) else dt.date.fromisoformat(value)


@dataclass
class BacktestConfig:
    """
    Configuration for a backtest run.

    ``start`` and ``end`` define the trading calendar the run is measured
    against, and are deliberately not inferred from the alpha file. The
    rebalance grid is anchored at the first trading day in that window, so two
    signals whose data happens to begin on different dates still rebalance on
    the same dates as long as they share a ``start``/``end``.
    """

    # Required fields
    signal_name: str
    data_path: str
    start: dt.date
    end: dt.date

    # Environment/paths
    project_root: str
    byu_email: str

    # Optional with defaults
    target_active_risk: float = 0.05
    constraints: list[str] = field(default_factory=lambda: list(FUND_CONSTRAINTS))
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    # Robustness checks: trading days to lag alpha by, trading days between rebalances
    lag: int = 0
    rebalance_period: int = 1

    # Output configuration
    output_dir: str | None = None
    logs_dir: str | None = None

    def __post_init__(self) -> None:
        """Coerce loosely-typed YAML values and set derived paths."""
        self.start = _as_date(self.start)
        self.end = _as_date(self.end)
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        if self.lag < 0:
            raise ValueError(f"lag must be >= 0, got {self.lag}")
        if self.rebalance_period < 1:
            raise ValueError(
                f"rebalance_period must be >= 1, got {self.rebalance_period}"
            )

        suffix = (
            ""
            if (self.lag, self.rebalance_period) == (0, 1)
            else f"/lag{self.lag}_rebal{self.rebalance_period}"
        )
        base = f"{self.signal_name}/{self.target_active_risk}{suffix}"
        if self.output_dir is None:
            self.output_dir = f"{self.project_root}/weights/{base}"
        if self.logs_dir is None:
            self.logs_dir = f"logs/{base}"

        # Convert slurm dict to SlurmConfig if needed
        if isinstance(self.slurm, dict):
            self.slurm = SlurmConfig(**self.slurm)

    @property
    def years(self) -> list[int]:
        """Calendar years spanned by the run. One SLURM array task per year."""
        return list(range(self.start.year, self.end.year + 1))

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
