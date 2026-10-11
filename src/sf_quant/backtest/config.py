"""Backtest configuration, loaded from a flat YAML file.

The fund's mandate is not configurable: every backtest is long-only, fully
invested, zero active beta, and sized to 5% tracking error.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import yaml

ACTIVE_RISK_TARGET = 0.05  # the mandate's annualized tracking error


class ConfigError(ValueError):
    """A config is malformed or uses an unsupported option."""


@dataclass(frozen=True)
class SlurmConfig:
    n_cpus: int = 8
    mem: str = "32G"
    time: str = "06:00:00"
    email: str | None = None


@dataclass(frozen=True)
class BacktestConfig:
    name: str
    start: dt.date
    end: dt.date
    rebalance: str = "monthly"  # monthly | weekly | daily
    execution_lag_days: int = 1  # alpha formed at close t, traded at close t + lag
    # neutral: exactly zero exposure to every factor the signals aren't allowed to keep.
    # risk_model: no exact neutrality; the risk model alone discourages factor bets.
    factor_control: str = "neutral"
    bps_grid: list[float] = field(default_factory=lambda: [0.0, 5.0, 10.0, 20.0])
    min_alpha_coverage: float = 0.5
    aum: float = 1_000_000.0
    n_workers: int = 1
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    def __post_init__(self) -> None:
        checks = [
            (self.end >= self.start, "end must be on or after start"),
            (self.rebalance in ("monthly", "weekly", "daily"), "rebalance must be monthly, weekly, or daily"),
            (self.execution_lag_days >= 0, "execution_lag_days must be non-negative"),
            (self.factor_control in ("neutral", "risk_model"), "factor_control must be neutral or risk_model"),
            (all(bps >= 0 for bps in self.bps_grid), "bps_grid values must be non-negative"),
            (0 <= self.min_alpha_coverage <= 1, "min_alpha_coverage must be in [0, 1]"),
            (self.n_workers >= 1, "n_workers must be at least 1"),
        ]
        for ok, message in checks:
            if not ok:
                raise ConfigError(message + ".")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        data = dict(data)
        for key in ("start", "end"):
            if isinstance(data.get(key), str):
                data[key] = dt.date.fromisoformat(data[key])
        if "bps_grid" in data:
            data["bps_grid"] = [float(bps) for bps in data["bps_grid"]]
        try:
            if isinstance(data.get("slurm"), dict):
                data["slurm"] = SlurmConfig(**data["slurm"])
            return cls(**data)
        except TypeError as e:  # unknown or missing keys
            raise ConfigError(str(e)) from None

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def resolved(self) -> dict[str, Any]:
        """Every field, defaults included, as YAML-safe types."""
        return {k: v.isoformat() if isinstance(v, dt.date) else v for k, v in dataclasses.asdict(self).items()}

    def hash(self) -> str:
        """Short hash of everything that affects results (not n_workers or slurm)."""
        payload = {k: v for k, v in self.resolved().items() if k not in ("n_workers", "slurm")}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
