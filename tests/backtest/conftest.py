"""Shared helpers: configs, alpha panels, and a hand-checkable four-asset market."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from sf_quant.testing import PanelMarketData, synthetic_market

from sf_quant.backtest import BacktestConfig, run_backtest


def config(md, **overrides) -> BacktestConfig:
    return BacktestConfig.from_dict(
        {"name": "test", "start": md.days[0], "end": md.days[-1]} | overrides
    )


def run(md, alphas, **overrides):
    return run_backtest(alphas, config(md, **overrides), md)


def random_alphas(md, seed=0, ic=0.05) -> pl.DataFrame:
    """Grinold–Kahn alphas from pure noise: IC * sigma * Z."""
    z = np.random.default_rng(seed).normal(size=(len(md.days), len(md.ids)))
    return pl.DataFrame(
        {
            "date": [d for d in md.days for _ in md.ids],
            "barrid": md.ids * len(md.days),
            "alpha": (ic * np.sqrt(md.delta) * z).ravel(),
        }
    )


@pytest.fixture(scope="session")
def market():
    return synthetic_market(n_assets=150, n_days=504, seed=1)


def tiny_market(spikes: dict[dt.date, dict[str, float]], days: list[dt.date]) -> PanelMarketData:
    """Four assets, one country factor, constant equal benchmark; returns zero except ``spikes``."""
    ids = ["A", "B", "C", "D"]
    return PanelMarketData(
        returns=pl.DataFrame(
            [{"date": d, "barrid": i, "return": spikes.get(d, {}).get(i, 0.0)} for d in days for i in ids]
        ),
        benchmark=pl.DataFrame([{"date": d, "barrid": i, "weight": 0.25} for d in days for i in ids]),
        ids=ids,
        factor_exposures=np.ones((4, 1)),
        factor_covariance=np.array([[0.16**2]]),
        specific_variance=np.full(4, 0.30**2),
        factor_names=["COUNTRY"],
    )


def tiny_alphas(days: list[dt.date]) -> pl.DataFrame:
    """Constant alphas favoring A and against D."""
    values = {"A": 0.02, "B": 0.005, "C": -0.005, "D": -0.02}
    return pl.DataFrame([{"date": d, "barrid": i, "alpha": a} for d in days for i, a in values.items()])
