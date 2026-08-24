"""Transaction cost models for portfolio optimization.

Provides a protocol and concrete implementations for estimating one-way
trading costs per unit traded. These costs are used by the optimizer to
penalize turnover in the objective function.
"""

from typing import Protocol
import datetime as dt

import numpy as np
import polars as pl

from sf_quant.data.assets import load_assets_by_date


class CostModel(Protocol):
    """Protocol for transaction cost estimators."""

    def estimate(self, date_: dt.date, barrids: list[str]) -> np.ndarray:
        """Return one-way cost per unit traded, shape (N,), in decimal."""
        ...


class FixedCost:
    """Constant cost per unit traded for all assets."""

    def __init__(self, bps: float = 5.0):
        self._cost = bps / 10_000

    def estimate(self, date_: dt.date, barrids: list[str]) -> np.ndarray:
        return np.full(len(barrids), self._cost)


class SpreadCost:
    """Cost proportional to volatility / sqrt(liquidity).

    Based on Kyle/Glosten-Milgrom market microstructure:
        spread_i ~ sigma_i / sqrt(ADV_i)

    A single free parameter k is calibrated so the cross-sectional median
    cost equals `target_median_bps`.
    """

    def __init__(self, target_median_bps: float = 7.5):
        self._target_median = target_median_bps / 10_000

    def estimate(self, date_: dt.date, barrids: list[str]) -> np.ndarray:
        df = load_assets_by_date(
            date_,
            in_universe=False,
            columns=["barrid", "total_risk", "average_daily_volume_60"],
        )

        barrids_df = pl.DataFrame({"barrid": barrids})
        df = barrids_df.join(df, on="barrid", how="left").fill_null(0)

        sigma = df["total_risk"].to_numpy() / 100  # percent -> decimal
        adv = df["average_daily_volume_60"].to_numpy().astype(np.float64)
        adv = np.maximum(adv, 1.0)

        raw = sigma / np.sqrt(adv)
        median_raw = np.median(raw[raw > 0]) if np.any(raw > 0) else 1.0
        k = self._target_median / median_raw

        return k * raw
