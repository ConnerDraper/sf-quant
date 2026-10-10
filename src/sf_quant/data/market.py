"""Market data interface shared by the signal library and the backtester.

Conventions for every implementation:

- Benchmark weights dated d are weights at the close of d and sum to one.
  Their dates are the trading calendar and their members are the universe.
- Returns are decimal. The return on date d runs from the previous close to
  the close of d. The specific return is the part not explained by Barra
  factors: u_d = r_d - B_{d-1} f_d.
- Risk model quantities are annualized and decimal.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol

import polars as pl

from .covariance_matrix import RiskModel


class MarketData(Protocol):
    def benchmark_weights(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        """``date``, ``barrid``, ``weight``."""
        ...

    def asset_returns(self, start: dt.date, end: dt.date, barrids: list[str]) -> pl.DataFrame:
        """``date``, ``barrid``, ``return``."""
        ...

    def specific_returns(self, start: dt.date, end: dt.date, barrids: list[str]) -> pl.DataFrame:
        """``date``, ``barrid``, ``specific_return``."""
        ...

    def factor_returns(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        """``date`` plus one column per factor."""
        ...

    def exposures(self, date_: dt.date, barrids: list[str]) -> pl.DataFrame:
        """``barrid`` plus one column per factor, at the close of ``date_``."""
        ...

    def prices(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        """``date``, ``barrid``, ``price``."""
        ...

    def risk_model(self, date_: dt.date, barrids: list[str]) -> RiskModel: ...


def trading_days(data: MarketData, start: dt.date, end: dt.date) -> list[dt.date]:
    """The trading calendar: dates with benchmark weights."""
    return data.benchmark_weights(start, end)["date"].unique().sort().to_list()


def rebalance_dates(days: list[dt.date], frequency: str) -> list[dt.date]:
    """Last trading day of each month or ISO week, or every day."""
    period = {
        "daily": lambda d: d,
        "weekly": lambda d: d.isocalendar()[:2],
        "monthly": lambda d: (d.year, d.month),
    }[frequency]
    return [d for i, d in enumerate(days) if i + 1 == len(days) or period(days[i + 1]) != period(d)]


class SfQuantMarketData:
    """
    The sf-quant data layer on the BYU cluster. The universe is the
    ``in_universe`` flag and the benchmark is cap-weighted over it. sf-quant
    stores returns in percent; they are converted to decimal here.
    """

    def benchmark_weights(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        from .benchmark import load_benchmark

        return load_benchmark(start, end).select("date", "barrid", "weight")

    def asset_returns(self, start: dt.date, end: dt.date, barrids: list[str]) -> pl.DataFrame:
        # Not filtered on in_universe: names that leave the universe
        # (including delistings) still earn returns until the next rebalance.
        from ._tables import assets_table

        return (
            assets_table.scan()
            .filter(pl.col("date").is_between(start, end), pl.col("barrid").is_in(barrids))
            .select("date", "barrid", pl.col("return").truediv(100))
            .collect()
        )

    def specific_returns(self, start: dt.date, end: dt.date, barrids: list[str]) -> pl.DataFrame:
        """u_d = r_d - B_{d-1} f_d, computed a year at a time to bound memory."""
        from ._factors import factors
        from ._tables import exposures_table

        f = self.factor_returns(start - dt.timedelta(days=10), end).rename({k: f"{k}_f" for k in factors})
        out = []
        for year in range(start.year, end.year + 1):
            lo, hi = max(start, dt.date(year, 1, 1)), min(end, dt.date(year, 12, 31))
            exposures = (
                exposures_table.scan()
                .filter(
                    pl.col("date").is_between(lo - dt.timedelta(days=10), hi),
                    pl.col("barrid").is_in(barrids),
                )
                .select("date", "barrid", *factors)
                .collect()
                .sort("barrid", "date")
                .with_columns(pl.col(factors).fill_null(0).shift(1).over("barrid"))  # prior close
            )
            out.append(
                self.asset_returns(lo, hi, barrids)
                .join(exposures, on=["date", "barrid"], how="inner")
                .join(f, on="date", how="inner")
                .select(
                    "date",
                    "barrid",
                    (pl.col("return") - pl.sum_horizontal(pl.col(k) * pl.col(f"{k}_f") for k in factors))
                    .alias("specific_return"),
                )
                .drop_nulls()
            )
        return pl.concat(out)

    def factor_returns(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        from .factors import load_factors

        return load_factors(start, end).with_columns(pl.exclude("date").truediv(100))

    def exposures(self, date_: dt.date, barrids: list[str]) -> pl.DataFrame:
        from ._factors import factors
        from .exposures import load_exposures_by_date

        return (
            pl.DataFrame({"barrid": barrids})
            .join(load_exposures_by_date(date_).select("barrid", *factors), on="barrid", how="left")
            .fill_null(0)
        )

    def prices(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        from .assets import load_assets

        return load_assets(start, end, columns=["date", "barrid", "price"], in_universe=True)

    def risk_model(self, date_: dt.date, barrids: list[str]) -> RiskModel:
        from .covariance_matrix import load_risk_model

        return load_risk_model(date_, barrids)
