"""Market data access.

The backtester loads every market input itself, so every run uses the same
risk model, betas, and benchmark. Conventions for any implementation:

- Benchmark weights dated d are weights at the close of d and sum to one.
  Their dates are the trading calendar and their members are the universe.
- Returns are decimal. The return on date d runs from the previous close to
  the close of d.
- Risk model quantities are annualized and decimal.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol

import polars as pl
from sf_quant.data import RiskModel


class MarketData(Protocol):
    def benchmark_weights(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        """``date``, ``barrid``, ``weight``."""
        ...

    def asset_returns(self, start: dt.date, end: dt.date, barrids: list[str]) -> pl.DataFrame:
        """``date``, ``barrid``, ``return``."""
        ...

    def factor_returns(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        """``date`` plus one column per factor."""
        ...

    def prices(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        """``date``, ``barrid``, ``price``."""
        ...

    def risk_model(self, date_: dt.date, barrids: list[str]) -> RiskModel: ...


class SfQuantMarketData:
    """
    The sf-quant data layer on the BYU cluster. The universe is sf-quant's
    ``in_universe`` flag and the benchmark is cap-weighted over it. sf-quant
    stores returns in percent; they are converted to decimal here.
    """

    def benchmark_weights(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        import sf_quant.data as sfd

        return sfd.load_benchmark(start, end).select("date", "barrid", "weight")

    def asset_returns(self, start: dt.date, end: dt.date, barrids: list[str]) -> pl.DataFrame:
        # Not filtered on in_universe: names that leave the universe
        # (including delistings) still earn returns until the next rebalance.
        from sf_quant.data._tables import assets_table

        return (
            assets_table.scan()
            .filter(pl.col("date").is_between(start, end), pl.col("barrid").is_in(barrids))
            .select("date", "barrid", pl.col("return").truediv(100))
            .collect()
        )

    def factor_returns(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        import sf_quant.data as sfd

        return sfd.load_factors(start, end).with_columns(pl.exclude("date").truediv(100))

    def prices(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        import sf_quant.data as sfd

        return sfd.load_assets(start, end, columns=["date", "barrid", "price"], in_universe=True)

    def risk_model(self, date_: dt.date, barrids: list[str]) -> RiskModel:
        import sf_quant.data as sfd

        return sfd.load_risk_model(date_, barrids)
