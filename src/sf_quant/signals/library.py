"""Default signals."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

from sf_quant.data.market import MarketData
from sf_quant.optimizer.mandate import benchmark_betas

from .base import INDUSTRY_FACTORS, Signal


def _history_start(first: dt.date, trading_days: int) -> dt.date:
    """A calendar date at least ``trading_days`` trading days before ``first``."""
    return first - dt.timedelta(days=int(trading_days * 7 / 5) + 15)


def _rolling_specific(data: MarketData, universe: pl.DataFrame, history: int, score: pl.Expr) -> pl.DataFrame:
    dates = universe["date"].unique().sort()
    u = data.specific_returns(
        _history_start(dates[0], history), dates[-1], universe["barrid"].unique().to_list()
    ).sort("barrid", "date")
    scored = u.with_columns(score.over("barrid").alias("score")).select("date", "barrid", "score")
    return universe.join(scored, on=["date", "barrid"], how="inner").drop_nulls("score")


@dataclass(frozen=True)
class IdiosyncraticMomentum(Signal):
    """
    Trailing specific (Barra-residual) return over months 12 to 2, divided by
    its volatility. Skipping the last month avoids short-term reversal.
    """

    lookback: int = 252
    skip: int = 21

    def compute(self, data: MarketData, universe: pl.DataFrame) -> pl.DataFrame:
        n = self.lookback - self.skip
        u = pl.col("specific_return")
        return _rolling_specific(data, universe, self.lookback, (u.rolling_sum(n) / u.rolling_std(n)).shift(self.skip))


@dataclass(frozen=True)
class ShortTermReversal(Signal):
    """Minus the trailing specific return over the last ``lookback`` days."""

    lookback: int = 21

    def compute(self, data: MarketData, universe: pl.DataFrame) -> pl.DataFrame:
        return _rolling_specific(data, universe, self.lookback, -pl.col("specific_return").rolling_sum(self.lookback))


@dataclass(frozen=True)
class IndustryMomentum(Signal):
    """
    Each stock's industry exposure times its industries' trailing
    ``lookback``-day factor returns. This is purely an industry bet, so it
    keeps the industry factors.
    """

    allowed_factors: frozenset[str] = INDUSTRY_FACTORS
    lookback: int = 126

    def compute(self, data: MarketData, universe: pl.DataFrame) -> pl.DataFrame:
        dates = universe["date"].unique().sort().to_list()
        f = data.factor_returns(_history_start(dates[0], self.lookback), dates[-1]).sort("date")
        industries = sorted(INDUSTRY_FACTORS & set(f.columns))
        trailing = f.select("date", *(pl.col(k).rolling_sum(self.lookback) for k in industries))

        out = []
        for date_, members in universe.partition_by("date", as_dict=True, maintain_order=True).items():
            row = trailing.filter(pl.col("date") == date_[0])
            if row.is_empty() or row.select(industries).null_count().sum_horizontal().item():
                continue
            ids = members["barrid"].to_list()
            exposures = data.exposures(date_[0], ids).select(industries).to_numpy()
            out.append(pl.DataFrame({"date": date_[0], "barrid": ids, "score": exposures @ row.select(industries).to_numpy()[0]}))
        return pl.concat(out) if out else universe.head(0).with_columns(score=pl.lit(0.0))


@dataclass(frozen=True)
class BettingAgainstBeta(Signal):
    """
    Minus each stock's beta to the benchmark, from the same risk model the
    optimizer uses (beta = Vb / b'Vb).

    A worked example of a signal the mandate erases. Any alpha linear in
    beta earns exactly zero under sum(h) = 0 and beta'h = 0, so this score
    is worth roughly nothing to the fund; only its nonlinear part, or where
    it disagrees with the risk model's beta, can survive. The low-beta
    premium is paid for leverage, which a fully invested, beta-locked fund
    can't use.
    """

    allowed_factors: frozenset[str] = frozenset({"USSLOWL_BETA"})

    def compute(self, data: MarketData, universe: pl.DataFrame) -> pl.DataFrame:
        dates = universe["date"].unique().sort().to_list()
        bench = data.benchmark_weights(dates[0], dates[-1])
        out = []
        for date_, members in universe.partition_by("date", as_dict=True, maintain_order=True).items():
            w = members.join(bench.filter(pl.col("date") == date_[0]), on=["date", "barrid"], how="left")
            rm = data.risk_model(date_[0], w["barrid"].to_list())
            b = w.filter(pl.col("barrid").is_in(rm.ids))["weight"].fill_null(0).to_numpy()
            if not b.sum() > 0:
                continue
            beta = benchmark_betas(rm.factor_exposures, rm.factor_covariance, rm.specific_variance, b / b.sum())
            out.append(pl.DataFrame({"date": date_[0], "barrid": rm.ids, "score": -beta}))
        return pl.concat(out) if out else universe.head(0).with_columns(score=pl.lit(0.0))
