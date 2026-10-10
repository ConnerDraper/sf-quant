"""Writing your own signal.

A signal is a frozen dataclass that subclasses ``Signal`` and implements
``compute``. The library handles everything after that: winsorizing,
z-scoring, removing the Barra factors the signal may not keep, and turning the
score into an alpha.
"""

import datetime as dt
from dataclasses import dataclass

import polars as pl

from sf_quant.data import MarketData
from sf_quant.signals import Signal


@dataclass(frozen=True)
class LowSpecificVolatility(Signal):
    """
    Stocks whose specific (Barra-residual) returns have been calm are expected
    to outperform: minus the volatility of specific returns over ``lookback``
    trading days.
    """

    lookback: int = 63  # parameters are dataclass fields: LowSpecificVolatility(lookback=126)

    def compute(self, data: MarketData, universe: pl.DataFrame) -> pl.DataFrame:
        # ``universe`` has one row per (date, barrid) the library needs a score for.
        # Return date, barrid, score, where a larger score means a higher expected return.
        dates = universe["date"].unique().sort()
        history_start = dates[0] - dt.timedelta(days=self.lookback * 7 // 5 + 15)  # room for the first window
        returns = data.specific_returns(history_start, dates[-1], universe["barrid"].unique().to_list())

        # No look-ahead: the score for date t may use only data known by the
        # close of t. A backward rolling window ending at t satisfies that.
        scores = returns.sort("barrid", "date").with_columns(
            score=-pl.col("specific_return").rolling_std(self.lookback).over("barrid")
        )
        # Stocks without a score simply get no alpha.
        return universe.join(scores.select("date", "barrid", "score"), on=["date", "barrid"]).drop_nulls("score")


# Factor bets. By default a signal keeps no Barra factor: its score is made
# orthogonal to all of them, and the portfolio is held neutral to them. To let
# this signal keep its low-residual-volatility tilt, capped at 0.10 exposure:
#
#   from sf_quant.signals import Constraints, caps
#   LowSpecificVolatility(
#       allowed_factors=frozenset({"USSLOWL_RESVOL"}),
#       constraints=Constraints(factor_caps=caps({"USSLOWL_RESVOL"}, 0.10)),
#   )
#
# Sets combine with set algebra, e.g. STYLE_FACTORS | {"USSLOWL_BETA"}.
