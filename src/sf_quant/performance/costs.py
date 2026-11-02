"""Half-spread trading costs of a weight history.

Each trade crosses half the bid-ask spread (buying at the ask or selling at the
bid against a mid-price mark), so it costs

.. math::

    c_i = |\\Delta w_i| \\, \\frac{s_i}{2 p_i}

where :math:`\\Delta w_i` is the new weight minus the weight the previous
portfolio drifted to over its holding period, and :math:`s_i / 2 p_i` the
half-spread at the close the trade is made at (see
:func:`sf_quant.data.load_half_spreads`).

**Trades before 2010-11-22 cost 0.** Barra's bid-ask spreads start on that
date and the data lake has no other source of quotes, so every trade before
then, and the odd name-day after it without a spread, is costed at 0 and a
:class:`MissingSpreadWarning` says how much of the trading that covers.
"""

import sys
import warnings
from pathlib import Path

import polars as pl

from sf_quant.data.spreads import SPREAD_START, _trade_half_spreads
from sf_quant.performance.returns import _period_returns


class MissingSpreadWarning(UserWarning):
    """Some trades have no bid-ask spread, so they are costed at 0."""


def _trades(weights: pl.DataFrame, weight: str) -> pl.DataFrame:
    """Every name's trade into every weight date after the first, with its half-spread.

    The trade is the new weight minus the drifted weight, a missing row being a
    weight of 0, so opening and closing a position are trades too. The first
    date is the starting portfolio and isn't a trade. A name with no asset row
    on the day the trade is made isn't trading (it delisted, or its data ended),
    so nothing is traded and it has no row.
    """
    dates = weights["date"].unique().sort()
    held = weights.select("date", "barrid", weight)

    drifted = (
        held.join(_period_returns(held["barrid"], held["date"]), on=["date", "barrid"], how="left")
        .with_columns(pl.col("return").fill_null(0.0))
        .with_columns(
            pl.col(weight)
            .mul(pl.col("return").add(1))
            .truediv(pl.col(weight).mul(pl.col("return")).sum().over("date").add(1))
            .alias("drifted")
        )
        .join(pl.DataFrame({"date": dates, "next": dates.shift(-1)}), on="date")
        .drop_nulls("next")
        .select(pl.col("next").alias("date"), "barrid", "drifted")
    )

    trades = (
        held.filter(pl.col("date") != dates.min())
        .join(drifted, on=["date", "barrid"], how="full", coalesce=True)
        .select(
            "date",
            "barrid",
            pl.col(weight).fill_null(0.0).sub(pl.col("drifted").fill_null(0.0)).alias("trade"),
        )
        .filter(pl.col("trade") != 0)
    )
    if trades.is_empty():
        return trades.with_columns(pl.lit(None, dtype=pl.Float64).alias("half_spread"))

    spreads = _trade_half_spreads(dates, trades["barrid"].unique().to_list())
    return trades.join(spreads, on=["date", "barrid"], how="inner").sort("date", "barrid")


def generate_trading_costs(weights: pl.DataFrame, weight: str = "weight") -> pl.DataFrame:
    """
    Half-spread cost of every trade a weight history makes.

    Parameters
    ----------
    weights : pl.DataFrame
        Weights with ``date``, ``barrid`` and the ``weight`` column, as from a
        backtest.
    weight : str, default "weight"
        The weight column to trade. Total weights give what the fund pays;
        active weights give the cost of the active positions alone (their
        drift is then only approximate, since it ignores the benchmark).

    Returns
    -------
    pl.DataFrame
        One row per non-zero trade:

        - ``date`` (date): the weight date the trade puts on.
        - ``barrid`` (str): the asset.
        - ``trade`` (float): new weight minus the weight the previous
          portfolio drifted to.
        - ``half_spread`` (float): half the bid-ask spread over price, at the
          previous trading day's close; null where Barra has no spread, which
          is every name before 2010-11-22.
        - ``cost`` (float): ``|trade| * half_spread``, as a fraction of
          capital; 0 where ``half_spread`` is null.

    Warns
    -----
    MissingSpreadWarning
        If any trade has no spread and so costs 0.

    Notes
    -----
    - The first date is the starting portfolio, so it has no trades.
    - A position that ends because the name stopped trading (a delisting) is
      paid out, not traded, and costs nothing.

    Examples
    --------
    >>> import sf_quant.performance as sfp
    >>> costs = sfp.generate_trading_costs(weights)   # doctest: +SKIP
    """
    costs = _trades(weights, weight).with_columns(
        pl.col("trade").abs().mul(pl.col("half_spread")).fill_null(0.0).alias("cost")
    )
    _warn_missing_spreads(costs)
    return costs


def _warn_missing_spreads(costs: pl.DataFrame) -> None:
    missing = costs.filter(pl.col("half_spread").is_null())
    if not missing.height:
        return
    share = missing["trade"].abs().sum() / costs["trade"].abs().sum()
    before = missing.filter(pl.col("date") <= SPREAD_START)
    message = (
        f"Trading costs are 0 for {share:.1%} of traded weight, because those trades have "
        f"no bid-ask spread ({missing['date'].n_unique():,} of {costs['date'].n_unique():,} "
        f"trade dates, {missing['date'].min()} to {missing['date'].max()})."
    )
    if before.height:
        message += (
            f" Barra's spreads start on {SPREAD_START} and the data lake has no other source of "
            f"quotes, so every trade before {SPREAD_START} costs 0: net returns before then are "
            "really gross returns."
        )
    if before.height < missing.height:
        message += (
            f" After {SPREAD_START}, {missing.height - before.height:,} trades lack a spread "
            "for individual names and cost 0 too."
        )
    warnings.warn(message, MissingSpreadWarning, stacklevel=_outside_sf_quant())


def _outside_sf_quant() -> int:
    """The stacklevel of the first caller outside sf_quant, so a warning points at the user's code."""
    package = str(Path(__file__).resolve().parents[1])
    frame, level = sys._getframe(1), 1
    while frame is not None and frame.f_code.co_filename.startswith(package):
        frame, level = frame.f_back, level + 1
    return level


def _period_costs(weights: pl.DataFrame) -> pl.DataFrame:
    """Total cost per weight date (``date``, ``cost``) of trading total weights."""
    return generate_trading_costs(weights, "weight").group_by("date").agg(pl.col("cost").sum())
