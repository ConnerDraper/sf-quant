"""Half-spread trading costs per name and day.

Buying at the ask or selling at the bid, against a mid-price mark, costs half
the bid-ask spread: as a fraction of the amount traded,

.. math::

    c_i = \\frac{s_i}{2 p_i}

where :math:`s_i` is Barra's 30-day average daily bid-ask spread in dollars
(the day's spread where the average is missing) and :math:`p_i` the price.

**Barra's spreads start on 2010-11-22** (the 30-day average on 2011-06-30), and
the data lake has no other source of quotes, so half-spreads before then are
null.
"""

import datetime as dt

import polars as pl

from ._tables import assets_table

#: Barra's 30-day average of the daily bid-ask spread, in dollars.
SPREAD = "average_daily_bid_ask_spread_30"
#: Barra's bid-ask spread on the day, in dollars: the fallback where the
#: average is missing, which it is until 2011-06-30.
DAILY_SPREAD = "bid_ask_spread"
#: The first date Barra has bid-ask spreads.
SPREAD_START = dt.date(2010, 11, 22)


def load_half_spreads(
    start: dt.date, end: dt.date, barrids: list[str] | None = None
) -> pl.DataFrame:
    """
    Load each asset's half-spread, the cost of trading it as a fraction of the amount traded.

    Parameters
    ----------
    start : datetime.date
        First date to load (inclusive).
    end : datetime.date
        Last date to load (inclusive).
    barrids : list of str, optional
        Restrict to these assets, filtered while the files are scanned.
        Defaults to every asset, in the universe or not.

    Returns
    -------
    pl.DataFrame
        One row per asset row in the period:

        - ``date`` (date)
        - ``barrid`` (str)
        - ``half_spread`` (float): spread / (2 × price), a decimal; null where
          Barra has no spread, which is every name before 2010-11-22.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> sfd.load_half_spreads(dt.date(2024, 1, 2), dt.date(2024, 1, 31))   # doctest: +SKIP
    """
    frame = assets_table.scan().filter(pl.col("date").is_between(start, end))
    if barrids is not None:
        frame = frame.filter(pl.col("barrid").is_in(pl.Series(barrids, dtype=pl.String).implode()))
    return (
        frame.select(
            "date",
            "barrid",
            pl.coalesce(SPREAD, DAILY_SPREAD).truediv(pl.col("price").mul(2)).alias("half_spread"),
        )
        .sort("barrid", "date")
        .collect()
    )


def _trade_half_spreads(weight_dates: pl.Series, barrids: list[str]) -> pl.DataFrame:
    """
    The half-spread a trade into each weight date's weights pays, keyed by that weight date.

    Weights dated t earn day t's return, so they are put on at the previous
    trading day's close, and that close's spread is what the trade pays. A name
    with no asset row that day isn't trading (it has delisted, or its data has
    ended) and gets no row. The backtester and the cost accounting both use
    this, so the cost the optimizer plans for is the cost it is charged.
    """
    dates = weight_dates.unique().sort()
    # Two weeks covers the trading day before the first weight date.
    start, end = dates.min() - dt.timedelta(days=14), dates.max()
    calendar = (
        assets_table.scan()
        .filter(pl.col("date").is_between(start, end))
        .select(pl.col("date").unique().sort())
        .collect()
    )
    trade_days = pl.DataFrame({"date": dates}).join_asof(
        calendar.with_columns(pl.col("date").alias("trade_day")),
        on="date",
        strategy="backward",
        allow_exact_matches=False,
    )
    spreads = load_half_spreads(start, end, barrids).rename({"date": "trade_day"})
    return (
        trade_days.drop_nulls("trade_day")
        .join(spreads, on="trade_day", how="inner")
        .select("date", "barrid", "half_spread")
        .sort("date", "barrid")
    )
