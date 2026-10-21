import polars as pl

from sf_quant.data.assets import load_trading_days

TRADING_DAYS_PER_YEAR = 252


def infer_periods_per_year(dates: pl.Series) -> float:
    """
    Return periods per year implied by the spacing of formation dates.

    Returns from :func:`generate_active_returns_from_weights` come one per
    holding period, not one per day, so a ``rebalance_period=21`` run has
    about 12 periods a year. Annualizing those at 252 would overstate its
    Sharpe by about √21, and since rebalancing is often the axis under test,
    that would read as "rebalancing less improves the signal".

    The spacing is the median number of trading days between consecutive
    formation dates, measured on the trading calendar, so weekends and
    holidays don't count.

    Parameters
    ----------
    dates : pl.Series
        Formation dates, e.g. the ``date`` column of a returns frame.

    Returns
    -------
    float
        ``252 / N`` for an N-trading-day holding period.

    Examples
    --------
    >>> import sf_quant.performance as sfp
    >>> returns = sfp.generate_active_returns_from_weights(weights)   # doctest: +SKIP
    >>> sfp.infer_periods_per_year(returns["date"])   # 12.0 for rebalance_period=21  # doctest: +SKIP
    """
    formation = dates.unique().sort()
    if formation.len() < 2:
        raise ValueError("need at least two formation dates to infer a holding period")
    calendar = pl.DataFrame({"date": load_trading_days(formation.min(), formation.max())}).with_row_index("t")
    positions = pl.DataFrame({"date": formation}).join(calendar, on="date", how="inner")["t"]
    if positions.len() < 2:
        raise ValueError("formation dates are not on the trading calendar")
    spacing = positions.diff().drop_nulls().median()
    return TRADING_DAYS_PER_YEAR / spacing
