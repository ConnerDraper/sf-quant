import polars as pl


def generate_drawdown_from_returns(returns: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate drawdowns for each portfolio over time.

    Parameters
    ----------
        returns (pl.DataFrame): Portfolio returns.
            Must include the following columns:
            - ``date`` (date): The observation date.
            - ``return`` (float): Daily portfolio return.

    Returns
    -------
        pl.DataFrame: A DataFrame with columns:
            - ``date`` (date): The observation date.
            - ``drawdown`` (float): Drawdown from peak (≤ 0).

    Notes
    -----
        - Drawdown is calculated as (current_value / peak_value) - 1
        - Values are ≤ 0, where 0 indicates a new peak
        - Calculated for a single portfolio time series


    Examples
    --------
    >>> import polars as pl
    >>> import sf_quant.performance as sfp
    >>> import datetime as dt
    >>> weights = pl.DataFrame(
    ...     {
    ...         'date': [dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 3)],
    ...         'barrid': ['USA06Z1', 'USA0771', 'USA06Z1', 'USA0771'],
    ...         'weight': [0.5, 0.5, 0.3, 0.7]
    ...     }
    ... )
    >>> returns = sfp.generate_returns_from_weights(weights)
    >>> drawdowns = sfp.generate_drawdown_from_returns(returns)
    >>> drawdowns
    """
    result = (
        returns.sort("date")
        .with_columns(pl.col("return").add(1).cum_prod().alias("portfolio_value"))
        .with_columns(pl.col("portfolio_value").cum_max().alias("peak"))
        .with_columns((pl.col("portfolio_value") / pl.col("peak") - 1).alias("drawdown"))
        .select("date", "drawdown")
    )
    return result
