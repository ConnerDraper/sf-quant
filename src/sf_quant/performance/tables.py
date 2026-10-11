import polars as pl
import polars.selectors as cs


def generate_multi_returns_summary_table(returns: pl.DataFrame, wide: bool = True) -> pl.DataFrame:
    """
    Generate a summary statistics table for multiple portfolio returns.

    This function calculates performance metrics for each portfolio (total, benchmark, active),
    including mean return, volatility, total return, and Sharpe ratio. Results can be
    returned either in wide format (one row per portfolio) or in long format
    (statistics transposed into rows).

    Parameters
    ----------
    returns : pl.DataFrame
        Portfolio returns.
        Must include the following columns:

        - ``date`` (date): The observation date.
        - ``portfolio`` (str): Portfolio name or identifier.
        - ``return`` (float): Daily portfolio return.
    wide : bool, optional
        If ``True`` (default), return the summary in wide
        format with one row per portfolio. If ``False``, return the summary
        transposed with statistics as rows.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing portfolio summary statistics.
        The exact structure depends on the ``wide`` argument:

        **Wide format (default):**

        - ``Portfolio`` (str): Portfolio name.
        - ``Count`` (int): Number of days in the sample.
        - ``Mean Return (%)`` (float): Annualized mean return (in percent).
        - ``Volatility (%)`` (float): Annualized volatility (in percent).
        - ``Total Return (%)`` (float): Total return over the period (in percent).
        - ``Sharpe`` (float): Sharpe ratio.

        **Long format (wide=False):**

        - ``statistics`` (str): Statistic name.
        - One column per portfolio containing the respective values.

    Notes
    -----
    - Annualization assumes 252 trading days per year.
    - Returns are converted to percentages for readability.
    - Sharpe ratio is calculated as ``mean_return / volatility``.

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
    >>> summary = sfp.generate_summary_table(returns)
    >>> summary
    """
    summary_wide = (
        returns.group_by("portfolio")
        .agg(
            pl.col("date").n_unique().alias("n_days"),
            pl.col("return").mean().alias("mean_return"),
            pl.col("return").std().alias("volatility"),
            pl.col("return").add(1).product().sub(1).alias("total_return"),
        )
        .with_columns(
            pl.col("mean_return").mul(252),
            pl.col("volatility").mul(pl.lit(252).sqrt()),
        )
        .with_columns(pl.col("mean_return").truediv("volatility").alias("sharpe"))
        .with_columns(pl.col("mean_return", "volatility", "total_return").mul(100))
        .sort("portfolio")
        .with_columns(pl.col("portfolio").str.to_titlecase())
        .rename(
            {
                "portfolio": "Portfolio",
                "n_days": "Count",
                "mean_return": "Mean Return (%)",
                "volatility": "Volatility (%)",
                "total_return": "Total Return (%)",
                "sharpe": "Sharpe",
            }
        )
        .with_columns(cs.float().round(2))
    )

    if wide:
        return summary_wide

    else:
        return (
            summary_wide.drop("Portfolio")
            .transpose(
                include_header=True,
                header_name="statistics",
                column_names=summary_wide["Portfolio"],
            )
            .with_columns(cs.float().round(2))
        )


def generate_returns_summary_table(returns: pl.DataFrame) -> pl.DataFrame:
    """
    Generate a summary statistics table for single portfolio returns.

    This function calculates performance metrics for a single portfolio,
    including mean return, volatility, total return, and Sharpe ratio.

    Parameters
    ----------
    returns : pl.DataFrame
        Portfolio returns.
        Must include the following columns:

        - ``date`` (date): The observation date.
        - ``return`` (float): Daily portfolio return.

    Returns
    -------
    pl.DataFrame
        A single-row DataFrame containing portfolio summary statistics:

        - ``Count`` (int): Number of days in the sample.
        - ``Mean Return (%)`` (float): Annualized mean return (in percent).
        - ``Volatility (%)`` (float): Annualized volatility (in percent).
        - ``Total Return (%)`` (float): Total return over the period (in percent).
        - ``Sharpe`` (float): Sharpe ratio.

    Notes
    -----
    - Annualization assumes 252 trading days per year.
    - Returns are converted to percentages for readability.
    - Sharpe ratio is calculated as ``mean_return / volatility``.

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
    >>> summary = sfp.generate_returns_summary_table(returns)
    >>> summary
    """
    return (
        returns.select(
            pl.col("date").n_unique().alias("Count"),
            pl.col("return").mean().mul(252).alias("Mean Return (%)"),
            pl.col("return").std().mul(pl.lit(252).sqrt()).alias("Volatility (%)"),
            pl.col("return").add(1).product().sub(1).mul(100).alias("Total Return (%)"),
        )
        .with_columns(pl.col("Mean Return (%)").truediv("Volatility (%)").alias("Sharpe"))
        .with_columns(pl.col("Mean Return (%)", "Volatility (%)").mul(100))
        .with_columns(cs.float().round(2))
    )


def generate_drawdown_summary_table(drawdown: pl.DataFrame) -> pl.DataFrame:
    """
    Generate a summary statistics table for portfolio drawdowns.

    This function calculates summary metrics for portfolio drawdowns,
    including mean, min (max drawdown), current drawdown, and longest drawdown period.

    Parameters
    ----------
    drawdown : pl.DataFrame
        Portfolio drawdowns.
        Must include the following columns:

        - ``date`` (date): The observation date.
        - ``drawdown`` (float): Daily portfolio drawdown (≤ 0).

    Returns
    -------
    pl.DataFrame
        A single-row DataFrame containing drawdown summary statistics:

        - ``Count`` (int): Number of days in the sample.
        - ``Mean Drawdown (%)`` (float): Average drawdown (in percent).
        - ``Max Drawdown (%)`` (float): Maximum drawdown (in percent).
        - ``Current Drawdown (%)`` (float): Most recent drawdown (in percent).
        - ``Longest Drawdown (days)`` (int): Longest consecutive period in drawdown.

    Notes
    -----
    - Drawdowns are expressed as negative percentages.
    - A drawdown of 0% indicates the portfolio is at a new peak.
    - Max Drawdown represents the largest peak-to-trough decline.
    - Longest Drawdown counts consecutive days where drawdown < 0.

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
    >>> drawdown = sfp.generate_drawdown_from_returns(returns)
    >>> summary = sfp.generate_drawdown_summary_table(drawdown)
    >>> summary
    """
    # Calculate longest drawdown period
    longest_dd = (
        drawdown.with_columns(
            # Create groups where drawdown < 0 (in drawdown)
            (pl.col("drawdown") < 0).alias("in_drawdown")
        )
        .with_columns(
            # Create run ID for consecutive drawdown periods
            (pl.col("in_drawdown") != pl.col("in_drawdown").shift(1)).cum_sum().alias("run_id")
        )
        .filter(pl.col("in_drawdown"))  # Only keep drawdown periods
        .group_by("run_id")
        .agg(pl.len().alias("duration"))
        .select(pl.col("duration").max().alias("longest_drawdown"))
        .item()
    )

    # Handle case where there's no drawdown
    if longest_dd is None:
        longest_dd = 0

    return (
        drawdown.select(
            pl.col("date").n_unique().alias("Count"),
            pl.col("drawdown").mean().mul(100).alias("Mean Drawdown (%)"),
            pl.col("drawdown").min().mul(100).alias("Max Drawdown (%)"),
            pl.col("drawdown").last().mul(100).alias("Current Drawdown (%)"),
        )
        .with_columns(pl.lit(longest_dd).alias("Longest Drawdown (days)"))
        .with_columns(cs.float().round(2))
    )
