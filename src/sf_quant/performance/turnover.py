import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt

from sf_quant.performance.costs import _trades
from sf_quant.performance.leverage import generate_leverage_from_weights


def _compute_turnover(weights: pl.DataFrame) -> pl.DataFrame:
    # Generate leverage from weights
    leverage = generate_leverage_from_weights(weights)

    traded = _trades(weights, "weight").group_by("date").agg(
        pl.col("trade").abs().sum().alias("two_sided_turnover")
    )

    return (
        leverage.join(traded, on="date", how="left")
        # The first date is the starting portfolio; building it isn't turnover.
        .filter(pl.col("date") != weights["date"].min())
        .with_columns(pl.col("two_sided_turnover").fill_null(0.0))
        .sort("date")
        .with_columns(
            (pl.col("two_sided_turnover") / pl.col("leverage")).alias("two_sided_turnover")
        )
        .with_columns(pl.col("two_sided_turnover").rolling_mean(252))
        .drop("leverage")
    )


def get_turnover_stats(weights: pl.DataFrame) -> pl.DataFrame:
    """
    Compute summary statistics for two-sided portfolio turnover.

    Calculates the rolling 252-day mean two-sided turnover and summarises
    it with mean, min, and max statistics.

    Parameters
    ----------
    weights : pl.DataFrame
        Portfolio weights containing:

        - ``date`` (date): The observation date.
        - ``barrid`` (str): Security identifier.
        - ``weight`` (float): Portfolio weight.

    Returns
    -------
    pl.DataFrame
        Summary statistics with columns:

        - ``Mean Turnover`` (float): Mean of the rolling two-sided turnover.
        - ``Min Turnover`` (float): Minimum of the rolling two-sided turnover.
        - ``Max Turnover`` (float): Maximum of the rolling two-sided turnover.

    Notes
    -----
    - Two-sided turnover is the sum of absolute trades per date, divided by
      leverage. A trade is the weight minus the weight the previous
      portfolio drifted to with returns, the same trades
      :func:`sf_quant.performance.generate_trading_costs` charges, so this
      loads returns from the data lake. Opening and closing positions count;
      a position ended by a delisting doesn't, since nothing is traded.
    - The first date is the starting portfolio and isn't counted.
    - Rolling window is 252 trading days.
    - Null values (warm-up period) are excluded from summary statistics.

    Examples
    --------
    >>> import sf_quant.performance as sfp
    >>> sfp.get_turnover_stats(weights)   # doctest: +SKIP
    """
    turnover = _compute_turnover(weights)

    return (
        turnover.drop_nulls("two_sided_turnover")
        .select(
            pl.col("two_sided_turnover").mean().alias("Mean Turnover"),
            pl.col("two_sided_turnover").min().alias("Min Turnover"),
            pl.col("two_sided_turnover").max().alias("Max Turnover"),
        )
        .with_columns(cs.float().round(4))
    )


def plot_turnover(
    weights: pl.DataFrame,
    title: str,
    subtitle: str | None = None,
    file_name: str | None = None,
) -> None:
    """
    Plot rolling two-sided turnover over time.

    Parameters
    ----------
    weights : pl.DataFrame
        Portfolio weights containing:

        - ``date`` (date): The observation date.
        - ``barrid`` (str): Security identifier.
        - ``weight`` (float): Portfolio weight.
    title : str
        The chart's main title.
    subtitle : str or None, optional
        The chart's subtitle. Defaults to ``None``.
    file_name : str or None, optional
        If not ``None``, saves the chart to the given file path.
        Defaults to displaying the chart interactively.

    Returns
    -------
    None

    Notes
    -----
    - Two-sided turnover is the sum of absolute trades per date against
      drifted weights, divided by leverage, as in :func:`get_turnover_stats`.
    - Rolling window is 252 trading days.
    """
    turnover = _compute_turnover(weights)

    plt.figure(figsize=(10, 6))
    plt.plot(turnover["date"], turnover["two_sided_turnover"])

    plt.suptitle(title)
    plt.title(subtitle)

    plt.xlabel(None)
    plt.ylabel("Two-Sided Turnover (Rolling 252-Day Mean)")

    plt.grid()
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()
