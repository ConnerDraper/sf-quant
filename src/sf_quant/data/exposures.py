import datetime as dt
from functools import lru_cache

import polars as pl

from ._config import register_cache
from ._tables import exposures_table, assets_table


def load_exposures(start: dt.date, end: dt.date, in_universe: bool, columns: list) -> pl.DataFrame:
    """
    Load a Polars DataFrame of factor exposures data between two dates.

    Parameters
    ----------
    start : datetime.date
        Start date (inclusive) of the data frame.
    end : datetime.date
        End date (inclusive) of the data frame.
    in_universe : bool
        If ``True``, restrict to assets that are in the universe.
        If ``False``, include all assets.
    columns : list of str
        List of column names to include in the result.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing factor exposures data between the specified dates,
        filtered by universe membership if requested, with only the
        selected columns.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 12, 31)
    >>> columns = [
    ...     "date",
    ...     "barrid",
    ...     "USSLOWL_MOMENTUM",
    ...     "USSLOWL_VALUE"
    ... ]
    >>> df = sfd.load_exposures(
    ...     start=start,
    ...     end=end,
    ...     in_universe=True,
    ...     columns=columns
    ... )
    >>> df.head()
    """
    if in_universe:
        return (
            assets_table.scan()
            .filter(pl.col("in_universe"))
            .select("date", "barrid")
            .join(other=exposures_table.scan(), on=["date", "barrid"], how="left")
            .filter(pl.col("date").is_between(start, end))
            .select(columns)
            .sort("barrid", "date")
            .collect()
        )

    else:
        return (
            exposures_table.scan()
            .filter(pl.col("date").is_between(start, end))
            .select(columns)
            .sort("barrid", "date")
            .collect()
        )


@lru_cache(maxsize=64)
def load_exposures_by_date(date_: dt.date) -> pl.DataFrame:
    """
    Load all factor exposures for a single date.

    The returned frame is cached and shared between callers, so treat it as
    read-only. Call :func:`sf_quant.data.clear_cache` to drop it.
    """
    return exposures_table.scan(date_.year).filter(pl.col("date").eq(date_)).sort("barrid", "date").collect()


register_cache(load_exposures_by_date.cache_clear)


def get_exposures_columns() -> str:
    """
    Return the available columns in the exposures dataset.

    This function provides a schema of all barra factor exposures that can be
    retrieved with :func:`load_exposures`. The output is a table listing each
    column name along with its corresponding data type.

    Returns
    -------
    str
        A string representation of a polars data frame containing the
        column names and types for the exposures table.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_exposures_columns()
    """
    return exposures_table.columns()
