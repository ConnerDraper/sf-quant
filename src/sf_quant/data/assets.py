import datetime as dt
from functools import lru_cache

import polars as pl

from ._config import register_cache
from ._tables import assets_table


def load_assets(start: dt.date, end: dt.date, columns: list[str], in_universe: bool | None = None) -> pl.DataFrame:
    """
    Load a Polars DataFrame of assets data between two dates.

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
        A DataFrame containing asset data between the specified dates,
        filtered by universe membership if requested, with only the
        selected columns.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 12, 31)
    >>> columns = ["barrid", "date", "price"]
    >>> df = sfd.load_assets(
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
            .filter(pl.col("date").is_between(start, end), pl.col("in_universe"))
            .sort(["barrid", "date"])
            .select(columns)
            .collect()
        )

    else:
        return (
            assets_table.scan()
            .filter(pl.col("date").is_between(start, end))
            .sort(["barrid", "date"])
            .select(columns)
            .collect()
        )


def load_assets_by_date(date_: dt.date, in_universe: bool, columns: list[str]) -> pl.DataFrame:
    """
    Load a Polars DataFrame of assets data for a single date.

    Parameters
    ----------
    date_ : datetime.date
        Date of the data frame.
    in_universe : bool
        If ``True``, restrict to assets that are in the universe.
        If ``False``, include all assets.
    columns : list of str
        List of column names to include in the result.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing asset data on the specified date,
        filtered by universe membership if requested, with only the
        selected columns.

    Examples
    --------
    >>> import sf_quant as sf
    >>> import datetime as dt
    >>> date_ = dt.date(2024, 1, 3)
    >>> columns = ["barrid", "date", "price"]
    >>> df = sf.data.load_assets_by_date(
    ...     date_=date_,
    ...     in_universe=True,
    ...     columns=columns
    ... )
    >>> df.head()

    Notes
    -----
    The returned frame is cached and shared between callers, so treat it as
    read-only. Call :func:`sf_quant.data.clear_cache` to drop it.
    """
    return _assets_by_date(date_, in_universe, tuple(columns))


@lru_cache(maxsize=64)
def _assets_by_date(date_: dt.date, in_universe: bool, columns: tuple[str, ...]) -> pl.DataFrame:
    """Cached body of :func:`load_assets_by_date` (``columns`` made hashable)."""
    if in_universe:
        return (
            assets_table.scan()
            .filter(pl.col("date").eq(date_), pl.col("in_universe"))
            .sort(["barrid", "date"])
            .select(list(columns))
            .collect()
        )
    else:
        return (
            assets_table.scan(date_.year)
            .filter(pl.col("date").eq(date_))
            .sort(["barrid", "date"])
            .select(list(columns))
            .collect()
        )


register_cache(_assets_by_date.cache_clear)


def get_assets_columns() -> str:
    """
    Return the available columns in the assets dataset.

    This function provides a schema of all asset-level fields that can be
    retrieved with :func:`load_assets`. The output is a table listing each
    column name along with its corresponding data type.

    Returns
    -------
    str
        A string representation of a polars data frame containing the
        column names and types for the assets table.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_assets_columns()
    """
    return assets_table.columns()
