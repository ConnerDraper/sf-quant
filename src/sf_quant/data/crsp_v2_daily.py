import datetime as dt
import polars as pl

from ._views import crsp_v2_daily_clean, crsp_v2_daily_table


def load_crsp_v2_daily(start: dt.date, end: dt.date, columns: list[str]) -> pl.DataFrame:
    """
    Load a Polars DataFrame of CRSP v2 daily data between two dates.

    Parameters
    ----------
    start : datetime.date
        Start date (inclusive) of the data frame.
    end : datetime.date
        End date (inclusive) of the data frame.
    columns : list of str
        List of column names to include in the result.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing CRSP v2 daily data between the specified dates,
        sorted by ``permno`` and ``date``, with only the selected columns.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 1, 31)
    >>> columns = ["permno", "date", "ret"]
    >>> df = sfd.load_crsp_v2_daily(
    ...     start=start,
    ...     end=end,
    ...     columns=columns
    ... )
    >>> df.head()
    """
    return (
        crsp_v2_daily_clean()
        .filter(pl.col("date").is_between(start, end))
        .sort(["permno", "date"])
        .select(columns)
        .collect()
    )


def get_crsp_v2_daily_columns() -> str:
    """
    Return the available columns in the CRSP v2 daily dataset.

    This function provides a schema of all CRSP v2 daily fields that can be
    retrieved with :func:`load_crsp_v2_daily`. The output lists each column name
    along with its corresponding data type.

    Returns
    -------
    str
        A string representation of a polars data frame containing the
        column names and types for the CRSP v2 daily table.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_crsp_v2_daily_columns()
    """
    return crsp_v2_daily_table.columns()
