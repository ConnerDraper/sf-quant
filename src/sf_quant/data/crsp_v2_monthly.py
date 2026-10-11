import datetime as dt
import polars as pl

from ._views import crsp_v2_monthly_clean, crsp_v2_monthly_table


def load_crsp_v2_monthly(start: dt.date, end: dt.date, columns: list[str]) -> pl.DataFrame:
    """
    Load a Polars DataFrame of CRSP v2 monthly data between two dates.

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
        A DataFrame containing CRSP v2 monthly data between the specified dates,
        sorted by ``permno`` and ``date``, with only the selected columns.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 12, 31)
    >>> columns = ["permno", "date", "ret"]
    >>> df = sfd.load_crsp_v2_monthly(
    ...     start=start,
    ...     end=end,
    ...     columns=columns
    ... )
    >>> df.head()
    """
    return (
        crsp_v2_monthly_clean()
        .filter(pl.col("date").is_between(start, end))
        .sort(["permno", "date"])
        .select(columns)
        .collect()
    )


def get_crsp_v2_monthly_columns() -> str:
    """
    Return the available columns in the CRSP monthly dataset.

    This function returns a string representation of the CRSP monthly table schema
    as exposed by Polars. It can be used to inspect which columns are available
    for use with :func:`load_crsp_v2_monthly`.

    Returns
    -------
    str
        A string representation of the column names and types.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_crsp_v2_monthly_columns()
    """
    return crsp_v2_monthly_table.columns()
