import datetime as dt
import polars as pl

from ._tables import (
    Table,
    crsp_daily_table,
    crsp_monthly_table,
    crsp_v2_daily_table,
    crsp_v2_monthly_table,
)
from ._views import crsp_clean


def _load(table: Table, monthly: bool, start: dt.date, end: dt.date, columns: list[str]) -> pl.DataFrame:
    return (
        crsp_clean(table, monthly)
        .filter(pl.col("date").is_between(start, end))
        .sort(["permno", "date"])
        .select(columns)
        .collect()
    )


def load_crsp_daily(start: dt.date, end: dt.date, columns: list[str]) -> pl.DataFrame:
    """
    Load a Polars DataFrame of CRSP daily data between two dates.

    Rows are common stock (share code 10 or 11) on NYSE, AMEX, or Nasdaq
    (exchange code 1-3) with a valid return; share and exchange codes come
    from the CRSP events table, forward-filled per ``permno``. Price is
    reported as an absolute value.

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
        A DataFrame containing CRSP daily data between the specified dates,
        sorted by ``permno`` and ``date``, with only the selected columns.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> df = sfd.load_crsp_daily(
    ...     start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 31), columns=["date", "permno", "ret"]
    ... )   # doctest: +SKIP
    """
    return _load(crsp_daily_table, False, start, end, columns)


def load_crsp_v2_daily(start: dt.date, end: dt.date, columns: list[str]) -> pl.DataFrame:
    """
    Load a Polars DataFrame of CRSP v2 daily data between two dates.

    Identical to :func:`load_crsp_daily`, reading the CRSP v2 daily table.

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
    >>> df = sfd.load_crsp_v2_daily(
    ...     start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 31), columns=["date", "permno", "ret"]
    ... )   # doctest: +SKIP
    """
    return _load(crsp_v2_daily_table, False, start, end, columns)


def load_crsp_monthly(start: dt.date, end: dt.date, columns: list[str]) -> pl.DataFrame:
    """
    Load a Polars DataFrame of CRSP monthly data between two dates.

    Each month takes the last event recorded in that month. Otherwise the
    filters match :func:`load_crsp_daily`.

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
        A DataFrame containing CRSP monthly data between the specified dates,
        sorted by ``permno`` and ``date``, with only the selected columns.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> df = sfd.load_crsp_monthly(
    ...     start=dt.date(2024, 1, 1), end=dt.date(2024, 12, 31), columns=["date", "permno", "ret"]
    ... )   # doctest: +SKIP
    """
    return _load(crsp_monthly_table, True, start, end, columns)


def load_crsp_v2_monthly(start: dt.date, end: dt.date, columns: list[str]) -> pl.DataFrame:
    """
    Load a Polars DataFrame of CRSP v2 monthly data between two dates.

    Identical to :func:`load_crsp_monthly`, reading the CRSP v2 monthly table.

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
    >>> df = sfd.load_crsp_v2_monthly(
    ...     start=dt.date(2024, 1, 1), end=dt.date(2024, 12, 31), columns=["date", "permno", "ret"]
    ... )   # doctest: +SKIP
    """
    return _load(crsp_v2_monthly_table, True, start, end, columns)


def get_crsp_daily_columns() -> str:
    """
    Return the columns available to :func:`load_crsp_daily`.

    Returns
    -------
    str
        A printed Polars DataFrame of every column name and its data type.
    """
    return crsp_daily_table.columns()


def get_crsp_v2_daily_columns() -> str:
    """
    Return the columns available to :func:`load_crsp_v2_daily`.

    Returns
    -------
    str
        A printed Polars DataFrame of every column name and its data type.
    """
    return crsp_v2_daily_table.columns()


def get_crsp_monthly_columns() -> str:
    """
    Return the columns available to :func:`load_crsp_monthly`.

    Returns
    -------
    str
        A printed Polars DataFrame of every column name and its data type.
    """
    return crsp_monthly_table.columns()


def get_crsp_v2_monthly_columns() -> str:
    """
    Return the columns available to :func:`load_crsp_v2_monthly`.

    Returns
    -------
    str
        A printed Polars DataFrame of every column name and its data type.
    """
    return crsp_v2_monthly_table.columns()
