"""Loaders for the four CRSP datasets: daily and monthly, each in two vintages.

All four narrow a cleaned view to a date range and collect it. The cleaning
rules — share code, exchange, the absolute value of price, and CRSP's
missing-return codes — live once in :mod:`sf_quant.data._views`; only the
table differs between these loaders.
"""

import datetime as dt

import polars as pl

from ._tables import crsp_daily_table, crsp_monthly_table, crsp_v2_daily_table, crsp_v2_monthly_table
from ._views import crsp_daily_clean, crsp_monthly_clean, crsp_v2_daily_clean, crsp_v2_monthly_clean


def _load(clean: pl.LazyFrame, start: dt.date, end: dt.date, columns: list[str]) -> pl.DataFrame:
    """Narrow a cleaned CRSP view to ``[start, end]`` and the given columns."""
    return clean.filter(pl.col("date").is_between(start, end)).sort(["permno", "date"]).select(columns).collect()


def load_crsp_daily(start: dt.date, end: dt.date, columns: list[str]) -> pl.DataFrame:
    """
    Load a Polars DataFrame of CRSP daily data between two dates.

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
    ...     start=dt.date(2024, 1, 1),
    ...     end=dt.date(2024, 1, 31),
    ...     columns=["permno", "date", "ret"],
    ... )
    """
    return _load(crsp_daily_clean(), start, end, columns)


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
    >>> df = sfd.load_crsp_v2_daily(
    ...     start=dt.date(2024, 1, 1),
    ...     end=dt.date(2024, 1, 31),
    ...     columns=["permno", "date", "ret"],
    ... )
    """
    return _load(crsp_v2_daily_clean(), start, end, columns)


def load_crsp_monthly(start: dt.date, end: dt.date, columns: list[str]) -> pl.DataFrame:
    """
    Load a Polars DataFrame of CRSP monthly data between two dates.

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
    ...     start=dt.date(2024, 1, 1),
    ...     end=dt.date(2024, 12, 31),
    ...     columns=["permno", "date", "ret"],
    ... )
    """
    return _load(crsp_monthly_clean(), start, end, columns)


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
        A DataFrame containing CRSP v2 monthly data between the specified
        dates, sorted by ``permno`` and ``date``, with only the selected
        columns.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> df = sfd.load_crsp_v2_monthly(
    ...     start=dt.date(2024, 1, 1),
    ...     end=dt.date(2024, 12, 31),
    ...     columns=["permno", "date", "ret"],
    ... )
    """
    return _load(crsp_v2_monthly_clean(), start, end, columns)


def get_crsp_daily_columns() -> str:
    """
    Return the available columns in the CRSP daily dataset.

    The output lists each column name available to :func:`load_crsp_daily`
    along with its data type.

    Returns
    -------
    str
        A string representation of a polars data frame containing the column
        names and types for the CRSP daily table.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_crsp_daily_columns()
    """
    return crsp_daily_table.columns()


def get_crsp_v2_daily_columns() -> str:
    """
    Return the available columns in the CRSP v2 daily dataset.

    The output lists each column name available to :func:`load_crsp_v2_daily`
    along with its data type.

    Returns
    -------
    str
        A string representation of a polars data frame containing the column
        names and types for the CRSP v2 daily table.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_crsp_v2_daily_columns()
    """
    return crsp_v2_daily_table.columns()


def get_crsp_monthly_columns() -> str:
    """
    Return the available columns in the CRSP monthly dataset.

    The output lists each column name available to :func:`load_crsp_monthly`
    along with its data type.

    Returns
    -------
    str
        A string representation of a polars data frame containing the column
        names and types for the CRSP monthly table.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_crsp_monthly_columns()
    """
    return crsp_monthly_table.columns()


def get_crsp_v2_monthly_columns() -> str:
    """
    Return the available columns in the CRSP v2 monthly dataset.

    The output lists each column name available to
    :func:`load_crsp_v2_monthly` along with its data type.

    Returns
    -------
    str
        A string representation of a polars data frame containing the column
        names and types for the CRSP v2 monthly table.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_crsp_v2_monthly_columns()
    """
    return crsp_v2_monthly_table.columns()
