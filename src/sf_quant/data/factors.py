import datetime as dt
import polars as pl
from typing import Literal

from ._tables import factors_table
from ._factors import factors, style_factors, sector_factors


def load_factors(start: dt.date, end: dt.date, factors: list[str] | None = None) -> pl.DataFrame:
    """
    Load a Polars DataFrame of factor returns(%) between two dates.

    Parameters
    ----------
    start : datetime.date
        Start date (inclusive) of the data frame.
    end : datetime.date
        End date (inclusive) of the data frame.
    factors : list of str, optional
        List of factor names to select.
        - If provided, only the specified factors are included in the output.
        - If None (default), all available factors are included.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing factor returns(%) between the specified dates.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 12, 31)
    >>> df = sfd.load_factors(
    ...     start=start,
    ...     end=end
    ... )
    >>> df.head()
    """
    frame = factors_table.scan().filter(pl.col("date").is_between(start, end)).sort("date")
    if factors:  # an empty list means every factor, as it always has
        frame = frame.select(["date"] + factors)
    return frame.collect()


def get_factors_columns() -> str:
    """
    Return the available columns in the factors dataset.

    This function provides a schema of all factors that can be
    retrieved with :func:`load_factors`. The output is a table listing each
    column name along with its corresponding data type.

    Returns
    -------
    str
        A string representation of a polars data frame containing the
        column names and types for the factors table.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_factors_columns()
    """
    return factors_table.columns()


def get_factor_names(type: Literal["style", "sector"] | None = None) -> list[str]:
    """
    Return the list of available factor names.

    This function provides the names of factors that can be retrieved
    with :func:`load_factors`. By default, all factor names are returned.
    Optionally, the list can be restricted to either style or sector factors.

    Parameters
    ----------
    type : {"style", "sector"}, optional
        The category of factors to return.
        - "style" : return only style factor names
        - "sector": return only sector/industry factor names
        - None (default): return all available factors

    Returns
    -------
    list of str
        A list of factor names matching the specified type.

    Raises
    ------
    ValueError
        If `type` is provided but is not one of {"style", "sector", None}.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_factor_names()
    ["USSLOWL_BETA", "USSLOWL_VALUE", "USSLOWL_AERODEF", ...]

    >>> sfd.get_factor_names("style")
    ["USSLOWL_BETA", "USSLOWL_VALUE", "USSLOWL_MOMENTUM", ...]

    >>> sfd.get_factor_names("sector")
    ["USSLOWL_AERODEF", "USSLOWL_AIRLINES", "USSLOWL_BANKS", ...]
    """

    if type == "style":
        return style_factors
    if type == "sector":
        return sector_factors
    if type is None:
        return factors
    raise ValueError(f"Invalid type: {type!r}. Must be 'style', 'sector', or None.")
