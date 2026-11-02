"""CRSP delisting returns, resolved to one return per delisted barrid.

When a stock stops trading, its last move is the delisting return: the value
of a share at the delisting (a deal price, a liquidation payout, or what
little a failed firm's shares fetch) against its last traded price. Barra's
return series simply ends, so without this the final return counts as 0.

Resolution, per delisting:

- CRSP's ``delret`` when it exists, whatever the reason.
- Missing, and the delisting was for performance (bankruptcy, failing listing
  rules): Shumway (1997) and Shumway & Warther (1999), −30% on NYSE/AMEX and
  −55% on Nasdaq.
- Missing, and a merger, exchange or liquidation: 0, since those usually close
  at a known deal value near the last price.
- Missing and unclassified: 0, with a warning counting them.
"""

import datetime as dt
import warnings

import polars as pl

from ._tables import crsp_delisting_table
from .crsp_link import CrspTablesMissingError, load_barrid_permno_link

#: CIZ ``delactiontype`` -> category, checked against the full table on
#: 2026-10-21 (``delisting_type_counts``): MER 11,049 delistings, mean delret
#: +0.9%; GDR 7,545, mean -13.8%, 1,103 missing, reasons bankruptcy /
#: insufficient capital / delinquent filing / low price; GLI 2,781, mean -0.1%;
#: GEX 560, mean +1.1%. Unlisted values fall into "other" with a warning.
CATEGORY_BY_ACTION = {
    "MER": "merger",
    "GEX": "exchange",
    "GLI": "liquidation",
    "GDR": "performance",
}

#: Shumway imputation for a missing performance delisting return, by CIZ
#: ``primaryexch``. Any other exchange uses DEFAULT_PERFORMANCE_RETURN.
PERFORMANCE_RETURN_BY_EXCHANGE = {"N": -0.30, "A": -0.30, "Q": -0.55}
DEFAULT_PERFORMANCE_RETURN = -0.30


def _scan_delistings() -> pl.LazyFrame:
    try:
        frame = crsp_delisting_table.scan()
        frame.collect_schema()
    except (FileNotFoundError, pl.exceptions.ComputeError) as error:
        raise CrspTablesMissingError(
            "CRSP delisting data not found. Delisting returns need "
            f"{crsp_delisting_table._file_path()} and the crsp_security_info table "
            "in the data lake. Pass delistings=False to the return functions to "
            "run without them."
        ) from error
    return frame


def delisting_type_counts() -> pl.DataFrame:
    """Counts of every (action, reason, return-missing) type, to check the category map."""
    return (
        _scan_delistings()
        .group_by("delactiontype", "delreasontype", "delretmisstype")
        .agg(pl.len().alias("count"), pl.col("delret").is_null().sum().alias("missing_delret"))
        .sort("count", descending=True)
        .collect()
    )


def load_delistings(start: dt.date, end: dt.date) -> pl.DataFrame:
    """
    One resolved delisting return per delisted barrid, for last trading days
    between ``start`` and ``end``.

    Parameters
    ----------
    start, end : datetime.date
        Window on the last trading date (inclusive).

    Returns
    -------
    pl.DataFrame
        - ``barrid``, ``permno``
        - ``last_trading_date``: the last day the stock traded (CIZ
          ``delistingdt``)
        - ``delisting_return_date``: the day CRSP applies the delisting return
          (``deldlydt``), usually the next trading day
        - ``category``: merger, exchange, liquidation, performance or other
        - ``delret``: CRSP's delisting return, decimal (null if missing)
        - ``delret_used``: the resolved return, decimal (see module docstring)
        - ``delret_source``: ``"crsp"``, ``"shumway"`` or ``"zero"``

    Raises
    ------
    CrspTablesMissingError
        If the CRSP tables are not in the data lake.

    Examples
    --------
    >>> import datetime as dt, sf_quant.data as sfd
    >>> sfd.load_delistings(dt.date(2008, 1, 1), dt.date(2009, 12, 31))   # doctest: +SKIP
    """
    category = (
        pl.col("delactiontype")
        .replace_strict(CATEGORY_BY_ACTION, default="other", return_dtype=pl.String)
        .alias("category")
    )
    performance_return = pl.col("primaryexch").replace_strict(
        PERFORMANCE_RETURN_BY_EXCHANGE, default=DEFAULT_PERFORMANCE_RETURN, return_dtype=pl.Float64
    )

    delistings = (
        _scan_delistings()
        .with_columns(pl.col("delistingdt").alias("last_trading_date"))
        .filter(pl.col("last_trading_date").is_between(start, end))
        .select(
            "permno",
            "last_trading_date",
            pl.col("deldlydt").alias("delisting_return_date"),
            "delactiontype",
            category,
            pl.col("delret").cast(pl.Float64),
            "primaryexch",
        )
        .with_columns(
            pl.when(pl.col("delret").is_not_null()).then(pl.col("delret"))
            .when(pl.col("category") == "performance").then(performance_return)
            .otherwise(0.0)
            .alias("delret_used"),
            pl.when(pl.col("delret").is_not_null()).then(pl.lit("crsp"))
            .when(pl.col("category") == "performance").then(pl.lit("shumway"))
            .otherwise(pl.lit("zero"))
            .alias("delret_source"),
        )
        .collect()
    )

    unmapped = delistings.filter(pl.col("category") == "other")
    if unmapped.height:
        types = sorted(unmapped["delactiontype"].cast(pl.String).fill_null("null").unique().to_list())
        missing = unmapped["delret"].null_count()
        warnings.warn(
            f"{unmapped.height:,} delistings have an unmapped action type {types}; "
            f"{missing:,} of them have no delret and are set to 0",
            stacklevel=2,
        )

    link = load_barrid_permno_link()
    return (
        delistings.join(link, on="permno", how="inner")
        .filter(pl.col("last_trading_date").is_between(pl.col("start_date"), pl.col("end_date")))
        .select(
            "barrid", "permno", "last_trading_date", "delisting_return_date", "category",
            "delret", "delret_used", "delret_source",
        )
        .unique(["barrid", "last_trading_date"])
        .sort("last_trading_date", "barrid")
    )
