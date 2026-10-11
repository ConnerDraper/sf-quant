import polars as pl

from ._tables import (
    Table,
    assets_table,
    crsp_daily_table,
    crsp_events_table,
    crsp_monthly_table,
    crsp_v2_monthly_table,
    crsp_v2_daily_table,
    ff_table,
)

COMMON_SHARE_CODES = [10, 11]
MAIN_EXCHANGE_CODES = [1, 2, 3]  # NYSE, AMEX, NASDAQ
MISSING_RETURNS = [-66.0, -77.0, -88.0, -99.0]  # CRSP's missing-return codes
EVENT_COLUMNS = ["ticker", "shrcd", "exchcd"]


def crsp_events_monthly():
    return (
        crsp_events_table.scan()
        .select(
            pl.col("date").dt.strftime("%Y-%m").alias("month_date"),
            "permno",
            "ticker",
            "shrcd",
            "exchcd",
        )
        .group_by(["month_date", "permno"])
        .agg(pl.col("ticker").last(), pl.col("shrcd").last(), pl.col("exchcd").last())
    )


def _clean(joined: pl.LazyFrame, *, keep_missing_events: bool) -> pl.LazyFrame:
    """
    The CRSP cleaning rules, applied to data already joined to its events.

    Keep US common shares on NYSE, AMEX, or NASDAQ, carrying event data
    forward over gaps. Monthly data keeps rows whose event data is missing
    entirely; daily data does not.
    """
    share_codes = COMMON_SHARE_CODES + ([None] if keep_missing_events else [])
    exchange_codes = MAIN_EXCHANGE_CODES + ([None] if keep_missing_events else [])
    return (
        joined.sort(["permno", "date"])
        .with_columns(pl.col(column).fill_null(strategy="forward").over("permno") for column in EVENT_COLUMNS)
        .filter(pl.col("shrcd").is_in(share_codes), pl.col("exchcd").is_in(exchange_codes))
        .with_columns(pl.col("prc").abs())  # CRSP stores a bid/ask average as a negative price
        .filter(~pl.col("ret").is_in(MISSING_RETURNS))
        .sort(["permno", "date"])
    )


def _monthly_clean(table: Table) -> pl.LazyFrame:
    """Events are aggregated to the month, then joined on the month."""
    return _clean(
        table.scan()
        .with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month_date"))
        .join(crsp_events_monthly(), on=["month_date", "permno"], how="left"),
        keep_missing_events=True,
    )


def _daily_clean(table: Table) -> pl.LazyFrame:
    return _clean(
        table.scan().join(crsp_events_table.scan(), on=["date", "permno"], how="left"),
        keep_missing_events=False,
    )


def crsp_monthly_clean():
    return _monthly_clean(crsp_monthly_table)


def crsp_v2_monthly_clean():
    return _monthly_clean(crsp_v2_monthly_table)


def crsp_daily_clean():
    return _daily_clean(crsp_daily_table)


def crsp_v2_daily_clean():
    return _daily_clean(crsp_v2_daily_table)


def benchmark():
    return (
        assets_table.scan()
        .filter(pl.col("in_universe"))
        .select(
            "date",
            "barrid",
            pl.col("market_cap").truediv(pl.col("market_cap").sum()).over("date").alias("weight"),
        )
    )


def benchmark_returns():
    # The return on date t is earned by the weight set at the close of t-1.
    # Cap weights at close t already include return t, so pairing them with
    # the same day's return overweights that day's winners.
    return (
        benchmark()
        .sort("barrid", "date")
        .with_columns(pl.col("weight").shift(1).over("barrid"))
        .join(
            assets_table.scan().filter(pl.col("in_universe")).select("date", "barrid", "return"),
            on=["date", "barrid"],
            how="inner",
        )
        .drop_nulls("weight")
    )


def fama_french():
    return ff_table.scan()
