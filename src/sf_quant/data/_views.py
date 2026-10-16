import polars as pl

from ._tables import Table, assets_table, crsp_events_table, ff_table


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


def crsp_clean(table: Table, monthly: bool) -> pl.LazyFrame:
    """CRSP returns with share/exchange codes attached and the standard filters.

    Monthly rows take the last event of their month; daily rows take the event
    on their date. Either way, codes are forward-filled per permno, and only
    common stock (share code 10/11) on NYSE/AMEX/Nasdaq (exchange 1-3) with a
    real return is kept, with the bid/ask-midpoint sign removed from price.
    """
    if monthly:
        frame = (
            table.scan()
            .with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month_date"))
            .join(crsp_events_monthly(), on=["month_date", "permno"], how="left")
        )
        share_codes, exchange_codes = [10, 11, None], [1, 2, 3, None]
    else:
        frame = table.scan().join(crsp_events_table.scan(), on=["date", "permno"], how="left")
        share_codes, exchange_codes = [10, 11], [1, 2, 3]

    return (
        frame.sort(["permno", "date"])
        .with_columns(
            pl.col("ticker").fill_null(strategy="forward").over("permno"),
            pl.col("shrcd").fill_null(strategy="forward").over("permno"),
            pl.col("exchcd").fill_null(strategy="forward").over("permno"),
        )
        .filter(pl.col("shrcd").is_in(share_codes), pl.col("exchcd").is_in(exchange_codes))
        .with_columns(pl.col("prc").abs())
        .filter(~pl.col("ret").is_in([-66.0, -77.0, -88.0, -99.0]))
        .sort(["permno", "date"])
    )


def benchmark(*columns: str) -> pl.LazyFrame:
    """Cap weights over the universe on each date, plus any asset ``columns``."""
    return (
        assets_table.scan()
        .filter(pl.col('in_universe'))
        .select(
            "date",
            "barrid",
            pl.col("market_cap")
            .truediv(pl.col("market_cap").sum())
            .over("date")
            .alias("weight"),
            *columns,
        )
    )


def fama_french():
    return ff_table.scan()
