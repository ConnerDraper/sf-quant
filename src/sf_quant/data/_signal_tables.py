"""Shared queries for the signal-shaped tables: signals, scores, and alphas.

Each holds one row per (date, barrid, signal_name), so their loaders differ
only in the table they read.
"""

import datetime as dt

import polars as pl

from ._tables import Table


def load_range(table: Table, start: dt.date, end: dt.date, names: list[str] | None) -> pl.DataFrame:
    frame = table.scan().filter(pl.col("date").is_between(start, end))
    if names is not None:
        frame = frame.filter(pl.col("signal_name").is_in(names))
    return frame.sort(["barrid", "date"]).collect()


def load_date(table: Table, date_: dt.date, names: list[str] | None) -> pl.DataFrame:
    # Without names, only the date's year file is scanned; with names, every
    # year's file is, as before.
    frame = table.scan(date_.year if names is None else None).filter(pl.col("date").eq(date_))
    if names is not None:
        frame = frame.filter(pl.col("signal_name").is_in(names))
    return frame.sort(["barrid", "date"]).collect()


def signal_names(table: Table) -> list[str]:
    return table.scan().select("signal_name").unique().collect()["signal_name"].to_list()
