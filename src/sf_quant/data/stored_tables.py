"""
Loaders for the research pipeline's stored tables: signals, scores, alphas,
and composite alphas. These read what the pipeline saved; to build signals
from market data, use :mod:`sf_quant.signals`.

The signals, scores, and alphas tables share one long layout keyed by
``date``, ``barrid``, and ``signal_name``, with one value column each:
``signal_value``, ``score``, and ``alpha_value``.
"""

import datetime as dt

import polars as pl

from ._tables import Table, alphas_table, composite_alphas_table, scores_table, signals_table


def _load(table: Table, start: dt.date, end: dt.date, names: list[str] | None) -> pl.DataFrame:
    frame = table.scan().filter(pl.col("date").is_between(start, end))
    if names is not None:
        frame = frame.filter(pl.col("signal_name").is_in(names))
    return frame.sort(["barrid", "date"]).collect()


def _load_by_date(table: Table, date_: dt.date, names: list[str] | None) -> pl.DataFrame:
    frame = table.scan(date_.year).filter(pl.col("date").eq(date_))
    if names is not None:
        frame = frame.filter(pl.col("signal_name").is_in(names))
    return frame.sort(["barrid", "date"]).collect()


def _names(table: Table) -> list[str]:
    return table.scan().select("signal_name").unique().collect()["signal_name"].to_list()


def load_signals(start: dt.date, end: dt.date, names: list[str] | None = None) -> pl.DataFrame:
    """
    Stored signals between ``start`` and ``end`` (inclusive), optionally only
    the named ones. Columns: ``date``, ``barrid``, ``signal_name``,
    ``signal_value``.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> df = sfd.load_signals(dt.date(2024, 1, 1), dt.date(2024, 12, 31), names=["momentum"])
    """
    return _load(signals_table, start, end, names)


def load_signals_by_date(date_: dt.date, names: list[str] | None = None) -> pl.DataFrame:
    """Stored signals on one date, optionally only the named ones."""
    return _load_by_date(signals_table, date_, names)


def get_signal_names() -> list[str]:
    """Names of the stored signals."""
    return _names(signals_table)


def load_scores(start: dt.date, end: dt.date, names: list[str] | None = None) -> pl.DataFrame:
    """
    Stored scores between ``start`` and ``end`` (inclusive), optionally only
    the named ones. Columns: ``date``, ``barrid``, ``signal_name``,
    ``score``.
    """
    return _load(scores_table, start, end, names)


def load_scores_by_date(date_: dt.date, names: list[str] | None = None) -> pl.DataFrame:
    """Stored scores on one date, optionally only the named ones."""
    return _load_by_date(scores_table, date_, names)


def get_score_names() -> list[str]:
    """Names of the stored scores."""
    return _names(scores_table)


def load_alphas(start: dt.date, end: dt.date, names: list[str] | None = None) -> pl.DataFrame:
    """
    Stored alphas between ``start`` and ``end`` (inclusive), optionally only
    the named ones. Columns: ``date``, ``barrid``, ``signal_name``,
    ``alpha_value``.
    """
    return _load(alphas_table, start, end, names)


def load_alphas_by_date(date_: dt.date, names: list[str] | None = None) -> pl.DataFrame:
    """Stored alphas on one date, optionally only the named ones."""
    return _load_by_date(alphas_table, date_, names)


def get_alpha_names() -> list[str]:
    """Names of the stored alphas."""
    return _names(alphas_table)


def load_composite_alphas(start: dt.date, end: dt.date) -> pl.DataFrame:
    """Stored composite alphas between ``start`` and ``end`` (inclusive). Columns: ``date``, ``barrid``, ``alpha``."""
    return composite_alphas_table.scan().filter(pl.col("date").is_between(start, end)).sort(["barrid", "date"]).collect()


def load_composite_alphas_by_date(date_: dt.date) -> pl.DataFrame:
    """Stored composite alphas on one date."""
    return composite_alphas_table.scan(date_.year).filter(pl.col("date").eq(date_)).sort(["barrid", "date"]).collect()
