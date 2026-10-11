"""Fixtures shared by the whole suite.

``local_database`` writes a tiny year-partitioned parquet store to a temp
directory and points the package at it, so the data-layer loaders can be
tested without the cluster.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

import sf_quant.data as sfd
from sf_quant.data._config import _config
from sf_quant.data._factors import factors

YEAR = 2024
DATES = [dt.date(YEAR, 1, 2), dt.date(YEAR, 1, 3)]
BARRIDS = ["USA0001", "USA0002", "USA0003"]


@dataclass(frozen=True)
class Database:
    """What the ``local_database`` fixture wrote, and where."""

    root: Path
    database: str
    dates: list[dt.date]
    barrids: list[str]


def _write(root: Path, database: str, table: str, frame: pl.DataFrame) -> None:
    """Write one table where ``Table._file_path`` expects to find it."""
    directory = root / "groups" / "grp_quant" / "database" / database / table
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / f"{table}_{YEAR}.parquet")


@pytest.fixture
def local_database(tmp_path: Path):
    """A minimal on-disk database, with the package pointed at it.

    Yields a :class:`Database`; restores the real environment afterwards.
    """
    database = "testdb"
    rows = [(d, b) for d in DATES for b in BARRIDS]
    date_column = pl.Series("date", [d for d, _ in rows])
    barrid_column = pl.Series("barrid", [b for _, b in rows])

    _write(
        tmp_path,
        database,
        "exposures",
        pl.DataFrame([date_column, barrid_column]).with_columns(*(pl.lit(0.1).alias(name) for name in factors)),
    )
    _write(
        tmp_path,
        database,
        "assets",
        pl.DataFrame([date_column, barrid_column])
        .with_columns(
            specific_risk=pl.lit(25.0),
            in_universe=pl.lit(True),
            market_cap=pl.lit(1e9),
            price=pl.lit(10.0),
        )
        .with_columns(pl.lit(0.5).alias("return")),
    )
    _write(
        tmp_path,
        database,
        "covariances",
        pl.DataFrame(
            {"date": [DATES[0]] * len(factors) + [DATES[1]] * len(factors), "factor_1": factors * 2}
        ).with_columns(*(pl.lit(1.0).alias(name) for name in factors)),
    )

    original = dict(_config)
    sfd.env(root=str(tmp_path), database=database)  # also clears the caches
    yield Database(tmp_path, database, list(DATES), list(BARRIDS))
    _config.update(original)
    sfd.clear_cache()
