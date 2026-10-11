"""Fixtures shared by the whole suite.

``local_database`` writes a tiny year-partitioned parquet store to a temp
directory and points the package at it, so the data-layer loaders can be
tested without the cluster.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import pytest

import sf_quant.data as sfd
from sf_quant.data._config import _config
from sf_quant.data._factors import factors

YEAR = 2024
DATES = [dt.date(YEAR, 1, 2), dt.date(YEAR, 1, 3)]
BARRIDS = ["USA0001", "USA0002", "USA0003"]
# Present in assets and exposures but flagged out of the universe, so the
# in_universe branches of the loaders have something to exclude.
OUT_OF_UNIVERSE = "USA0004"
ALL_BARRIDS = [*BARRIDS, OUT_OF_UNIVERSE]

# CRSP. Three days in one month, and permnos chosen to exercise every
# cleaning rule in _views: share code, exchange code, the absolute value of
# price, CRSP's missing-return codes, and the forward fill of event data.
CRSP_DAYS = [dt.date(YEAR, 1, 2), dt.date(YEAR, 1, 3), dt.date(YEAR, 1, 4)]
MONTH_END = dt.date(YEAR, 1, 31)
KEPT, DROPPED_SHRCD, MISSING_RETURN = 10001, 10002, 10003


@dataclass(frozen=True)
class Database:
    """What the ``local_database`` fixture wrote, and where."""

    root: Path
    database: str
    dates: list[dt.date]
    barrids: list[str]
    crsp_days: list[dt.date] = field(default_factory=lambda: list(CRSP_DAYS))
    month_end: dt.date = MONTH_END
    out_of_universe: str = OUT_OF_UNIVERSE


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
    rows = [(d, b) for d in DATES for b in ALL_BARRIDS]
    date_column = pl.Series("date", [d for d, _ in rows])
    barrid_column = pl.Series("barrid", [b for _, b in rows])
    in_universe = pl.Series("in_universe", [b != OUT_OF_UNIVERSE for _, b in rows])

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
            in_universe,
            specific_risk=pl.lit(25.0),
            market_cap=pl.lit(1e9),
            price=pl.lit(10.0),
        )
        .with_columns(pl.lit(0.5).alias("return")),
    )
    _write(
        tmp_path,
        database,
        "factors",
        pl.DataFrame({"date": DATES}).with_columns(*(pl.lit(0.25).alias(name) for name in factors)),
    )
    _write(
        tmp_path,
        database,
        "covariances",
        pl.DataFrame(
            {"date": [DATES[0]] * len(factors) + [DATES[1]] * len(factors), "factor_1": factors * 2}
        ).with_columns(*(pl.lit(1.0).alias(name) for name in factors)),
    )

    # CRSP events. KEPT has no row on the middle day, so the clean views have
    # to forward fill it rather than drop the day.
    events = [
        (CRSP_DAYS[0], KEPT, "AAA", 10, 1),
        (CRSP_DAYS[2], KEPT, "AAA", 10, 1),
        *((d, DROPPED_SHRCD, "BBB", 73, 1) for d in CRSP_DAYS),
        *((d, MISSING_RETURN, "CCC", 11, 2) for d in CRSP_DAYS),
    ]
    _write(
        tmp_path,
        database,
        "crsp_events",
        pl.DataFrame(
            {
                "date": [e[0] for e in events],
                "permno": [e[1] for e in events],
                "ticker": [e[2] for e in events],
                "shrcd": [e[3] for e in events],
                "exchcd": [e[4] for e in events],
            }
        ),
    )

    # Daily prices and returns. KEPT's middle price is negative, which CRSP
    # uses for a bid/ask average; MISSING_RETURN's middle return is -88.0, one
    # of CRSP's missing codes.
    daily = [
        *((d, KEPT, 0.01 * (i + 1), -11.0 if i == 1 else 10.0 + i) for i, d in enumerate(CRSP_DAYS)),
        *((d, DROPPED_SHRCD, 0.05, 20.0) for d in CRSP_DAYS),
        *((d, MISSING_RETURN, -88.0 if i == 1 else 0.02, 30.0) for i, d in enumerate(CRSP_DAYS)),
    ]
    daily_frame = pl.DataFrame(
        {
            "date": [r[0] for r in daily],
            "permno": [r[1] for r in daily],
            "ret": [r[2] for r in daily],
            "prc": [r[3] for r in daily],
        }
    )
    monthly_frame = pl.DataFrame(
        {
            "date": [MONTH_END] * 3,
            "permno": [KEPT, DROPPED_SHRCD, MISSING_RETURN],
            "ret": [0.05, 0.06, -99.0],
            "prc": [-20.0, 25.0, 35.0],
        }
    )
    for table, frame in [
        ("crsp_daily", daily_frame),
        ("crsp_v2_daily", daily_frame),
        ("crsp_monthly", monthly_frame),
        ("crsp_v2_monthly", monthly_frame),
    ]:
        _write(tmp_path, database, table, frame)

    original = dict(_config)
    sfd.env(root=str(tmp_path), database=database)  # also clears the caches
    yield Database(tmp_path, database, list(DATES), list(BARRIDS))
    _config.update(original)
    sfd.clear_cache()
