"""What the assets, exposures, and factors loaders select and filter.

Each of these had its query written out twice, once per branch of a flag.
These tests pin the behavior of both branches so the duplication can be
removed without changing what the loaders return.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

import sf_quant.data as sfd

ID = ["date", "barrid"]


# --- in_universe ------------------------------------------------------------


def test_load_assets_in_universe_excludes_the_flagged_name(local_database):
    start, end = local_database.dates[0], local_database.dates[-1]
    out = sfd.load_assets(start, end, ["barrid"], in_universe=True)
    assert sorted(set(out["barrid"])) == local_database.barrids


@pytest.mark.parametrize("in_universe", [False, None])
def test_load_assets_without_the_universe_filter_returns_everything(local_database, in_universe):
    start, end = local_database.dates[0], local_database.dates[-1]
    out = sfd.load_assets(start, end, ["barrid"], in_universe=in_universe)
    assert local_database.out_of_universe in set(out["barrid"])


def test_load_assets_defaults_to_the_whole_table(local_database):
    """The flag is optional and defaults to no filter."""
    start, end = local_database.dates[0], local_database.dates[-1]
    assert local_database.out_of_universe in set(sfd.load_assets(start, end, ["barrid"])["barrid"])


@pytest.mark.parametrize(
    "in_universe, expect_flagged",
    [(True, False), (False, True)],
)
def test_load_assets_by_date_honors_the_universe_flag(local_database, in_universe, expect_flagged):
    out = sfd.load_assets_by_date(local_database.dates[0], in_universe, ["barrid"])
    assert (local_database.out_of_universe in set(out["barrid"])) is expect_flagged


def test_load_assets_by_date_returns_only_that_date(local_database):
    date_ = local_database.dates[0]
    assert sfd.load_assets_by_date(date_, False, ["date"])["date"].unique().to_list() == [date_]


# --- year partitioning ------------------------------------------------------
#
# load_assets_by_date reads the one file named for the date's year. Its two
# branches used to disagree: the in_universe branch globbed every year while
# the other scanned one. These cover the assumption that makes the single
# scan correct, because if it is wrong the loader returns an empty frame
# rather than raising.


def _add_year(local_database, table: str, date_: dt.date, barrids: list[str]) -> None:
    """Write a second year's file into an existing table."""
    directory = local_database.root / "groups" / "grp_quant" / "database" / local_database.database / table
    pl.DataFrame(
        {
            "date": [date_] * len(barrids),
            "barrid": barrids,
            "in_universe": [True] * len(barrids),
            "specific_risk": [25.0] * len(barrids),
            "market_cap": [1e9] * len(barrids),
            "price": [10.0] * len(barrids),
            "return": [0.5] * len(barrids),
        }
    ).write_parquet(directory / f"{table}_{date_.year}.parquet")
    sfd.clear_cache()


@pytest.mark.parametrize("in_universe", [True, False])
def test_load_assets_by_date_finds_a_date_in_another_year(local_database, in_universe):
    earlier = dt.date(local_database.dates[0].year - 1, 6, 15)
    expected = ["USA0009"]
    _add_year(local_database, "assets", earlier, expected)

    out = sfd.load_assets_by_date(earlier, in_universe, ["date", "barrid"])
    assert out["barrid"].to_list() == expected
    assert out["date"].unique().to_list() == [earlier]


@pytest.mark.parametrize("in_universe", [True, False])
def test_load_assets_by_date_opens_only_that_year(local_database, in_universe, monkeypatch):
    """Both branches must read one file, not glob every year."""
    earlier = dt.date(local_database.dates[0].year - 1, 6, 15)
    _add_year(local_database, "assets", earlier, ["USA0009"])

    paths: list[str] = []
    original = pl.scan_parquet
    monkeypatch.setattr(pl, "scan_parquet", lambda *a, **k: paths.append(str(a[0])) or original(*a, **k))

    sfd.load_assets_by_date(local_database.dates[0], in_universe, ["barrid"])
    assets_paths = [p for p in paths if "assets" in p]
    assert assets_paths, "expected the assets table to be scanned"
    assert all("*" not in p for p in assets_paths), f"globbed every year: {assets_paths}"
    assert all(str(local_database.dates[0].year) in p for p in assets_paths)


@pytest.mark.parametrize(
    "in_universe, expect_flagged",
    [(True, False), (False, True)],
)
def test_load_exposures_honors_the_universe_flag(local_database, in_universe, expect_flagged):
    start, end = local_database.dates[0], local_database.dates[-1]
    out = sfd.load_exposures(start, end, in_universe, ID)
    assert (local_database.out_of_universe in set(out["barrid"])) is expect_flagged


def test_load_exposures_respects_the_date_range(local_database):
    date_ = local_database.dates[0]
    assert sfd.load_exposures(date_, date_, False, ID)["date"].unique().to_list() == [date_]


# --- factor selection -------------------------------------------------------


def test_load_factors_returns_every_factor_by_default(local_database):
    start, end = local_database.dates[0], local_database.dates[-1]
    out = sfd.load_factors(start, end)
    assert out.columns[0] == "date"
    assert out.width == len(sfd.get_factor_names()) + 1


def test_load_factors_selects_the_named_factors(local_database):
    start, end = local_database.dates[0], local_database.dates[-1]
    out = sfd.load_factors(start, end, ["USSLOWL_BETA", "USSLOWL_VALUE"])
    assert out.columns == ["date", "USSLOWL_BETA", "USSLOWL_VALUE"]


def test_load_factors_treats_an_empty_list_as_every_factor(local_database):
    """
    An empty list is falsy and takes the no-selection branch. Surprising, but
    long-standing: pin it so the flattening does not quietly turn it into a
    frame with only a date column.
    """
    start, end = local_database.dates[0], local_database.dates[-1]
    assert sfd.load_factors(start, end, []).width == len(sfd.get_factor_names()) + 1


def test_load_factors_is_sorted_by_date(local_database):
    start, end = local_database.dates[0], local_database.dates[-1]
    out = sfd.load_factors(start, end)
    assert out["date"].to_list() == sorted(out["date"].to_list())
