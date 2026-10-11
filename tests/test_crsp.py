"""What the CRSP loaders return, rule by rule.

The four datasets (daily and monthly, each in two vintages) share one set of
cleaning rules: keep US common shares on NYSE/AMEX/NASDAQ, carry event data
forward over gaps, take the absolute value of price because CRSP stores a
bid/ask average as negative, and drop CRSP's missing-return codes.

These are characterization tests: they pin the behavior the loaders had
before the four copies were merged, so the merge has to preserve it.
"""

from __future__ import annotations

import pytest

import sf_quant.data as sfd

COLUMNS = ["date", "permno", "ret", "prc"]

DAILY_LOADERS = [
    pytest.param(sfd.load_crsp_daily, id="daily"),
    pytest.param(sfd.load_crsp_v2_daily, id="v2_daily"),
]
MONTHLY_LOADERS = [
    pytest.param(sfd.load_crsp_monthly, id="monthly"),
    pytest.param(sfd.load_crsp_v2_monthly, id="v2_monthly"),
]
COLUMN_GETTERS = [
    pytest.param(sfd.get_crsp_daily_columns, id="daily"),
    pytest.param(sfd.get_crsp_v2_daily_columns, id="v2_daily"),
    pytest.param(sfd.get_crsp_monthly_columns, id="monthly"),
    pytest.param(sfd.get_crsp_v2_monthly_columns, id="v2_monthly"),
]


def _daily(load, local_database):
    days = local_database.crsp_days
    return load(days[0], days[-1], COLUMNS)


@pytest.mark.parametrize("load", DAILY_LOADERS)
def test_daily_applies_every_cleaning_rule(load, local_database):
    assert _daily(load, local_database).to_dicts() == [
        # 10001 is kept on all three days: the middle day has no event row and
        # is carried forward, and its negative price comes back positive.
        {"date": local_database.crsp_days[0], "permno": 10001, "ret": 0.01, "prc": 10.0},
        {"date": local_database.crsp_days[1], "permno": 10001, "ret": 0.02, "prc": 11.0},
        {"date": local_database.crsp_days[2], "permno": 10001, "ret": 0.03, "prc": 12.0},
        # 10003 loses only its middle day, where the return is CRSP's -88.0.
        {"date": local_database.crsp_days[0], "permno": 10003, "ret": 0.02, "prc": 30.0},
        {"date": local_database.crsp_days[2], "permno": 10003, "ret": 0.02, "prc": 30.0},
    ]


@pytest.mark.parametrize("load", DAILY_LOADERS)
def test_daily_drops_a_share_code_outside_common_stock(load, local_database):
    assert 10002 not in _daily(load, local_database)["permno"]


@pytest.mark.parametrize("load", MONTHLY_LOADERS)
def test_monthly_applies_every_cleaning_rule(load, local_database):
    month_end = local_database.month_end
    # Event data is aggregated to the month before the join, so 10001 survives
    # on its month-end row; 10002 fails the share code and 10003's return
    # is CRSP's -99.0.
    assert load(month_end, month_end, COLUMNS).to_dicts() == [
        {"date": month_end, "permno": 10001, "ret": 0.05, "prc": 20.0},
    ]


@pytest.mark.parametrize("load", DAILY_LOADERS)
def test_daily_respects_the_date_range(load, local_database):
    days = local_database.crsp_days
    out = load(days[0], days[0], COLUMNS)
    assert out["date"].unique().to_list() == [days[0]]


@pytest.mark.parametrize("load", DAILY_LOADERS)
def test_daily_returns_only_the_requested_columns(load, local_database):
    assert load(*[local_database.crsp_days[i] for i in (0, -1)], ["permno", "ret"]).columns == ["permno", "ret"]


@pytest.mark.parametrize("load", DAILY_LOADERS)
def test_daily_is_sorted_by_permno_then_date(load, local_database):
    out = _daily(load, local_database)
    assert out.sort(["permno", "date"]).equals(out)


@pytest.mark.parametrize("columns", COLUMN_GETTERS)
def test_column_listing_names_the_dataset_schema(columns, local_database):
    listing = columns()
    assert isinstance(listing, str)
    for name in ("date", "permno", "ret", "prc"):
        assert name in listing


def test_the_two_vintages_are_loaded_from_different_tables(local_database):
    """
    The v2 loaders must read the v2 tables. Merging the copies makes it easy
    to wire a loader to the wrong table, and the fixture gives both vintages
    the same contents, so only the paths distinguish them.
    """
    from sf_quant.data._tables import crsp_daily_table, crsp_v2_daily_table

    assert crsp_daily_table._name == "crsp_daily"
    assert crsp_v2_daily_table._name == "crsp_v2_daily"
    assert "crsp_v2_daily" in crsp_v2_daily_table._file_path()
    assert "crsp_v2_daily" not in crsp_daily_table._file_path()
