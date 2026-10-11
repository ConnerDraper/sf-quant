"""The by-date loaders cache their parquet reads, and the environment clears them.

A research run asks for the same date's risk data from several places — the
signal pipeline's contexts, a signal that needs betas, and the backtester's
solve — so these reads land dozens of times on a handful of dates.
"""

from __future__ import annotations

import pytest

import sf_quant.data as sfd
from sf_quant.data.assets import _assets_by_date
from sf_quant.data.covariances import load_covariances_by_date
from sf_quant.data.exposures import load_exposures_by_date

CACHED = [
    pytest.param(lambda d: load_exposures_by_date(d), load_exposures_by_date, id="exposures"),
    pytest.param(lambda d: load_covariances_by_date(d), load_covariances_by_date, id="covariances"),
    pytest.param(
        lambda d: sfd.load_assets_by_date(d, False, ["barrid", "specific_risk"]),
        _assets_by_date,
        id="assets",
    ),
]


@pytest.mark.parametrize("load, cache", CACHED)
def test_second_read_of_a_date_is_a_cache_hit(local_database, load, cache):
    sfd.clear_cache()
    before = cache.cache_info().hits
    first = load(local_database.dates[0])
    second = load(local_database.dates[0])
    assert cache.cache_info().hits == before + 1
    assert first.equals(second)


@pytest.mark.parametrize("load, cache", CACHED)
def test_distinct_dates_are_cached_separately(local_database, load, cache):
    sfd.clear_cache()
    load(local_database.dates[0])
    load(local_database.dates[1])
    assert cache.cache_info().currsize == 2
    assert cache.cache_info().hits == 0


@pytest.mark.parametrize("load, cache", CACHED)
def test_cached_read_does_not_touch_the_filesystem(local_database, load, cache):
    """Delete the parquet after the first read: a second read must still work."""
    sfd.clear_cache()
    expected = load(local_database.dates[0])
    for path in local_database.root.rglob("*.parquet"):
        path.unlink()
    assert load(local_database.dates[0]).equals(expected)


@pytest.mark.parametrize("load, cache", CACHED)
def test_env_clears_the_cache(local_database, load, cache):
    """Pointing at another database must drop frames read from the previous one."""
    sfd.clear_cache()
    load(local_database.dates[0])
    assert cache.cache_info().currsize == 1
    sfd.env(database="somewhere_else")
    assert cache.cache_info().currsize == 0


def test_risk_model_reads_each_table_once_per_date(local_database, monkeypatch):
    """The whole point: repeated load_risk_model calls for one date read once."""
    import polars as pl

    dates, barrids = local_database.dates, local_database.barrids
    sfd.clear_cache()
    scans = []
    original = pl.scan_parquet
    monkeypatch.setattr(pl, "scan_parquet", lambda *a, **kw: scans.append(a[0]) or original(*a, **kw))

    first = sfd.load_risk_model(dates[0], list(barrids))
    after_first = len(scans)
    # Same date, a different subset of assets: a (date, barrids) key would miss,
    # but the underlying reads are keyed by date alone.
    sfd.load_risk_model(dates[0], list(barrids[:2]))

    assert after_first > 0, "the first risk model should read from disk"
    assert len(scans) == after_first, "the second should read nothing"
    assert first.ids == list(barrids)
