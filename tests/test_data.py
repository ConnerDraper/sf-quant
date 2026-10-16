"""Data-loader tests against the synthetic lake in conftest.py.

These pin what each loader returns today -- rows, columns, order, and the
cleaning rules -- so that refactoring the loaders can be checked against them.
"""

import numpy as np
import polars as pl
import pytest

import sf_quant.data as sfd
from sf_quant.data._factors import factors

from conftest import BARRIDS, DATES, MONTH_ENDS, SIGNAL_NAMES, factor_covariance

START, END = DATES[0], DATES[-1]


def test_trading_days_are_the_universe_dates(lake):
    assert sfd.load_trading_days(START, END).to_list() == DATES


@pytest.mark.parametrize("in_universe, rows", [(True, 20), (None, 24), (False, 24)])
def test_load_assets(lake, in_universe, rows):
    out = sfd.load_assets(START, END, ["date", "barrid", "in_universe"], in_universe=in_universe)
    assert out.columns == ["date", "barrid", "in_universe"]
    assert out.height == rows
    assert out.equals(out.sort("barrid", "date"))
    if in_universe:
        assert out["in_universe"].all()


@pytest.mark.parametrize("date", [DATES[1], DATES[4]])  # one date in each year
def test_load_assets_by_date(lake, date):
    universe = sfd.load_assets_by_date(date, True, ["date", "barrid"])
    everything = sfd.load_assets_by_date(date, False, ["date", "barrid"])
    assert universe["barrid"].to_list() == BARRIDS[:3]
    assert everything["barrid"].to_list() == BARRIDS
    assert (everything["date"] == date).all()


def test_load_exposures(lake):
    universe = sfd.load_exposures(START, END, True, ["date", "barrid", factors[0]])
    everything = sfd.load_exposures(START, END, False, [factors[0]])
    assert universe.columns == everything.columns == ["date", "barrid", factors[0]]
    assert (universe.height, everything.height) == (20, 24)
    assert universe.equals(universe.sort("barrid", "date"))
    assert sfd.load_exposures_by_date(DATES[2]).height == len(BARRIDS)


def test_load_factors(lake):
    assert sfd.load_factors(START, END, factors[:2]).columns == ["date", *factors[:2]]
    assert sfd.load_factors(START, END).columns == ["date", *factors]
    # An empty list is falsy, so it selects every factor too.
    assert sfd.load_factors(START, END, []).columns == ["date", *factors]


@pytest.mark.parametrize(
    "load, by_date, names, value",
    [
        (sfd.load_signals, sfd.load_signals_by_date, sfd.get_signal_names, "signal_value"),
        (sfd.load_scores, sfd.load_scores_by_date, sfd.get_score_names, "score"),
        (sfd.load_alphas, sfd.load_alphas_by_date, sfd.get_alpha_names, "alpha"),
    ],
)
def test_signal_tables(lake, load, by_date, names, value):
    everything = load(START, END)
    one = load(START, END, ["reversal"])
    assert everything.columns == ["date", "barrid", "signal_name", value]
    assert everything.height == len(DATES) * len(BARRIDS) * len(SIGNAL_NAMES)
    assert one["signal_name"].unique().to_list() == ["reversal"]
    assert one.equals(one.sort("barrid", "date"))
    assert by_date(DATES[3]).height == len(BARRIDS) * len(SIGNAL_NAMES)
    assert by_date(DATES[3], ["momentum"])["signal_name"].unique().to_list() == ["momentum"]
    assert sorted(names()) == SIGNAL_NAMES


def test_benchmark_is_cap_weighted_over_the_universe(lake):
    weights = sfd.load_benchmark(START, END)
    assert weights.columns == ["date", "barrid", "weight"]
    assert weights.height == 20
    sums = weights.group_by("date").agg(pl.col("weight").sum())["weight"]
    assert sums.to_list() == pytest.approx([1.0] * len(DATES))

    returns = sfd.load_benchmark_returns(START, END).sort("date")
    assets = sfd.load_assets(START, END, ["date", "barrid", "return"], in_universe=True)
    expected = (
        weights.join(assets, on=["date", "barrid"])
        .group_by("date")
        .agg((pl.col("weight") * pl.col("return")).sum().alias("bmk_return"))
        .sort("date")
    )
    assert returns.columns == ["date", "bmk_return"]
    assert returns["bmk_return"].to_list() == pytest.approx(expected["bmk_return"].to_list())


CRSP_COLUMNS = ["date", "permno", "ret", "prc", "ticker", "shrcd", "exchcd"]


@pytest.mark.parametrize("load", [sfd.load_crsp_daily, sfd.load_crsp_v2_daily])
def test_daily_crsp_cleaning(lake, load):
    out = load(START, END, CRSP_COLUMNS)
    # 10003 (share code 12), 10004 (exchange 4) and 10005 (no events) drop out,
    # as does 10002's missing-return code on the second date.
    assert out.filter(pl.col("permno") == 10001).height == len(DATES)
    assert out.filter(pl.col("permno") == 10002)["date"].to_list() == [d for d in DATES if d != DATES[1]]
    assert set(out["permno"]) == {10001, 10002}
    assert (out["prc"] > 0).all()
    # Events exist only on each year's first date; the rest are forward-filled.
    assert out["ticker"].null_count() == 0
    assert out.equals(out.sort("permno", "date"))


@pytest.mark.parametrize("load", [sfd.load_crsp_monthly, sfd.load_crsp_v2_monthly])
def test_monthly_crsp_cleaning(lake, load):
    out = load(START, MONTH_ENDS[-1], CRSP_COLUMNS)
    assert out.select("date", "permno").rows() == [
        (MONTH_ENDS[0], 10001), (MONTH_ENDS[1], 10001), (MONTH_ENDS[0], 10002)
    ]
    assert (out["prc"] > 0).all()
    assert out["shrcd"].to_list() == [10, 10, 11]


def test_fama_french(lake):
    out = sfd.load_fama_french(START, END)
    assert out.columns == ["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    assert out.height == len(DATES)


@pytest.mark.parametrize("get", [sfd.get_assets_columns, sfd.get_fama_french_columns])
def test_column_listings_show_every_column(lake, get):
    listing = get()
    assert "date" in listing and "…" not in listing


def test_covariance_matrix_is_the_factor_model(lake):
    date = DATES[3]
    out = sfd.construct_covariance_matrix(date, BARRIDS)
    assert out.columns == ["barrid", *BARRIDS]
    assert out["barrid"].to_list() == BARRIDS

    exposures = (
        sfd.load_exposures_by_date(date).sort("barrid").select(factors).fill_null(0).to_numpy()
    )
    specific = sfd.load_assets_by_date(date, False, ["specific_risk"])["specific_risk"].to_numpy()
    expected = (exposures @ factor_covariance(date) @ exposures.T + np.diag(specific**2)) / 100**2
    np.testing.assert_allclose(out.drop("barrid").to_numpy(), expected, rtol=1e-12)

    B, F, D = sfd.construct_factor_model_components(date, BARRIDS)
    np.testing.assert_allclose(B @ F @ B.T + np.diag(D), expected, rtol=1e-12)
