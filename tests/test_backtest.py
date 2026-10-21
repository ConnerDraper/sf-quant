"""backtest() tests. Synthetic risk model throughout, so these need no data lake.

loky workers are fresh interpreters that don't see a monkeypatched risk model,
so the parallel path runs here through joblib's threading backend. That covers
the dispatch itself -- splitting by date, ordering, reassembly. loky's output
was checked bit-for-bit against the sequential path on a real year of data in
research/parallel-backends.
"""

import datetime as dt
import importlib

import joblib
import numpy as np
import polars as pl
import pytest

from sf_quant.backtester import backtest

# sf_quant.backtester re-exports the backtest *function* under the module's
# name, so ``import sf_quant.backtester.backtest as m`` would bind the function.
backtest_module = importlib.import_module("sf_quant.backtester.backtest")

N, K = 30, 4
BARRIDS = [f"USA{i:04d}" for i in range(N)]
DATES = [dt.date(2024, 1, 2) + dt.timedelta(days=i) for i in range(10)]


def fake_components(date_, barrids):
    """A well-conditioned factor model per date, returned in ``barrids`` order."""
    rng = np.random.default_rng(date_.toordinal())
    exposures = rng.normal(size=(N, K))
    root = rng.normal(size=(K, K))
    covariance = root @ root.T / K + np.eye(K) * 0.02
    specific = rng.uniform(0.02, 0.09, size=N) ** 2
    rows = [BARRIDS.index(b) for b in barrids]
    return exposures[rows], covariance, specific[rows]


def panel() -> pl.DataFrame:
    """Ten dates with a universe that changes date to date, rows shuffled."""
    rng = np.random.default_rng(0)
    frames = []
    for i, date_ in enumerate(DATES):
        names = BARRIDS[: N - i % 3]
        cap = rng.pareto(1.5, size=len(names)) + 1.0
        frames.append(
            pl.DataFrame(
                {
                    "date": [date_] * len(names),
                    "barrid": names,
                    "alpha": rng.normal(scale=0.02, size=len(names)),
                    "benchmark_weight": cap / cap.sum(),
                    "predicted_beta": rng.uniform(0.6, 1.5, size=len(names)),
                }
            )
        )
    return pl.concat(frames).sample(fraction=1.0, shuffle=True, seed=1)


@pytest.fixture(autouse=True)
def synthetic_risk_model(monkeypatch):
    monkeypatch.setattr(backtest_module, "construct_factor_model_components", fake_components)


def test_parallel_matches_sequential():
    data = panel()
    sequential = backtest(data, n_cpus=1)
    with joblib.parallel_config(backend="threading"):
        parallel = backtest(data, n_cpus=4)
    assert parallel.equals(sequential)


def test_one_portfolio_per_date_with_every_name():
    data = panel()
    with joblib.parallel_config(backend="threading"):
        out = backtest(data, n_cpus=4)

    assert out["date"].unique().sort().to_list() == DATES
    expected = data.select("date", "barrid").sort("date", "barrid")
    assert out.select("date", "barrid").equals(expected)
    sums = out.group_by("date").agg(pl.col("active_weight").sum())["active_weight"]
    assert np.allclose(sums.to_numpy(), 0.0, atol=1e-7)


def lake_alphas() -> pl.DataFrame:
    """An alpha on every (date, barrid) of the synthetic lake, in or out of universe."""
    from conftest import BARRIDS as LAKE_BARRIDS, DATES as LAKE_DATES

    rng = np.random.default_rng(2)
    return pl.DataFrame(
        {
            "date": [d for d in LAKE_DATES for _ in LAKE_BARRIDS],
            "barrid": LAKE_BARRIDS * len(LAKE_DATES),
            "alpha": rng.normal(scale=0.02, size=len(LAKE_DATES) * len(LAKE_BARRIDS)),
        }
    )


def test_attach_joins_beta_and_benchmark_and_drops_out_of_universe(lake):
    from sf_quant.backtester import attach_beta_and_benchmark
    import sf_quant.data as sfd

    alphas = lake_alphas()
    with pytest.warns(UserWarning, match="dropped 4 of 24 rows"):
        out = attach_beta_and_benchmark(alphas)

    universe = sfd.load_assets(
        alphas["date"].min(), alphas["date"].max(), ["date", "barrid", "predicted_beta"], in_universe=True
    )
    assert out.select("date", "barrid").sort("date", "barrid").equals(
        universe.select("date", "barrid").sort("date", "barrid")
    )
    sums = out.group_by("date").agg(pl.col("benchmark_weight").sum())["benchmark_weight"]
    assert sums.to_list() == pytest.approx([1.0] * sums.len())
    assert out["predicted_beta"].null_count() == 0


def test_attach_keeps_columns_the_caller_supplied(lake):
    from sf_quant.backtester import attach_beta_and_benchmark

    alphas = lake_alphas().with_columns(predicted_beta=pl.lit(1.0), benchmark_weight=pl.lit(None, pl.Float64))
    out = attach_beta_and_benchmark(alphas)
    assert out.height == alphas.height  # no universe join when beta is supplied
    assert (out["predicted_beta"] == 1.0).all()
    assert (out["benchmark_weight"] == 0.0).all()  # supplied nulls become 0


def test_backtest_needs_only_date_barrid_alpha(lake):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = backtest(lake_alphas(), n_cpus=1)
    assert out.columns == ["date", "barrid", "active_weight", "benchmark_weight", "active_risk"]
    sums = out.group_by("date").agg(pl.col("active_weight").sum())["active_weight"]
    assert np.allclose(sums.to_numpy(), 0.0, atol=1e-6)
