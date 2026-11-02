"""Parallel backtester tests. Synthetic risk model throughout, so these need no data lake.

These cover the thread-pool dispatch -- splitting by date, ordering,
reassembly: the parallel paths must reproduce single-worker results exactly.
"""

import datetime as dt
import importlib

import numpy as np
import polars as pl
import pytest

import sf_quant.optimizer as sfo
from sf_quant.backtester import (
    backtest_parallel,
    backtest_sequential,
    dynamic_backtest_parallel,
)

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
    for name in ("sequential", "parallel", "dynamic_parallel"):
        module = importlib.import_module(f"sf_quant.backtester.{name}")
        monkeypatch.setattr(module, "construct_factor_model_components", fake_components)


def test_parallel_matches_sequential():
    data = panel()
    constraints = [sfo.FullInvestment()]
    sequential = backtest_sequential(data, constraints).sort("date", "barrid")
    parallel = backtest_parallel(data, constraints, n_cpus=4).sort("date", "barrid")
    assert parallel.equals(sequential)


def test_dynamic_parallel_matches_one_worker():
    data = panel()
    constraints = [sfo.ZeroInvestment(), sfo.LongOnly(), sfo.ZeroBeta()]
    one = dynamic_backtest_parallel(data, constraints, n_cpus=1)
    four = dynamic_backtest_parallel(data, constraints, n_cpus=4)
    assert four.equals(one)
    assert one["date"].unique().sort().to_list() == DATES
    assert one.select("date", "barrid").equals(
        data.select("date", "barrid").sort("date", "barrid")
    )


def stub_costs(monkeypatch, returns=0.0, half_spread=0.002):
    """Every name returns ``returns`` over every period and pays ``half_spread`` to trade."""
    module = importlib.import_module("sf_quant.backtester.dynamic_parallel")
    grid = pl.DataFrame({"date": [d for d in DATES for _ in BARRIDS], "barrid": BARRIDS * len(DATES)})
    monkeypatch.setattr(
        module, "_period_returns", lambda barrids, dates: grid.with_columns(pl.lit(returns).alias("return"))
    )
    monkeypatch.setattr(
        module,
        "_trade_half_spreads",
        lambda dates, barrids: grid.with_columns(pl.lit(half_spread).alias("half_spread")),
    )
    return module


def total_trading(weights: pl.DataFrame) -> float:
    """Sum over dates of |w_t - w_{t-1}|, total weights, with zero returns."""
    wide = weights.pivot(on="barrid", index="date", values="weight").sort("date").fill_null(0.0)
    values = wide.drop("date").to_numpy()
    return float(np.abs(np.diff(values, axis=0)).sum())


MANDATE = [sfo.ZeroInvestment(), sfo.LongOnly(), sfo.ZeroBeta()]


def expected_alpha(weights: pl.DataFrame, data: pl.DataFrame) -> np.ndarray:
    """Each date's alpha'h."""
    return (
        weights.join(data.select("date", "barrid", "alpha"), on=["date", "barrid"])
        .group_by("date")
        .agg(pl.col("weight").mul("alpha").sum())
        .sort("date")["weight"]
        .to_numpy()
    )


def test_a_tiny_cost_aversion_gives_the_cost_blind_portfolios(monkeypatch):
    stub_costs(monkeypatch)
    data = panel()
    blind = dynamic_backtest_parallel(data, MANDATE, n_cpus=1)
    aware = dynamic_backtest_parallel(data, MANDATE, cost_aversion=1e-9)
    assert aware.select("date", "barrid").equals(blind.select("date", "barrid"))
    # The cost term adds variables, so the solver stops at a slightly different
    # point: objectives agree to ~3e-7, single weights to ~1e-4.
    np.testing.assert_allclose(expected_alpha(aware, data), expected_alpha(blind, data), rtol=1e-6)
    np.testing.assert_allclose(aware["weight"], blind["weight"], atol=2e-4)
    assert aware["gamma"].null_count() == aware.height


def test_cost_aversion_cuts_trading(monkeypatch):
    stub_costs(monkeypatch)
    data = panel()
    traded = [
        total_trading(dynamic_backtest_parallel(data, MANDATE, active_weights=False, cost_aversion=lam))
        for lam in (1e-9, 2.0, 8.0)
    ]
    assert traded[0] > traded[1] > traded[2]


def test_trades_are_charged_against_the_drifted_holdings(monkeypatch):
    module = stub_costs(monkeypatch)
    # Name i returns i/1000 over every period.
    returns = pl.DataFrame(
        {
            "date": [d for d in DATES for _ in BARRIDS],
            "barrid": BARRIDS * len(DATES),
            "return": [i / 1000 for _ in DATES for i in range(N)],
        }
    )
    monkeypatch.setattr(module, "_period_returns", lambda barrids, dates: returns)
    seen = []
    solve = module.cost_aware_mve_optimizer

    def spy(**kwargs):
        seen.append(kwargs["previous_weights"])
        return solve(**kwargs)

    monkeypatch.setattr(module, "cost_aware_mve_optimizer", spy)
    data = panel()
    out = dynamic_backtest_parallel(data, MANDATE, active_weights=False, cost_aversion=1.0)

    assert not seen[0].any()  # the first date is the starting portfolio
    held = out.filter(pl.col("date") == DATES[0]).join(
        returns.filter(pl.col("date") == DATES[0]), on=["date", "barrid"]
    )
    growth = 1 + (held["weight"] * held["return"]).sum()
    drifted = held.select("barrid", (pl.col("weight") * (1 + pl.col("return")) / growth).alias("drifted"))
    expected = (
        data.filter(pl.col("date") == DATES[1])
        .sort("barrid")
        .join(drifted, on="barrid", how="left")
        .select(pl.col("drifted").fill_null(0.0) - pl.col("benchmark_weight"))
        .to_series()
        .to_numpy()
    )
    np.testing.assert_allclose(seen[1], expected, atol=1e-12)
