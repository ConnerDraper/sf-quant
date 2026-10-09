"""Return timing, drift, turnover, and costs on a hand-checkable market."""

import datetime as dt

import polars as pl
import pytest
from conftest import run, tiny_alphas, tiny_market
from synthetic import business_days

DAYS = business_days(dt.date(2024, 1, 29), 30)  # Jan 29 .. Mar 8, 2024
FORMATION, EXECUTION = dt.date(2024, 1, 31), dt.date(2024, 2, 1)
SPIKES = {
    FORMATION: {"A": 0.10},  # before the portfolio exists
    EXECUTION: {"A": 0.10},  # earned by the pre-trade holdings (none)
    dt.date(2024, 2, 2): {"A": 0.10},  # first day the new holdings earn
    dt.date(2024, 2, 5): {"B": 0.05},  # earned by drifted holdings
}


def tiny_run(lag=1, spikes=SPIKES, **overrides):
    return run(tiny_market(spikes, DAYS), tiny_alphas(DAYS), execution_lag_days=lag, **overrides)


def target(result, formation=FORMATION):
    w = result.weights.filter(pl.col("formation_date") == formation)
    return dict(zip(w["barrid"], w["weight"]))


def day(result, date_):
    return result.returns.filter(pl.col("date") == date_).row(0, named=True)


def test_weights_formed_at_t_first_earn_on_t_plus_lag_plus_one():
    result = tiny_run(lag=1)
    w = target(result)
    assert w["A"] > 0.25  # the alpha tilts toward A
    assert result.returns["date"].min() == dt.date(2024, 2, 2)
    first = day(result, dt.date(2024, 2, 2))
    assert first["portfolio"] == pytest.approx(w["A"] * 0.10)
    assert first["benchmark"] == pytest.approx(0.25 * 0.10)
    assert first["active"] == pytest.approx((w["A"] - 0.25) * 0.10)


def test_zero_lag_earns_from_the_next_day():
    result = tiny_run(lag=0)
    assert result.returns["date"].min() == EXECUTION
    assert day(result, EXECUTION)["portfolio"] == pytest.approx(target(result)["A"] * 0.10)


def test_holdings_drift_between_rebalances():
    result = tiny_run()
    w = target(result)
    assert day(result, dt.date(2024, 2, 5))["portfolio"] == pytest.approx(w["B"] / (1 + 0.10 * w["A"]) * 0.05)


def test_turnover_is_measured_against_drifted_weights():
    result = tiny_run()
    w1, w2 = target(result), target(result, dt.date(2024, 2, 29))
    value = dict(w1) | {"A": w1["A"] * 1.10, "B": w1["B"] * 1.05}
    total = sum(value.values())
    expected = sum(abs(w2[k] - value[k] / total) for k in w2)

    diagnostics = result.diagnostics.sort("formation_date")
    assert diagnostics["turnover"].to_list() == pytest.approx([1.0, expected])  # first is the build from cash
    assert diagnostics["initial"].to_list() == [True, False]


def test_costs_are_charged_on_execution_days_only():
    result = tiny_run(bps_grid=[0, 10])
    traded = result.diagnostics.filter(~pl.col("initial")).row(0, named=True)
    r = result.returns
    on = r.filter(pl.col("date") == traded["execution_date"]).row(0, named=True)
    assert on["net_active_10bps"] == pytest.approx(on["active"] - 0.001 * traded["turnover"])
    off = r.filter(pl.col("date") != traded["execution_date"])
    assert (off["net_active_10bps"] == off["active"]).all()
    assert (r["net_active_0bps"] == r["active"]).all()


def test_breakeven_cost_zeroes_net_active_return():
    result = tiny_run(spikes=SPIKES | {dt.date(2024, 3, 4): {"A": 0.02}})
    s = result.summary
    traded = result.diagnostics.filter(~pl.col("initial"))["turnover"].sum()
    net = s["realized_active_return"] - s["breakeven_bps"] / 1e4 * traded / (s["n_days"] / 252)
    assert net == pytest.approx(0.0, abs=1e-12)
