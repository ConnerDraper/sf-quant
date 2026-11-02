"""Trading-cost tests. The assets table and benchmark are stubbed, so these need no data lake."""

import datetime as dt

import polars as pl
import pytest

import sf_quant.data.spreads as spreads_module
import sf_quant.performance.returns as returns_module
from sf_quant.performance import (
    MissingSpreadWarning,
    generate_multi_returns_from_weights,
    generate_returns_from_weights,
    generate_trading_costs,
)

DATES = [dt.date(2024, 1, d) for d in (2, 3, 4, 5)]
# Returns in percent; price and spreads in dollars. A's half-spread is
# 0.02 / (2 * 10) = 0.001 every day, B's 0.004, C's 0.002 but missing on DATES[1].
ASSETS = pl.DataFrame(
    {
        "date": DATES * 3,
        "barrid": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        "return": [0.0, 10.0, 0.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "in_universe": [True] * 12,
        "price": [10.0] * 12,
        "average_daily_bid_ask_spread_30": [0.02] * 4 + [0.08] * 4 + [0.04, None, 0.04, 0.04],
        "bid_ask_spread": [0.02] * 4 + [0.08] * 4 + [0.04, None, 0.04, 0.04],
    }
)


NO_DELISTINGS = pl.DataFrame(
    schema={"barrid": pl.String, "last_trading_date": pl.Date, "delret_used": pl.Float64}
)


class StubTable:
    def scan(self, year=None):
        return ASSETS.lazy()


@pytest.fixture(autouse=True)
def stub_assets(monkeypatch):
    monkeypatch.setattr(returns_module, "assets_table", StubTable())
    monkeypatch.setattr(spreads_module, "assets_table", StubTable())
    monkeypatch.setattr(returns_module, "load_delistings", lambda start, end: NO_DELISTINGS)


def weights_frame(rows):
    return pl.DataFrame(rows, schema=["date", "barrid", "weight"], orient="row")


# Half A, half B on DATES[1]; A +10% and B -10% that day, so the portfolio
# drifts to A 0.55, B 0.45. On DATES[2] it moves to all A, closing B.
WEIGHTS = weights_frame(
    [
        (DATES[1], "A", 0.5),
        (DATES[1], "B", 0.5),
        (DATES[2], "A", 1.0),
    ]
)


def test_trades_are_measured_against_drifted_weights_and_cost_half_the_spread():
    out = generate_trading_costs(WEIGHTS)
    by_name = {r["barrid"]: r for r in out.iter_rows(named=True)}
    assert out["date"].unique().to_list() == [DATES[2]]  # the first date isn't a trade
    assert by_name["A"]["trade"] == pytest.approx(1.0 - 0.55)
    assert by_name["B"]["trade"] == pytest.approx(0.0 - 0.45)  # closing is a trade
    assert by_name["A"]["cost"] == pytest.approx(0.45 * 0.001)
    assert by_name["B"]["cost"] == pytest.approx(0.45 * 0.004)


def test_net_returns_subtract_the_period_cost():
    gross = generate_returns_from_weights(WEIGHTS, costs=False)
    net = generate_returns_from_weights(WEIGHTS)
    cost = 0.45 * 0.001 + 0.45 * 0.004
    assert net["return"].to_list() == pytest.approx([gross["return"][0], gross["return"][1] - cost])


def test_the_benchmark_trades_for_free(monkeypatch):
    benchmark = pl.DataFrame({"date": DATES[1:3], "barrid": ["A", "A"], "weight": [1.0, 1.0]})
    monkeypatch.setattr(returns_module, "load_benchmark", lambda start, end: benchmark)
    gross = generate_multi_returns_from_weights(WEIGHTS, costs=False)
    net = generate_multi_returns_from_weights(WEIGHTS)
    paid = gross.join(net, on=["date", "portfolio"], suffix="_net").filter(pl.col("date") == DATES[2])
    paid = dict(zip(paid["portfolio"], paid["return"] - paid["return_net"]))
    cost = 0.45 * 0.001 + 0.45 * 0.004
    assert paid == pytest.approx({"total": cost, "active": cost, "benchmark": 0.0})


def test_a_trade_without_a_spread_costs_nothing_and_warns():
    weights = weights_frame([(DATES[1], "A", 1.0), (DATES[2], "C", 1.0)])
    with pytest.warns(MissingSpreadWarning, match="0 for 50.0% of traded weight"):
        out = generate_trading_costs(weights)
    assert out.filter(pl.col("barrid") == "C")["cost"].item() == 0.0
    assert out.filter(pl.col("barrid") == "A")["cost"].item() == pytest.approx(1.0 * 0.001)  # closing A


def test_turnover_counts_opening_and_closing_positions(monkeypatch):
    """All in A one day, all in B the next: every day sells 1 and buys 1."""
    days = [dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(260)]
    assets = pl.DataFrame(
        {
            "date": days * 2,
            "barrid": ["A"] * 260 + ["B"] * 260,
            "return": [0.0] * 520,
            "price": [10.0] * 520,
            "average_daily_bid_ask_spread_30": [0.02] * 520,
            "bid_ask_spread": [0.02] * 520,
        }
    )

    class LongStub:
        def scan(self, year=None):
            return assets.lazy()

    monkeypatch.setattr(returns_module, "assets_table", LongStub())
    monkeypatch.setattr(spreads_module, "assets_table", LongStub())
    weights = pl.DataFrame({"date": days, "barrid": ["A", "B"] * 130, "weight": [1.0] * 260})

    from sf_quant.performance.turnover import _compute_turnover

    out = _compute_turnover(weights).drop_nulls()
    assert out.height == 259 - 251  # the first date isn't counted; 252-day warm-up
    assert out["two_sided_turnover"].to_list() == pytest.approx([2.0] * out.height)
