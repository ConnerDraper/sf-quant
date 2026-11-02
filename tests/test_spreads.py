"""Half-spread tests. The assets table is stubbed, so these need no data lake."""

import datetime as dt

import polars as pl
import pytest

import sf_quant.data.spreads as spreads_module
from sf_quant.data import load_half_spreads

DATES = [dt.date(2024, 1, d) for d in (2, 3, 4)]
# Prices in dollars and spreads in dollars, as stored. B has no 30-day average
# on DATES[0], so the day's spread stands in; C has no spread at all; B has no
# row on DATES[1].
ASSETS = pl.DataFrame(
    {
        "date": DATES + [DATES[0], DATES[2]] + DATES,
        "barrid": ["A"] * 3 + ["B"] * 2 + ["C"] * 3,
        "price": [10.0, 20.0, 40.0, 50.0, 50.0, 5.0, 5.0, 5.0],
        "average_daily_bid_ask_spread_30": [0.02, 0.02, 0.02, None, 0.1, None, None, None],
        "bid_ask_spread": [0.04, 0.04, 0.04, 0.5, 0.5, None, None, None],
    }
)


class StubTable:
    def scan(self, year=None):
        return ASSETS.lazy()


@pytest.fixture(autouse=True)
def stub_assets(monkeypatch):
    monkeypatch.setattr(spreads_module, "assets_table", StubTable())


def test_half_spread_is_the_average_spread_over_twice_the_price():
    out = load_half_spreads(DATES[0], DATES[2], ["A", "B", "C"])
    by_row = {(r["barrid"], r["date"]): r["half_spread"] for r in out.iter_rows(named=True)}
    assert by_row[("A", DATES[1])] == pytest.approx(0.02 / (2 * 20.0))
    assert by_row[("B", DATES[0])] == pytest.approx(0.5 / (2 * 50.0))  # the day's spread
    assert by_row[("B", DATES[2])] == pytest.approx(0.1 / (2 * 50.0))
    assert by_row[("C", DATES[0])] is None


def test_trades_pay_the_previous_trading_days_spread():
    out = spreads_module._trade_half_spreads(pl.Series(DATES), ["A", "B"])
    by_row = {(r["barrid"], r["date"]): r["half_spread"] for r in out.iter_rows(named=True)}
    # Weights dated DATES[2] are put on at DATES[1]'s close, A's price then 20.
    assert by_row[("A", DATES[2])] == pytest.approx(0.02 / (2 * 20.0))
    assert by_row[("B", DATES[1])] == pytest.approx(0.5 / (2 * 50.0))
    # B has no row on DATES[1], so it isn't trading into DATES[2].
    assert ("B", DATES[2]) not in by_row
    # DATES[0] has no earlier trading day in the stub, so nothing trades into it.
    assert out["date"].min() == DATES[1]
