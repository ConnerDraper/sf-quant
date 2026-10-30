"""Period-return tests. The assets table is stubbed, so these need no data lake."""

import datetime as dt

import polars as pl
import pytest

import sf_quant.performance.returns as returns_module
from sf_quant.performance import (
    generate_multi_returns_from_weights,
    generate_returns_from_weights,
)

DATES = [dt.date(2024, 1, d) for d in (2, 3, 4, 5, 8, 9, 10)]
BARRIDS = ["A", "B"]

# Daily simple returns, in percent as the assets table stores them.
DAILY = {
    "A": [10.0, 1.0, 2.0, -3.0, 4.0, -1.0, 5.0],
    "B": [-8.0, 2.0, -1.0, 1.0, -2.0, 3.0, -4.0],
}
# B drops out of the universe on DATES[2] through DATES[4].
B_OUT = {2, 3, 4}


class StubTable:
    def __init__(self, frame):
        self.frame = frame

    def scan(self, year=None):
        return self.frame.lazy()


@pytest.fixture
def stub_assets(monkeypatch):
    frame = pl.DataFrame(
        {
            "date": DATES * len(BARRIDS),
            "barrid": [b for b in BARRIDS for _ in DATES],
            "return": [r for b in BARRIDS for r in DAILY[b]],
            "in_universe": [b != "B" or i not in B_OUT for b in BARRIDS for i in range(len(DATES))],
        }
    )
    monkeypatch.setattr(returns_module, "assets_table", StubTable(frame))


def total_weights_on(dates, values):
    return pl.DataFrame(
        {
            "date": [d for d in dates for _ in BARRIDS],
            "barrid": BARRIDS * len(dates),
            "weight": [v for pair in values for v in pair],
        }
    )


def compound(barrid, days):
    out = 1.0
    for i in days:
        out *= 1 + DAILY[barrid][i] / 100
    return out - 1


def test_total_returns_exclude_the_formation_day(stub_assets):
    out = generate_returns_from_weights(
        total_weights_on([DATES[0], DATES[3], DATES[6]], [[1.0, 0.0]] * 3)
    )
    assert out["date"].to_list() == [DATES[0], DATES[3]]
    assert out["return"].to_list() == pytest.approx(
        [compound("A", [1, 2, 3]), compound("A", [4, 5, 6])]
    )


def test_multi_returns_exclude_the_formation_day(stub_assets, monkeypatch):
    benchmark = pl.DataFrame(
        {
            "date": [d for d in DATES for _ in BARRIDS],
            "barrid": BARRIDS * len(DATES),
            "weight": [0.5, 0.5] * len(DATES),
        }
    )
    monkeypatch.setattr(
        returns_module, "load_benchmark", lambda **kwargs: benchmark.clone()
    )

    out = generate_multi_returns_from_weights(
        total_weights_on([DATES[0], DATES[3], DATES[6]], [[1.0, 0.0]] * 3)
    )
    assert out["date"].unique().sort().to_list() == [DATES[0], DATES[3]]

    first = out.filter(pl.col("date") == DATES[0])
    by_portfolio = dict(zip(first["portfolio"], first["return"]))
    total = compound("A", [1, 2, 3])
    bench = 0.5 * compound("A", [1, 2, 3]) + 0.5 * compound("B", [1, 2, 3])
    assert by_portfolio["total"] == pytest.approx(total)
    assert by_portfolio["benchmark"] == pytest.approx(bench)
    assert by_portfolio["active"] == pytest.approx(total - bench)


def test_held_names_earn_returns_after_leaving_the_universe(stub_assets):
    """Leaving the Russell indexes doesn't sell the position: B earns DATES[1..3]."""
    out = generate_returns_from_weights(
        total_weights_on([DATES[0], DATES[3], DATES[6]], [[0.0, 1.0]] * 3)
    )
    assert out["return"].to_list() == pytest.approx(
        [compound("B", [1, 2, 3]), compound("B", [4, 5, 6])]
    )
