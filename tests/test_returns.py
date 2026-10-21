"""Period-return tests. ``load_assets`` is stubbed, so these need no data lake."""

import datetime as dt

import polars as pl
import pytest

import sf_quant.performance.returns as returns_module
from sf_quant.performance import (
    generate_active_returns_from_weights,
    generate_multi_returns_from_weights,
    generate_returns_from_weights,
)

DATES = [dt.date(2024, 1, d) for d in (2, 3, 4, 5, 8, 9, 10)]
BARRIDS = ["A", "B"]

# Daily simple returns, in percent as load_assets delivers them.
DAILY = {
    "A": [10.0, 1.0, 2.0, -3.0, 4.0, -1.0, 5.0],
    "B": [-8.0, 2.0, -1.0, 1.0, -2.0, 3.0, -4.0],
}


@pytest.fixture
def stub_assets(monkeypatch):
    frame = pl.DataFrame(
        {
            "date": DATES * len(BARRIDS),
            "barrid": [b for b in BARRIDS for _ in DATES],
            "return": [r for b in BARRIDS for r in DAILY[b]],
        }
    )
    monkeypatch.setattr(
        returns_module, "load_assets", lambda **kwargs: frame.clone()
    )


def weights_on(dates, values):
    return pl.DataFrame(
        {
            "date": [d for d in dates for _ in BARRIDS],
            "barrid": BARRIDS * len(dates),
            "active_weight": [v for pair in values for v in pair],
            "benchmark_weight": [0.5, 0.5] * len(dates),
            "active_risk": [0.05] * (len(dates) * len(BARRIDS)),
        }
    )


def compound(barrid, days):
    out = 1.0
    for i in days:
        out *= 1 + DAILY[barrid][i] / 100
    return out - 1


def test_period_return_is_buy_and_hold(stub_assets):
    """Sum h_i * compounded_i is the exact P&L of a position left alone."""
    h = {"A": 0.3, "B": -0.3}
    out = generate_active_returns_from_weights(
        weights_on([DATES[0], DATES[3], DATES[6]], [[0.3, -0.3]] * 3)
    )

    # Day 0's portfolio is held over days 1-3, day 3's over days 4-6.
    expected_first = sum(h[b] * compound(b, [1, 2, 3]) for b in BARRIDS)
    expected_second = sum(h[b] * compound(b, [4, 5, 6]) for b in BARRIDS)

    assert out["date"].to_list() == [DATES[0], DATES[3]]
    assert out["return"].to_list() == pytest.approx([expected_first, expected_second])

    # Independently: simulate holding the dollars and let them drift.
    terminal = sum(h[b] * (1 + compound(b, [1, 2, 3])) for b in BARRIDS)
    assert out["return"][0] == pytest.approx(terminal - sum(h.values()))


def test_formation_day_return_is_excluded(stub_assets):
    """Day 0 carries a +10% outlier for A; crediting it would be look-ahead."""
    out = generate_active_returns_from_weights(
        weights_on([DATES[0], DATES[3], DATES[6]], [[1.0, 0.0]] * 3)
    )
    assert out["return"][0] == pytest.approx(compound("A", [1, 2, 3]))
    assert out["return"][0] != pytest.approx(compound("A", [0, 1, 2, 3]))


def test_daily_grid_gives_one_day_periods(stub_assets):
    out = generate_active_returns_from_weights(
        weights_on(DATES, [[1.0, 0.0]] * len(DATES))
    )
    # K dates produce K-1 periods; the last formation date has no closing date.
    assert out["date"].to_list() == DATES[:-1]
    assert out["return"].to_list() == pytest.approx(
        [DAILY["A"][i] / 100 for i in range(1, len(DATES))]
    )


def test_name_absent_from_the_period_contributes_nothing(stub_assets, monkeypatch):
    frame = pl.DataFrame(
        {
            "date": DATES + DATES[:1],
            "barrid": ["A"] * len(DATES) + ["B"],
            "return": DAILY["A"] + [DAILY["B"][0]],
        }
    )
    monkeypatch.setattr(returns_module, "load_assets", lambda **kwargs: frame.clone())

    out = generate_active_returns_from_weights(
        weights_on([DATES[0], DATES[3]], [[0.3, -0.3]] * 2)
    )
    # B has no observations after day 0, so only A's leg is earned.
    assert out["return"][0] == pytest.approx(0.3 * compound("A", [1, 2, 3]))


def test_single_rebalance_date_is_rejected(stub_assets):
    with pytest.raises(ValueError, match="at least two rebalance dates"):
        generate_active_returns_from_weights(weights_on([DATES[0]], [[0.3, -0.3]]))


def total_weights_on(dates, values):
    return pl.DataFrame(
        {
            "date": [d for d in dates for _ in BARRIDS],
            "barrid": BARRIDS * len(dates),
            "weight": [v for pair in values for v in pair],
        }
    )


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


def test_cumulative_path_books_each_period_at_its_end(stub_assets):
    """Formed on days 0, 2, 4: period returns land on days 2 and 4, flat between."""
    from sf_quant.performance import generate_cumulative_log_returns
    import math

    formation = [DATES[0], DATES[2], DATES[4]]
    weights = weights_on(formation, [(0.3, -0.3)] * 3)
    path = generate_cumulative_log_returns(weights, dates=pl.Series(DATES[:5]))

    first = 0.3 * compound("A", [1, 2]) - 0.3 * compound("B", [1, 2])
    second = 0.3 * compound("A", [3, 4]) - 0.3 * compound("B", [3, 4])
    expected = [0.0, 0.0, math.log1p(first), math.log1p(first), math.log1p(first) + math.log1p(second)]
    assert path["date"].to_list() == DATES[:5]
    assert path["cumulative_log_return"].to_list() == pytest.approx(expected)
