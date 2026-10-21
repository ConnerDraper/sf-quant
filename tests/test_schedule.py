import datetime as dt

import polars as pl
import pytest

from sf_quant.backtester import collapse, rebalance_dates, rebalance_phases

# Ten consecutive weekdays, two assets. alpha encodes the day index so the test
# can assert exactly which date a value travelled from.
DATES = [
    dt.date(2024, 1, 2),
    dt.date(2024, 1, 3),
    dt.date(2024, 1, 4),
    dt.date(2024, 1, 5),
    dt.date(2024, 1, 8),
    dt.date(2024, 1, 9),
    dt.date(2024, 1, 10),
    dt.date(2024, 1, 11),
    dt.date(2024, 1, 12),
    dt.date(2024, 1, 15),
]
CALENDAR = pl.Series("date", DATES)


def panel(barrids=("A", "B"), skip: dict[str, list[dt.date]] | None = None):
    """Daily panel where alpha == the day's index in DATES."""
    skip = skip or {}
    rows = [
        {
            "date": d,
            "barrid": b,
            "alpha": float(i),
            "benchmark_weight": 0.5,
            "predicted_beta": 1.0 + i / 100,
        }
        for b in barrids
        for i, d in enumerate(DATES)
        if d not in skip.get(b, [])
    ]
    return pl.DataFrame(rows).lazy()


def test_rebalance_dates_anchored_at_first_trading_day():
    assert rebalance_dates(CALENDAR, 1).to_list() == DATES
    assert rebalance_dates(CALENDAR, 3).to_list() == [DATES[0], DATES[3], DATES[6], DATES[9]]
    # Order and duplicates in the input must not move the grid.
    shuffled = pl.Series("date", list(reversed(DATES)) + DATES)
    assert rebalance_dates(shuffled, 3).to_list() == rebalance_dates(CALENDAR, 3).to_list()


def test_rebalance_period_below_one_rejected():
    with pytest.raises(ValueError, match="rebalance_period must be >= 1"):
        rebalance_dates(CALENDAR, 0)


def test_defaults_are_an_identity_on_the_date_axis():
    out = collapse(panel(), dates=CALENDAR).collect()
    assert out["date"].unique().sort().to_list() == DATES
    assert out.height == 20
    # alpha still matches its own date
    a = out.filter(pl.col("barrid") == "A").sort("date")
    assert a["alpha"].to_list() == [float(i) for i in range(10)]


def test_collapse_keeps_only_rebalance_dates():
    out = collapse(panel(), dates=CALENDAR, rebalance_period=3).collect()
    assert out["date"].unique().sort().to_list() == [DATES[0], DATES[3], DATES[6], DATES[9]]


def test_grid_is_identical_whether_built_on_full_index_or_a_slice():
    """Sharding must filter collapse's output, never slice the panel first."""
    full = collapse(panel(), dates=CALENDAR, rebalance_period=3).collect()
    sharded = full.filter(pl.col("date") >= DATES[5])

    # Slicing the panel first re-anchors the grid and gives different dates.
    resliced = collapse(
        panel().filter(pl.col("date") >= DATES[5]),
        dates=CALENDAR.filter(CALENDAR >= DATES[5]),
        rebalance_period=3,
    ).collect()

    assert sharded["date"].unique().sort().to_list() == [DATES[6], DATES[9]]
    assert resliced["date"].unique().sort().to_list() == [DATES[5], DATES[8]]


def test_lag_moves_alpha_but_not_benchmark_or_beta():
    out = (
        collapse(panel(), dates=CALENDAR, rebalance_period=3, lag=2)
        .collect()
        .filter(pl.col("barrid") == "A")
        .sort("date")
    )
    # Day 0 has no source two days back, so the grid starts at day 3.
    assert out["date"].to_list() == [DATES[3], DATES[6], DATES[9]]
    # alpha comes from t - 2 ...
    assert out["alpha"].to_list() == [1.0, 4.0, 7.0]
    # ... while predicted_beta is observed on the rebalance date itself.
    assert out["predicted_beta"].to_list() == pytest.approx([1.03, 1.06, 1.09])


def test_lag_is_in_trading_days_not_rows():
    """An asset with a gap gets dropped, never a silently stale alpha."""
    out = collapse(
        panel(skip={"B": [DATES[2]]}), dates=CALENDAR, rebalance_period=1, lag=1
    ).collect()

    a = out.filter(pl.col("barrid") == "A").sort("date")
    b = out.filter(pl.col("barrid") == "B").sort("date")

    # A is lagged one trading day throughout.
    assert a["date"].to_list() == DATES[1:]
    assert a["alpha"].to_list() == [float(i) for i in range(9)]

    # B has no observation on DATES[2], so that date has no row at all, and
    # DATES[3] has no alpha one trading day back. Both drop out; everywhere else
    # the lag is still exactly one trading day.
    assert b["date"].to_list() == [DATES[1], *DATES[4:]]
    assert b["alpha"].to_list() == [0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    # DATES[1] is the last alpha B had before the gap. A positional shift over
    # B's own rows would have carried it forward to DATES[3]; it must not appear.
    assert 1.0 not in b["alpha"].to_list()


def test_null_alpha_rows_are_dropped():
    data = panel().with_columns(
        pl.when((pl.col("barrid") == "B") & (pl.col("date") == DATES[3]))
        .then(None)
        .otherwise(pl.col("alpha"))
        .alias("alpha")
    )
    out = collapse(data, dates=CALENDAR).collect()
    assert out.filter(pl.col("date") == DATES[3])["barrid"].to_list() == ["A"]
    assert out["alpha"].null_count() == 0


def test_null_benchmark_weight_filled_with_zero():
    """A name outside the benchmark has b = 0; a left-joined null must not reach cvxpy."""
    data = panel().with_columns(
        pl.when(pl.col("barrid") == "B")
        .then(None)
        .otherwise(pl.col("benchmark_weight"))
        .alias("benchmark_weight")
    )
    out = collapse(data, dates=CALENDAR).collect()
    assert out["benchmark_weight"].null_count() == 0
    assert out.filter(pl.col("barrid") == "B")["benchmark_weight"].unique().to_list() == [0.0]


def test_negative_lag_rejected():
    with pytest.raises(ValueError, match="lag must be >= 0"):
        collapse(panel(), dates=CALENDAR, lag=-1).collect()


@pytest.mark.parametrize("period", [1, 3, 4])
def test_phases_are_the_grids_a_direct_run_would_solve(period):
    daily = collapse(panel(), dates=CALENDAR).collect()
    phases = rebalance_phases(daily, dates=CALENDAR, rebalance_period=period)

    assert len(phases) == period
    for p, frame in enumerate(phases):
        direct = collapse(panel(), dates=CALENDAR[p:], rebalance_period=period).collect()
        assert frame.sort("date", "barrid").equals(direct.select(daily.columns).sort("date", "barrid"))
    # Together the phases use every day exactly once.
    days = sorted(d for frame in phases for d in frame["date"].unique().to_list())
    assert days == DATES


def test_collapse_joins_on_both_keys_never_barrid_alone():
    """A join on barrid alone pairs every date of a name with every other."""
    plan = collapse(panel(), dates=CALENDAR, rebalance_period=2, lag=1).explain()
    assert 'ON: [col("barrid")]' not in plan
