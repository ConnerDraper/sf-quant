"""Schedule, invariants, failed solves, determinism, and outputs."""

import dataclasses
import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from conftest import config, random_alphas, run
from sf_quant.testing import business_days, synthetic_market

from sf_quant.backtest import BacktestConfig, InvariantError, backtest
from sf_quant.backtest.backtest import Rebalance, schedule
from sf_quant.backtest.cli import sbatch_script


@pytest.fixture
def alphas(market):
    return random_alphas(market, seed=3)


def test_schedule_monthly_weekly_and_lag():
    days = business_days(dt.date(2024, 1, 29), 30)  # Jan 29 .. Mar 8, 2024
    # Mar 8 is the last day: nothing left to earn on, so it is dropped.
    assert schedule(days, "monthly", 1) == [
        Rebalance(dt.date(2024, 1, 31), dt.date(2024, 2, 1)),
        Rebalance(dt.date(2024, 2, 29), dt.date(2024, 3, 1)),
    ]
    assert schedule(days, "monthly", 0)[0] == Rebalance(dt.date(2024, 1, 31), dt.date(2024, 1, 31))
    assert [r.formation for r in schedule(days, "weekly", 1)[:2]] == [dt.date(2024, 2, 2), dt.date(2024, 2, 9)]


# Invariants


def test_duplicate_rows_raise(market, alphas):
    with pytest.raises(InvariantError, match="duplicate"):
        run(market, pl.concat([alphas, alphas.head(1)]))


def test_non_finite_alpha_raises(market, alphas):
    with pytest.raises(InvariantError, match="non-finite"):
        run(market, alphas.with_columns(pl.col("alpha").replace(alphas["alpha"][0], float("nan"))))


def test_low_coverage_raises(market, alphas):
    with pytest.raises(InvariantError, match="coverage"):
        run(market, alphas.filter(pl.col("barrid") < "S0050"))


def test_units_check_catches_percent_risk_model(alphas):
    md = synthetic_market(n_assets=150, n_days=504, seed=1)
    md.F, md.delta = md.F * 100**2, md.delta * 100**2
    with pytest.raises(InvariantError, match="annualized decimal"):
        run(md, alphas)


def test_mandate_holds_on_every_rebalance(market, alphas):
    result = run(market, alphas)
    by_date = result.weights.group_by("formation_date").agg(
        pl.col("active_weight").sum().abs().alias("sum_h"),
        pl.col("weight").min().alias("min_w"),
        (pl.col("weight").sum() - 1).abs().alias("sum_w_error"),
    )
    assert by_date["sum_h"].max() < 1e-6
    assert by_date["min_w"].min() >= 0
    assert by_date["sum_w_error"].max() < 1e-6
    assert (result.diagnostics["active_beta"].abs() < 1e-6).all()


# Failed solves


@pytest.fixture
def second_solve_fails(monkeypatch):
    real, calls = backtest.mandate_optimizer, []

    def flaky(*args, **kwargs):
        calls.append(1)
        result = real(*args, **kwargs)
        if len(calls) == 2:
            return dataclasses.replace(result, status="infeasible", active_weights=np.zeros_like(result.active_weights))
        return result

    monkeypatch.setattr(backtest, "mandate_optimizer", flaky)


def test_failed_rebalance_holds_previous_weights(market, alphas, second_solve_fails):
    result = run(market, alphas)
    assert result.summary["failed_rebalances"] == 1
    assert result.diagnostics.filter(pl.col("solver_status") == "infeasible")["turnover"].item() == 0.0
    # The previous holdings kept drifting through the failed date: no gap in daily returns.
    first = market.days.index(result.returns["date"].min())
    assert result.returns["date"].to_list() == market.days[first:]


def test_names_missing_risk_data_are_held_at_benchmark(alphas):
    md = synthetic_market(n_assets=150, n_days=504, seed=1)
    md.missing_risk = {"S0003"}
    result = run(md, alphas)
    s3 = result.weights.filter(pl.col("barrid") == "S0003")
    assert (s3["active_weight"] == 0).all()
    assert (s3["weight"] == s3["benchmark_weight"]).all()
    assert (result.diagnostics["n_missing_risk"] == 1).all()


# Determinism and outputs


def test_parallel_and_serial_give_identical_results(market, alphas):
    serial, parallel = run(market, alphas), run(market, alphas, n_workers=2)
    assert serial.weights.equals(parallel.weights)
    assert serial.returns.equals(parallel.returns)


def test_save_writes_every_output_file(market, alphas, tmp_path):
    out = run(market, alphas).save(tmp_path / "run")
    assert sorted(p.name for p in out.iterdir()) == [
        "config.resolved.yml",
        "diagnostics.parquet",
        "factor_exposures.parquet",
        "factor_returns.parquet",
        "returns.parquet",
        "summary.json",
        "weights.parquet",
    ]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["metadata"]["config_hash"] == config(market).hash()
    assert BacktestConfig.from_yaml(out / "config.resolved.yml") == config(market)


def test_sbatch_script_is_single_node():
    cfg = BacktestConfig.from_dict(
        {"name": "mom", "start": "2010-01-01", "end": "2020-12-31", "slurm": {"n_cpus": 16, "email": "x@byu.edu"}}
    )
    script = sbatch_script(cfg, Path("config.yml"), Path("alphas.parquet"), Path("runs/mom"))
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --cpus-per-task=16" in script
    assert "#SBATCH --array" not in script
    assert "sf_quant.backtest.cli run" in script
