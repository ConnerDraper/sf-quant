"""
Canary tests from the handoff. They catch timing bugs unit tests miss.

Ten years of synthetic data keeps the IR standard error near
1 / sqrt(10) ≈ 0.32, so the thresholds are about three standard errors.
"""

import polars as pl
import pytest
from conftest import random_alphas, run
from synthetic import synthetic_market

from sf_quant.backtest.backtest import schedule

N_DAYS = 2520


@pytest.fixture(scope="module")
def long_market():
    return synthetic_market(n_assets=150, n_days=N_DAYS, seed=7)


def test_perfect_foresight_is_absurd(long_market):
    """Alpha equal to next period's realized specific return must look absurdly good."""
    md, days = long_market, long_market.days
    rebalances = schedule(days, "monthly", 1)
    windows = []
    for k, r in enumerate(rebalances):
        end = rebalances[k + 1].execution if k + 1 < len(rebalances) else days[-1]
        windows.append(
            md.specific.filter(pl.col("date") > r.execution, pl.col("date") <= end)
            .group_by("barrid")
            .agg(pl.col("specific_return").sum().alias("alpha"))
            .with_columns(pl.lit(r.formation).alias("date"))
        )
    alphas = pl.concat(windows).select("date", "barrid", "alpha")

    assert run(md, alphas).summary["ir_gross"] > 3


def test_same_day_leak_is_nothing_special(long_market):
    """Alpha equal to the formation day's own return is stale by execution. A great IR means look-ahead."""
    alphas = long_market.specific.rename({"specific_return": "alpha"})
    assert abs(run(long_market, alphas).summary["ir_gross"]) < 1.0


def test_random_alpha_has_no_skill_and_hits_the_risk_target(long_market):
    summary = run(long_market, random_alphas(long_market, seed=11)).summary
    assert abs(summary["ir_gross"]) < 1.0
    assert summary["mean_ex_ante_te"] == pytest.approx(0.05, rel=1e-4)
    assert 0.85 < summary["realized_to_ex_ante_te"] < 1.15
    assert abs(summary["realized_active_beta"]) < 0.05
    assert summary["failed_rebalances"] == 0
