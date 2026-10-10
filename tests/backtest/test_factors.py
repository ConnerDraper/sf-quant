"""Factor control, per-factor attribution, signal constraints, and candidate comparison."""

from dataclasses import dataclass

import numpy as np
import polars as pl
import pytest
import sf_quant.signals as sg
from conftest import config, random_alphas, run
from synthetic import INDUSTRIES, synthetic_market

from sf_quant.backtest import run_backtest
from sf_quant.backtest.compare import compare


@dataclass(frozen=True)
class Planted(sg.Signal):
    def compute(self, data, universe):
        return universe.join(data.planted_scores, on=["date", "barrid"])


@pytest.fixture(scope="module")
def planted_market():
    return synthetic_market(n_assets=150, n_days=1260, seed=4, planted_ic=0.12, industry_drift=0.5)


def after_warmup(md, **overrides):
    """Start once the signals' lookbacks (up to 252 days) have history."""
    return config(md, start=md.days[260], **overrides)


def test_factor_contributions_sum_to_active_factor(market):
    result = run(market, random_alphas(market), factor_control="risk_model")
    by_day = result.factor_returns.group_by("date").agg(pl.col("contribution").sum()).sort("date")
    joined = result.returns.join(by_day, on="date")
    np.testing.assert_allclose(joined["contribution"], joined["active_factor"], atol=1e-14)
    assert set(result.summary["factors"]) == set(market.factor_names)


def test_neutral_control_zeroes_every_factor(market):
    result = run(market, random_alphas(market))
    assert result.factor_exposures["active_exposure"].abs().max() < 1e-7
    style = result.factor_exposures.filter(pl.col("factor") == "USSLOWL_MOMENTUM")
    assert style["shadow_price"].is_finite().all()
    # Only drift between rebalances leaves any factor exposure.
    assert result.summary["factor_share_of_active_variance"] < 0.01


def test_industry_momentum_keeps_capped_industry_bets(planted_market):
    md = planted_market
    signal = sg.IndustryMomentum(constraints=sg.Constraints(factor_caps=sg.caps(sg.INDUSTRY_FACTORS, 0.02)))
    panel = sg.build_alpha(signal, md, md.days[0], md.days[-1])
    result = run_backtest(panel, after_warmup(md), md)
    exposures = result.factor_exposures
    industries = exposures.filter(pl.col("factor").is_in(INDUSTRIES))["active_exposure"]
    others = exposures.filter(~pl.col("factor").is_in(INDUSTRIES))["active_exposure"]
    assert industries.abs().max() > 0.01
    assert industries.abs().max() <= 0.02 + 1e-7
    assert others.abs().max() < 1e-7
    assert result.summary["metadata"]["allowed_factors"] == sorted(sg.INDUSTRY_FACTORS)


def test_the_mandate_erases_most_of_betting_against_beta(planted_market):
    # Alpha linear in beta earns nothing under sum(h) = 0 and beta'h = 0; only the
    # nonlinear remainder survives, so BAB keeps far less of its signal than the others.
    md = planted_market
    cfg = after_warmup(md)

    def transfer(signal):
        result = run_backtest(sg.build_alpha(signal, md, md.days[0], md.days[-1]), cfg, md)
        assert result.diagnostics["active_beta"].abs().max() < 1e-6
        return result.diagnostics["transfer_coefficient"].mean()

    assert transfer(sg.BettingAgainstBeta()) < 0.5 * transfer(Planted())


def test_compare_rewards_new_information_not_duplicates(planted_market):
    md = planted_market
    cfg = after_warmup(md)
    adds = compare([sg.ShortTermReversal()], Planted(), cfg, md, ic_shrink_obs=6)
    assert adds.ir_gain > 0.2
    assert set(adds.summary["run"]) == {"baseline", "with_candidate", "candidate"}
    assert adds.candidate_in_composite["marginal_ic"][-1] > 0

    duplicate = compare([Planted()], Planted(), cfg, md, ic_shrink_obs=6)
    assert abs(duplicate.ir_gain) < 0.05
