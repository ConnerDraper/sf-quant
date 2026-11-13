"""calibrate_gamma, the gamma term of cost_aware_mve_optimizer, MaxShort and GroupNeutral; synthetic problems only."""

import numpy as np
import pytest

from sf_quant.optimizer import (GroupNeutral, LongOnly, MaxShort, ZeroBeta, ZeroInvestment, calibrate_gamma,
                                cost_aware_mve_optimizer)


def problem(n=40, k=3, seed=4):
    rng = np.random.default_rng(seed)
    root = rng.normal(size=(k, k)) * 0.1
    return {
        "ids": [f"A{i:03d}" for i in range(n)],
        "alphas": rng.normal(0, 0.01, n),
        "factor_exposures": rng.normal(size=(n, k)),
        "factor_covariance": root @ root.T + 0.01 * np.eye(k),
        "specific_risk": rng.uniform(0.02, 0.2, n) ** 2,
        "betas": rng.uniform(0.5, 1.5, n),
        "benchmark_weights": rng.dirichlet(np.ones(n)),
    }


def covariance(p):
    B = p["factor_exposures"]
    return B @ p["factor_covariance"] @ B.T + np.diag(p["specific_risk"])


def risk_model(p):
    return {k: p[k] for k in ("alphas", "factor_exposures", "factor_covariance", "specific_risk")}


def test_calibrated_gamma_matches_a_dense_inverse():
    p = problem()
    V = covariance(p)
    dense = np.sqrt(p["alphas"] @ np.linalg.solve(V, p["alphas"])) / 0.02
    assert calibrate_gamma(**risk_model(p), target_active_risk=0.02) == pytest.approx(dense, rel=1e-10)


def test_the_unconstrained_portfolio_lands_on_the_target():
    p = problem()
    V = covariance(p)
    gamma = calibrate_gamma(**risk_model(p), target_active_risk=0.01)
    h = np.linalg.solve(V, p["alphas"]) / gamma
    assert np.sqrt(h @ V @ h) == pytest.approx(0.01, rel=1e-10)


def test_zero_alphas_give_zero_gamma():
    p = problem()
    assert calibrate_gamma(**{**risk_model(p), "alphas": np.zeros(len(p["ids"]))}) == 0.0


def optimize(p, constraints, gamma, cap=0.05, lam=0.0, previous=None):
    n = len(p["ids"])
    out = cost_aware_mve_optimizer(
        **p, constraints=constraints, previous_weights=np.zeros(n) if previous is None else previous,
        half_spreads=np.full(n, 0.001), cost_aversion=lam, target_active_risk=cap, gamma=gamma)
    return out["weight"].to_numpy(), out


def test_gamma_zero_changes_nothing():
    p = problem()
    n = len(p["ids"])
    default = cost_aware_mve_optimizer(**p, constraints=[ZeroInvestment()], previous_weights=np.zeros(n),
                                       half_spreads=np.full(n, 0.001), cost_aversion=0.0)["weight"].to_numpy()
    h, _ = optimize(p, [ZeroInvestment()], 0.0)
    assert h == pytest.approx(default, abs=1e-9)


def test_with_a_slack_cap_the_solution_is_v_inverse_alpha_over_gamma():
    p = problem()
    gamma = calibrate_gamma(**risk_model(p), target_active_risk=0.01)
    h, out = optimize(p, [], gamma, cap=0.05)
    assert h == pytest.approx(np.linalg.solve(covariance(p), p["alphas"]) / gamma, abs=1e-5)
    assert out["active_risk"][0] == pytest.approx(0.01, rel=1e-3)


def test_a_larger_gamma_takes_less_risk():
    p = problem()
    gamma = calibrate_gamma(**risk_model(p), target_active_risk=0.01)
    _, low = optimize(p, [ZeroInvestment()], gamma)
    _, high = optimize(p, [ZeroInvestment()], 4 * gamma)
    assert high["active_risk"][0] < low["active_risk"][0]


def test_max_short_bounds_total_weight():
    p = problem()
    h, _ = optimize(p, [ZeroInvestment(), ZeroBeta(), MaxShort(0.002)], 0.0)
    assert (h + p["benchmark_weights"]).min() >= -0.002 - 1e-7
    assert (h + p["benchmark_weights"]).min() < -0.001  # the limit binds here


def test_max_short_zero_is_long_only():
    p = problem()
    a, _ = optimize(p, [ZeroInvestment(), ZeroBeta(), MaxShort(0.0)], 0.0)
    b, _ = optimize(p, [ZeroInvestment(), ZeroBeta(), LongOnly()], 0.0)
    assert a == pytest.approx(b, abs=1e-6)


def test_max_short_rejects_a_negative_limit():
    with pytest.raises(ValueError):
        MaxShort(-0.01)


def test_group_neutral_nets_each_group_to_zero():
    p = problem()
    groups = np.arange(len(p["ids"])) % 5
    h, _ = optimize(p, [ZeroInvestment(), GroupNeutral(groups)], 0.0)
    for g in range(5):
        assert abs(h[groups == g].sum()) < 1e-7
    assert np.abs(h).sum() > 0.01  # still takes positions
