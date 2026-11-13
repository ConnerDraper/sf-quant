"""calibrate_gamma, the gamma term of cost_aware_mve_optimizer, MaxShort and GroupNeutral; synthetic problems only."""

import numpy as np
import pytest

from sf_quant.optimizer import calibrate_gamma


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
