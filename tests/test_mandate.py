"""Tests for the mandate optimizer against the Silver Fund manual, Chapter 3."""

import numpy as np
import pytest

from sf_quant.optimizer import (
    active_variance,
    benchmark_betas,
    covariance_times,
    equality_direction,
    mandate_optimizer,
)


def make_model(n=80, k=5, seed=0, benchmark="equal"):
    """Barra-shaped synthetic risk model in annualized decimal units."""
    rng = np.random.default_rng(seed)
    B = rng.normal(0, 1, (n, k))
    B[:, 0] = 1.0  # market/country factor
    A = rng.normal(0, 0.05, (k, k))
    F = A @ A.T + np.diag(np.full(k, 0.002))
    F[0, 0] += 0.025  # ~16% market vol
    delta = rng.uniform(0.15, 0.45, n) ** 2
    if benchmark == "equal":
        b = np.full(n, 1 / n)
    else:  # cap-weighted with a long tail of tiny weights
        caps = rng.lognormal(0, 2, n)
        b = caps / caps.sum()
    alphas = 0.05 * np.sqrt(delta) * rng.normal(0, 1, n)
    return alphas, B, F, delta, b


def dense_v(B, F, delta):
    return B @ F @ B.T + np.diag(delta)


def solve(alphas, B, F, delta, b, target=0.05, **kwargs):
    ids = [f"A{i}" for i in range(len(alphas))]
    return mandate_optimizer(ids, alphas, B, F, delta, b, target_active_risk=target, **kwargs)


def test_benchmark_betas_match_dense_formula():
    _, B, F, delta, b = make_model()
    V = dense_v(B, F, delta)
    beta = benchmark_betas(B, F, delta, b)
    np.testing.assert_allclose(beta, V @ b / (b @ V @ b), rtol=1e-12)
    assert b @ beta == pytest.approx(1.0)


def test_covariance_helpers_match_dense():
    alphas, B, F, delta, _ = make_model()
    V = dense_v(B, F, delta)
    np.testing.assert_allclose(covariance_times(B, F, delta, alphas), V @ alphas, rtol=1e-12)
    assert active_variance(alphas, B, F, delta) == pytest.approx(alphas @ V @ alphas)


def test_equality_direction_manual_problem_20():
    # Manual 3.6 Problem 20: V = I, r = (0.10, 0.05, 0.02), C = [1 1 1].
    r = np.array([0.10, 0.05, 0.02])
    d = equality_direction(r, np.zeros((3, 1)), np.zeros((1, 1)), np.ones(3), np.ones((1, 3)))
    np.testing.assert_allclose(d, r - r.mean(), atol=1e-14)
    # Scale to 10% tracking error.
    h = 0.10 / np.sqrt(d @ d) * d
    assert np.sqrt(h @ h) == pytest.approx(0.10)


def test_matches_closed_form_when_long_only_is_slack():
    alphas, B, F, delta, b = make_model(benchmark="equal")
    target = 0.005  # small enough that no fund weight approaches zero
    beta = benchmark_betas(B, F, delta, b)
    d = equality_direction(alphas, B, F, delta, np.vstack([np.ones_like(b), beta]))
    expected = target / np.sqrt(active_variance(d, B, F, delta)) * d

    result = solve(alphas, B, F, delta, b, target=target)

    assert result.ok
    assert (b + result.active_weights).min() > 1e-3  # long-only bound is slack
    # Interior-point accuracy: 1e-6 is about $1 per position on $1M AUM.
    np.testing.assert_allclose(result.active_weights, expected, atol=1e-6)


@pytest.mark.parametrize("benchmark", ["equal", "cap"])
def test_constraints_hold_and_risk_binds(benchmark):
    alphas, B, F, delta, b = make_model(benchmark=benchmark)
    result = solve(alphas, B, F, delta, b)
    h = result.active_weights
    beta = benchmark_betas(B, F, delta, b)

    assert result.ok
    assert abs(h.sum()) < 1e-8
    assert abs(beta @ h) < 1e-8
    assert abs(result.active_beta) < 1e-8
    assert (b + h).min() > -1e-8
    assert result.ex_ante_te == pytest.approx(0.05, rel=1e-5)
    assert result.te_shortfall == pytest.approx(0.0, abs=1e-6)
    assert result.expected_active_return == pytest.approx(alphas @ h)
    np.testing.assert_allclose(result.active_factor_exposures, B.T @ h)


def test_scale_invariance():
    alphas, B, F, delta, b = make_model(benchmark="cap")
    h1 = solve(alphas, B, F, delta, b).active_weights
    h2 = solve(10 * alphas, B, F, delta, b).active_weights
    np.testing.assert_allclose(h1, h2, atol=1e-6)


def test_zero_alpha_holds_benchmark():
    _, B, F, delta, b = make_model()
    result = solve(np.zeros(len(b)), B, F, delta, b)
    assert result.ok
    np.testing.assert_array_equal(result.active_weights, 0.0)
    assert result.te_shortfall == pytest.approx(0.05)


def test_total_fund_variance_is_constant():
    # Manual 3.5: zero active beta and a fixed TE make fund variance b'Vb + TE^2.
    alphas, B, F, delta, b = make_model(benchmark="cap")
    h = solve(alphas, B, F, delta, b).active_weights
    V = dense_v(B, F, delta)
    w = b + h
    assert w @ V @ w == pytest.approx(b @ V @ b + 0.05**2, rel=1e-6)


def test_long_only_binds_with_small_benchmark_weights():
    alphas, B, F, delta, b = make_model(n=150, benchmark="cap", seed=3)
    result = solve(alphas, B, F, delta, b)
    beta = benchmark_betas(B, F, delta, b)
    d = equality_direction(alphas, B, F, delta, np.vstack([np.ones_like(b), beta]))
    unconstrained_ir = (alphas @ d) / np.sqrt(active_variance(d, B, F, delta))

    assert result.ok
    assert result.n_at_long_only_bound > 0
    assert result.expected_active_return / result.ex_ante_te <= unconstrained_ir + 1e-9


def test_factor_bounds_are_respected():
    alphas, B, F, delta, b = make_model(benchmark="cap")
    free = solve(alphas, B, F, delta, b)
    k = int(np.argmax(np.abs(free.active_factor_exposures[1:]))) + 1
    cap = 0.25 * abs(free.active_factor_exposures[k])

    bounded = solve(alphas, B, F, delta, b, factor_bounds={k: cap})

    assert bounded.ok
    assert abs(bounded.active_factor_exposures[k]) <= cap + 1e-8


def test_rejects_bad_inputs():
    alphas, B, F, delta, b = make_model()
    with pytest.raises(ValueError):
        solve(alphas[:-1], B, F, delta, b)
    with pytest.raises(ValueError):
        solve(alphas, B, F, -delta, b)
    with pytest.raises(ValueError):
        solve(alphas, B, F, delta, b, target=0.0)


def make_barra_like(n=120, n_ind=4, n_style=3, seed=5):
    """Country column of ones plus 0/1 industries (which sum to the country column) and styles."""
    rng = np.random.default_rng(seed)
    B = np.zeros((n, 1 + n_ind + n_style))
    B[:, 0] = 1.0
    B[np.arange(n), 1 + rng.integers(0, n_ind, n)] = 1.0
    B[:, 1 + n_ind :] = rng.normal(0, 1, (n, n_style))
    F = np.diag([0.16**2] + [0.05**2] * n_ind + [0.04**2] * n_style)
    delta = rng.uniform(0.2, 0.45, n) ** 2
    b = np.full(n, 1 / n)
    alphas = 0.05 * np.sqrt(delta) * rng.normal(0, 1, n)
    return alphas, B, F, delta, b


def test_neutral_factors_are_exactly_zero():
    alphas, B, F, delta, b = make_barra_like()
    alphas = alphas + 0.02 * B[:, 5]  # alpha that wants style exposure
    result = solve(alphas, B, F, delta, b, neutral_factors=[5, 6])
    assert result.ok
    np.testing.assert_allclose(result.active_factor_exposures[[5, 6]], 0, atol=1e-8)
    assert result.ex_ante_te == pytest.approx(0.05, rel=1e-5)


def test_full_neutrality_with_collinear_country_and_industries():
    # Country = iota = sum of industries: three redundant constraint rows.
    alphas, B, F, delta, b = make_barra_like()
    result = solve(alphas, B, F, delta, b, neutral_factors=list(range(B.shape[1])))
    assert result.ok
    np.testing.assert_allclose(result.active_factor_exposures, 0, atol=1e-7)
    # The country row duplicates sum(h) = 0, so it has no shadow price of its own.
    assert np.isnan(result.factor_shadow_prices[0])
    assert np.isfinite(result.factor_shadow_prices[5:]).all()


def test_shadow_price_matches_relaxing_the_constraint():
    alphas, B, F, delta, b = make_barra_like()
    alphas = alphas + 0.02 * B[:, 5]
    neutral = solve(alphas, B, F, delta, b, neutral_factors=[5])
    price = neutral.factor_shadow_prices[5]
    assert price > 0  # the alpha wants positive exposure to factor 5

    eps = 1e-3
    relaxed = solve(alphas, B, F, delta, b, factor_bounds={5: eps})
    gain = relaxed.expected_active_return - neutral.expected_active_return
    assert gain == pytest.approx(price * eps, rel=0.05)
    assert np.isnan(neutral.factor_shadow_prices[6])  # not held neutral


def test_alpha_linear_in_beta_is_worthless_under_the_mandate():
    # Betting against beta: alpha = c(1 - beta). Sum(h) = 0 and beta'h = 0 make h'alpha = 0.
    _, B, F, delta, b = make_barra_like()
    alphas = 0.03 * (1 - benchmark_betas(B, F, delta, b))
    mandate = solve(alphas, B, F, delta, b)
    relaxed = solve(alphas, B, F, delta, b, zero_active_beta=False)
    assert abs(mandate.expected_active_return) < 1e-8
    assert relaxed.expected_active_return > 1e-4  # positive once beta may move
    assert relaxed.active_beta < -0.01
