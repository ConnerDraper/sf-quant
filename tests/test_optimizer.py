"""Optimizer tests. Synthetic risk model throughout, so these need no data lake."""

import numpy as np
import pytest

from sf_quant.optimizer import (
    DEFAULT_CONSTRAINTS,
    LongOnly,
    ZeroBeta,
    ZeroInvestment,
    active_mve_optimizer,
)

TARGET = 0.05
N, K = 40, 5


def model(seed: int = 0):
    """A well-conditioned factor model plus a cap-weighted benchmark."""
    rng = np.random.default_rng(seed)
    ids = [f"USA{i:04d}" for i in range(N)]
    factor_exposures = rng.normal(size=(N, K))
    root = rng.normal(size=(K, K))
    factor_covariance = root @ root.T / K + np.eye(K) * 0.02
    specific_risk = rng.uniform(0.02, 0.09, size=N) ** 2
    cap = rng.pareto(1.5, size=N) + 1.0
    benchmark_weights = cap / cap.sum()
    betas = rng.uniform(0.6, 1.5, size=N)
    alphas = rng.normal(scale=0.02, size=N)
    return dict(
        ids=ids,
        alphas=alphas,
        factor_exposures=factor_exposures,
        factor_covariance=factor_covariance,
        specific_risk=specific_risk,
        benchmark_weights=benchmark_weights,
        betas=betas,
    )


def test_defaults_satisfy_the_active_mandate():
    m = model()
    out = active_mve_optimizer(**m, target_active_risk=TARGET)
    h = out["active_weight"].to_numpy()

    assert out.columns == ["barrid", "active_weight", "active_risk"]
    assert np.isclose(h.sum(), 0.0, atol=1e-7)
    assert np.all(h >= -m["benchmark_weights"] - 1e-7)
    assert np.isclose(m["betas"] @ h, 0.0, atol=1e-7)


def test_gamma_is_gone():
    out = active_mve_optimizer(**model(), target_active_risk=TARGET)
    assert "gamma" not in out.columns


def test_risk_budget_binds_at_the_target():
    out = active_mve_optimizer(**model(), target_active_risk=TARGET)
    # Linear objective on a convex ball: the budget is always spent in full.
    assert out["active_risk"][0] == pytest.approx(TARGET, rel=1e-3)


@pytest.mark.parametrize("target", [0.01, 0.02, 0.05, 0.10])
def test_active_risk_tracks_the_requested_target(target):
    m = model()
    out = active_mve_optimizer(**m, target_active_risk=target)
    h = out["active_weight"].to_numpy()

    loadings = m["factor_exposures"].T @ h
    variance = (
        loadings @ m["factor_covariance"] @ loadings
        + np.sum(m["specific_risk"] * h**2)
    )
    assert np.sqrt(variance) == pytest.approx(target, rel=1e-3)


def test_total_portfolio_is_long_only_and_fully_invested():
    m = model()
    out = active_mve_optimizer(**m, target_active_risk=TARGET)
    w = out["active_weight"].to_numpy() + m["benchmark_weights"]

    assert np.all(w >= -1e-7)
    assert np.isclose(w.sum(), 1.0, atol=1e-7)
    # Unit beta follows from zero active beta plus a unit-beta benchmark.
    assert m["betas"] @ w == pytest.approx(m["betas"] @ m["benchmark_weights"])


def test_dropping_zero_beta_frees_active_beta():
    m = model()
    constrained = active_mve_optimizer(**m, target_active_risk=TARGET)
    free = active_mve_optimizer(
        **m,
        target_active_risk=TARGET,
        constraints=(ZeroInvestment(), LongOnly()),
    )

    beta_free = m["betas"] @ free["active_weight"].to_numpy()
    assert abs(beta_free) > 1e-4
    # Relaxing a constraint cannot reduce the attainable alpha.
    assert m["alphas"] @ free["active_weight"].to_numpy() >= (
        m["alphas"] @ constrained["active_weight"].to_numpy() - 1e-9
    )


def test_long_only_can_be_dropped_to_allow_true_shorts():
    m = model()
    out = active_mve_optimizer(
        **m,
        target_active_risk=TARGET,
        constraints=(ZeroInvestment(), ZeroBeta()),
    )
    w = out["active_weight"].to_numpy() + m["benchmark_weights"]
    assert w.min() < 0


def test_zero_beta_without_betas_is_rejected():
    m = model()
    m["betas"] = None
    with pytest.raises(ValueError, match="ZeroBeta requires betas"):
        active_mve_optimizer(**m, target_active_risk=TARGET)


def test_non_finite_benchmark_weight_is_rejected():
    """A null benchmark_weight left-joined onto the universe arrives as NaN."""
    m = model()
    m["benchmark_weights"] = m["benchmark_weights"].copy()
    m["benchmark_weights"][3] = np.nan
    with pytest.raises(ValueError, match="benchmark_weights has 1 non-finite"):
        active_mve_optimizer(**m, target_active_risk=TARGET)


def test_length_mismatch_is_rejected():
    m = model()
    m["alphas"] = m["alphas"][:-1]
    with pytest.raises(ValueError, match="length mismatch"):
        active_mve_optimizer(**m, target_active_risk=TARGET)


def test_non_positive_target_is_rejected():
    with pytest.raises(ValueError, match="target_active_risk must be positive"):
        active_mve_optimizer(**model(), target_active_risk=0.0)


def test_default_constraints_are_the_fund_mandate():
    assert [type(c).__name__ for c in DEFAULT_CONSTRAINTS] == [
        "ZeroInvestment",
        "ZeroBeta",
        "LongOnly",
    ]


EQUALITY_SETS = [
    (ZeroInvestment(), ZeroBeta()),
    (ZeroInvestment(),),
    (ZeroBeta(),),
    (),
]


@pytest.mark.parametrize("constraints", EQUALITY_SETS, ids=lambda c: "+".join(type(x).__name__ for x in c) or "none")
def test_closed_form_matches_the_numerical_solver(constraints):
    """Manual §3.3 and §3.4 solve the same problem when nothing can bind."""
    from sf_quant.optimizer.optimizers import _numerical

    m = model()
    closed = active_mve_optimizer(**m, target_active_risk=TARGET, constraints=constraints)
    h = closed["active_weight"].to_numpy()
    numerical = _numerical(
        m["alphas"], m["factor_exposures"], m["factor_covariance"], m["specific_risk"],
        m["benchmark_weights"], m["betas"], constraints, TARGET,
    )

    assert closed["active_risk"][0] == pytest.approx(TARGET, rel=1e-12)
    for c in constraints:
        assert c.violation(h, benchmark_weights=m["benchmark_weights"], betas=m["betas"]) < 1e-12
    # The optimum is unique, so the two agree. CLARABEL may overshoot the risk
    # budget by its own tolerance (~1e-8), so compare alpha per unit of risk.
    assert np.allclose(h, numerical, atol=1e-5)
    numerical_risk = np.sqrt(
        numerical @ m["factor_exposures"] @ m["factor_covariance"] @ m["factor_exposures"].T @ numerical
        + np.sum(m["specific_risk"] * numerical**2)
    )
    assert m["alphas"] @ h / TARGET >= (m["alphas"] @ numerical / numerical_risk) * (1 - 1e-9)


def test_closed_form_is_the_manual_formula():
    """d = V⁻¹[I − Cᵀ(CV⁻¹Cᵀ)⁻¹CV⁻¹]α with an explicit inverse, as in Manual §3.3.4."""
    m = model()
    V = m["factor_exposures"] @ m["factor_covariance"] @ m["factor_exposures"].T + np.diag(m["specific_risk"])
    C = np.vstack([np.ones(N), m["betas"]])
    V_inv = np.linalg.inv(V)
    d = V_inv @ (np.eye(N) - C.T @ np.linalg.inv(C @ V_inv @ C.T) @ C @ V_inv) @ m["alphas"]
    expected = d * TARGET / np.sqrt(d @ V @ d)

    out = active_mve_optimizer(**m, target_active_risk=TARGET, constraints=(ZeroInvestment(), ZeroBeta()))
    assert np.allclose(out["active_weight"].to_numpy(), expected, rtol=1e-8, atol=1e-12)


def test_long_only_still_goes_through_the_numerical_solver():
    m = model()
    out = active_mve_optimizer(**m, target_active_risk=TARGET)
    w = out["active_weight"].to_numpy() + m["benchmark_weights"]
    # The closed form ignores the bound and would short; this must not.
    assert w.min() >= -1e-6


def test_violation_is_zero_only_when_satisfied():
    m = model()
    b, betas = m["benchmark_weights"], m["betas"]
    h = np.zeros(N)
    h[0], h[1] = 0.01, -0.01
    assert ZeroInvestment().violation(h, benchmark_weights=b, betas=betas) == 0.0
    assert ZeroBeta().violation(h, benchmark_weights=b, betas=betas) == pytest.approx(abs(0.01 * (betas[0] - betas[1])))
    assert LongOnly().violation(-2 * b, benchmark_weights=b, betas=betas) == pytest.approx(b.max())
