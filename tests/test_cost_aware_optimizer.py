"""cost_aware_mve_optimizer on synthetic problems; no data lake needed."""

import numpy as np
import pytest

from sf_quant.optimizer import (
    LongOnly,
    ZeroBeta,
    ZeroInvestment,
    cost_aware_mve_optimizer,
    dynamic_mve_optimizer,
)

MANDATE = [ZeroInvestment(), LongOnly(), ZeroBeta()]


def problem(n=60, k=4, seed=1):
    rng = np.random.default_rng(seed)
    root = rng.normal(size=(k, k)) * 0.1
    return {
        "ids": [f"A{i:03d}" for i in range(n)],
        "alphas": rng.normal(0, 0.01, n),
        "factor_exposures": rng.normal(size=(n, k)),
        "factor_covariance": root @ root.T,
        "specific_risk": rng.uniform(0.02, 0.2, n) ** 2,
        "betas": rng.uniform(0.5, 1.5, n),
        "benchmark_weights": rng.dirichlet(np.ones(n)),
    }


def solve(p, previous, lam, alphas=None):
    n = len(p["ids"])
    spreads = np.random.default_rng(3).uniform(0.0005, 0.01, n)
    out = cost_aware_mve_optimizer(
        **{**p, "alphas": p["alphas"] if alphas is None else alphas},
        constraints=MANDATE, previous_weights=previous, half_spreads=spreads, cost_aversion=lam,
    )
    return out["weight"].to_numpy(), out, spreads


def test_without_a_cost_term_it_matches_dynamic_mve_optimizer():
    p = problem()
    ours, _, _ = solve(p, np.zeros(len(p["ids"])), 0.0)
    theirs = dynamic_mve_optimizer(**p, constraints=MANDATE)["weight"].to_numpy()
    assert ours == pytest.approx(theirs, abs=5e-5)  # solver precision; weights are ~1e-2


def test_solutions_meet_the_mandate_and_the_risk_cap():
    p = problem()
    h, out, _ = solve(p, np.zeros(len(p["ids"])), 10.0)
    assert abs(h.sum()) < 1e-6
    assert abs(p["betas"] @ h) < 1e-6
    assert (h + p["benchmark_weights"]).min() > -1e-7
    assert out["active_risk"][0] <= 0.05 * (1 + 1e-4)


def test_turnover_does_not_rise_with_cost_aversion():
    p = problem()
    previous, _, _ = solve(p, np.zeros(len(p["ids"])), 0.0)
    new_alphas = p["alphas"] + np.random.default_rng(2).normal(0, 0.01, len(p["ids"]))
    turnover = [np.abs(solve(p, previous, lam, new_alphas)[0] - previous).sum() for lam in (0.0, 1.0, 10.0, 100.0)]
    assert all(t1 <= t0 + 1e-4 for t0, t1 in zip(turnover, turnover[1:]))
    assert turnover[-1] < turnover[0]


def test_reported_cost_is_half_spread_times_trades():
    p = problem()
    previous = np.zeros(len(p["ids"]))
    h, out, spreads = solve(p, previous, 5.0)
    assert out["cost"][0] == pytest.approx(float(spreads @ np.abs(h - previous)))


def test_a_clarabel_failure_falls_back_to_scs(monkeypatch):
    import cvxpy as cp

    p = problem()
    expected, _, _ = solve(p, np.zeros(len(p["ids"])), 10.0)
    original = cp.Problem.solve

    def clarabel_fails(self, *args, solver=None, **kwargs):
        if solver == "CLARABEL":
            raise cp.error.SolverError("Solver 'CLARABEL' failed.")
        return original(self, *args, solver=solver, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", clarabel_fails)
    h, out, _ = solve(p, np.zeros(len(p["ids"])), 10.0)
    assert h == pytest.approx(expected, abs=1e-3)  # SCS is less precise than CLARABEL
    assert abs(h.sum()) < 1e-4 and out["active_risk"][0] <= 0.05 * (1 + 1e-3)

