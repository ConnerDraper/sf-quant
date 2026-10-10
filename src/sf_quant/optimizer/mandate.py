"""
Mandate optimizer for the Silver Fund.

Solves the problem in Chapter 3 of the Silver Fund manual directly in active
weights. There is no risk-aversion parameter: the active portfolio is sized to
the target tracking error by the risk constraint itself.

.. math::

    \\max_h \\; h^T \\alpha \\quad \\text{s.t.} \\quad
    \\iota^T h = 0, \\quad \\beta^T h = 0, \\quad
    h^T V h \\le \\bar\\sigma_a^2, \\quad w_b + h \\ge 0

with :math:`V = B F B^T + \\text{diag}(\\delta)` and
:math:`\\beta = V w_b / (w_b^T V w_b)` (manual 3.3.2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import polars as pl

_OK_STATUSES = ("optimal", "optimal_inaccurate")


@dataclass(frozen=True)
class MandateResult:
    """
    Output of :func:`mandate_optimizer`.

    Attributes
    ----------
    ids : list[str]
        Asset identifiers, in the order passed to the optimizer.
    active_weights : np.ndarray
        Active weights h, shape (n_assets,). All zeros if the solve failed.
    status : str
        CVXPY solver status, e.g. ``"optimal"`` or ``"infeasible"``.
    solve_time : float
        Wall-clock seconds spent building and solving the problem.
    ex_ante_te : float
        Predicted annualized tracking error, :math:`\\sqrt{h^T V h}`.
    te_shortfall : float
        ``max(target - ex_ante_te, 0)``. Positive when the constraints keep
        the portfolio from reaching the risk target.
    active_beta : float
        :math:`\\beta^T h`. Should be about zero.
    expected_active_return : float
        :math:`h^T \\alpha`.
    n_at_long_only_bound : int
        Number of benchmark names (positive benchmark weight) whose fund
        weight is driven to zero.
    active_factor_exposures : np.ndarray
        :math:`B^T h`, shape (n_factors,).
    factor_shadow_prices : np.ndarray
        For each factor held neutral, the change in expected annual active
        return per unit of active exposure if its neutrality were relaxed:
        what that constraint costs. NaN for factors not held neutral or
        dropped as redundant, shape (n_factors,).
    """

    ids: list[str]
    active_weights: np.ndarray
    status: str
    solve_time: float
    ex_ante_te: float
    te_shortfall: float
    active_beta: float
    expected_active_return: float
    n_at_long_only_bound: int
    active_factor_exposures: np.ndarray
    factor_shadow_prices: np.ndarray

    @property
    def ok(self) -> bool:
        """Whether the solver returned a usable solution."""
        return self.status in _OK_STATUSES

    def to_frame(self) -> pl.DataFrame:
        """Return ``barrid`` and ``active_weight`` as a Polars DataFrame."""
        return pl.DataFrame({"barrid": self.ids, "active_weight": self.active_weights})


def covariance_times(
    factor_exposures: np.ndarray,
    factor_covariance: np.ndarray,
    specific_variance: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """
    Compute :math:`V x` for :math:`V = B F B^T + \\text{diag}(\\delta)`
    without forming the N x N matrix.
    """
    return factor_exposures @ (factor_covariance @ (factor_exposures.T @ x)) + specific_variance * x


def active_variance(
    active_weights: np.ndarray,
    factor_exposures: np.ndarray,
    factor_covariance: np.ndarray,
    specific_variance: np.ndarray,
) -> float:
    """Compute :math:`h^T V h` using the factor decomposition."""
    loadings = factor_exposures.T @ active_weights
    return float(
        loadings @ factor_covariance @ loadings
        + np.sum(specific_variance * active_weights**2)
    )


def benchmark_betas(
    factor_exposures: np.ndarray,
    factor_covariance: np.ndarray,
    specific_variance: np.ndarray,
    benchmark_weights: np.ndarray,
) -> np.ndarray:
    """
    Compute asset betas to the benchmark from the risk model.

    .. math::

        \\beta = \\frac{V w_b}{w_b^T V w_b}

    This is manual equation 3.3.2. Using the same V for beta and for risk is
    what makes total-fund variance constant across feasible portfolios
    (manual 3.5).

    Parameters
    ----------
    factor_exposures : np.ndarray
        B, shape (n_assets, n_factors).
    factor_covariance : np.ndarray
        F, shape (n_factors, n_factors), annualized decimal units.
    specific_variance : np.ndarray
        Diagonal of the specific covariance, shape (n_assets,), annualized
        decimal variance.
    benchmark_weights : np.ndarray
        Benchmark weights, shape (n_assets,).

    Returns
    -------
    np.ndarray
        Betas, shape (n_assets,). ``benchmark_weights @ betas == 1``.
    """
    vb = covariance_times(factor_exposures, factor_covariance, specific_variance, benchmark_weights)
    benchmark_variance = float(benchmark_weights @ vb)
    if not benchmark_variance > 0:
        raise ValueError("Benchmark variance must be positive to compute betas.")
    return vb / benchmark_variance


def equality_direction(
    alphas: np.ndarray,
    factor_exposures: np.ndarray,
    factor_covariance: np.ndarray,
    specific_variance: np.ndarray,
    constraint_matrix: np.ndarray,
) -> np.ndarray:
    """
    Closed-form optimal direction under linear equality constraints only.

    .. math::

        d = V^{-1}\\left[I - C^T (C V^{-1} C^T)^{-1} C V^{-1}\\right] \\alpha

    This is manual equation 3.2.9. Scale it by
    :math:`\\bar\\sigma_a / \\sqrt{d^T V d}` to hit the risk target
    (equation 3.2.10). It ignores the long-only bound, so it equals the
    :func:`mandate_optimizer` solution only when that bound is slack.

    :math:`V^{-1}` is applied with the Woodbury identity, which requires
    strictly positive specific variance.
    """
    if np.any(specific_variance <= 0):
        raise ValueError("equality_direction requires strictly positive specific variance.")

    B, F = factor_exposures, factor_covariance
    d_inv = 1.0 / specific_variance
    n_factors = F.shape[0]
    inner = np.eye(n_factors) + (B.T * d_inv) @ B @ F

    def solve_v(x: np.ndarray) -> np.ndarray:
        y = d_inv[:, None] * x
        return y - d_inv[:, None] * (B @ (F @ np.linalg.solve(inner, B.T @ y)))

    C = np.atleast_2d(constraint_matrix)
    g_alpha = solve_v(alphas[:, None])[:, 0]
    g_c = solve_v(C.T)
    correction = g_c @ np.linalg.solve(C @ g_c, C @ g_alpha)
    return g_alpha - correction


def mandate_optimizer(
    ids: list[str],
    alphas: np.ndarray,
    factor_exposures: np.ndarray,
    factor_covariance: np.ndarray,
    specific_variance: np.ndarray,
    benchmark_weights: np.ndarray,
    target_active_risk: float = 0.05,
    factor_bounds: dict[int, float] | None = None,
    neutral_factors: list[int] | None = None,
) -> MandateResult:
    """
    Solve the Silver Fund mandate problem for one rebalance date.

    .. math::

        \\max_h \\; h^T \\alpha \\quad \\text{s.t.} \\quad
        \\iota^T h = 0, \\quad \\beta^T h = 0, \\quad
        h^T V h \\le \\bar\\sigma_a^2, \\quad w_b + h \\ge 0,
        \\quad |B_k^T h| \\le c_k

    The risk constraint is an inequality. It gives the manual's portfolio
    whenever it binds; when it does not, long-only has used up the
    opportunity set and the gap is reported as ``te_shortfall``.

    Multiplying every alpha by one positive constant leaves the solution
    unchanged, so the IC used to build alphas affects reported expected
    return but not weights.

    Parameters
    ----------
    ids : list[str]
        Asset identifiers.
    alphas : np.ndarray
        Expected residual returns, shape (n_assets,), annualized decimal.
    factor_exposures : np.ndarray
        B, shape (n_assets, n_factors).
    factor_covariance : np.ndarray
        F, shape (n_factors, n_factors), annualized decimal units.
    specific_variance : np.ndarray
        Specific variance per asset, shape (n_assets,), annualized decimal.
    benchmark_weights : np.ndarray
        Benchmark weights, shape (n_assets,). Used for beta and for the
        long-only bound ``h >= -benchmark_weights``.
    target_active_risk : float, optional
        Annualized tracking-error target. Default 0.05.
    factor_bounds : dict[int, float], optional
        Optional caps on absolute active factor exposure, keyed by the
        column index of ``factor_exposures``.
    neutral_factors : list[int], optional
        Column indices of factors whose active exposure must be exactly zero.
        Constraints that are implied by others (for example the country
        factor, which equals the full-investment row when every asset has
        exposure 1) are dropped before solving.

    Returns
    -------
    MandateResult
        Active weights plus solve diagnostics. Never raises on solver
        failure; check ``result.ok``.

    Examples
    --------
    >>> import numpy as np
    >>> import sf_quant.optimizer as sfo
    >>> rng = np.random.default_rng(0)
    >>> n, k = 50, 3
    >>> result = sfo.mandate_optimizer(
    ...     ids=[f"A{i}" for i in range(n)],
    ...     alphas=rng.normal(0, 0.01, n),
    ...     factor_exposures=rng.normal(0, 1, (n, k)),
    ...     factor_covariance=np.diag([0.03, 0.01, 0.01]),
    ...     specific_variance=np.full(n, 0.09),
    ...     benchmark_weights=np.full(n, 1 / n),
    ... )
    >>> round(result.ex_ante_te, 4)
    0.05
    """
    start = time.perf_counter()

    alphas = np.asarray(alphas, dtype=float)
    B = np.asarray(factor_exposures, dtype=float)
    F = np.asarray(factor_covariance, dtype=float)
    delta = np.asarray(specific_variance, dtype=float)
    b = np.asarray(benchmark_weights, dtype=float)
    n_assets, n_factors = B.shape

    _validate_inputs(ids, alphas, B, F, delta, b, target_active_risk)

    beta = benchmark_betas(B, F, delta, b)

    if not np.any(alphas):
        h, status, shadow = np.zeros(n_assets), "optimal", np.full(n_factors, np.nan)
    else:
        h, status, shadow = _solve(
            alphas, B, F, delta, b, beta, target_active_risk, factor_bounds, neutral_factors or []
        )

    return _result(ids, h, status, time.perf_counter() - start, alphas, B, F, delta, b, beta, target_active_risk, shadow)


def _validate_inputs(ids, alphas, B, F, delta, b, target_active_risk) -> None:
    n_assets = len(ids)
    if alphas.shape != (n_assets,) or delta.shape != (n_assets,) or b.shape != (n_assets,):
        raise ValueError("alphas, specific_variance, and benchmark_weights must have shape (n_assets,).")
    if B.shape[0] != n_assets or F.shape != (B.shape[1], B.shape[1]):
        raise ValueError("factor_exposures must be (n_assets, n_factors) and factor_covariance (n_factors, n_factors).")
    for name, arr in (("alphas", alphas), ("factor_exposures", B), ("factor_covariance", F),
                      ("specific_variance", delta), ("benchmark_weights", b)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")
    if np.any(delta < 0):
        raise ValueError("specific_variance must be non-negative.")
    if np.any(b < 0):
        raise ValueError("benchmark_weights must be non-negative.")
    if not target_active_risk > 0:
        raise ValueError("target_active_risk must be positive.")


def _factor_root(F: np.ndarray) -> np.ndarray:
    """Return L with F = L L^T, clipping tiny negative eigenvalues to zero."""
    eigenvalues, eigenvectors = np.linalg.eigh((F + F.T) / 2)
    return eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))


def _solve(alphas, B, F, delta, b, beta, target_active_risk, factor_bounds, neutral_factors):
    n_assets, n_factors = B.shape

    h = cp.Variable(n_assets)
    y = cp.Variable(n_factors)
    L = _factor_root(F)

    # Alpha and beta are normalized for conditioning; scale invariance means
    # neither changes the solution.
    alpha_norm = np.linalg.norm(alphas)
    objective = cp.Maximize(h @ (alphas / alpha_norm))
    constraints = [
        y == B.T @ h,
        cp.sum(h) == 0,
        cp.SOC(
            cp.Constant(target_active_risk),
            cp.hstack([L.T @ y, cp.multiply(np.sqrt(delta), h)]),
        ),
        h >= -b,
        h @ (beta / np.linalg.norm(beta)) == 0,
    ]
    for k, cap in (factor_bounds or {}).items():
        constraints.append(cp.abs(y[k]) <= cap)

    kept = _independent([np.ones(n_assets), beta], [B[:, k] for k in neutral_factors])
    neutral = {k: y[k] == 0 for k, keep in zip(neutral_factors, kept) if keep}
    constraints.extend(neutral.values())

    shadow = np.full(n_factors, np.nan)
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver="CLARABEL")
    except cp.SolverError:
        return np.zeros(n_assets), "solver_error", shadow

    if problem.status not in _OK_STATUSES or h.value is None:
        return np.zeros(n_assets), problem.status, shadow
    for k, constraint in neutral.items():
        shadow[k] = float(constraint.dual_value) * alpha_norm
    return np.asarray(h.value, dtype=float), problem.status, shadow


def _independent(base: list[np.ndarray], candidates: list[np.ndarray], tol: float = 1e-8) -> list[bool]:
    """
    For each candidate constraint row, whether it adds a new direction to
    the span of ``base`` and the candidates kept before it (Gram-Schmidt).
    Redundant equality rows make the solver's KKT system singular.
    """
    basis: list[np.ndarray] = []

    def add(v: np.ndarray) -> bool:
        r = v.astype(float)
        for _ in range(2):  # re-orthogonalize for stability
            for q in basis:
                r = r - (q @ r) * q
        if np.linalg.norm(r) <= tol * max(np.linalg.norm(v), 1.0):
            return False
        basis.append(r / np.linalg.norm(r))
        return True

    for v in base:
        add(v)
    return [add(v) for v in candidates]


def _result(ids, h, status, solve_time, alphas, B, F, delta, b, beta, target_active_risk, shadow) -> MandateResult:
    te = float(np.sqrt(max(active_variance(h, B, F, delta), 0.0)))
    at_bound = (b > 0) & (b + h <= 1e-8)
    return MandateResult(
        ids=list(ids),
        active_weights=h,
        status=status,
        solve_time=solve_time,
        ex_ante_te=te,
        te_shortfall=max(target_active_risk - te, 0.0),
        active_beta=float(beta @ h),
        expected_active_return=float(alphas @ h),
        n_at_long_only_bound=int(at_bound.sum()),
        active_factor_exposures=B.T @ h,
        factor_shadow_prices=shadow,
    )
