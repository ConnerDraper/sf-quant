"""The active-space optimizer: maximize alpha at a target active risk.

There is deliberately no gamma (risk-aversion) parameter anywhere in this
package. A portfolio is sized by ``target_active_risk``; the tracking-error
target is what implicitly sets the multiplier on risk each date, never the
other way round. As Manual §3.2 puts it, that multiplier "is the scalar
multiplier on the tracking-error target, not a risk-aversion parameter." If you
find yourself passing or tuning a gamma, you are doing it wrong.

Two solvers sit behind ``active_mve_optimizer``, following Manual Ch. 3:

- Linear equality constraints only (``ZeroInvestment``, ``ZeroBeta``): the
  closed form of §3.2–3.3. No numerical solver.
- Any inequality (``LongOnly``): a numerical solve, as in §3.4. That path uses
  CVXPY, because an inequality that binds has no closed form.
"""

from collections.abc import Sequence

import cvxpy as cp
import numpy as np
import polars as pl

from .constraints import DEFAULT_CONSTRAINTS, Constraint

# Manual §3.3.6: after solving, confirm the restrictions that define the
# problem. The closed form is exact to rounding; CLARABEL stops within its own
# tolerance of feasible, so it is held to a looser bound.
CLOSED_FORM_TOLERANCE = 1e-9
NUMERICAL_TOLERANCE = 1e-6
RISK_TOLERANCE = 1e-3  # relative, on active risk vs. its target


def active_mve_optimizer(
    ids: list[str],
    alphas: np.ndarray,
    factor_exposures: np.ndarray,
    factor_covariance: np.ndarray,
    specific_risk: np.ndarray,
    benchmark_weights: np.ndarray,
    target_active_risk: float = 0.05,
    betas: np.ndarray | None = None,
    constraints: Sequence[Constraint] = DEFAULT_CONSTRAINTS,
) -> pl.DataFrame:
    """
    Maximize active return subject to a tracking-error budget.

    Solves for the active overlay ``h`` directly:

    .. math::

        \\max_h \\alpha^T h
        \\quad \\text{s.t.} \\quad
        h^T \\Sigma h \\le \\sigma_{target}^2

    where :math:`\\Sigma = B F B^T + \\text{diag}(D)`, plus whatever
    ``constraints`` are supplied. The total portfolio is ``w = b + h``.

    The objective is linear and the risk budget is a convex ball, so the
    tracking-error constraint binds at the optimum and the solution sits near a
    vertex of the feasible set — expect most names to land exactly on the
    ``LongOnly`` bound ``h = -b``.

    When every constraint is a linear equality (each has an ``equality_row``),
    the solution is the closed form of Manual §3.2–3.3:

    .. math::

        d = V^{-1}\\left[I - C^T (C V^{-1} C^T)^{-1} C V^{-1}\\right] \\alpha,
        \\qquad
        h^* = \\frac{\\bar\\sigma_a}{\\sqrt{d^T V d}}\\, d

    Otherwise (``LongOnly``) it is solved numerically with CVXPY, as in §3.4.
    There is no gamma: see the module docstring.

    Parameters
    ----------
    ids : list[str]
        Asset identifiers (barrids), in the same row order as every array below.
    alphas : np.ndarray
        Expected active returns, shape (n_assets,).
    factor_exposures : np.ndarray
        B matrix of shape (n_assets, n_factors).
    factor_covariance : np.ndarray
        F matrix of shape (n_factors, n_factors), in decimal space.
    specific_risk : np.ndarray
        D vector of shape (n_assets,), idiosyncratic *variance* in decimal space.
    benchmark_weights : np.ndarray
        Benchmark weights ``b`` of shape (n_assets,). Names absent from the
        benchmark must be 0, not null.
    target_active_risk : float, default 0.05
        Annualized tracking-error budget, in decimal (0.05 is 5%).
    betas : np.ndarray, optional
        Predicted betas of shape (n_assets,). Required by ``ZeroBeta``.
    constraints : Sequence[Constraint], default DEFAULT_CONSTRAINTS
        Active-space constraints. Defaults to the fund mandate of
        ``ZeroInvestment``, ``ZeroBeta`` and ``LongOnly``.

    Returns
    -------
    pl.DataFrame
        Columns:

        - ``barrid`` : str, asset identifier.
        - ``active_weight`` : float, the optimized overlay ``h``.
        - ``active_risk`` : float, realized annualized active risk.

    Raises
    ------
    ValueError
        If array lengths disagree, if ``alphas`` or ``benchmark_weights``
        contain non-finite values, if the solver does not converge, or if the
        solution fails the Manual §3.3.6 checks.

    Examples
    --------
    >>> import numpy as np
    >>> import sf_quant.optimizer as sfo
    >>> weights = sfo.active_mve_optimizer(
    ...     ids=['USA06Z1', 'USA0771'],
    ...     alphas=np.array([0.01, -0.01]),
    ...     factor_exposures=np.array([[0.8, 0.5], [1.2, 0.3]]),
    ...     factor_covariance=np.array([[0.04, 0.01], [0.01, 0.02]]),
    ...     specific_risk=np.array([0.02, 0.03]),
    ...     benchmark_weights=np.array([0.6, 0.4]),
    ...     betas=np.array([0.9, 1.1]),
    ...     target_active_risk=0.05,
    ... )
    """
    n_assets = len(ids)
    if not all(
        len(arr) == n_assets for arr in (alphas, specific_risk, benchmark_weights)
    ):
        raise ValueError(
            f"length mismatch: {n_assets} ids, {len(alphas)} alphas, "
            f"{len(specific_risk)} specific_risk, {len(benchmark_weights)} benchmark_weights"
        )
    if factor_exposures.shape[0] != n_assets:
        raise ValueError(
            f"factor_exposures has {factor_exposures.shape[0]} rows, expected {n_assets}"
        )
    if betas is not None and len(betas) != n_assets:
        raise ValueError(f"betas has length {len(betas)}, expected {n_assets}")

    # A null benchmark_weight left-joined onto the universe arrives here as NaN and
    # would silently poison the LongOnly bound, so reject it rather than solve.
    for name, arr in (("alphas", alphas), ("benchmark_weights", benchmark_weights)):
        if not np.all(np.isfinite(arr)):
            bad = [ids[i] for i in np.flatnonzero(~np.isfinite(arr))]
            raise ValueError(
                f"{name} has {len(bad)} non-finite values (barrids: {bad[:10]}); "
                "fill nulls before optimizing"
            )

    if target_active_risk <= 0:
        raise ValueError(f"target_active_risk must be positive, got {target_active_risk}")

    alphas = np.asarray(alphas, dtype=float)
    if all(hasattr(c, "equality_row") for c in constraints):
        rows = [
            c.equality_row(benchmark_weights=benchmark_weights, betas=betas)
            for c in constraints
        ]
        h = _closed_form(alphas, factor_exposures, factor_covariance, specific_risk, rows, target_active_risk)
        tolerance = CLOSED_FORM_TOLERANCE
    else:
        h = _numerical(
            alphas, factor_exposures, factor_covariance, specific_risk,
            benchmark_weights, betas, constraints, target_active_risk,
        )
        tolerance = NUMERICAL_TOLERANCE

    active_risk = _active_risk(h, factor_exposures, factor_covariance, specific_risk)
    _check(h, active_risk, constraints, benchmark_weights, betas, target_active_risk, tolerance)

    return pl.DataFrame(
        {
            "barrid": ids,
            "active_weight": h,
            "active_risk": active_risk,
        }
    )


def sigma_inverse(
    v: np.ndarray, factor_exposures: np.ndarray, factor_covariance: np.ndarray, specific_risk: np.ndarray
) -> np.ndarray:
    """
    :math:`V^{-1} v` for :math:`V = B F B^T + \\text{diag}(D)`, without forming V.

    Uses the Woodbury identity in the form that never inverts F (the loader
    fills missing covariances with 0, so F can be singular):

    .. math::

        V^{-1} v = D^{-1} v - D^{-1} B (I + F B^T D^{-1} B)^{-1} F B^T D^{-1} v

    ``v`` may be a vector or a matrix of column vectors. This is the linear-solve
    form Manual §3.3.4 recommends over explicit inverses at scale.
    """
    B, F, D = factor_exposures, factor_covariance, specific_risk
    d_inv_v = v / D[:, None] if v.ndim == 2 else v / D
    d_inv_b = B / D[:, None]
    core = np.eye(F.shape[0]) + F @ (B.T @ d_inv_b)
    return d_inv_v - d_inv_b @ np.linalg.solve(core, F @ (B.T @ d_inv_v))


def _closed_form(alphas, factor_exposures, factor_covariance, specific_risk, rows, target_active_risk):
    """Manual §3.2–3.3: the constrained direction, scaled to the tracking-error target."""
    if np.any(specific_risk <= 0):
        raise ValueError("specific_risk must be positive for the closed form")

    g = sigma_inverse(alphas, factor_exposures, factor_covariance, specific_risk)  # unconstrained direction V⁻¹α
    if rows:
        C = np.vstack(rows)
        W = sigma_inverse(C.T, factor_exposures, factor_covariance, specific_risk)  # V⁻¹Cᵀ
        eta = np.linalg.solve(C @ W, C @ g)  # (CV⁻¹Cᵀ)⁻¹ C g
        d = g - W @ eta
    else:
        d = g

    d_variance = _active_risk(d, factor_exposures, factor_covariance, specific_risk) ** 2  # dᵀVd
    if not d_variance > 0:
        raise ValueError("alphas lie entirely in the span of the constraints; no active bet is possible")
    return d * (target_active_risk / np.sqrt(d_variance))


def _numerical(
    alphas, factor_exposures, factor_covariance, specific_risk,
    benchmark_weights, betas, constraints, target_active_risk,
):
    """Manual §3.4: inequality constraints can bind, so solve numerically."""
    h = cp.Variable(len(alphas))

    factor_loadings = factor_exposures.T @ h
    # Factor risk as an explicit sum of squares, not cp.quad_form: CVXPY's
    # quad_form rewrite left CLARABEL stuck (InsufficientProgress /
    # NumericalError) on dates that solve cleanly in this form. The root comes
    # from an eigendecomposition rather than Cholesky because the loader fills
    # missing covariances with 0, so F can be singular.
    eigenvalues, eigenvectors = np.linalg.eigh(factor_covariance)
    factor_root = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    active_variance = cp.sum_squares(factor_root.T @ factor_loadings) + cp.sum_squares(
        cp.multiply(np.sqrt(specific_risk), h)
    )

    cvxpy_constraints = [active_variance <= target_active_risk**2]
    cvxpy_constraints += [
        constraint(h, benchmark_weights=benchmark_weights, betas=betas)
        for constraint in constraints
    ]

    problem = cp.Problem(cp.Maximize(alphas @ h), cvxpy_constraints)
    # One thread per solve: backtest() runs one solve per worker, and the
    # solver's own auto-threading would oversubscribe the CPUs.
    problem.solve(solver="CLARABEL", max_threads=1)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise ValueError(f"Optimization failed. Status: {problem.status}")

    return h.value


def _active_risk(h, factor_exposures, factor_covariance, specific_risk) -> float:
    """√(hᵀVh), Manual eq. 2.4.11."""
    loadings = factor_exposures.T @ h
    variance = loadings @ factor_covariance @ loadings + np.sum(specific_risk * np.square(h))
    return float(np.sqrt(max(0.0, variance)))


def _check(h, active_risk, constraints, benchmark_weights, betas, target_active_risk, tolerance) -> None:
    """Manual §3.3.6 / §3.4.5: confirm every restriction that defines the problem."""
    failures = []
    for c in constraints:
        gap = c.violation(h, benchmark_weights=benchmark_weights, betas=betas)
        if gap > tolerance:
            failures.append(f"{type(c).__name__} violated by {gap:.3g}")
    if abs(active_risk - target_active_risk) > RISK_TOLERANCE * target_active_risk:
        failures.append(f"active risk {active_risk:.6g} vs. target {target_active_risk:.6g}")
    if failures:
        raise ValueError("solution fails the Manual §3.3.6 checks: " + "; ".join(failures))
