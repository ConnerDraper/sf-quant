from collections.abc import Sequence

import cvxpy as cp
import numpy as np
import polars as pl

from .constraints import DEFAULT_CONSTRAINTS, Constraint


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
        contain non-finite values, or if the solver does not converge.

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

    h = cp.Variable(n_assets)

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

    active_weights = h.value

    factor_loadings = factor_exposures.T @ active_weights
    realized_variance = (
        factor_loadings @ factor_covariance @ factor_loadings
        + np.sum(specific_risk * np.square(active_weights))
    )

    return pl.DataFrame(
        {
            "barrid": ids,
            "active_weight": active_weights,
            "active_risk": float(np.sqrt(max(0.0, realized_variance))),
        }
    )
