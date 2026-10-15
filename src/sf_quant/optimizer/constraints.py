"""Constraints for the active-space optimizer.

Every constraint is stated in terms of the active weight vector ``h``, where the
total portfolio is ``w = b + h`` for benchmark weights ``b``. Total-space
constraints are not expressible here and are not needed: because ``sum(h) == 0``
and ``betas @ h == 0`` are enforced by default, full investment
(``sum(w) == sum(b) == 1``) and unit beta (``betas @ w == betas @ b == 1``)
follow automatically.
"""

from typing import Protocol

import cvxpy as cp
import numpy as np


class Constraint(Protocol):
    """
    Protocol for active-space portfolio constraints.

    Implementations receive the active weight variable plus the benchmark
    weights and predicted betas for the same date, in the same row order, and
    return a single CVXPY constraint.
    """

    def __call__(
        self,
        active_weights: cp.Variable,
        *,
        benchmark_weights: np.ndarray,
        betas: np.ndarray | None = None,
    ) -> cp.Constraint:
        """
        Apply the constraint to the active weights.

        Parameters
        ----------
        active_weights : cp.Variable
            Active weight variable ``h`` of shape (n_assets,).
        benchmark_weights : np.ndarray
            Benchmark weights ``b`` of shape (n_assets,).
        betas : np.ndarray, optional
            Predicted betas of shape (n_assets,). Required by ``ZeroBeta``.

        Returns
        -------
        cp.Constraint
            A CVXPY constraint object.
        """
        ...


class ZeroInvestment(Constraint):
    """
    Enforces dollar neutrality of the active overlay: ``sum(h) == 0``.

    Combined with a fully invested benchmark this also makes the total
    portfolio fully invested, since ``sum(w) = sum(b) + 0 = 1``.

    Examples
    --------
    >>> import cvxpy as cp, numpy as np
    >>> from sf_quant.optimizer import ZeroInvestment
    >>> h = cp.Variable(3)
    >>> constraint = ZeroInvestment()(h, benchmark_weights=np.array([0.2, 0.3, 0.5]))
    """

    def __call__(
        self,
        active_weights: cp.Variable,
        *,
        benchmark_weights: np.ndarray,
        betas: np.ndarray | None = None,
    ) -> cp.Constraint:
        """Apply the zero-investment constraint."""
        return cp.sum(active_weights) == 0


class ZeroBeta(Constraint):
    """
    Enforces zero active market beta: ``betas @ h == 0``.

    Because the benchmark has unit beta by construction, this also pins the
    total portfolio at unit beta.

    Raises
    ------
    ValueError
        If ``betas`` is not provided.

    Examples
    --------
    >>> import cvxpy as cp, numpy as np
    >>> from sf_quant.optimizer import ZeroBeta
    >>> h = cp.Variable(3)
    >>> constraint = ZeroBeta()(
    ...     h,
    ...     benchmark_weights=np.array([0.2, 0.3, 0.5]),
    ...     betas=np.array([0.5, 1.2, 0.8]),
    ... )
    """

    def __call__(
        self,
        active_weights: cp.Variable,
        *,
        benchmark_weights: np.ndarray,
        betas: np.ndarray | None = None,
    ) -> cp.Constraint:
        """Apply the zero-active-beta constraint."""
        if betas is None:
            raise ValueError("ZeroBeta requires betas")
        return betas @ active_weights == 0


class LongOnly(Constraint):
    """
    Forbids shorting beyond the benchmark: ``h >= -b``, i.e. ``w >= 0``.

    This is the active-space statement of long-only. It permits underweighting
    a name all the way to zero but never past it.

    Examples
    --------
    >>> import cvxpy as cp, numpy as np
    >>> from sf_quant.optimizer import LongOnly
    >>> h = cp.Variable(3)
    >>> constraint = LongOnly()(h, benchmark_weights=np.array([0.2, 0.3, 0.5]))
    """

    def __call__(
        self,
        active_weights: cp.Variable,
        *,
        benchmark_weights: np.ndarray,
        betas: np.ndarray | None = None,
    ) -> cp.Constraint:
        """Apply the benchmark-relative long-only bound."""
        return active_weights >= -benchmark_weights


#: The Silver Fund active mandate, applied unless a caller passes its own list.
DEFAULT_CONSTRAINTS: tuple[Constraint, ...] = (
    ZeroInvestment(),
    ZeroBeta(),
    LongOnly(),
)
