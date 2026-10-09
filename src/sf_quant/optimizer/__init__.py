"""
The ``optimizer`` module provides functionality for portfolio optimization
and risk management. It exposes a clean public API for constructing optimized
portfolios, computing weights, and evaluating constraints, while keeping
internal mathematical and solver details hidden.

:func:`mandate_optimizer` solves the Silver Fund mandate problem (manual
Chapter 3). The gamma-based ``mve_optimizer`` and ``dynamic_mve_optimizer``
are deprecated.
"""

from .optimizers import mve_optimizer, dynamic_mve_optimizer
from .constraints import LongOnly, FullInvestment, ZeroInvestment, NoBuyingOnMargin, UnitBeta, ZeroBeta
from .mandate import (
    MandateResult,
    active_variance,
    benchmark_betas,
    covariance_times,
    equality_direction,
    mandate_optimizer,
)

__all__ = [
    "mandate_optimizer",
    "MandateResult",
    "benchmark_betas",
    "equality_direction",
    "active_variance",
    "covariance_times",
    "mve_optimizer",
    "dynamic_mve_optimizer",
    "LongOnly",
    "FullInvestment",
    "ZeroInvestment",
    "NoBuyingOnMargin",
    "UnitBeta",
    "ZeroBeta",
]
