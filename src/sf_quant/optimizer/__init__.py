"""
The ``optimizer`` module solves the Silver Fund mandate problem (manual
Chapter 3) with :func:`mandate_optimizer`, plus the risk-model helpers it
is built on.
"""

from .mandate import (
    MandateResult,
    active_variance,
    benchmark_betas,
    covariance_times,
    equality_direction,
    mandate_optimizer,
)

__all__ = [
    "MandateResult",
    "active_variance",
    "benchmark_betas",
    "covariance_times",
    "equality_direction",
    "mandate_optimizer",
]
