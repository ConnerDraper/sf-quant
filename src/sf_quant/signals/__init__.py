"""
The ``signals`` module turns stock-level signals into alphas for the
mandate optimizer.

- :class:`Signal` is the interface; :mod:`~sf_quant.signals.library` has
  defaults (idiosyncratic momentum, short-term reversal, industry momentum,
  betting against beta).
- Each signal declares the Barra factors it may keep (``allowed_factors``);
  its score is residualized against all others.
- :func:`build_alpha` makes single-signal alphas, :func:`build_composite`
  combines signals with estimated ICs, and :func:`factor_loss` shows what
  orthogonalizing against each factor gives up.
"""

from .alpha import AlphaPanel, FactorLoss, build_alpha, estimate_ic, factor_loss, residualize, standardize
from .base import (
    ALL_FACTORS,
    COUNTRY_FACTOR,
    INDUSTRY_FACTORS,
    NO_FACTORS,
    STYLE_FACTORS,
    Constraints,
    Signal,
    caps,
)
from .composite import build_composite, combination_weights
from .library import BettingAgainstBeta, IdiosyncraticMomentum, IndustryMomentum, ShortTermReversal

__all__ = [
    "ALL_FACTORS",
    "COUNTRY_FACTOR",
    "INDUSTRY_FACTORS",
    "NO_FACTORS",
    "STYLE_FACTORS",
    "AlphaPanel",
    "BettingAgainstBeta",
    "Constraints",
    "FactorLoss",
    "IdiosyncraticMomentum",
    "IndustryMomentum",
    "ShortTermReversal",
    "Signal",
    "build_alpha",
    "build_composite",
    "combination_weights",
    "caps",
    "estimate_ic",
    "factor_loss",
    "residualize",
    "standardize",
]
