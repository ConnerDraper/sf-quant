"""The Signal interface, Barra factor sets, and signal-level constraints."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import polars as pl

from sf_quant.data._factors import factors, sector_factors, style_factors
from sf_quant.data.market import MarketData

# Barra USSLOWL factor sets. Combine them with set algebra,
# e.g. STYLE_FACTORS | {"USSLOWL_BETA"} or ALL_FACTORS - INDUSTRY_FACTORS.
ALL_FACTORS = frozenset(factors)
STYLE_FACTORS = frozenset(style_factors)
INDUSTRY_FACTORS = frozenset(sector_factors)
COUNTRY_FACTOR = frozenset({"USSLOWL_COUNTRY"})
NO_FACTORS: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Constraints:
    """
    Portfolio constraints a signal brings into the composite.

    Under ``profile: mandate`` these can only add to the fund's constraints:
    ``zero_active_beta=False`` is a relaxation honored only under
    ``profile: experimental``.
    """

    factor_caps: Mapping[str, float] = field(default_factory=dict)  # max |active exposure| per factor
    zero_active_beta: bool = True

    def combine(self, other: Constraints) -> Constraints:
        """Constraints for a composite: the tighter cap per factor; zero beta unless a signal relaxes it."""
        caps_ = dict(self.factor_caps)
        for name, cap in other.factor_caps.items():
            caps_[name] = min(cap, caps_.get(name, cap))
        return Constraints(caps_, self.zero_active_beta and other.zero_active_beta)


def caps(factor_set: Iterable[str], cap: float) -> dict[str, float]:
    """The same cap for every factor in a set, e.g. ``caps(INDUSTRY_FACTORS, 0.05)``."""
    return {name: cap for name in factor_set}


@dataclass(frozen=True)
class Signal(ABC):
    """
    A rule that scores stocks using only information available at each close.

    Subclasses are frozen dataclasses, so parameters are set per instance,
    e.g. ``IndustryMomentum(lookback=252)``.

    Attributes
    ----------
    allowed_factors : frozenset[str]
        Barra factors this signal may keep. Its score is residualized against
        every other factor, and the portfolio is held neutral to factors no
        signal in the composite is allowed to keep. Default: none.
    constraints : Constraints
        Extra portfolio constraints that come with the signal.
    """

    allowed_factors: frozenset[str] = NO_FACTORS
    constraints: Constraints = field(default_factory=Constraints)

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def compute(self, data: MarketData, universe: pl.DataFrame) -> pl.DataFrame:
        """
        Raw scores for each (``date``, ``barrid``) row of ``universe``.

        Returns ``date``, ``barrid``, ``score``, where a larger score means a
        higher expected return. The score for date t may use only data
        available by the close of t. Rows can be missing; they get no alpha.
        """
