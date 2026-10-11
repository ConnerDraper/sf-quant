"""CachedMarketData serves nested specific-return requests from one computation.

Each signal reaches back by its own lookback and the IC target window sits
inside both, so a run asks for the same specific returns over several nested
ranges. Keyed on the range those all miss; covering them with one frame does
not.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

import sf_quant.signals as sg
from sf_quant.data.market import CachedMarketData, cached

from synthetic import synthetic_market

WIDE = (dt.date(2010, 1, 4), dt.date(2012, 6, 1))
INSIDE = (dt.date(2011, 3, 1), dt.date(2012, 1, 10))


class Counting:
    """Wraps a market, counting the calls that reach it."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = {}

    def __getattr__(self, name):
        if name == "_inner":
            raise AttributeError(name)
        attribute = getattr(self._inner, name)
        if not callable(attribute):
            return attribute

        def counted(*args, **kwargs):
            self.calls[name] = self.calls.get(name, 0) + 1
            return attribute(*args, **kwargs)

        return counted


@pytest.fixture(scope="module")
def market():
    return synthetic_market(n_assets=120, n_days=756)


@pytest.fixture(scope="module")
def ids(market):
    start, end = dt.date(1900, 1, 1), dt.date(2100, 1, 1)
    return sorted(market.benchmark_weights(start, end)["barrid"].unique().to_list())


def test_a_range_inside_a_cached_one_is_not_recomputed(market, ids):
    counting = Counting(market)
    data = CachedMarketData(counting)
    data.specific_returns(*WIDE, ids)
    data.specific_returns(*INSIDE, ids)
    assert counting.calls["specific_returns"] == 1


def test_slice_of_a_wider_computation_matches_computing_it_directly(market, ids):
    """
    The one risky part: u_d = r_d - B_{d-1} f_d is row-local apart from the
    one-day exposure shift, which the loader already pads for, so widening the
    range must not change a row it already returned.
    """
    data = CachedMarketData(market)
    data.specific_returns(*WIDE, ids)
    from_cache = data.specific_returns(*INSIDE, ids).sort("barrid", "date")
    direct = market.specific_returns(*INSIDE, ids).sort("barrid", "date")
    assert from_cache.equals(direct)


def test_widening_is_order_independent(market, ids):
    """Asking narrow first then wide still ends up with one covering frame."""
    counting = Counting(market)
    data = CachedMarketData(counting)
    data.specific_returns(*INSIDE, ids)
    data.specific_returns(*WIDE, ids)  # recomputes, widened
    data.specific_returns(*INSIDE, ids)  # now covered
    assert counting.calls["specific_returns"] == 2


def test_a_different_asset_set_is_cached_separately(market, ids):
    counting = Counting(market)
    data = CachedMarketData(counting)
    data.specific_returns(*WIDE, ids)
    data.specific_returns(*WIDE, ids[:10])
    assert counting.calls["specific_returns"] == 2


@pytest.mark.parametrize(
    "method, args",
    [
        ("benchmark_weights", WIDE),
        ("factor_returns", WIDE),
        ("prices", WIDE),
    ],
)
def test_other_protocol_methods_pass_through(market, method, args):
    data = CachedMarketData(market)
    assert getattr(data, method)(*args).equals(getattr(market, method)(*args))


def test_extra_attributes_pass_through(market):
    """A market may carry more than the protocol; signals use it (planted_scores)."""
    data = CachedMarketData(market)
    assert isinstance(data.planted_scores, pl.DataFrame)


def test_cached_is_idempotent(market):
    once = cached(market)
    assert cached(once) is once


def test_pickling_drops_the_cache(market, ids):
    """Workers must not receive a copy of every cached frame."""
    import pickle

    data = CachedMarketData(market)
    data.specific_returns(*WIDE, ids)
    assert data._specific
    assert not pickle.loads(pickle.dumps(data))._specific


def test_caching_does_not_change_the_alphas(market):
    """
    End to end: the composite built through the cache matches the one built
    without it. Not bit-exact — summing 76 factor columns over a wider frame
    reassociates the floating point — so compare within tolerance.
    """
    import sf_quant.signals.composite as composite

    start, end = dt.date(1900, 1, 1), dt.date(2100, 1, 1)
    days = market.benchmark_weights(start, end)["date"]
    window = (days.min(), days.max())
    signals = [sg.IdiosyncraticMomentum(), sg.ShortTermReversal()]

    real = composite.cached
    composite.cached = lambda data: data  # bypass the wrapper
    try:
        bare = sg.build_composite(signals, market, *window)
    finally:
        composite.cached = real
    through_cache = sg.build_composite(signals, market, *window)

    joined = bare.alphas.join(through_cache.alphas, on=["date", "barrid"], suffix="_cached")
    assert joined.height == bare.alphas.height
    assert joined.select((pl.col("alpha") - pl.col("alpha_cached")).abs().max()).item() < 1e-12
