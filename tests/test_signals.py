"""Signals, residualization, targets, the factor report, and the composite."""

from dataclasses import dataclass

import numpy as np
import polars as pl
import pytest

import sf_quant.signals as sg
from sf_quant.signals.alpha import contexts, residual_scores, target_returns, universe_frame
from sf_quant.testing import INDUSTRIES, PanelMarketData, synthetic_market


@dataclass(frozen=True)
class Planted(sg.Signal):
    """Reads the synthetic market's planted scores."""

    def compute(self, data, universe):
        return universe.join(data.planted_scores, on=["date", "barrid"])


@dataclass(frozen=True)
class Noise(sg.Signal):
    seed: int = 0

    def compute(self, data, universe):
        return universe.with_columns(score=pl.Series(np.random.default_rng(self.seed).normal(size=universe.height)))


@pytest.fixture(scope="module")
def market():
    return synthetic_market(n_assets=150, n_days=1008, seed=2, planted_ic=0.10, industry_drift=0.5)


def span(md):
    return md.days[0], md.days[-1]


def truncate(md: PanelMarketData, end) -> PanelMarketData:
    """The same market, with every panel cut off after ``end``."""
    cut = lambda df: df.filter(pl.col("date") <= end)  # noqa: E731
    out = PanelMarketData(
        cut(md.returns), cut(md.benchmark), md.ids, md.B, md.F, md.delta, md.factor_names,
        factor_returns=cut(md.factors), specific_returns=cut(md.specific), prices=cut(md.price_panel),
    )
    out.planted_scores = md.planted_scores
    return out


def test_factor_sets_partition_the_barra_model():
    assert sg.STYLE_FACTORS | sg.INDUSTRY_FACTORS | sg.COUNTRY_FACTOR == sg.ALL_FACTORS
    assert not sg.STYLE_FACTORS & sg.INDUSTRY_FACTORS
    assert len(sg.ALL_FACTORS) == 76


def test_constraints_combine_to_the_tightest():
    a = sg.Constraints(factor_caps={"X": 0.10, "Y": 0.05})
    b = sg.Constraints(factor_caps={"X": 0.02})
    assert a.combine(b).factor_caps == {"X": 0.02, "Y": 0.05}


@pytest.mark.parametrize(
    "signal",
    [sg.IdiosyncraticMomentum(), sg.ShortTermReversal(), sg.IndustryMomentum(), sg.BettingAgainstBeta()],
    ids=lambda s: s.name,
)
def test_signals_use_no_future_data(market, signal):
    t = market.days[600]
    universe = pl.DataFrame({"date": t, "barrid": market.ids})
    full = signal.compute(market, universe).sort("barrid")
    cut = signal.compute(truncate(market, t), universe).sort("barrid")
    assert full.height > 0
    assert full.equals(cut)


def test_residual_scores_are_orthogonal_to_removed_factors(market):
    _, ctxs = contexts(market, *span(market), "monthly")
    z = residual_scores(sg.IdiosyncraticMomentum().compute(market, universe_frame(ctxs)), ctxs, sg.NO_FACTORS)
    ctx = ctxs[-1]
    day = z.filter(pl.col("date") == ctx.date)
    rows = [ctx.rm.ids.index(x) for x in day["barrid"]]
    w = ctx.sigma(sg.NO_FACTORS)[rows] / ctx.rm.specific_variance[rows]
    B = ctx.rm.factor_exposures[rows]
    np.testing.assert_allclose(B.T @ (w * day["z"].to_numpy()), 0, atol=1e-8)
    assert day["z"].std() == pytest.approx(1.0, rel=1e-2)


def test_industry_momentum_is_zero_without_its_industries(market):
    _, ctxs = contexts(market, *span(market), "monthly")
    scores = sg.IndustryMomentum().compute(market, universe_frame(ctxs))
    assert (residual_scores(scores, ctxs, sg.NO_FACTORS)["z"] == 0).all()
    assert residual_scores(scores, ctxs, sg.INDUSTRY_FACTORS)["z"].abs().max() > 0.5


def test_target_with_no_allowed_factors_is_barra_specific(market):
    days, ctxs = contexts(market, *span(market), "monthly")
    ctx = ctxs[3]
    np.testing.assert_allclose(ctx.sigma(sg.NO_FACTORS), np.sqrt(ctx.rm.specific_variance))
    target = target_returns(market, days, [ctx], 21, 1, sg.NO_FACTORS)
    window = days[ctx.index + 2 : ctx.index + 23]
    u = market.specific.filter(pl.col("date").is_between(window[0], window[-1]), pl.col("barrid") == "S0000")
    expected = u["specific_return"].sum() / np.sqrt(market.delta[0])
    assert target.filter(pl.col("barrid") == "S0000")["target"].item() == pytest.approx(expected)


def test_industry_momentum_ic_comes_from_its_industries(market):
    days, ctxs = contexts(market, *span(market), "monthly")
    z = residual_scores(sg.IndustryMomentum().compute(market, universe_frame(ctxs)), ctxs, sg.INDUSTRY_FACTORS)
    specific_only = sg.estimate_ic(z, target_returns(market, days, ctxs, 21, 1, sg.NO_FACTORS))["ic"].mean()
    with_industries = sg.estimate_ic(z, target_returns(market, days, ctxs, 21, 1, sg.INDUSTRY_FACTORS))["ic"].mean()
    assert abs(specific_only) < 0.03
    assert with_industries > 0.05


@pytest.fixture(scope="module")
def industry_report(market):
    return sg.factor_report(sg.IndustryMomentum(), market, *span(market))


def test_factor_report_shares_add_to_one(industry_report):
    table = industry_report.table
    for p in ("before", "after"):
        for share in ("return_share", "ex_ante_risk_share", "realized_risk_share"):
            assert table[f"{share}_{p}"].sum() == pytest.approx(1.0)


def test_factor_report_industry_momentum_keeps_only_its_industries(industry_report):
    table = industry_report.table
    industries = table.filter(pl.col("factor").is_in(INDUSTRIES))
    assert not industries["removed"].any()
    assert industries["return_share_before"].sum() > 0.6
    assert industries["return_share_after"].sum() > 0.6
    # Every other factor is held neutral after orthogonalizing, so it earns nothing.
    removed = table.filter(pl.col("removed"))
    assert removed["annual_return_after"].abs().max() < 1e-10
    assert removed["ex_ante_risk_share_after"].abs().max() < 1e-10


def test_factor_report_residual_signal_is_all_specific_after(market):
    start = market.days[260]
    report = sg.factor_report(sg.IdiosyncraticMomentum(), market, start, market.days[-1])
    table = report.table
    factors = table.filter(pl.col("factor") != "specific")
    assert factors["annual_return_after"].abs().max() < 1e-10
    specific = table.filter(pl.col("factor") == "specific")
    assert specific["ex_ante_risk_share_after"].item() == pytest.approx(1.0)
    assert specific["realized_risk_share_after"].item() == pytest.approx(1.0)


def test_factor_report_plot_writes_a_file(industry_report, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    path = tmp_path / "report.png"
    industry_report.plot(file_name=str(path))
    assert path.stat().st_size > 0


def test_combination_weights_math():
    ic = np.array([0.04, 0.03, 0.02])
    v, ic_c = sg.combination_weights(ic, np.eye(3))
    np.testing.assert_allclose(v, ic)  # uncorrelated: weights proportional to IC
    assert ic_c**2 == pytest.approx((ic**2).sum())

    K, rho = 4, 0.3
    R = np.full((K, K), rho) + (1 - rho) * np.eye(K)
    _, ic_c = sg.combination_weights(np.full(K, 0.05), R)
    assert ic_c**2 == pytest.approx(0.05**2 * K / (1 + (K - 1) * rho))  # IC^2 times effective breadth


def test_composite_weights_the_real_signal_over_noise(market):
    panel = sg.build_composite([Planted(), Noise()], market, *span(market), ic_shrink_obs=6)
    d = panel.diagnostics
    first_two = d["date"].unique().sort()[:2].to_list()
    assert (d.filter(pl.col("date").is_in(first_two))["ic"] == 0).all()  # no completed windows yet

    last = d.filter(pl.col("date") == d["date"].max())
    planted, noise = (last.filter(pl.col("signal") == s).row(0, named=True) for s in ("Planted", "Noise"))
    assert planted["ic"] > 0.03
    assert planted["weight"] > 3 * abs(noise["weight"])
    assert planted["marginal_ic"] > 0
    assert panel.alphas.height > 0 and panel.allowed_factors == sg.NO_FACTORS


def test_duplicate_signals_split_the_weight(market):
    panel = sg.build_composite([Planted(), Planted()], market, *span(market), ic_shrink_obs=6)
    last = panel.diagnostics.filter(pl.col("date") == panel.diagnostics["date"].max())
    w = last["weight"].to_numpy()
    assert w[0] == pytest.approx(w[1])
    assert set(last["signal"]) == {"Planted", "Planted_2"}


def test_composite_carries_the_union_of_allowances(market):
    capped = sg.IndustryMomentum(constraints=sg.Constraints(factor_caps={"USSLOWL_BANKS": 0.02}))
    panel = sg.build_composite([capped, sg.BettingAgainstBeta()], market, *span(market))
    assert panel.allowed_factors == sg.INDUSTRY_FACTORS | {"USSLOWL_BETA"}
    assert panel.constraints.factor_caps == {"USSLOWL_BANKS": 0.02}
