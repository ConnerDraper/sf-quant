"""Transfer coefficient (Clarke, de Silva & Thorley 2002) against the synthetic lake."""

import numpy as np
import polars as pl
import pytest

import sf_quant.data as sfd
from sf_quant.performance import generate_transfer_coefficients
from conftest import BARRIDS, DATES


def frames(seed=0):
    rng = np.random.default_rng(seed)
    keys = {"date": [d for d in DATES for _ in BARRIDS], "barrid": BARRIDS * len(DATES)}
    weights = pl.DataFrame({**keys, "active_weight": rng.normal(size=len(keys["date"]))})
    alphas = pl.DataFrame({**keys, "alpha": rng.normal(size=len(keys["date"]))})
    return weights, alphas


def test_tc_is_the_risk_adjusted_correlation(lake):
    weights, alphas = frames()
    out = generate_transfer_coefficients(weights, alphas)

    risk = sfd.load_assets(DATES[0], DATES[-1], ["date", "barrid", "specific_risk"])
    for date_ in DATES:
        h = weights.filter(pl.col("date") == date_).sort("barrid")["active_weight"].to_numpy()
        a = alphas.filter(pl.col("date") == date_).sort("barrid")["alpha"].to_numpy()
        s = risk.filter(pl.col("date") == date_).sort("barrid")["specific_risk"].to_numpy() / 100
        expected = np.corrcoef(h * s, a / s)[0, 1]
        got = out.filter(pl.col("date") == date_)["transfer_coefficient"].item()
        assert got == pytest.approx(expected)


def test_unconstrained_diagonal_optimum_has_tc_of_one(lake):
    """With diagonal risk, h ∝ α/σ² is the unconstrained optimum, so TC = 1."""
    _, alphas = frames(1)
    risk = sfd.load_assets(DATES[0], DATES[-1], ["date", "barrid", "specific_risk"])
    weights = alphas.join(risk, on=["date", "barrid"]).select(
        "date", "barrid", (pl.col("alpha") / (pl.col("specific_risk") / 100) ** 2).alias("active_weight")
    )
    out = generate_transfer_coefficients(weights, alphas)
    assert out["transfer_coefficient"].to_numpy() == pytest.approx(1.0)
