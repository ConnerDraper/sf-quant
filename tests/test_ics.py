"""generate_alpha_ics must pair an alpha dated t with returns after t."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from sf_quant.performance.ics import generate_alpha_ics
from sf_quant.testing import business_days


def panel(n_days=60, n_assets=50, seed=0):
    rng = np.random.default_rng(seed)
    days = business_days(dt.date(2024, 1, 2), n_days)
    rets = pl.DataFrame(
        {
            "date": [d for d in days for _ in range(n_assets)],
            "barrid": [f"A{i}" for i in range(n_assets)] * n_days,
            "return": rng.normal(0, 0.02, n_days * n_assets),
        }
    )
    return rets


def test_same_day_return_is_not_predictive():
    rets = panel()
    alphas = rets.rename({"return": "alpha"})
    ics = generate_alpha_ics(alphas, rets, method="pearson", window=1)
    assert abs(ics["ic"].mean()) < 0.05


def test_next_day_return_is_perfectly_predictive():
    rets = panel()
    alphas = rets.sort("barrid", "date").with_columns(
        pl.col("return").shift(-1).over("barrid").alias("alpha")
    ).drop("return").drop_nulls()
    ics = generate_alpha_ics(alphas, rets, method="pearson", window=1)
    assert ics["ic"].mean() == pytest.approx(1.0)
