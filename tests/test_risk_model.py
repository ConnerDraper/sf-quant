"""Tests for load_risk_model without the BYU data store."""

import datetime as dt

import numpy as np
import polars as pl

import sf_quant.data.covariance_matrix as cm
from sf_quant.data._factors import factors


def test_load_risk_model_reports_missing_ids(monkeypatch):
    date_ = dt.date(2024, 1, 3)
    k = len(factors)
    exposures = pl.DataFrame({"date": [date_] * 3, "barrid": ["A", "B", "C"]} | {f: [1.0, 0.5, None] for f in factors})
    assets = pl.DataFrame({"barrid": ["A", "B", "D"], "specific_risk": [30.0, None, 25.0]})
    factor_cov = pl.DataFrame({"factor_1": factors} | {f: np.eye(k)[:, i] * 4.0 for i, f in enumerate(factors)})

    monkeypatch.setattr(cm, "load_exposures_by_date", lambda d: exposures)
    monkeypatch.setattr(cm, "load_assets_by_date", lambda d, in_universe, columns: assets)
    monkeypatch.setattr(cm, "_construct_factor_covariance_matrix", lambda d: factor_cov)

    rm = cm.load_risk_model(date_, ["A", "B", "C", "D", "E"])

    # B has no specific risk; D has no exposure row; E has neither.
    # C has an exposure row (nulls are non-membership in a factor) but no specific risk.
    assert rm.ids == ["A"]
    assert rm.missing_ids == ["B", "C", "D", "E"]
    assert rm.factor_exposures.shape == (1, k)
    np.testing.assert_allclose(rm.specific_variance, [0.09])
    np.testing.assert_allclose(np.diag(rm.factor_covariance), 4.0 / 100**2)
