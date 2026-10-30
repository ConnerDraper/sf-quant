"""Data loader tests. Table.scan is stubbed, so these need no data lake."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

import sf_quant.data as sfd
import sf_quant.data.covariance_matrix as cov_module
from sf_quant.data._factors import factors
from sf_quant.data._tables import Table


def test_by_date_loaders_read_only_the_dates_year(monkeypatch):
    scanned = []

    def scan(self, year=None):
        scanned.append((self._name, year))
        return pl.LazyFrame(
            schema={
                "date": pl.Date,
                "barrid": pl.String,
                "in_universe": pl.Boolean,
                "signal_name": pl.String,
            }
        )

    monkeypatch.setattr(Table, "scan", scan)
    date = dt.date(2024, 3, 5)
    for in_universe in (True, False):
        sfd.load_assets_by_date(date, in_universe, ["barrid"])
    for load in (sfd.load_signals_by_date, sfd.load_scores_by_date, sfd.load_alphas_by_date):
        for names in (None, ["momentum"]):
            load(date, names)
    assert len(scanned) == 8
    assert all(year == date.year for _, year in scanned)


BARRIDS = ["USA0001", "USA0002", "USA0003"]
SPECIFIC_RISK = np.array([20.0, 35.0, 50.0])  # percent, as stored


def stub_risk_model(monkeypatch, date, exposure_barrids=BARRIDS, specific_barrids=BARRIDS):
    """Stub the three risk tables; returns B, F (percent^2) in BARRIDS order."""
    rng = np.random.default_rng(0)
    B = rng.normal(size=(len(BARRIDS), len(factors)))
    root = rng.normal(size=(len(factors), len(factors)))
    F = root @ root.T  # percent^2, as stored

    keep = [BARRIDS.index(b) for b in exposure_barrids]
    exposures = pl.DataFrame({"date": [date] * len(keep), "barrid": exposure_barrids}).with_columns(
        pl.Series(f, B[keep, j]) for j, f in enumerate(factors)
    )
    upper = np.where(np.triu(np.ones_like(F)) > 0, F, np.nan)  # stored upper-triangular
    covariances = pl.DataFrame({"date": [date] * len(factors), "factor_1": factors}).with_columns(
        pl.Series(f, upper[:, j]) for j, f in enumerate(factors)
    )
    keep = [BARRIDS.index(b) for b in specific_barrids]
    assets = pl.DataFrame(
        {"date": [date] * len(keep), "barrid": specific_barrids, "specific_risk": SPECIFIC_RISK[keep]}
    )

    monkeypatch.setattr(cov_module, "load_exposures_by_date", lambda d: exposures)
    monkeypatch.setattr(cov_module, "load_covariances_by_date", lambda d: covariances)
    monkeypatch.setattr(cov_module, "load_assets_by_date", lambda d, in_universe, columns: assets)
    return B, F


def test_covariance_matrix_is_the_factor_model(monkeypatch):
    date = dt.date(2024, 3, 5)
    B, F = stub_risk_model(monkeypatch, date)

    out = cov_module.construct_covariance_matrix(date, BARRIDS)
    expected = (B @ F @ B.T + np.diag(SPECIFIC_RISK**2)) / 100**2
    assert out.columns == ["barrid", *BARRIDS]
    np.testing.assert_allclose(out.drop("barrid").to_numpy(), expected, rtol=1e-12)


def test_components_follow_the_order_barrids_are_given_in(monkeypatch):
    date = dt.date(2024, 3, 5)
    B, F = stub_risk_model(monkeypatch, date)
    order = ["USA0003", "USA0001", "USA0002"]
    rows = [BARRIDS.index(b) for b in order]

    exposures, factor_covariance, specific = cov_module.construct_factor_model_components(date, order)
    np.testing.assert_allclose(exposures, B[rows])
    np.testing.assert_allclose(factor_covariance, F / 100**2)
    np.testing.assert_allclose(specific, SPECIFIC_RISK[rows] ** 2 / 100**2)


@pytest.mark.parametrize("missing_from", ["exposures", "specific_risk"])
def test_a_barrid_without_risk_data_raises(monkeypatch, missing_from):
    date = dt.date(2024, 3, 5)
    present = BARRIDS[:2]
    if missing_from == "exposures":
        stub_risk_model(monkeypatch, date, exposure_barrids=present)
    else:
        stub_risk_model(monkeypatch, date, specific_barrids=present)

    with pytest.raises(ValueError, match=f"{missing_from} missing for 1 of 3 barrids.*USA0003"):
        cov_module.construct_factor_model_components(date, BARRIDS)


def test_column_listings_restore_the_callers_row_setting(monkeypatch):
    import sf_quant.data.fama_french as ff_module

    schema = {"date": pl.Date, "barrid": pl.String}
    monkeypatch.setattr(Table, "scan", lambda self, year=None: pl.LazyFrame(schema=schema))
    monkeypatch.setattr(ff_module, "fama_french", lambda: pl.LazyFrame(schema=schema))

    with pl.Config(tbl_rows=25):
        for listing in (sfd.get_assets_columns, sfd.get_fama_french_columns):
            assert "barrid" in listing()
            assert pl.Config.state(if_set=True)["POLARS_FMT_MAX_ROWS"] == "25"
