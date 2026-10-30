"""Data loader tests. Table.scan is stubbed, so these need no data lake."""

import datetime as dt

import polars as pl

import sf_quant.data as sfd
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


def test_covariance_matrix_is_the_factor_model(monkeypatch):
    import numpy as np

    import sf_quant.data.covariance_matrix as cov_module
    from sf_quant.data._factors import factors

    date = dt.date(2024, 3, 5)
    barrids = ["USA0001", "USA0002", "USA0003"]
    rng = np.random.default_rng(0)
    B = rng.normal(size=(len(barrids), len(factors)))
    root = rng.normal(size=(len(factors), len(factors)))
    F = root @ root.T  # percent^2, as stored
    specific_risk = np.array([20.0, 35.0, 50.0])  # percent, as stored

    exposures = pl.DataFrame({"date": [date] * 3, "barrid": barrids}).with_columns(
        pl.Series(f, B[:, j]) for j, f in enumerate(factors)
    )
    upper = np.where(np.triu(np.ones_like(F)) > 0, F, np.nan)  # stored upper-triangular
    covariances = pl.DataFrame({"date": [date] * len(factors), "factor_1": factors}).with_columns(
        pl.Series(f, upper[:, j]) for j, f in enumerate(factors)
    )
    assets = pl.DataFrame({"date": [date] * 3, "barrid": barrids, "specific_risk": specific_risk})

    monkeypatch.setattr(cov_module, "load_exposures_by_date", lambda d: exposures)
    monkeypatch.setattr(cov_module, "load_covariances_by_date", lambda d: covariances)
    monkeypatch.setattr(cov_module, "load_assets_by_date", lambda d, in_universe, columns: assets)

    out = cov_module.construct_covariance_matrix(date, barrids)
    expected = (B @ F @ B.T + np.diag(specific_risk**2)) / 100**2
    assert out.columns == ["barrid", *barrids]
    np.testing.assert_allclose(out.drop("barrid").to_numpy(), expected, rtol=1e-12)
