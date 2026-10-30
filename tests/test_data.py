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
