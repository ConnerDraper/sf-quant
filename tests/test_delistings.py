"""CRSP link and delisting resolution. The CRSP and Barra ID tables are stubbed, so these need no data lake."""

import datetime as dt

import polars as pl
import pytest

import sf_quant.data as sfd
import sf_quant.data.crsp_link as link_module
import sf_quant.data.delistings as delistings_module

DATES = [
    dt.date(2023, 12, 27), dt.date(2023, 12, 28), dt.date(2023, 12, 29),
    dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4),
]
FAR = dt.date(2075, 12, 31)

# Historical CUSIPs. USA0001 changes CUSIP on 2023-12-29; LOCALID rows are noise.
BARRA_IDS = pl.DataFrame(
    [
        ("USA0001", "CUSIP", "111111119", dt.date(2000, 1, 1), dt.date(2023, 12, 28)),
        ("USA0001", "CUSIP", "222222229", dt.date(2023, 12, 29), FAR),
        ("USA0002", "CUSIP", "333333339", dt.date(2000, 1, 1), FAR),
        ("USA0003", "CUSIP", "444444449", dt.date(2000, 1, 1), FAR),
        ("USA0004", "CUSIP", "555555559", dt.date(2000, 1, 1), FAR),
        ("USA0005", "CUSIP", "666666669", dt.date(2000, 1, 1), FAR),
        ("USA0006", "CUSIP", "777777779", dt.date(2000, 1, 1), FAR),
        ("USA0001", "LOCALID", "111111119", dt.date(2000, 1, 1), FAR),
    ],
    schema=["barrid", "asset_id_type", "asset_id", "start_date", "end_date"],
    orient="row",
)
SECURITY_INFO = pl.DataFrame(
    [
        (10001, "11111111", dt.date(1990, 1, 1), dt.date(2023, 12, 28)),
        (10001, "22222222", dt.date(2023, 12, 29), FAR),
        (10002, "33333333", dt.date(1990, 1, 1), FAR),
        (10003, "44444444", dt.date(1990, 1, 1), FAR),
        (10004, "55555555", dt.date(1990, 1, 1), FAR),
        (10005, "66666666", dt.date(1990, 1, 1), FAR),
        (10006, "77777777", dt.date(1990, 1, 1), FAR),
    ],
    schema=["permno", "cusip", "secinfostartdt", "secinfoenddt"],
    orient="row",
)
# One delisting per resolution branch: permno, last trading day,
# delisting-return day, delret, action, exchange.
DELISTINGS = pl.DataFrame(
    [
        (10001, DATES[4], DATES[5], None, "GLI", "N"),  # liquidation, missing -> 0
        (10002, DATES[3], DATES[4], 0.25, "MER", "N"),  # merger with a return -> 0.25
        (10003, DATES[3], DATES[4], None, "GDR", "N"),  # performance, NYSE, missing -> -0.30
        (10004, DATES[3], DATES[4], None, "GDR", "Q"),  # performance, Nasdaq, missing -> -0.55
        (10005, DATES[3], DATES[4], None, "XYZ", "N"),  # unmapped, missing -> 0 with a warning
        (10006, dt.date(2010, 6, 30), dt.date(2010, 7, 1), -0.8, "GDR", "Q"),  # outside the window
    ],
    schema={
        "permno": pl.Int64, "delistingdt": pl.Date, "deldlydt": pl.Date, "delret": pl.Float64,
        "delactiontype": pl.String, "primaryexch": pl.String,
    },
    orient="row",
).with_columns(
    delretmisstype=pl.when(pl.col("delret").is_null()).then(pl.lit("NA")).otherwise(None),
    delreasontype=pl.lit("R"),
)


class StubTable:
    def __init__(self, frame):
        self.frame = frame

    def scan(self, year=None):
        return self.frame.lazy()

    def _file_path(self, year=None):
        return "stub.parquet"


class MissingTable(StubTable):
    def scan(self, year=None):
        raise FileNotFoundError("stub.parquet")


@pytest.fixture(autouse=True)
def stub_tables(monkeypatch):
    monkeypatch.setattr(link_module, "barra_ids_table", StubTable(BARRA_IDS))
    monkeypatch.setattr(link_module, "crsp_security_info_table", StubTable(SECURITY_INFO))
    monkeypatch.setattr(delistings_module, "crsp_delisting_table", StubTable(DELISTINGS))


def test_link_follows_a_cusip_change():
    link = sfd.load_barrid_permno_link()
    usa1 = link.filter(pl.col("barrid") == "USA0001").sort("start_date")
    assert usa1["permno"].to_list() == [10001, 10001]
    assert usa1["end_date"][0] < usa1["start_date"][1]
    assert set(link["barrid"]) == {f"USA000{i}" for i in range(1, 7)}  # LOCALID rows ignored


def test_each_resolution_branch():
    with pytest.warns(UserWarning, match="unmapped action type"):
        out = sfd.load_delistings(DATES[0], DATES[-1])
    got = {
        row["barrid"]: (row["category"], row["delret_used"], row["delret_source"])
        for row in out.iter_rows(named=True)
    }
    assert got == {
        "USA0001": ("liquidation", 0.0, "zero"),
        "USA0002": ("merger", 0.25, "crsp"),
        "USA0003": ("performance", -0.30, "shumway"),
        "USA0004": ("performance", -0.55, "shumway"),
        "USA0005": ("other", 0.0, "zero"),
    }  # USA0006 delisted in 2010, outside the window


def test_the_link_in_force_on_the_last_trading_day_is_used():
    with pytest.warns(UserWarning):
        out = sfd.load_delistings(DATES[0], DATES[-1])
    assert out.filter(pl.col("barrid") == "USA0001")["permno"].item() == 10001


def test_type_counts_cover_every_delisting():
    counts = sfd.delisting_type_counts()
    assert counts["count"].sum() == 6
    assert counts["missing_delret"].sum() == 4


def test_missing_tables_raise_a_clear_error(monkeypatch):
    monkeypatch.setattr(delistings_module, "crsp_delisting_table", MissingTable(DELISTINGS))
    with pytest.raises(sfd.CrspTablesMissingError, match="delistings=False"):
        sfd.load_delistings(DATES[0], DATES[-1])
