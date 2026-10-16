"""A synthetic data lake: every table sf_quant.data reads, year-partitioned like
the real store, written to a temp directory and pointed at with sfd.env().

Small and deterministic, with the cases the loaders must handle written in:
names that leave the universe, null exposures, an upper-triangular factor
covariance in shuffled row order, and CRSP rows that each cleaning rule drops.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import sf_quant.data._config as data_config
from sf_quant.data._factors import factors

DATABASE = "test"
DATES = [
    dt.date(2023, 12, 27), dt.date(2023, 12, 28), dt.date(2023, 12, 29),
    dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4),
]
BARRIDS = ["USA0001", "USA0002", "USA0003", "USA0004"]
SIGNAL_NAMES = ["momentum", "reversal"]


def write_years(root: Path, name: str, frame: pl.DataFrame, sep: str = "_") -> None:
    directory = root / "groups" / "grp_quant" / "database" / DATABASE / name
    directory.mkdir(parents=True, exist_ok=True)
    by_year = frame.with_columns(pl.col("date").dt.year().alias("year"))
    for (year,), part in by_year.partition_by("year", as_dict=True, include_key=False).items():
        part.write_parquet(directory / f"{name}{sep}{year}.parquet")


def assets() -> pl.DataFrame:
    rows = []
    for t, date in enumerate(DATES):
        for i, barrid in enumerate(BARRIDS):
            rows.append(
                {
                    "date": date,
                    "barrid": barrid,
                    # USA0004 is in the universe only on the first and last dates.
                    "in_universe": i < 3 or t in (0, len(DATES) - 1),
                    "market_cap": 100.0 * (i + 1) + t,
                    "return": 0.5 * (i - 1.5) + 0.1 * t,
                    "specific_risk": 20.0 + 5 * i + t,
                    "predicted_beta": 0.8 + 0.1 * i,
                    "price": 10.0 + i + t,
                    "ticker": f"T{i}",
                }
            )
    return pl.DataFrame(rows)


def factor_covariance(date: dt.date) -> np.ndarray:
    rng = np.random.default_rng(date.toordinal())
    root = rng.normal(size=(len(factors), len(factors)))
    return root @ root.T / len(factors) + np.eye(len(factors))


def exposures() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    frame = pl.DataFrame(
        {
            "date": [d for d in DATES for _ in BARRIDS],
            "barrid": BARRIDS * len(DATES),
            **{f: rng.normal(size=len(DATES) * len(BARRIDS)) for f in factors},
        }
    )
    # A null exposure, which the covariance loader fills with 0.
    return frame.with_columns(
        pl.when((pl.col("barrid") == "USA0002") & (pl.col("date") == DATES[3]))
        .then(None)
        .otherwise(pl.col(factors[5]))
        .alias(factors[5])
    )


def covariances() -> pl.DataFrame:
    frames = []
    for date in DATES:
        cov = factor_covariance(date)
        upper = np.where(np.triu(np.ones_like(cov, dtype=bool)), cov, np.nan)
        frame = pl.DataFrame(
            {"date": [date] * len(factors), "factor_1": factors, **dict(zip(factors, upper.T))}
        ).fill_nan(None)
        # Stored in an arbitrary row order; the loader sorts.
        frames.append(frame.sample(fraction=1.0, shuffle=True, seed=date.toordinal()))
    return pl.concat(frames)


def factor_returns() -> pl.DataFrame:
    rng = np.random.default_rng(1)
    return pl.DataFrame({"date": DATES, **{f: rng.normal(size=len(DATES)) for f in factors}})


def signal_table(value: str) -> pl.DataFrame:
    rng = np.random.default_rng(len(value))
    n = len(DATES) * len(BARRIDS) * len(SIGNAL_NAMES)
    return pl.DataFrame(
        {
            "date": [d for d in DATES for _ in BARRIDS for _ in SIGNAL_NAMES],
            "barrid": [b for _ in DATES for b in BARRIDS for _ in SIGNAL_NAMES],
            "signal_name": SIGNAL_NAMES * (len(DATES) * len(BARRIDS)),
            value: rng.normal(size=n),
        }
    )


PERMNOS = [10001, 10002, 10003, 10004, 10005]


def crsp_returns(dates: list[dt.date]) -> pl.DataFrame:
    rows = []
    for t, date in enumerate(dates):
        for permno in PERMNOS:
            ret = 0.01 * (permno - 10003) + 0.001 * t
            if permno == 10002 and t == 1:
                ret = -99.0  # a CRSP missing-return code
            rows.append(
                {
                    "date": date,
                    "permno": permno,
                    "ret": ret,
                    # CRSP marks a bid/ask midpoint with a negative price.
                    "prc": -20.0 if permno == 10001 else 15.0 + t,
                }
            )
    return pl.DataFrame(rows)


def crsp_events() -> pl.DataFrame:
    """Events only on the first date of each year: everything later forward-fills.

    10003 is not common stock (shrcd 12), 10004 trades on an excluded exchange
    (exchcd 4), and 10005 never has an event at all.
    """
    rows = []
    for date in (DATES[0], DATES[3]):
        for permno, shrcd, exchcd in [(10001, 10, 1), (10002, 11, 3), (10003, 12, 1), (10004, 10, 4)]:
            rows.append(
                {"date": date, "permno": permno, "ticker": f"P{permno % 100}", "shrcd": shrcd, "exchcd": exchcd}
            )
    return pl.DataFrame(rows)


MONTH_ENDS = [dt.date(2023, 12, 29), dt.date(2024, 1, 31)]


def fama_french() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": DATES,
            **{c: [0.001 * (k + 1) * (t + 1) for t in range(len(DATES))]
               for k, c in enumerate(["mkt_rf", "smb", "hml", "rmw", "cma", "rf"])},
        }
    )


@pytest.fixture
def lake(tmp_path, monkeypatch):
    write_years(tmp_path, "assets", assets())
    write_years(tmp_path, "exposures", exposures())
    write_years(tmp_path, "covariances", covariances())
    write_years(tmp_path, "factors", factor_returns())
    write_years(tmp_path, "signals", signal_table("signal_value"))
    write_years(tmp_path, "scores", signal_table("score"))
    write_years(tmp_path, "alphas", signal_table("alpha"))
    write_years(tmp_path, "crsp_daily", crsp_returns(DATES))
    write_years(tmp_path, "crsp_v2_daily", crsp_returns(DATES))
    write_years(tmp_path, "crsp_monthly", crsp_returns(MONTH_ENDS))
    write_years(tmp_path, "crsp_v2_monthly", crsp_returns(MONTH_ENDS))
    write_years(tmp_path, "crsp_events", crsp_events())
    write_years(tmp_path, "ff5_factors", fama_french(), sep="")
    monkeypatch.setitem(data_config._config, "root", str(tmp_path))
    monkeypatch.setitem(data_config._config, "database", DATABASE)
    return tmp_path
