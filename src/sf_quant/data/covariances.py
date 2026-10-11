import datetime as dt
from functools import lru_cache

import polars as pl

from ._config import register_cache
from ._tables import covariances_table


@lru_cache(maxsize=64)
def load_covariances_by_date(date_: dt.date) -> pl.DataFrame:
    """
    Factor covariances for one date.

    The returned frame is cached and shared between callers, so treat it as
    read-only. Call :func:`sf_quant.data.clear_cache` to drop it.
    """
    return covariances_table.scan(date_.year).filter(pl.col("date").eq(date_)).sort("date").collect()


register_cache(load_covariances_by_date.cache_clear)


def get_covariances_columns() -> str:
    return covariances_table.columns()
