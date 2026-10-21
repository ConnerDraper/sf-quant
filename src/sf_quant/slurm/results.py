"""Read a year-sharded backtest back, refusing to return it incomplete."""

from pathlib import Path

import polars as pl

import sf_quant.data as sfd
from sf_quant.backtester import collapse
from sf_quant.slurm.config import BacktestConfig


class MissingYearsError(Exception):
    """A sharded run is missing year files it should have written."""


def expected_years(config: BacktestConfig) -> list[int]:
    """
    Years a run should have written a weights file for.

    Built exactly the way the worker builds its grid: collapse the alpha panel
    on the full run calendar, then look at which years have rows. A year can
    legitimately have none (a signal starting in late December, lagged 21
    days), and the worker writes no file for it, so it isn't expected here.
    """
    dates = sfd.load_trading_days(config.start, config.end)
    grid = collapse(
        pl.scan_parquet(config.data_path).select("date", "barrid", "alpha"),
        dates=dates,
        rebalance_period=config.rebalance_period,
        lag=config.lag,
    )
    years = grid.select(pl.col("date").dt.year().unique()).collect()["date"]
    return sorted(years.to_list())


def load_weights(config: BacktestConfig) -> pl.DataFrame:
    """
    All of a run's weights, after checking every expected year was written.

    Parameters
    ----------
    config : BacktestConfig
        The config the run was submitted with; its ``output_dir`` holds one
        ``{year}.parquet`` per array task.

    Returns
    -------
    pl.DataFrame
        The concatenated weights, sorted by ``date`` and ``barrid``.

    Raises
    ------
    MissingYearsError
        Naming the years that have no file, e.g. because their array task
        failed or is still running. A run with holes would otherwise be
        analyzed as if the missing years were never traded.
    """
    directory = Path(config.output_dir)
    found = {int(f.stem) for f in directory.glob("*.parquet")}
    missing = sorted(set(expected_years(config)) - found)
    if missing:
        raise MissingYearsError(
            f"{config.signal_name}: no weights for {missing} in {directory}; "
            "check those array tasks' logs before analyzing"
        )
    return pl.read_parquet(sorted(directory.glob("*.parquet"))).sort("date", "barrid")
