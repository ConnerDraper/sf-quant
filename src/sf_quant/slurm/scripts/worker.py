"""Worker for a single SLURM array task.

Each task owns one calendar year. The rebalance grid and the lag are built
against the full run calendar first and only then filtered to this task's year,
so every shard agrees on the grid and the lag can reach back across the January
boundary. Slicing the panel by year before collapsing would do neither.
"""

import argparse
import datetime as dt

import polars as pl

import sf_quant.data as sfd
from sf_quant.backtester import backtest, collapse
from sf_quant.slurm.scripts.utils import get_constraints


def run_backtest_by_year(
    data_path: str,
    year: int,
    start: dt.date,
    end: dt.date,
    output_dir: str,
    n_cpus: int,
    constraints: list[str],
    target_active_risk: float,
    lag: int = 0,
    rebalance_period: int = 1,
) -> None:
    """Collapse the panel to the rebalance grid, solve this year, write a parquet."""
    dates = sfd.load_trading_days(start, end)
    print(f"Calendar: {dates.len()} trading days, {dates.min()} to {dates.max()}")

    grid = collapse(
        pl.scan_parquet(data_path),
        dates=dates,
        rebalance_period=rebalance_period,
        lag=lag,
    )

    year_data = grid.filter(pl.col("date").dt.year() == year).collect()

    if year_data.is_empty():
        print(f"Year {year}: no rebalance dates with data. Nothing to write.")
        return

    n_dates = year_data["date"].n_unique()
    print(f"Year {year}: {n_dates} rebalance dates, {len(year_data)} rows")

    weights = backtest(
        year_data,
        constraints=get_constraints(constraints),
        target_active_risk=target_active_risk,
        n_cpus=n_cpus,
    )

    output_path = f"{output_dir}/{year}.parquet"
    weights.write_parquet(output_path)
    print(f"Wrote {len(weights)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backtest for a single year.")
    parser.add_argument("--data_path", type=str, required=True, help="Alpha parquet.")
    parser.add_argument("--year", type=int, required=True, help="Year to process.")
    parser.add_argument(
        "--start", type=dt.date.fromisoformat, required=True, help="Run start date."
    )
    parser.add_argument(
        "--end", type=dt.date.fromisoformat, required=True, help="Run end date."
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output dir.")
    parser.add_argument("--n_cpus", type=int, default=8, help="CPUs for Ray.")
    parser.add_argument(
        "--target_active_risk",
        type=float,
        default=0.05,
        help="Annualized tracking-error budget.",
    )
    parser.add_argument(
        "--constraints",
        type=str,
        nargs="+",
        required=True,
        help="Constraint names (e.g. ZeroInvestment ZeroBeta LongOnly).",
    )
    parser.add_argument(
        "--lag", type=int, default=0, help="Trading days to lag alpha by."
    )
    parser.add_argument(
        "--rebalance_period",
        type=int,
        default=1,
        help="Rebalance every N trading days.",
    )

    args = parser.parse_args()

    run_backtest_by_year(
        data_path=args.data_path,
        year=args.year,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        n_cpus=args.n_cpus,
        constraints=args.constraints,
        target_active_risk=args.target_active_risk,
        lag=args.lag,
        rebalance_period=args.rebalance_period,
    )


if __name__ == "__main__":
    main()
