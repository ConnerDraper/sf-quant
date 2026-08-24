"""Cost-aware MVO worker script for SLURM jobs.

This script runs a sequential, cost-aware backtest over the full date range.
Unlike the parallel workers that split by year, this maintains continuity
of warm-starting and turnover cost tracking across the entire period.
"""

import argparse

import polars as pl
import sf_quant.backtester as sfb
import sf_quant.costs as sfc

from sf_backtester.scripts.utils import get_constraints


def build_cost_model(
    cost_type: str,
    target_median_bps: float,
    fixed_bps: float,
) -> sfc.CostModel:
    """Instantiate a cost model from config parameters."""
    if cost_type == "spread":
        return sfc.SpreadCost(target_median_bps=target_median_bps)
    elif cost_type == "fixed":
        return sfc.FixedCost(bps=fixed_bps)
    else:
        raise ValueError(f"Unknown cost model type: {cost_type!r}. Use 'spread' or 'fixed'.")


def run_cost_aware_backtest(
    df: pl.LazyFrame,
    gamma: float,
    output_dir: str,
    constraints: list[str],
    cost_type: str,
    target_median_bps: float,
    fixed_bps: float,
) -> None:
    """Run a sequential cost-aware backtest over all dates."""
    data = df.collect()
    print(f"Running cost-aware sequential backtest: {len(data)} rows")

    constraint_objects = get_constraints(constraints)
    cost_model = build_cost_model(cost_type, target_median_bps, fixed_bps)

    weights = sfb.backtest_sequential(
        data=data,
        constraints=constraint_objects,
        gamma=gamma,
        cost_model=cost_model,
    )

    output_path = f"{output_dir}/weights.parquet"
    weights.write_parquet(output_path)
    print(f"Wrote weights to {output_path}")


def main() -> None:
    """Main entry point for the cost-aware worker script."""
    parser = argparse.ArgumentParser(
        description="Run cost-aware sequential MVO backtest."
    )

    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to parquet file containing the data.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        required=True,
        help="Risk aversion parameter for MVO.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write output parquet file.",
    )
    parser.add_argument(
        "--constraints",
        nargs="+",
        required=True,
        help="List of constraint names (e.g., ZeroBeta ZeroInvestment).",
    )
    parser.add_argument(
        "--cost_type",
        default="spread",
        help="Cost model type: 'spread' or 'fixed'.",
    )
    parser.add_argument(
        "--target_median_bps",
        type=float,
        default=7.5,
        help="Target median cost in basis points (for spread model).",
    )
    parser.add_argument(
        "--fixed_bps",
        type=float,
        default=5.0,
        help="Fixed cost in basis points (for fixed model).",
    )

    args = parser.parse_args()

    df = pl.scan_parquet(args.data_path)

    run_cost_aware_backtest(
        df=df,
        gamma=args.gamma,
        output_dir=args.output_dir,
        constraints=args.constraints,
        cost_type=args.cost_type,
        target_median_bps=args.target_median_bps,
        fixed_bps=args.fixed_bps,
    )


if __name__ == "__main__":
    main()
