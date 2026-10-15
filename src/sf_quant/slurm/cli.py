"""Command-line interface for SLURM-orchestrated backtests."""

import click

from sf_quant.slurm.runner import BacktestRunner

#: Overrides that feed into the derived output/logs paths, so a sweep writes to
#: a separate directory per setting instead of overwriting itself.
PATH_AFFECTING = ("target_active_risk", "lag", "rebalance_period")


@click.group()
def main() -> None:
    """sf-quant: multi-node parallelized backtesting."""


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the SBATCH script without submitting.",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    help="Override the data path from config.",
)
@click.option(
    "--target-active-risk",
    type=float,
    help="Override target_active_risk from config.",
)
@click.option("--lag", type=int, help="Override lag (trading days) from config.")
@click.option(
    "--rebalance-period",
    type=int,
    help="Override rebalance_period (trading days) from config.",
)
def run(
    config_path: str,
    dry_run: bool,
    data_path: str | None,
    target_active_risk: float | None,
    lag: int | None,
    rebalance_period: int | None,
) -> None:
    """Run a backtest on SLURM from a YAML config file.

    CONFIG_PATH is the path to a YAML configuration file.

    Example:
        sf_quant run config.yml
        sf_quant run config.yml --dry-run
        sf_quant run config.yml --target-active-risk 0.05
        sf_quant run config.yml --lag 5 --rebalance-period 21
    """
    runner = BacktestRunner.from_yaml(config_path)

    if data_path is not None:
        runner.config.data_path = data_path

    overrides = {
        "target_active_risk": target_active_risk,
        "lag": lag,
        "rebalance_period": rebalance_period,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if overrides:
        for key, value in overrides.items():
            setattr(runner.config, key, value)
        # Re-derive the output paths that depend on the overridden values.
        runner.config.output_dir = runner.config.logs_dir = None
        runner.config.__post_init__()

    runner.submit(dry_run=dry_run)
