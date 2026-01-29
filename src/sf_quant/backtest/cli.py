"""Command-line interface for sf-backtester."""

import click

from sf_backtester.runner import BacktestRunner


@click.command()
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
    "--gamma",
    type=float,
    help="Override gamma from config.",
)
def main(
    config_path: str,
    dry_run: bool,
    data_path: str | None,
    gamma: float | None,
) -> None:
    """Run a parallel backtest on SLURM from a YAML config file.

    CONFIG_PATH is the path to a YAML configuration file.

    Example:
        sf-backtester config.yml
        sf-backtester config.yml --dry-run
        sf-backtester config.yml --gamma 0.5
    """
    runner = BacktestRunner.from_yaml(config_path)

    # Apply overrides
    if data_path is not None:
        runner.config.data_path = data_path
    if gamma is not None:
        runner.config.gamma = gamma
        # Update output paths that depend on gamma
        runner.config.output_dir = (
            f"{runner.config.project_root}/weights/{runner.config.signal_name}/{gamma}"
        )
        runner.config.logs_dir = f"logs/{runner.config.signal_name}/{gamma}"

    runner.submit(dry_run=dry_run)


if __name__ == "__main__":
    main()
