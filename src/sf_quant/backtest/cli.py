"""Command-line interface: run a backtest here, or submit it as one SLURM job."""

from __future__ import annotations

import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import click
import polars as pl

from .backtest import run_backtest
from .config import BacktestConfig


@click.group()
def main() -> None:
    """Walk-forward backtests of the Silver Fund mandate."""


@main.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--alphas", "alphas_path", required=True, type=click.Path(exists=True), help="Parquet with date, barrid, alpha.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), help="Output directory (default runs/<name>).")
def run(config_path: str, alphas_path: str, out_dir: str | None) -> None:
    """Run a backtest in this process and save its outputs."""
    cfg = BacktestConfig.from_yaml(config_path)
    result = run_backtest(pl.read_parquet(alphas_path), cfg)
    out = result.save(out_dir or f"runs/{cfg.name}")
    s = result.summary
    click.echo(
        f"Saved to {out}. IR gross {s['ir_gross']:.2f}, realized TE {s['realized_te']:.2%}, "
        f"failed rebalances {s['failed_rebalances']}."
    )


@main.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--alphas", "alphas_path", required=True, type=click.Path(exists=True))
@click.option("--out", "out_dir", type=click.Path(file_okay=False))
@click.option("--dry-run", is_flag=True, help="Print the sbatch script without submitting.")
def submit(config_path: str, alphas_path: str, out_dir: str | None, dry_run: bool) -> None:
    """Submit `run` as a single-node SLURM job using the config's slurm settings."""
    cfg = BacktestConfig.from_yaml(config_path)
    script = sbatch_script(cfg, Path(config_path), Path(alphas_path), Path(out_dir or f"runs/{cfg.name}"))
    if dry_run:
        click.echo(script)
        return
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
    click.echo(subprocess.run(["sbatch", f.name], capture_output=True, text=True, check=True).stdout.strip())


def sbatch_script(cfg: BacktestConfig, config_path: Path, alphas_path: Path, out_dir: Path) -> str:
    slurm, logs = cfg.slurm, out_dir.resolve() / "logs"
    command = shlex.join(
        str(part)
        for part in (sys.executable, "-m", "sf_quant.backtest.cli", "run", config_path.resolve(),
                     "--alphas", alphas_path.resolve(), "--out", out_dir.resolve())
    )
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={cfg.name}",
        f"#SBATCH --output={logs}/%j.out",
        f"#SBATCH --error={logs}/%j.err",
        "#SBATCH --nodes=1",
        f"#SBATCH --cpus-per-task={slurm.n_cpus}",
        f"#SBATCH --mem={slurm.mem}",
        f"#SBATCH --time={slurm.time}",
        *([f"#SBATCH --mail-user={slurm.email}", "#SBATCH --mail-type=END,FAIL"] if slurm.email else []),
        f"mkdir -p {shlex.quote(str(logs))}",
        command,
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
