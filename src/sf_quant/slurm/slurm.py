"""SLURM script generation and job submission."""

import importlib.resources
import os
import subprocess
import tempfile
from pathlib import Path

from sf_quant.slurm.config import BacktestConfig


def get_worker_script_path() -> Path:
    """Get the path to the bundled worker script."""
    ref = importlib.resources.files("sf_quant.slurm.scripts").joinpath("worker.py")
    with importlib.resources.as_file(ref) as path:
        return Path(path)


def generate_sbatch_script(config: BacktestConfig, years: list[int]) -> str:
    """Generate the SBATCH script content, one array task per year."""
    num_years = len(years)
    years_str = " ".join(str(y) for y in years)
    constraints_str = " ".join(config.constraints)
    worker_script = get_worker_script_path()
    exclusive_line = "#SBATCH --exclusive" if config.slurm.exclusive else ""

    script = f"""#!/bin/bash
#SBATCH --job-name={config.signal_name}_backtest
#SBATCH --output={config.logs_dir}/backtest_%A_%a.out
#SBATCH --error={config.logs_dir}/backtest_%A_%a.err
#SBATCH --array=0-{num_years - 1}%{config.slurm.max_concurrent_jobs}
#SBATCH --cpus-per-task={config.slurm.n_cpus}
#SBATCH --mem={config.slurm.mem}
#SBATCH --time={config.slurm.time}
#SBATCH --mail-user={config.byu_email}
#SBATCH --mail-type={config.slurm.mail_type}
{exclusive_line}

# Don't let a crashing worker dump a multi-hundred-MB core file per task --
# these are shared, often-unattended nodes, and a crash storm shouldn't also
# become a disk-I/O storm.
ulimit -c 0

# backtest() runs one single-threaded solve per worker process; keep BLAS from
# spawning its own threads on top of that.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

DATA_PATH="{config.data_path}"
OUTPUT_DIR="{config.output_dir}"
TARGET_ACTIVE_RISK="{config.target_active_risk}"
N_CPUS="{config.slurm.n_cpus}"
CONSTRAINTS="{constraints_str}"

# Years to process
years=({years_str})

num_years=${{#years[@]}}

if [ $SLURM_ARRAY_TASK_ID -ge $num_years ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID is out of range (max $((num_years-1)))."
    exit 1
fi

year=${{years[$SLURM_ARRAY_TASK_ID]}}

source {config.project_root}/.venv/bin/activate
echo "Running year=$year"
srun --ntasks=1 --propagate=CORE python {worker_script} \\
    --data_path "$DATA_PATH" \\
    --year "$year" \\
    --start {config.start.isoformat()} \\
    --end {config.end.isoformat()} \\
    --output_dir "$OUTPUT_DIR" \\
    --n_cpus "$N_CPUS" \\
    --target_active_risk "$TARGET_ACTIVE_RISK" \\
    --lag {config.lag} \\
    --rebalance_period {config.rebalance_period} \\
    --constraints $CONSTRAINTS
"""
    return script


def submit_job(
    script_content: str, dry_run: bool = False
) -> subprocess.CompletedProcess | None:
    """Submit a SLURM job from script content.

    Args:
        script_content: The SBATCH script content.
        dry_run: If True, print the script instead of submitting.

    Returns:
        The subprocess result if submitted, None if dry_run.
    """
    if dry_run:
        print("=" * 60)
        print("DRY RUN - Would submit the following script:")
        print("=" * 60)
        print(script_content)
        print("=" * 60)
        return None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=True,
        )
        return result
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)
