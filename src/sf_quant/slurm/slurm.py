"""SLURM script generation and job submission."""

import importlib.resources
import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from sf_quant.data._config import _config as data_config
from sf_quant.slurm.config import BacktestConfig, SlurmConfig


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
{_lake_exports()}

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


def _lake_exports() -> str:
    """Export the submitting session's data lake, so compute nodes read the same one."""
    pairs = [f"{key.upper()}={data_config[key]}" for key in ("root", "database") if data_config[key]]
    return f"export {' '.join(pairs)}" if pairs else ""


def generate_command_script(
    command: str,
    *,
    job_name: str,
    logs_dir: str,
    byu_email: str,
    slurm: SlurmConfig | None = None,
    workdir: str | None = None,
) -> str:
    """An SBATCH script that runs one shell command on a compute node.

    The command runs from ``workdir`` (default: the current directory), with the
    same BLAS threading and data-lake settings as the backtest workers.
    """
    slurm = slurm or SlurmConfig()
    workdir = workdir or os.getcwd()
    exclusive_line = "#SBATCH --exclusive" if slurm.exclusive else ""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/{job_name}_%j.out
#SBATCH --cpus-per-task={slurm.n_cpus}
#SBATCH --mem={slurm.mem}
#SBATCH --time={slurm.time}
#SBATCH --mail-user={byu_email}
#SBATCH --mail-type={slurm.mail_type}
{exclusive_line}

ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
{_lake_exports()}

cd {workdir}
{command}
"""


def submit_job(
    script_content: str,
    dry_run: bool = False,
    after: Sequence[str | None] | None = None,
) -> str | None:
    """Submit a SLURM job from script content.

    Args:
        script_content: The SBATCH script content.
        dry_run: If True, print the script instead of submitting.
        after: Job ids this job waits for (``--dependency=afterok``). It runs
            only if all of them finish successfully. ``None`` entries (from
            dry runs) are ignored.

    Returns:
        The job id, or None on a dry run.
    """
    dependencies = [job for job in (after or []) if job]
    flags = [f"--dependency=afterok:{':'.join(dependencies)}"] if dependencies else []

    if dry_run:
        print("=" * 60)
        print("DRY RUN - Would submit the following script" + (f" with {flags[0]}" if flags else "") + ":")
        print("=" * 60)
        print(script_content)
        print("=" * 60)
        return None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        result = subprocess.run(
            ["sbatch", "--parsable", *flags, script_path],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)
    # --parsable prints "jobid" or "jobid;cluster".
    return result.stdout.strip().split(";")[0]


def submit_command(
    command: str,
    *,
    job_name: str,
    logs_dir: str,
    byu_email: str,
    slurm: SlurmConfig | None = None,
    workdir: str | None = None,
    after: Sequence[str | None] | None = None,
    dry_run: bool = False,
) -> str | None:
    """Queue any shell command as its own SLURM job; return its job id.

    Use it for the steps around a backtest (building alphas, analysis,
    plotting), chained with ``after``:

    >>> data = submit_command("python -m study data", job_name="data", ...)  # doctest: +SKIP
    >>> runs = submit_sweep(configs, after=[data])  # doctest: +SKIP
    >>> submit_command("python -m study analyze", job_name="analyze", after=runs, ...)  # doctest: +SKIP
    """
    os.makedirs(logs_dir, exist_ok=True)
    script = generate_command_script(
        command, job_name=job_name, logs_dir=logs_dir, byu_email=byu_email, slurm=slurm, workdir=workdir
    )
    return submit_job(script, dry_run=dry_run, after=after)
