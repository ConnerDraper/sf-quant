# sf-backtester

SLURM-based parallel backtesting for quantitative finance. Distributes MVO optimization across compute nodes, processing one year per task.

## Installation

```bash
pip install sf-backtester
```

## Usage

### CLI

```bash
# Run backtest
sf_backtester run config.yml

# Preview sbatch script without submitting
sf_backtester run config.yml --dry-run
```

### Python API

```python
from sf_backtester import BacktestRunner, BacktestConfig

slurm_config = SlurmConfig(
    n_cpus=8,
    mem="32G",
    time="03:00:00",
    mail_type="BEGIN,END,FAIL",
    max_concurrent_jobs=30,
)

config = BacktestConfig(
    signal_name="momentum",
    gamma=50,
    data_path="/path/to/alphas.parquet",
    project_root="/path/to/project",
    byu_email="you@byu.edu",
    constraints=["ZeroBeta", "ZeroInvestment"],
    slurm=slurm_config,
)

runner = BacktestRunner(config)

# Preview the sbatch script
print(runner.dry_run())

# Submit to SLURM
runner.submit()
```

Or load from YAML:

```python
from sf_backtester import BacktestRunner

runner = BacktestRunner.from_yaml("config.yml")

runner.submit()
```

You can also pass a DataFrame directly:

```python
from sf_backtester import BacktestRunner
import polars as pl

runner = BacktestRunner.from_yaml("config.yml")

data = pl.read_parquet("alphas.parquet")

runner.submit(data=data)
```

## Configuration

### YAML format

```yaml
signal_name: momentum
gamma: 500
data_path: /path/to/alphas.parquet
project_root: /path/to/project
email: you@byu.edu

constraints:
  - ZeroBeta
  - ZeroInvestment

slurm:
  n_cpus: 8
  mem: 32G
  time: "03:00:00"
  mail_type: BEGIN,END,FAIL
```

## Data format

Input parquet must have columns:
- `date`: Date column
- `barrid`: Asset identifier  
- `alpha`: Alpha signal values
- `predicted_beta`: Predicted beta values

Output is one parquet per year in `output_dir/{year}.parquet` containing portfolio weights.

## Publishing
1. Bump the version

```bash
uv version v*.*.*
```

2. Add changes (it can be just the version change)

```bash
git add .
git commit -m "Bumped version."
```

3. Tag the branch

```bash
git tag v*.*.*
```

4. Push to origin

```bash
git push --tags
```

