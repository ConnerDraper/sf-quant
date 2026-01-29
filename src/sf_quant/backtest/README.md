# sf-backtester

SLURM-based parallel backtesting for quantitative finance. Distributes MVO optimization across compute nodes, processing one year per task.

## Installation

```bash
pip install -e .
```

## Usage

### CLI

```bash
# Generate a config template
sf_backtester init config.yml

# Validate config
sf_backtester validate config.yml

# Run backtest
sf_backtester run config.yml

# Preview sbatch script without submitting
sf_backtester run config.yml --dry-run
```

### Python API

```python
from sf_backtester import BacktestRunner, BacktestConfig

config = BacktestConfig(
    signal_name="momentum",
    gamma=0.5,
    data_path="/path/to/alphas.parquet",
    project_root="/path/to/project",
    email="you@byu.edu",
    constraints=["ZeroBeta", "ZeroInvestment"],
)

runner = BacktestRunner(config)

# Preview the sbatch script
print(runner.dry_run())

# Submit to SLURM
runner.submit()
```

Or load from YAML:

```python
runner = BacktestRunner.from_yaml("config.yml")
runner.submit()
```

You can also pass a DataFrame directly:

```python
import polars as pl

data = pl.read_parquet("alphas.parquet")
runner.submit(data=data)
```

## Configuration

### YAML format

```yaml
signal_name: momentum
gamma: 0.5
data_path: /path/to/alphas.parquet
project_root: /home/user/research
email: you@byu.edu

constraints:
  - ZeroBeta
  - ZeroInvestment

slurm:
  n_cpus: 8
  mem: 32G
  time: "06:00:00"
  mail_type: BEGIN,END,FAIL

# Optional: register custom constraints
constraint_registry:
  ZeroBeta: sf_quant.optimizer.constraints.ZeroBeta
  ZeroInvestment: sf_quant.optimizer.constraints.ZeroInvestment
  MyCustom: mypackage.constraints.CustomConstraint
```

### Config fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `signal_name` | Yes | - | Name for the signal (used in output paths) |
| `gamma` | Yes | - | Risk aversion parameter for MVO |
| `data_path` | Yes | - | Path to parquet with alpha data |
| `project_root` | Yes | - | Project root directory |
| `email` | Yes | - | Email for SLURM notifications |
| `constraints` | No | `["ZeroInvestment"]` | List of constraint names |
| `output_dir` | No | `{project_root}/weights/{signal_name}/{gamma}` | Output directory |
| `logs_dir` | No | `{project_root}/logs/{signal_name}/{gamma}` | Logs directory |
| `slurm.n_cpus` | No | `8` | CPUs per task |
| `slurm.mem` | No | `32G` | Memory per task |
| `slurm.time` | No | `06:00:00` | Time limit |

## Data format

Input parquet must have columns:
- `date`: Date column
- `barrid`: Asset identifier  
- `alpha`: Alpha signal values
- `predicted_beta`: Predicted beta values

Output is one parquet per year in `output_dir/{year}.parquet` containing portfolio weights.

## Custom constraints

Register custom constraints by adding them to `constraint_registry`:

```yaml
constraint_registry:
  MyConstraint: mypackage.module.MyConstraintClass
```

The constraint class must be importable on the compute nodes and follow the `sf_quant.optimizer.constraints` interface.

## Requirements

- Python 3.10+
- SLURM cluster
- `sf_quant` package (for optimization)
- polars, pyyaml, click
