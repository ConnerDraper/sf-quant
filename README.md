# Silver Fund Quant

A quantitative finance library for the BYU Silver Fund quant team. Provides tools for data loading, portfolio optimization, backtesting, performance analysis, and signal research — all built on Polars DataFrames.

## Installation

```bash
pip install sf-quant
```

## Environment Configuration

`sf-quant` requires two environment variables to locate data: `ROOT` and `DATABASE`.

### Option 1: `.env` file or environment variables

Create a `.env` file in your project root (automatically loaded via `python-dotenv`):

```
ROOT=/path/to/root
DATABASE=your_database
```

Or export them in your shell:

```bash
export ROOT=/path/to/root
export DATABASE=your_database
```

### Option 2: Programmatic configuration

Set values at runtime using `sfd.env()`:

```python
import sf_quant.data as sfd

sfd.env(root="/path/to/root", database="your_database")
```

This will override any values set via environment variables.

## Quick Start

A complete workflow: build a momentum alpha, backtest it under the fund mandate, and summarize the active returns.

```python
import datetime as dt
import polars as pl
import sf_quant.data as sfd
import sf_quant.backtester as sfb
import sf_quant.performance as sfp

start, end = dt.date(2024, 1, 1), dt.date(2024, 12, 31)

# 1. Signal -> score -> alpha (Manual 4.6: alpha = IC * sigma * Z)
alphas = (
    sfd.load_assets(start, end, ["date", "barrid", "return", "specific_risk"], in_universe=True)
    .sort("barrid", "date")
    .with_columns(
        pl.col("return").truediv(100).log1p().rolling_sum(230).shift(21).over("barrid").alias("signal")
    )
    .drop_nulls("signal")
    .with_columns(
        ((pl.col("signal") - pl.col("signal").mean()) / pl.col("signal").std()).over("date").alias("score")
    )
    .select("date", "barrid", (0.05 * pl.col("score") * pl.col("specific_risk") / 100).alias("alpha"))
)

# 2. One active portfolio per date at 5% tracking error. Beta and benchmark
#    weights are joined on for you.
weights = sfb.backtest(alphas, target_active_risk=0.05)

# 3. Active returns, IR, and transfer coefficient
returns = sfp.generate_active_returns_from_weights(weights)
print(sfp.generate_returns_summary_table(returns))   # annualization inferred
print(sfp.generate_transfer_coefficients(weights, alphas)["transfer_coefficient"].mean())
```

**There is no gamma.** Portfolios are sized by `target_active_risk`; the tracking-error target sets the multiplier on risk each date, not the other way round (Manual 3.2). If you are passing or tuning a gamma, you are doing it wrong.

## Modules

### `sf_quant.data`

Load financial data as Polars DataFrames. Use `get_*_columns()` to inspect available columns for any dataset.

| Function | Description |
|---|---|
| `load_assets()` | Asset prices, returns, risk, and metadata for a date range |
| `load_assets_by_date()` | Asset data for a single date |
| `load_crsp_daily()` / `load_crsp_v2_daily()` | CRSP daily stock data |
| `load_crsp_monthly()` / `load_crsp_v2_monthly()` | CRSP monthly stock data |
| `load_exposures()` / `load_exposures_by_date()` | Barra factor exposures (77+ factors) |
| `load_covariances_by_date()` | Factor covariance matrix for a date |
| `construct_covariance_matrix()` | Build an asset-level covariance matrix from the factor model |
| `load_benchmark()` / `load_benchmark_returns()` | Benchmark weights and returns |
| `load_factors()` / `get_factor_names()` | Barra factor returns and metadata |
| `load_fama_french()` | Fama-French 5-factor data |

### `sf_quant.optimizer`

Maximize active return at a target active risk, in active space (`w = b + h`).

```python
import sf_quant.optimizer as sfo

weights = sfo.active_mve_optimizer(
    ids=barrids,
    alphas=alphas,
    factor_exposures=B,          # from sfd.construct_factor_model_components
    factor_covariance=F,
    specific_risk=D,
    benchmark_weights=b,
    betas=betas,
    target_active_risk=0.05,
    constraints=[sfo.ZeroInvestment(), sfo.ZeroBeta(), sfo.LongOnly()],
)
```

Two solvers sit behind one function, as in Manual Ch. 3:

- **Equality constraints only** (`ZeroInvestment`, `ZeroBeta`): the closed form of Manual 3.2–3.3. No numerical solver, so it's fast.
- **Any inequality** (`LongOnly`): solved numerically with CVXPY, as in Manual 3.4. A bound that binds has no closed form.

Either way the solution is checked against Manual 3.3.6 (active weights sum to 0, active beta is 0, tracking error equals the target), and a violation raises.

| Constraint | Effect |
|---|---|
| `ZeroInvestment()` | `sum(h) == 0`, so the fund stays fully invested |
| `ZeroBeta()` | `betas @ h == 0`, so the fund keeps unit beta |
| `LongOnly()` | `h >= -b`: no shorting beyond the benchmark |

The default is all three: the fund mandate. Drop `LongOnly` for a long/short portfolio.

### `sf_quant.backtester`

Solve one portfolio per date. Each date is independent, so rebalance periods and lags are reductions of the daily panel.

```python
import sf_quant.backtester as sfb

dates = sfd.load_trading_days(start, end)
grid = sfb.collapse(alphas.lazy(), dates=dates, rebalance_period=21, lag=5).collect()
weights = sfb.backtest(grid, target_active_risk=0.05, n_cpus=8)

# Every starting day of a 21-day rebalance, from one daily run
phases = sfb.rebalance_phases(daily_weights, dates=dates, rebalance_period=21)
```

| Function | Description |
|---|---|
| `backtest()` | One active portfolio per date; needs only `date`, `barrid`, `alpha` |
| `attach_beta_and_benchmark()` | The join `backtest()` does: in-universe `predicted_beta`, benchmark weight (0 if absent) |
| `collapse()` | Reduce a daily panel to a rebalance grid, optionally lagging alpha |
| `rebalance_dates()` | Every N-th trading day |
| `rebalance_phases()` | All N phases of an N-day rebalance, derived from daily weights |

### `sf_quant.slurm`

Run a backtest as one array task per year, chain jobs, and read the results back.

```python
from sf_quant.slurm import BacktestConfig, SlurmConfig, load_weights, submit_command, submit_sweep

configs = [
    BacktestConfig(signal_name="momentum", data_path="alphas.parquet", start=start, end=end,
                   project_root=".", byu_email="you@byu.edu", lag=lag)
    for lag in (0, 5, 21)
]
runs = submit_sweep(configs)                        # returns job ids
submit_command("python -m study analyze", job_name="analyze", logs_dir="logs",
               byu_email="you@byu.edu", after=runs)  # runs once all backtests succeed

weights = load_weights(configs[0])  # raises MissingYearsError if any year's task failed
```

`sf_quant run config.yml [--dry-run]` submits a single config from the command line.

### `sf_quant.performance`

Compute and visualize portfolio performance metrics.

| Function | Description |
|---|---|
| `generate_returns_from_weights()` | Portfolio returns from weight history |
| `generate_multi_returns_from_weights()` | Total, benchmark, and active returns |
| `generate_leverage_from_weights()` | Sum of absolute weights over time |
| `generate_drawdown_from_returns()` | Drawdown from peak |
| `generate_active_returns_from_weights()` | Active return per holding period |
| `generate_cumulative_log_returns()` | Cumulative log active return on a daily axis |
| `generate_returns_summary_table()` | Mean return, volatility, Sharpe/IR, total return; annualization inferred from the rebalance spacing |
| `infer_periods_per_year()` | 252 / N for an N-trading-day holding period |
| `generate_transfer_coefficients()` | Transfer coefficient per date (Clarke, de Silva & Thorley 2002) |
| `generate_alpha_ics()` | Information coefficients between signals and realized returns |
| `get_turnover_stats()` | Rolling turnover statistics |
| `generate_returns_chart()` | Plot cumulative returns |
| `generate_multi_returns_chart()` | Plot total/benchmark/active returns |
| `generate_leverage_chart()` | Plot leverage over time |
| `generate_drawdown_chart()` | Plot drawdowns |
| `generate_ic_chart()` | Plot information coefficients |
| `plot_turnover()` | Plot portfolio turnover |

### `sf_quant.research`

Tools for signal analysis and factor research.

| Function | Description |
|---|---|
| `generate_quantile_ports()` | Sort assets into quantile portfolios with long-short spread |
| `vol_scale_ports()` | Scale portfolio returns to target volatility |
| `beta_scale_ports()` | Scale portfolio returns to target beta |
| `run_ff_regression()` | Fama-French 5-factor regression |
| `run_quantile_ff_regression()` | FF regressions across quantile portfolios |
| `get_signal_stats()` | Summary statistics for a signal |
| `get_signal_distribution()` | Plot signal distribution |

## Documentation Development

To run a local server of the sphinx documentation run

```bash
uv run sphinx-autobuild docs docs/_build/html
```

## Release Process
1. Create PR
2. Merge PR(s)
3. Increment version in pyproject.toml
4. git tag v*.*.*
5. git push origin main --tags
6. Create a release and publish release notes (github)
7. uv build
8. uv publish
