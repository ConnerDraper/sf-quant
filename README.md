# Silver Fund Quant

A quantitative finance library for the BYU Silver Fund quant team. Provides tools for data loading, signal construction, portfolio optimization, performance analysis, and signal research — all built on Polars DataFrames. Backtests live in [sf-backtester](https://github.com/ConnerDraper/sf-backtester).

## Installation

```bash
pip install sf-quant
```

### Upgrading from 0.2

0.3.0 removes the gamma-based tools, which solved a different problem than the fund's mandate:
- `sf_quant.backtester` (`backtest_sequential`, `backtest_parallel`, `dynamic_backtest_parallel`)
- `mve_optimizer` and `dynamic_mve_optimizer`
- the constraint classes (`LongOnly`, `ZeroBeta`, …)
- `sf_quant.costs`
- `construct_factor_model_components` and `construct_covariance_matrix`
- the leverage tools (`generate_leverage_from_weights`, its chart and summary table): the fund is always fully invested and long-only, so leverage is always 1
- `generate_alpha_ics` and `generate_ic_chart`, which measured IC against total returns; use `sf_quant.signals.estimate_ic`, which measures it against the return the fund can actually earn
- `load_optimal_weights`

Use `sf_quant.optimizer.mandate_optimizer`, `sf_quant.data.load_risk_model`, and sf-backtester's `run_backtest` instead. To keep running old notebooks unchanged, pin `sf-quant<0.3`.

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

Build alphas from a signal, then solve the fund's mandate problem for one date. For full walk-forward backtests with returns, costs, and diagnostics, use [sf-backtester](https://github.com/ConnerDraper/sf-backtester).

```python
import datetime as dt

import numpy as np
import polars as pl
import sf_quant.optimizer as sfo
import sf_quant.signals as sg
from sf_quant.data import SfQuantMarketData

data = SfQuantMarketData()

# 1. Alphas: the score is residualized against Barra factors; alpha = IC * specific risk * Z
panel = sg.build_alpha(sg.IdiosyncraticMomentum(), data, dt.date(2023, 1, 1), dt.date(2024, 12, 31))

# 2. Solve the mandate problem on the last formation date
date_ = panel.alphas["date"].max()
alpha = dict(panel.alphas.filter(pl.col("date") == date_).select("barrid", "alpha").iter_rows())
bench = dict(data.benchmark_weights(date_, date_).select("barrid", "weight").iter_rows())
rm = data.risk_model(date_, sorted(bench))

result = sfo.mandate_optimizer(
    ids=rm.ids,
    alphas=np.array([alpha.get(x, 0.0) for x in rm.ids]),
    factor_exposures=rm.factor_exposures,
    factor_covariance=rm.factor_covariance,
    specific_variance=rm.specific_variance,
    benchmark_weights=np.array([bench[x] for x in rm.ids]),
    target_active_risk=0.05,
)
result.active_weights, result.ex_ante_te
```

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
| `load_risk_model()` | Barra B, F, and specific variance for one date (decimal, annualized), reporting names with missing risk data |
| `load_benchmark()` / `load_benchmark_returns()` | Benchmark weights and returns |
| `load_factors()` / `get_factor_names()` | Barra factor returns and metadata |
| `load_fama_french()` | Fama-French 5-factor data |

### `sf_quant.optimizer`

`mandate_optimizer` solves the Silver Fund mandate problem (manual Chapter 3)
in active weights, with no risk-aversion parameter:

```
max hᵀα  s.t.  Σh = 0,  βᵀh = 0,  hᵀVh ≤ σ̄²,  w_b + h ≥ 0
```

V is the Barra factor model `B F Bᵀ + diag(δ)`, and β = Vb / (bᵀVb) comes from
the same V (manual 3.3.2). The risk constraint sizes the active portfolio to
the tracking-error target directly. If long-only keeps the portfolio from
reaching the target, the gap is reported as `te_shortfall`.

```python
import sf_quant.data as sfd
import sf_quant.optimizer as sfo

rm = sfd.load_risk_model(date_, barrids)  # drops and reports names missing risk data
result = sfo.mandate_optimizer(
    ids=rm.ids,
    alphas=alphas,                     # aligned to rm.ids
    factor_exposures=rm.factor_exposures,
    factor_covariance=rm.factor_covariance,
    specific_variance=rm.specific_variance,
    benchmark_weights=benchmark_weights,
    target_active_risk=0.05,
)
result.active_weights, result.ex_ante_te, result.status
```

Optional arguments:
- `neutral_factors` holds factors at exactly zero active exposure. Redundant rows, like the country factor that duplicates Σh = 0, are dropped automatically.
- `factor_bounds` caps them.

`result.factor_shadow_prices` reports what each neutrality constraint costs in expected return.

### `sf_quant.signals`

Turn stock-level signals into alphas for the mandate optimizer.

```python
import sf_quant.signals as sg
from sf_quant.data import SfQuantMarketData

data = SfQuantMarketData()
signals = [sg.IdiosyncraticMomentum(), sg.ShortTermReversal(), sg.IndustryMomentum()]

panel = sg.build_composite(signals, data, start, end)        # alphas + allowed factors + constraints
loss = sg.factor_loss(sg.IdiosyncraticMomentum(), data, start, end)
loss.table                                                   # return given up to each Barra factor
```

- **Allowed factors.** Each `Signal` declares the Barra factors it may keep, with `NO_FACTORS` as the default. Sets compose: `ALL_FACTORS`, `STYLE_FACTORS`, `INDUSTRY_FACTORS`, `COUNTRY_FACTOR`, e.g. `STYLE_FACTORS | {"USSLOWL_BETA"}`. Scores are residualized against every other factor.
- **IC target.** IC and σ refer to the Barra specific return plus the allowed factors' returns. With no allowed factors, this is exactly specific return and specific risk.
- **Composite weights.** `build_composite` weights signals by v = R⁻¹·IC, using out-of-sample ICs shrunk toward zero and a shrunk signal correlation matrix R.
- **Defaults:**
  - `IdiosyncraticMomentum`
  - `ShortTermReversal`
  - `IndustryMomentum` (keeps industries)
  - `BettingAgainstBeta` (keeps `USSLOWL_BETA`). A worked example of a signal the mandate erases: any alpha linear in beta earns exactly zero under full investment and zero active beta.

`sf_quant.testing.synthetic_market` gives a Barra-shaped market with planted effects for tests.

### `sf_quant.performance`

Compute and visualize portfolio performance metrics.

| Function | Description |
|---|---|
| `generate_returns_from_weights()` | Portfolio returns from weight history |
| `generate_multi_returns_from_weights()` | Total, benchmark, and active returns |
| `generate_drawdown_from_returns()` | Drawdown from peak |
| `generate_returns_summary_table()` | Mean return, volatility, Sharpe, total return |
| `get_turnover_stats()` | Rolling turnover statistics |
| `generate_returns_chart()` | Plot cumulative returns |
| `generate_multi_returns_chart()` | Plot total/benchmark/active returns |
| `generate_drawdown_chart()` | Plot drawdowns |
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
