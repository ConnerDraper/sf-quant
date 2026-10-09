# sf-backtester

Walk-forward backtests of the BYU Silver Fund mandate. You supply an alpha panel and a config. The backtester loads the benchmark, Barra risk model, and returns itself, then solves the mandate problem at each rebalance and returns realized returns, costs, and diagnostics.

At each rebalance it solves (manual Chapter 3, via `sf_quant.optimizer.mandate_optimizer`):

```
max hᵀα  s.t.  Σh = 0,  βᵀh = 0,  hᵀVh ≤ 0.05²,  w_b + h ≥ 0
```

- There is no gamma. The active portfolio is sized to the 5% tracking-error target directly.
- β = Vb / (bᵀVb) is computed from the same Barra V. It is not Barra's predicted beta.
- Long-only, fully invested, and zero active beta are locked under `profile: mandate`.

> **0.3.0 is a breaking rebuild.** The gamma runners (`BacktestRunner`, `BacktestDynamicRunner`, `BacktestCostAwareRunner`), the `run-dynamic` command, the per-year SLURM array, the `predicted_beta` input column, and the constraint list are gone.

## Usage

```bash
sf_backtester run config.yml --alphas alphas.parquet              # in this process
sf_backtester submit config.yml --alphas alphas.parquet [--dry-run] # one SLURM node
```

```python
from sf_backtester import BacktestConfig, run_backtest

cfg = BacktestConfig.from_yaml("config.yml")
result = run_backtest(alphas, cfg)   # alphas: pl.DataFrame[date, barrid, alpha]

result.summary
result.diagnostics
result.save("runs/momentum/")
```

See [config.example.yml](config.example.yml) for every option and [example/](example/) for an end-to-end momentum run.

## Layout

| Module | Role |
|---|---|
| `config.py` | `BacktestConfig`: one flat dataclass, loaded from YAML and validated |
| `data.py` | `MarketData` protocol and `SfQuantMarketData` (the BYU data store) |
| `backtest.py` | `run_backtest`: schedule, invariant checks, one mandate solve per formation date |
| `accounting.py` | Daily returns with drift, turnover, costs, and the factor/specific split |
| `results.py` | `BacktestResult`, summary metrics, and `save()` |
| `cli.py` | `run` and `submit` |

## Input

| Column | Meaning |
|---|---|
| `date` | Formation date: the alpha uses only information available by that day's close |
| `barrid` | Asset ID |
| `alpha` | Expected residual return, annualized decimal (for example IC × specific risk × z) |

Universe members without an alpha get α = 0, so they can still be underweighted. Only alphas on rebalance formation dates are used.

## Timing

An alpha formed at the close of t trades at the close of t + `execution_lag_days` and earns returns from the next day on. Holdings drift with prices between rebalances. The benchmark return on day d uses benchmark weights from the prior close.

## Costs

Costs are charged on realized turnover (Σ|w_target − w_drifted|) at each bps level in `bps_grid`. They are never part of the optimizer. At about $1M AUM, market impact is negligible and only the spread matters. Spread cost scales with turnover, not fund size. The initial build from cash is not charged.

At our size, the real implementability limit is small positions. `diagnostics` counts positions under `min_position_dollars` and positions worth less than one share.

## Output (`result.save(dir)`)

| File | Contents |
|---|---|
| `weights.parquet` | Per rebalance and asset: alpha, benchmark, active, and fund weights |
| `returns.parquet` | Daily portfolio, benchmark, and active returns; active split into factor and specific parts; `net_active_<bps>bps` |
| `diagnostics.parquet` | Per rebalance: solver status, ex-ante TE, TE shortfall, active beta, transfer coefficient, long-only bound count, turnover, coverage, small-position counts |
| `factor_exposures.parquet` | Active factor exposure per rebalance (long format) |
| `summary.json` | Realized IR gross and net, TE, breakeven bps, realized/ex-ante TE, factor share of active variance, failures, and reproducibility metadata |
| `config.resolved.yml` | Config with every default filled in |

## Invariants

A run raises instead of producing numbers if any of these fails:

- duplicate (date, barrid) rows, or null or non-finite alphas
- alpha coverage below `min_alpha_coverage`
- √(bᵀVb) outside [0.08, 0.40] (the Barra units check)
- a solution that breaks Σh = 0, βᵀh = 0, w ≥ 0, or the TE limit

Names missing Barra risk data are held at benchmark weight and counted in `n_missing_risk`. Runs stop before `holdout_start` unless `unlock_holdout: true`.

## Development

```bash
uv sync          # uses the sibling ../sf-quant checkout (tool.uv.sources)
uv run pytest
```

The tests run on synthetic Barra-shaped data (`tests/synthetic.py`), so they don't need the BYU data store. They include the handoff's canaries: perfect foresight, same-day leak, and random alpha.

## Publishing

1. Bump the version: `uv version x.y.z`
2. Commit and tag: `git tag vx.y.z`
3. Push: `git push --tags`

Build with `uv build --no-sources` so the local sf-quant path isn't used.
