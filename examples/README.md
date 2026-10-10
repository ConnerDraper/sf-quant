# Examples

Four tasks you can copy, each a script that runs on the BYU cluster's data and writes its results to `out/<script name>/`. Nothing is printed, so they work as SLURM jobs.

| Script | Question it answers | Writes |
|---|---|---|
| [`my_signals.py`](my_signals.py) | How do I write a signal? (`LowSpecificVolatility`, used by the scripts below) | — |
| [`research_signal.py`](research_signal.py) | What does my signal earn and risk from each Barra factor, before and after orthogonalizing? | `factors.csv`, `ic.csv`, `factors.png` |
| [`backtest_signal.py`](backtest_signal.py) | How does it do under the fund's mandate? | `alphas.parquet` and the full run (returns, weights, diagnostics, factor files, `summary.json`) |
| [`plot_backtest.py`](plot_backtest.py) | What does a saved run look like? Reads `out/backtest_signal/` by default, or any run folder you pass | `active_return.png`, `factor_vs_specific.png`, `risk_and_transfer.png`, `top_factors.png` |
| [`add_signal_to_composite.py`](add_signal_to_composite.py) | Does adding it to the current signals raise the fund's IR? | `ir_comparison.csv`, `candidate_weight.csv`, `candidate_factors.png`, and each run |

All of them read their dates, rebalance frequency, and cost levels from [`config.yml`](config.yml).

## Setup

Copy `.env_example` to `.env` and fill in your paths, then run scripts from this folder:

```bash
cd examples
python research_signal.py
```

`plot_backtest.py` needs a run from `backtest_signal.py` (or `sf_backtest run`) first.

## Running on SLURM

Wrap any script in a batch job, for example:

```bash
#!/bin/bash
#SBATCH --nodes=1 --cpus-per-task=8 --mem=32G --time=06:00:00
cd ~/sf-quant/examples
python backtest_signal.py
```

For a backtest from an alphas file, the `sf_backtest` command writes and submits that job for you, using the `slurm` settings in `config.yml`:

```bash
sf_backtest submit config.yml --alphas out/backtest_signal/alphas.parquet --dry-run   # show the job script
sf_backtest submit config.yml --alphas out/backtest_signal/alphas.parquet             # submit it
```
