"""Plot cumulative gross and net active return from a saved run."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

run_dir = "runs/momentum"
returns = pl.read_parquet(f"{run_dir}/returns.parquet").sort("date")
summary = json.loads(Path(f"{run_dir}/summary.json").read_text())

for column in ["active", "net_active_10bps"]:
    plt.plot(returns["date"], returns[column].log1p().cum_sum() * 100, label=column)

plt.title(
    f"Momentum: IR {summary['ir_gross']:.2f} gross, {summary['ir_net']['10']:.2f} net of 10 bps; "
    f"TE {summary['realized_te']:.1%}"
)
plt.ylabel("Cumulative log active return (%)")
plt.legend()
plt.savefig(f"{run_dir}/active_return.png")
