"""Chart a saved backtest run without recomputing it.

Usage: python plot_backtest.py [run_dir]   (default out/backtest_signal; any
folder written by result.save or sf_backtest run works)

Writes to out/plot_backtest/:
  active_return.png       cumulative active return, gross and net of each cost level
  factor_vs_specific.png  cumulative active return from Barra factors vs. specific
  risk_and_transfer.png   ex-ante and realized tracking error, and transfer coefficient
  top_factors.png         annualized active return from each of the largest factors
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import polars as pl

run = Path(sys.argv[1] if len(sys.argv) > 1 else "out/backtest_signal")
out = Path("out/plot_backtest")
out.mkdir(parents=True, exist_ok=True)

returns = pl.read_parquet(run / "returns.parquet").sort("date")
diagnostics = pl.read_parquet(run / "diagnostics.parquet").sort("formation_date")
factor_returns = pl.read_parquet(run / "factor_returns.parquet")


def cumulative(column: str) -> pl.Series:
    return returns[column].cum_sum() * 100


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 5))
net = [c for c in returns.columns if c.startswith("net_active_") and c != "net_active_0bps"]  # 0 bps = gross
for column in ["active", *net]:
    label = "gross" if column == "active" else "net of " + column.removeprefix("net_active_").replace("bps", " bps")
    ax.plot(returns["date"], cumulative(column), label=label)
ax.set(title="Cumulative active return", ylabel="%")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "active_return.png")

fig, ax = plt.subplots(figsize=(10, 5))
for column, label in [("active_factor", "Barra factors"), ("active_specific", "specific")]:
    ax.plot(returns["date"], cumulative(column), label=label)
ax.set(title="Where the active return came from", ylabel="cumulative %")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "factor_vs_specific.png")

realized_te = returns["active"].rolling_std(63) * 252**0.5 * 100
fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
top.plot(diagnostics["formation_date"], diagnostics["ex_ante_te"] * 100, label="ex-ante (at each rebalance)")
top.plot(returns["date"], realized_te, label="realized (rolling 3 months)", alpha=0.7)
top.axhline(5, color="black", linestyle="--", linewidth=0.8, label="5% target")
top.set(title="Tracking error", ylabel="% per year")
top.legend()
top.grid(alpha=0.3)
bottom.plot(diagnostics["formation_date"], diagnostics["transfer_coefficient"])
bottom.set(title="Transfer coefficient: how much of the signal survives the constraints", ylim=(0, 1))
bottom.grid(alpha=0.3)
save(fig, "risk_and_transfer.png")

n_days = returns.height
top_factors = (
    factor_returns.group_by("factor")
    .agg((pl.col("contribution").sum() * 252 / n_days * 100).alias("annual"))
    .sort(pl.col("annual").abs(), descending=True)
    .head(15)
    .sort("annual")
)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_factors["factor"].str.replace("USSLOWL_", ""), top_factors["annual"])
ax.axvline(0, color="grey", linewidth=0.8)
ax.set(title="Active return from the 15 largest factors", xlabel="% per year")
ax.grid(axis="x", alpha=0.3)
save(fig, "top_factors.png")
