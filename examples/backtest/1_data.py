"""Build a momentum alpha panel: date, barrid, alpha.

Grinold-Kahn alpha = IC * specific risk * z-score (manual 4.6). The
backtester loads the benchmark, risk model, and betas itself, so the panel
carries alphas only. A production alpha builder should also residualize the
score against Barra exposures before standardizing (handoff, "Math spec").
"""

import datetime as dt

import polars as pl
import sf_quant.data as sfd

start = dt.date(2000, 1, 1)
end = dt.date(2024, 12, 31)
signal_name = "momentum"
price_filter = 5
IC = 0.05

data = sfd.load_assets(
    start=start,
    end=end,
    columns=["date", "barrid", "price", "return", "specific_risk"],
    in_universe=True,
).with_columns(pl.col("return", "specific_risk").truediv(100))

alphas = (
    data.sort("barrid", "date")
    .with_columns(
        pl.col("return").log1p().rolling_sum(230).shift(21).over("barrid").alias(signal_name),
        pl.col("price").shift(1).over("barrid").alias("prev_price"),
    )
    .filter(
        pl.col("prev_price").gt(price_filter),
        pl.col(signal_name).is_not_null(),
        pl.col("specific_risk").is_not_null(),
    )
    .with_columns(
        pl.col(signal_name)
        .sub(pl.col(signal_name).mean())
        .truediv(pl.col(signal_name).std())
        .over("date")
        .alias("score")
    )
    .select("date", "barrid", (IC * pl.col("specific_risk") * pl.col("score")).alias("alpha"))
    .sort("date", "barrid")
)

alphas.write_parquet(f"{signal_name}_alphas.parquet")
print(alphas)
