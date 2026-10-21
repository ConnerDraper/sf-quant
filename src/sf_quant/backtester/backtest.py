import datetime as dt
import os
import warnings
from collections.abc import Sequence

import polars as pl
import tqdm
from joblib import Parallel, delayed

from sf_quant.data.assets import load_assets
from sf_quant.data.benchmark import load_benchmark
from sf_quant.data.covariance_matrix import construct_factor_model_components
from sf_quant.optimizer.constraints import DEFAULT_CONSTRAINTS, Constraint
from sf_quant.optimizer.optimizers import active_mve_optimizer
from sf_quant.schema.active_portfolio_schema import ActivePortfolioSchema

REQUIRED_COLUMNS = ("date", "barrid", "alpha")


def attach_beta_and_benchmark(data: pl.DataFrame) -> pl.DataFrame:
    """
    Join ``predicted_beta`` and ``benchmark_weight`` onto ``data`` where absent.

    A column the caller already supplied is used as given. Otherwise:

    - ``predicted_beta`` comes from ``load_assets(in_universe=True)``, joined
      *inner*: a row with no in-universe beta that date is dropped. This is
      also what restricts the panel to the Barra universe, whose risk model is
      the one every solve uses.
    - ``benchmark_weight`` comes from ``load_benchmark``, joined left, with
      names outside the benchmark set to 0 (``LongOnly`` then floors them at
      ``h >= 0``).

    Both are observed on the row's own date. Run this after
    :func:`~sf_quant.backtester.collapse` (``backtest`` does), so a lagged alpha
    is paired with the rebalance date's beta and benchmark, which is what a
    signal lag means.

    Dropped rows are reported with a warning, never silently.
    """
    start, end = data["date"].min(), data["date"].max()
    out = data

    if "predicted_beta" not in out.columns:
        betas = load_assets(
            start=start, end=end, columns=["date", "barrid", "predicted_beta"], in_universe=True
        ).drop_nulls("predicted_beta")
        out = out.join(betas, on=["date", "barrid"], how="inner")
        dropped = data.height - out.height
        if dropped:
            warnings.warn(
                f"dropped {dropped:,} of {data.height:,} rows with no in-universe "
                "predicted_beta on their date",
                stacklevel=3,
            )

    if "benchmark_weight" not in out.columns:
        benchmark = load_benchmark(start, end).select(
            "date", "barrid", pl.col("weight").alias("benchmark_weight")
        )
        out = out.join(benchmark, on=["date", "barrid"], how="left")

    return out.with_columns(pl.col("benchmark_weight").fill_null(0.0))


def _construct_portfolio(
    date_: dt.date,
    data: pl.DataFrame,
    constraints: Sequence[Constraint],
    target_active_risk: float,
) -> pl.DataFrame:
    """Solve one date. Row order is set by ``barrid`` and shared with the risk model."""
    subset = data.filter(pl.col("date").eq(date_)).sort("barrid")
    barrids = subset["barrid"].to_list()

    factor_exposures, factor_covariance, specific_risk = (
        construct_factor_model_components(date_, barrids)
    )

    portfolio = active_mve_optimizer(
        ids=barrids,
        alphas=subset["alpha"].to_numpy(),
        factor_exposures=factor_exposures,
        factor_covariance=factor_covariance,
        specific_risk=specific_risk,
        benchmark_weights=subset["benchmark_weight"].to_numpy(),
        target_active_risk=target_active_risk,
        betas=subset["predicted_beta"].to_numpy(),
        constraints=constraints,
    )

    return portfolio.with_columns(
        pl.lit(date_).alias("date"),
        subset["benchmark_weight"],
    ).select("date", "barrid", "active_weight", "benchmark_weight", "active_risk")


def backtest(
    data: pl.DataFrame,
    constraints: Sequence[Constraint] = DEFAULT_CONSTRAINTS,
    *,
    target_active_risk: float = 0.05,
    n_cpus: int | None = None,
) -> pl.DataFrame:
    """
    Construct one active portfolio per distinct date in ``data``.

    Each date is solved independently against its own risk model. There is no
    schedule logic here: ``data`` is exactly the set of rows to optimize, so a
    daily panel gives daily rebalancing. For coarser rebalancing or a signal
    lag, reduce the panel with
    :func:`sf_quant.backtester.collapse` first.

    Parameters
    ----------
    data : pl.DataFrame
        Rows to optimize. Must contain ``date``, ``barrid`` and ``alpha``.
        ``predicted_beta`` and ``benchmark_weight`` are joined on from the data
        lake when absent (see :func:`attach_beta_and_benchmark`), which also
        drops rows outside the Barra universe; supply them to use your own.
    constraints : Sequence[Constraint], default DEFAULT_CONSTRAINTS
        Active-space constraints, defaulting to the fund mandate of
        ``ZeroInvestment``, ``ZeroBeta`` and ``LongOnly``.
    target_active_risk : float, default 0.05
        Annualized tracking-error budget, in decimal. This, not a gamma, sizes
        every portfolio; there is deliberately no gamma parameter. If you are
        passing or tuning one, you are doing it wrong (Manual §3.2).
    n_cpus : int, optional
        Worker processes, capped at the number of distinct dates. ``None`` uses
        ``os.cpu_count()``. Pass ``1`` to run in-process, which is what you
        want under a debugger. Dates are solved in parallel with joblib's loky
        process pool; see ``research/parallel-backends`` for why it beat Ray
        and a thread pool.

    Returns
    -------
    pl.DataFrame
        Validated against ``ActivePortfolioSchema``, with columns:

        - ``date`` : date, the rebalance date.
        - ``barrid`` : str, asset identifier.
        - ``active_weight`` : float, the overlay ``h``. Total weight is
          ``active_weight + benchmark_weight``.
        - ``benchmark_weight`` : float, ``b`` observed on ``date``.
        - ``active_risk`` : float, realized annualized active risk, constant
          within a date.

    Examples
    --------
    >>> import datetime as dt
    >>> import polars as pl
    >>> import sf_quant.data as sfd
    >>> import sf_quant.backtester as sfb
    >>> start, end = dt.date(2024, 1, 1), dt.date(2024, 12, 31)
    >>> dates = sfd.load_trading_days(start, end)
    >>> grid = sfb.collapse(
    ...     pl.scan_parquet("alphas.parquet"),
    ...     dates=dates,
    ...     rebalance_period=21,
    ...     lag=5,
    ... )
    >>> weights = sfb.backtest(grid.collect(), target_active_risk=0.05)
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    if data.is_empty():
        raise ValueError("data is empty; nothing to optimize")
    data = attach_beta_and_benchmark(data)
    if data.is_empty():
        raise ValueError("no rows left after joining the in-universe predicted_beta")

    # Each task gets only its own date's rows, so workers aren't sent the
    # whole panel once per date.
    parts = {key[0]: part for key, part in data.partition_by("date", as_dict=True).items()}
    dates = sorted(parts)
    n_cpus = min(len(dates), n_cpus or os.cpu_count())

    if n_cpus == 1:
        portfolio_list = [
            _construct_portfolio(date_, parts[date_], constraints, target_active_risk)
            for date_ in tqdm.tqdm(dates, desc="Computing portfolios")
        ]
    else:
        # joblib's default backend is loky: a process pool, so each solve's
        # Python-side setup runs in parallel too. parallel_config() can swap it.
        tasks = Parallel(n_jobs=n_cpus, return_as="generator")(
            delayed(_construct_portfolio)(date_, parts[date_], constraints, target_active_risk)
            for date_ in dates
        )
        portfolio_list = list(
            tqdm.tqdm(tasks, total=len(dates), desc=f"Computing portfolios with {n_cpus} workers")
        )

    return ActivePortfolioSchema.validate(
        pl.concat(portfolio_list).sort("date", "barrid")
    )
