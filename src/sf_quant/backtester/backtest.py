import datetime as dt
import os
from collections.abc import Sequence

import polars as pl
import tqdm
from joblib import Parallel, delayed

from sf_quant.data.covariance_matrix import construct_factor_model_components
from sf_quant.optimizer.constraints import DEFAULT_CONSTRAINTS, Constraint, ZeroBeta
from sf_quant.optimizer.optimizers import active_mve_optimizer
from sf_quant.schema.active_portfolio_schema import ActivePortfolioSchema

REQUIRED_COLUMNS = ("date", "barrid", "alpha", "benchmark_weight")


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
        betas=(
            subset["predicted_beta"].to_numpy()
            if "predicted_beta" in subset.columns
            else None
        ),
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
        Rows to optimize. Must contain ``date``, ``barrid``, ``alpha`` and
        ``benchmark_weight``; ``predicted_beta`` is required whenever
        ``ZeroBeta`` is among the constraints.
    constraints : Sequence[Constraint], default DEFAULT_CONSTRAINTS
        Active-space constraints, defaulting to the fund mandate of
        ``ZeroInvestment``, ``ZeroBeta`` and ``LongOnly``.
    target_active_risk : float, default 0.05
        Annualized tracking-error budget, in decimal.
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
    if "predicted_beta" not in data.columns and any(
        isinstance(c, ZeroBeta) for c in constraints
    ):
        raise ValueError(
            "ZeroBeta is in constraints but data has no 'predicted_beta' column; "
            "either add the column or drop ZeroBeta from constraints"
        )
    if data.is_empty():
        raise ValueError("data is empty; nothing to optimize")

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
