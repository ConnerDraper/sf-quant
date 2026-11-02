import numpy as np
import polars as pl
import os
import tqdm
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from sf_quant.data.covariance_matrix import construct_factor_model_components
from sf_quant.data.spreads import _trade_half_spreads
from sf_quant.optimizer.optimizers import cost_aware_mve_optimizer, dynamic_mve_optimizer
from sf_quant.optimizer.constraints import Constraint
from sf_quant.performance.returns import _period_returns


def _construct_portfolio(
    date_: dt.date,
    subset: pl.DataFrame,
    constraints: list[Constraint],
    target_active_risk: float,
    active_weights: bool = True,
):
    """
    Construct a single optimized portfolio for a given date.

    Parameters
    ----------
    date_ : dt.date
        The date for which to construct the portfolio.
    subset : pl.DataFrame
        Single-date slice with columns ``date``, ``barrid``, ``alpha``, and
        optionally ``predicted_beta`` and ``benchmark_weight``.
    constraints : list[Constraint]
        Portfolio constraints to enforce during optimization.
    target_active_risk : float
        Target annualized active risk for dual-variable gamma calibration.
    active_weights : bool, default True
        True returns pure active weights; False adds benchmark weights back to
        return total portfolio weights.

    Returns
    -------
    pl.DataFrame
        Portfolio weights for the given date, with columns ``date``, ``barrid``,
        ``weight``, ``gamma``, and ``active_risk``.
    """
    subset = subset.sort("barrid")
    barrids = subset["barrid"].to_list()
    alphas = subset["alpha"].to_numpy()

    betas = (
        subset["predicted_beta"].to_numpy()
        if "predicted_beta" in subset.columns
        else None
    )

    benchmark_weights = (
        subset["benchmark_weight"].to_numpy()
        if "benchmark_weight" in subset.columns
        else None
    )

    if benchmark_weights is None and not active_weights:
        raise ValueError(
            "Column 'benchmark_weight' must be present in data when active_weights is False."
        )

    (
        factor_exposures,
        factor_covariance,
        specific_risk,
    ) = construct_factor_model_components(date_, barrids)

    portfolio = dynamic_mve_optimizer(
        ids=barrids,
        alphas=alphas,
        factor_exposures=factor_exposures,
        factor_covariance=factor_covariance,
        specific_risk=specific_risk,
        constraints=constraints,
        target_active_risk=target_active_risk,
        betas=betas,
        benchmark_weights=benchmark_weights,
        active_weights=active_weights,
    )

    portfolio = portfolio.with_columns(pl.lit(date_).alias("date")).select(
        "date", "barrid", "weight", "gamma", "active_risk"
    )

    return portfolio


def dynamic_backtest_parallel(
    data: pl.DataFrame,
    constraints: list[Constraint],
    target_active_risk: float = 0.05,
    n_cpus: int | None = None,
    active_weights: bool = True,
    cost_aversion: float = 0.0,
) -> pl.DataFrame:
    """
    Run a parallelized backtest of portfolio optimization using a thread pool.

    Distributes single-date portfolio construction tasks across worker
    threads, solving mean-variance optimization problems in active space with
    explicitly supplied ``Constraint`` objects.

    With ``cost_aversion`` above 0 the backtest charges each trade half the
    bid-ask spread in the objective (:func:`sf_quant.optimizer.cost_aware_mve_optimizer`),
    so it only trades a name when the expected gain beats the cost. Each date's
    portfolio then depends on the one before, so the dates are solved in order
    rather than in parallel.

    Parameters
    ----------
    data : pl.DataFrame
        Input dataset containing:

        - ``date`` : datetime-like, the date of each observation.
        - ``barrid`` : str, unique identifier for each asset.
        - ``alpha`` : float, expected return (alpha) for each asset.
        - ``predicted_beta`` : float, optional, required if ``ZeroBeta`` or
          ``UnitBeta`` is included in ``constraints``.
        - ``benchmark_weight`` : float, optional, required if total-space
          constraints (such as ``LongOnly``, ``NoBuyingOnMargin``, or
          ``GrossLeverage``) are used relative to a benchmark or if
          ``active_weights`` is False.

    constraints : list[Constraint]
        List of portfolio constraints to enforce (e.g.,
        ``[ZeroInvestment(), LongOnly(), ZeroBeta()]`` for the Silver Fund
        mandate, ``[ZeroInvestment(), GrossLeverage(1.6)]`` for 130/30, or
        ``[ZeroInvestment()]`` for unconstrained signal research).
    target_active_risk : float, default 0.05
        Target annualized active risk cap (e.g., 0.05 for 5%).
    n_cpus : int, optional
        Number of worker threads. If ``None``, defaults to
        ``os.cpu_count()`` capped at the number of unique dates.
    active_weights : bool, default True
        True (default) outputs active weights :math:`h`. False outputs total
        portfolio weights :math:`w = h + b`.
    cost_aversion : float, default 0.0
        :math:`\\lambda`, the weight on half-spread trading costs in the
        objective. 0 gives the cost-blind portfolios, solved in parallel.
        Above 0, the cost of a trade is measured against what the portfolio
        holds before trading: the previous date's total weights drifted with
        their returns, less the date's benchmark weights. Half-spreads come
        from :func:`sf_quant.data.load_half_spreads` at the previous trading
        day's close, when the weights are put on; a name without one trades
        free. The first date is the starting portfolio and pays nothing.
        Needs the assets table for returns and spreads.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing optimized portfolio weights across all
        backtest dates, with columns ``date``, ``barrid``, ``weight``,
        ``gamma``, and ``active_risk``. ``gamma`` is null when
        ``cost_aversion`` is above 0: the cost term changes the objective, so
        no risk aversion alone reproduces the portfolio.
    """
    if not active_weights and "benchmark_weight" not in data.columns:
        raise ValueError(
            "Column 'benchmark_weight' must be present in data when active_weights is False."
        )

    partitions = {
        df["date"][0]: df for df in data.partition_by("date")
    }
    dates = sorted(partitions.keys())

    if not dates:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "barrid": pl.String,
                "weight": pl.Float64,
                "gamma": pl.Float64,
                "active_risk": pl.Float64,
            }
        )

    if cost_aversion > 0:
        return _cost_aware_backtest(
            partitions=partitions,
            dates=dates,
            constraints=constraints,
            target_active_risk=target_active_risk,
            active_weights=active_weights,
            cost_aversion=cost_aversion,
        )

    n_cpus = n_cpus or os.cpu_count() or 1
    n_cpus = min(len(dates), n_cpus)

    with ThreadPoolExecutor(max_workers=n_cpus) as executor:
        portfolio_futures = executor.map(
            lambda date_: _construct_portfolio(
                date_=date_,
                subset=partitions[date_],
                constraints=constraints,
                target_active_risk=target_active_risk,
                active_weights=active_weights,
            ),
            dates,
        )
        portfolio_list = list(
            tqdm.tqdm(
                portfolio_futures,
                total=len(dates),
                desc=f"Computing portfolios with {n_cpus} threads",
            )
        )

    return pl.concat(portfolio_list)

def _cost_aware_backtest(
    partitions: dict[dt.date, pl.DataFrame],
    dates: list[dt.date],
    constraints: list[Constraint],
    target_active_risk: float,
    active_weights: bool,
    cost_aversion: float,
) -> pl.DataFrame:
    """Solve each date in order, charging trades against the previous date's drifted holdings."""
    barrids = pl.concat([partitions[d].select("barrid") for d in dates])["barrid"].unique()
    period_returns = _period_returns(barrids, pl.Series(dates))
    half_spreads = _trade_half_spreads(pl.Series(dates), barrids.to_list())

    portfolio_list = []
    holdings = None  # previous date's total weights: barrid, total
    for date_ in tqdm.tqdm(dates, desc="Computing cost-aware portfolios in date order"):
        subset = partitions[date_].sort("barrid")
        ids = subset["barrid"]
        benchmark_weights = (
            subset["benchmark_weight"].to_numpy()
            if "benchmark_weight" in subset.columns
            else None
        )
        held_benchmark = benchmark_weights if benchmark_weights is not None else np.zeros(len(ids))

        if holdings is None:
            # The first date is the starting portfolio, not a trade from cash.
            previous_weights = np.zeros(len(ids))
        else:
            drifted = _drift(holdings, period_returns.filter(pl.col("date") == previous_date))
            previous_weights = (
                pl.DataFrame({"barrid": ids})
                .join(drifted, on="barrid", how="left")["drifted"]
                .fill_null(0.0)
                .to_numpy()
                - held_benchmark
            )

        spreads = (
            pl.DataFrame({"barrid": ids})
            .join(half_spreads.filter(pl.col("date") == date_), on="barrid", how="left")["half_spread"]
            .fill_null(0.0)
            .to_numpy()
        )

        factor_exposures, factor_covariance, specific_risk = construct_factor_model_components(
            date_, ids.to_list()
        )
        portfolio = cost_aware_mve_optimizer(
            ids=ids.to_list(),
            alphas=subset["alpha"].to_numpy(),
            factor_exposures=factor_exposures,
            factor_covariance=factor_covariance,
            specific_risk=specific_risk,
            constraints=constraints,
            previous_weights=previous_weights,
            half_spreads=spreads,
            cost_aversion=cost_aversion,
            target_active_risk=target_active_risk,
            betas=subset["predicted_beta"].to_numpy() if "predicted_beta" in subset.columns else None,
            benchmark_weights=benchmark_weights,
        )

        active = portfolio["weight"].to_numpy()
        holdings = pl.DataFrame({"barrid": ids, "total": active + held_benchmark})
        previous_date = date_
        portfolio_list.append(
            pl.DataFrame(
                {
                    "date": [date_] * len(ids),
                    "barrid": ids,
                    "weight": active if active_weights else active + held_benchmark,
                    "gamma": pl.Series([None] * len(ids), dtype=pl.Float64),
                    "active_risk": portfolio["active_risk"],
                }
            )
        )

    return pl.concat(portfolio_list)


def _drift(holdings: pl.DataFrame, period_returns: pl.DataFrame) -> pl.DataFrame:
    """Total weights after a holding period's returns: w (1 + r) / (1 + w'r)."""
    return (
        holdings.join(period_returns.select("barrid", "return"), on="barrid", how="left")
        .with_columns(pl.col("return").fill_null(0.0))
        .select(
            "barrid",
            pl.col("total")
            .mul(pl.col("return").add(1))
            .truediv(pl.col("total").mul(pl.col("return")).sum().add(1))
            .alias("drifted"),
        )
    )
