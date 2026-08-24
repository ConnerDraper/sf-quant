from __future__ import annotations

import numpy as np
import polars as pl
import tqdm

from sf_quant.costs import CostModel
from sf_quant.data.covariance_matrix import construct_factor_model_components
from sf_quant.optimizer.optimizers import mve_optimizer
from sf_quant.optimizer.constraints import Constraint
from sf_quant.schema.portfolio_schema import PortfolioSchema


def backtest_sequential(
    data: pl.DataFrame,
    constraints: list[Constraint],
    gamma: float = 2,
    cost_model: CostModel | None = None,
) -> pl.DataFrame:
    """
    Run a sequential backtest of portfolio optimization.

    This function iterates through unique dates in the input dataset,
    constructs the covariance matrix for each date, and solves a mean-
    variance optimization (MVE) problem subject to user-specified
    constraints. The optimized portfolios are concatenated into a
    single Polars DataFrame.

    When a ``cost_model`` is provided, the optimizer penalizes turnover
    using estimated transaction costs, and the previous day's optimal
    weights are used both as the turnover reference and as a warm-start
    for the solver.

    Parameters
    ----------
    data : pl.DataFrame
        Input dataset containing at least the following columns:

        - ``date`` : datetime-like, the date of each observation.
        - ``barrid`` : str, unique identifier for each asset.
        - ``alpha`` : float, expected return (alpha) for each asset.
        - ``predicted_beta`` : float, optional, factor exposures for constraints.

    constraints : list[Constraint]
        A list of constraints (e.g., :class:`~sf_quant.optimizer.constraints.FullInvestment`,
        :class:`~sf_quant.optimizer.constraints.LongOnly`) to be enforced in the optimization.

    gamma : float, optional
        Risk aversion parameter. Higher values penalize portfolio variance more strongly.
        Default is 2.

    cost_model : CostModel, optional
        A transaction cost estimator. When provided, enables cost-aware optimization
        with warm-starting from the previous period's weights.

    Returns
    -------
    pl.DataFrame
        A PortfolioSchema-validated Polars DataFrame containing optimized
        portfolio weights for each date, with the following columns:

        - ``date`` : datetime, backtest date.
        - ``barrid`` : str, asset identifier.
        - ``weight`` : float, optimized portfolio weight.

    Notes
    -----
    - The optimization is solved using :func:`~sf_quant.optimizer.optimizers.mve_optimizer`.
    - The factor model components are constructed using
      :func:`~sf_quant.data.covariance_matrix.construct_factor_model_components`.
    - Results are validated against :class:`~sf_quant.schema.portfolio_schema.PortfolioSchema`.
    - When ``cost_model`` is set, the backtest must run sequentially because each
      period depends on the previous period's weights.

    See Also
    --------
    backtest_parallel : Parallelized version using Ray for faster execution.
    dynamic_backtest_parallel : Parallel version with active risk calibration.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import sf_quant.backtester as sfb
    >>> import sf_quant.optimizer as sfo
    >>> import polars as pl
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 1, 10)
    >>> columns = [
    ...     'date',
    ...     'barrid',
    ... ]
    >>> data = (
    ...     sfd.load_assets(
    ...         start=start,
    ...         end=end,
    ...         in_universe=True,
    ...         columns=columns
    ...     )
    ...     .with_columns(
    ...         pl.lit(0).alias('alpha')
    ...     )
    ... )
    >>> constraints = [
    ...     sfo.FullInvestment()
    ... ]
    >>> weights = sfb.backtest_sequential(
    ...     data=data,
    ...     constraints=constraints,
    ...     gamma=2,
    ... )
    >>> weights.head()
    shape: (5, 3)
    ┌────────────┬─────────┬───────────┐
    │ date       ┆ barrid  ┆ weight    │
    │ ---        ┆ ---     ┆ ---       │
    │ date       ┆ str     ┆ f64       │
    ╞════════════╪═════════╪═══════════╡
    │ 2024-01-02 ┆ USA06Z1 ┆ -0.000639 │
    │ 2024-01-03 ┆ USA06Z1 ┆ -0.000644 │
    │ 2024-01-04 ┆ USA06Z1 ┆ -0.000641 │
    │ 2024-01-05 ┆ USA06Z1 ┆ -0.000666 │
    │ 2024-01-08 ┆ USA06Z1 ┆ -0.000666 │
    └────────────┴─────────┴───────────┘
    """
    dates = data["date"].unique().sort().to_list()
    portfolio_list = []
    prev_weights: np.ndarray | None = None

    for date_ in tqdm.tqdm(dates, "Running backtest"):
        subset = data.filter(pl.col("date").eq(date_)).sort("barrid")

        barrids = subset["barrid"].to_list()
        alphas = subset["alpha"].to_numpy()

        betas = (
            subset["predicted_beta"].to_numpy()
            if "predicted_beta" in subset.columns
            else None
        )

        (
            factor_exposures,
            factor_covariance,
            specific_risk,
        ) = construct_factor_model_components(date_, barrids)

        costs = cost_model.estimate(date_, barrids) if cost_model else None

        portfolio = mve_optimizer(
            ids=barrids,
            alphas=alphas,
            factor_exposures=factor_exposures,
            factor_covariance=factor_covariance,
            specific_risk=specific_risk,
            gamma=gamma,
            constraints=constraints,
            betas=betas,
            prev_weights=prev_weights,
            costs=costs,
            warm_start=prev_weights,
        )

        prev_weights = portfolio["weight"].to_numpy()

        portfolio = portfolio.with_columns(pl.lit(date_).alias("date")).select(
            "date", "barrid", "weight"
        )

        portfolio_list.append(portfolio)

    return PortfolioSchema.validate(pl.concat(portfolio_list).sort("barrid", "date"))
