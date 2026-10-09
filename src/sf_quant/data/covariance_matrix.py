import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl
from .exposures import load_exposures_by_date
from .covariances import load_covariances_by_date
from .assets import load_assets_by_date
from ._factors import factors


@dataclass(frozen=True)
class RiskModel:
    """
    Barra risk model for one date, restricted to assets with complete data.

    All quantities are annualized and in decimal units.

    Attributes
    ----------
    ids : list[str]
        Barrids with exposures and specific risk, in request order.
    factor_exposures : np.ndarray
        B, shape (n_assets, n_factors).
    factor_covariance : np.ndarray
        F, shape (n_factors, n_factors).
    specific_variance : np.ndarray
        Specific variance, shape (n_assets,).
    factor_names : list[str]
        Column names of ``factor_exposures``.
    missing_ids : list[str]
        Requested barrids with no exposure row or no positive specific risk.
    """

    ids: list[str]
    factor_exposures: np.ndarray
    factor_covariance: np.ndarray
    specific_variance: np.ndarray
    factor_names: list[str]
    missing_ids: list[str]


def load_risk_model(date_: dt.date, barrids: list[str]) -> RiskModel:
    """
    Load the Barra risk model for the given assets on one date.

    Unlike :func:`construct_factor_model_components`, this never fills
    missing risk data with zero. Assets without an exposure row or without a
    positive specific risk are dropped and listed in ``missing_ids``, so the
    caller decides how to treat them instead of the optimizer seeing them as
    riskless.

    Parameters
    ----------
    date_ : datetime.date
        Risk model date.
    barrids : list of str
        Requested assets. Must be unique.

    Returns
    -------
    RiskModel
        Risk model components in annualized decimal units.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> rm = sfd.load_risk_model(dt.date(2024, 1, 3), ['USA06Z1', 'USA0771'])
    >>> rm.factor_exposures.shape
    (2, 76)
    """
    request = pl.DataFrame({"barrid": barrids})
    if request["barrid"].is_duplicated().any():
        raise ValueError("barrids must be unique.")

    exposures = (
        load_exposures_by_date(date_)
        .select(["barrid"] + factors)
        .with_columns(pl.lit(True).alias("_has_exposures"))
    )
    specific = load_assets_by_date(
        date_, in_universe=False, columns=["barrid", "specific_risk"]
    )

    joined = (
        request.join(exposures, on="barrid", how="left")
        .join(specific, on="barrid", how="left")
        .with_columns(
            (
                pl.col("_has_exposures").is_not_null()
                & pl.col("specific_risk").is_not_null()
                & pl.col("specific_risk").gt(0)
            ).alias("_complete")
        )
    )

    complete = joined.filter(pl.col("_complete"))
    missing = joined.filter(~pl.col("_complete"))["barrid"].to_list()

    factor_cov = _construct_factor_covariance_matrix(date_).drop("factor_1").to_numpy()

    return RiskModel(
        ids=complete["barrid"].to_list(),
        factor_exposures=complete.select(factors).fill_null(0).to_numpy(),
        factor_covariance=factor_cov / 100**2,
        specific_variance=(complete["specific_risk"].to_numpy() / 100) ** 2,
        factor_names=list(factors),
        missing_ids=missing,
    )


def construct_factor_model_components(
    date_: dt.date, barrids: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the factor model components for the given assets.

    .. warning::
        Missing specific risk is filled with zero, which makes the asset look
        riskless to an optimizer, and assets missing from the exposures table
        are silently dropped. Prefer :func:`load_risk_model`.

    Returns
    -------
    tuple of np.ndarray
        - factor_exposures: shape (N, K)
        - factor_covariance: shape (K, K), in decimal space
        - specific_risk: shape (N,), variance in decimal space
    """
    exposures = _construct_factor_exposure_matrix(date_, barrids).drop("barrid").to_numpy()
    
    factor_cov = _construct_factor_covariance_matrix(date_).drop("factor_1").to_numpy()
    factor_cov = factor_cov / 100**2

    barrids_df = pl.DataFrame({"barrid": barrids})
    sr_df = load_assets_by_date(
        date_, in_universe=False, columns=["date", "barrid", "specific_risk"]
    )
    sr_df = barrids_df.join(sr_df, on=["barrid"], how="left").fill_null(0)
    
    specific_risk = np.power(sr_df["specific_risk"].to_numpy(), 2) / 100**2

    return exposures, factor_cov, specific_risk


def construct_covariance_matrix(date_: dt.date, barrids: list[str]) -> pl.DataFrame:
    """
    Construct the asset covariance matrix from a factor model.

    This function builds a covariance matrix for the given assets by
    combining factor exposures, factor covariances, and specific
    (idiosyncratic) risks. The resulting covariance matrix can be used
    as an input to mean-variance optimization.

    Parameters
    ----------
    date_ : datetime.date
        The date for which the covariance matrix is computed.
    barrids : list of str
        List of Barrid identifiers for the assets.

    Returns
    -------
    pl.DataFrame
        A square covariance matrix stored in a Polars DataFrame.

        - Rows and columns are indexed by ``barrid``.
        - Column ``barrid`` lists the asset identifiers.
        - Each subsequent column corresponds to the covariance of
          the row asset with the column asset.

    Notes
    -----
    The input factor covariance matrix is assumed to be positive
    semidefinite (PSD).

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> date_ = dt.date(2024, 1, 3)
    >>> barrids = ['USA06Z1', 'USA0771']
    >>> covariance_matrix = sfd.construct_covariance_matrix(
    ...     date_=date_,
    ...     barrids=barrids
    ... )
    >>> covariance_matrix
    shape: (2, 3)
    ┌─────────┬─────────────┬──────────────┐
    │ barrid  ┆ USA06Z1     ┆ USA0771      │
    │ ---     ┆ ---         ┆ ---          │
    │ str     ┆ f64         ┆ f64          │
    ╞═════════╪═════════════╪══════════════╡
    │ USA06Z1 ┆ 3224.338938 ┆ 697.641425   │
    │ USA0771 ┆ 697.641425  ┆ 11158.366868 │
    └─────────┴─────────────┴──────────────┘
    """
    # Load
    exposures_matrix = (
        _construct_factor_exposure_matrix(date_, barrids).drop("barrid").to_numpy()
    )
    covariance_matrix = (
        _construct_factor_covariance_matrix(date_).drop("factor_1").to_numpy()
    )
    idio_risk_matrix = (
        _construct_specific_risk_matrix(date_, barrids).drop("barrid").to_numpy()
    )

    # Compute covariance matrix
    covariance_matrix = (
        exposures_matrix @ covariance_matrix @ exposures_matrix.T + idio_risk_matrix
    )

    # Put in decimal space
    covariance_matrix /= 100**2

    # Package
    covariance_matrix = pl.DataFrame(
        {
            "barrid": barrids,
            **{id: covariance_matrix[:, i] for i, id in enumerate(barrids)},
        }
    )

    return covariance_matrix


def _construct_factor_exposure_matrix(
    date_: dt.date, barrids: list[str]
) -> pl.DataFrame:
    exp_mat = (
        load_exposures_by_date(date_)
        .drop("date")
        .filter(pl.col("barrid").is_in(barrids))
        .fill_null(0)
        .sort("barrid")
        .select(["barrid"] + factors)
    )

    return exp_mat


def _construct_factor_covariance_matrix(date_: dt.date) -> pl.DataFrame:
    # Load
    fc_df = load_covariances_by_date(date_).drop("date")

    # Sort headers and columns
    fc_df = fc_df.select(["factor_1"] + factors)

    fc_df = fc_df.filter(pl.col('factor_1').is_in(factors))

    fc_df = fc_df.sort("factor_1")

    # Convert from upper triangular to symetric
    utm = fc_df.drop("factor_1").to_numpy()
    cov_mat = np.where(np.isnan(utm), utm.T, utm)

    # Package
    cov_mat = pl.DataFrame(
        {
            "factor_1": factors,
            **{col: cov_mat[:, idx] for idx, col in enumerate(factors)},
        }
    )

    # Fill NaN (from Barra)
    cov_mat = cov_mat.fill_nan(0)

    return cov_mat


def _construct_specific_risk_matrix(date_: dt.date, barrids: list[str]) -> pl.DataFrame:
    # Barrids
    barrids_df = pl.DataFrame({"barrid": barrids})

    # Load
    sr_df = load_assets_by_date(
        date_, in_universe=False, columns=["date", "barrid", "specific_risk"]
    )

    # Filter
    sr_df = barrids_df.join(sr_df, on=["barrid"], how="left").fill_null(
        0
    )  # ask Brandon about this.

    # Convert vector to diagonal matrix
    diagonal = np.power(np.diag(sr_df["specific_risk"]), 2)

    # Package
    risk_matrix = pl.DataFrame(
        {
            "barrid": barrids,
            **{id: diagonal[:, i] for i, id in enumerate(barrids)},
        }
    )

    return risk_matrix
