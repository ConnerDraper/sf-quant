import warnings

import polars as pl

from ._tables import barra_ids_table, crsp_security_info_table


class CrspTablesMissingError(FileNotFoundError):
    """The CRSP tables delisting returns need are not in the data lake."""


def load_barrid_permno_link() -> pl.DataFrame:
    """
    Map Barra barrids to CRSP permnos through historical CUSIPs.

    Barra's ``barra_ids`` table holds every barrid's CUSIPs with the dates each
    was in force; CRSP's security-information history (``crsp_security_info``,
    from CRSP Stock v2 ``stksecurityinfohist``) does the same for permnos. A
    barrid and a permno are linked over the dates their 8-character CUSIPs
    match and both records are in force, so a CUSIP change on either side is
    followed rather than broken.

    Returns
    -------
    pl.DataFrame
        ``barrid``, ``permno``, ``start_date``, ``end_date``: one row per
        linked span.

    Warns
    -----
    UserWarning
        If a barrid links to more than one permno on the same dates. Those
        rows are kept, so the ambiguity stays visible downstream.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> link = sfd.load_barrid_permno_link()   # doctest: +SKIP
    """
    barra = (
        barra_ids_table.scan()
        .filter(pl.col("asset_id_type") == "CUSIP")
        .select(
            "barrid",
            pl.col("asset_id").str.slice(0, 8).alias("cusip8"),
            pl.col("start_date").alias("barra_start"),
            pl.col("end_date").alias("barra_end"),
        )
    )
    try:
        crsp = crsp_security_info_table.scan().select(
            "permno",
            pl.col("cusip").str.slice(0, 8).alias("cusip8"),
            pl.col("secinfostartdt").alias("crsp_start"),
            pl.col("secinfoenddt").alias("crsp_end"),
        )
        crsp.collect_schema()
    except (FileNotFoundError, pl.exceptions.ComputeError) as error:
        raise CrspTablesMissingError(
            f"CRSP security information not found at {crsp_security_info_table._file_path()}; "
            "it links CRSP permnos to barrids."
        ) from error

    link = (
        barra.join(crsp, on="cusip8", how="inner")
        .select(
            "barrid",
            "permno",
            pl.max_horizontal("barra_start", "crsp_start").alias("start_date"),
            pl.min_horizontal("barra_end", "crsp_end").alias("end_date"),
        )
        .filter(pl.col("start_date") <= pl.col("end_date"))
        .unique()
        .sort("barrid", "start_date")
        .collect()
    )

    overlaps = (
        link.join(link, on="barrid", suffix="_other")
        .filter(
            pl.col("permno") != pl.col("permno_other"),
            pl.col("start_date") <= pl.col("end_date_other"),
            pl.col("start_date_other") <= pl.col("end_date"),
        )["barrid"]
        .n_unique()
    )
    if overlaps:
        warnings.warn(
            f"{overlaps:,} barrids link to more than one permno on overlapping dates",
            stacklevel=2,
        )
    return link
