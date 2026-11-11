"""Duplicate assay correlation analysis."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from analysis.qc.flags import FAIL_DUPLICATE_ASSAY

logger = logging.getLogger(__name__)


def _no_duplicates(corr_threshold: float) -> dict:
    """Returns the result dict for a run with nothing to compare."""
    return {
        "assay_olink_map": {},
        "olink_ids_to_remove": [],
        "olink_metrics": [],
        "corr_plots": {},
        "corr_threshold": corr_threshold,
        "low_corr": [],
    }


def assay_duplicates(
    qc_instance,
    data_col: str | None = None,
    corr_threshold: float = 0.8,
) -> dict[str, list[str]]:
    """Correlates duplicate assays and fails all but the best-detected OlinkID.

    Duplicates are one assay name with several OlinkIDs. Each copy set keeps the
    OlinkID with the highest detection rate (ties go to the lexicographically first
    OlinkID); the others get ``FAIL_Duplicate_Assay``. The Spearman correlation between
    copies only labels the set as low-correlation; the keep-or-remove rule applies
    whether or not the copies share enough values for a correlation.

    Args:
        qc_instance: An OlinkQC instance with detection statistics.
        data_col: Data column for the correlation; defaults to ``qc_instance.data_col``.
        corr_threshold: Mean correlation below which an assay is low-correlation.

    Returns:
        A dict (empty values when there is no OlinkID column or no duplicate assay)
        with ``assay_olink_map``, ``olink_ids_to_remove``, ``olink_metrics`` (CV,
        detection rate and kept flag per OlinkID), ``corr_plots`` (the paired values of
        each set), ``corr_threshold`` and ``low_corr``.

    Raises:
        ValueError: If the detection statistics are missing or lack a detection rate
            column, or no copy of a duplicate assay has detection statistics.
    """
    if data_col is None:
        data_col = qc_instance.data_col

    if not qc_instance._has_column("OlinkID"):
        logger.warning("OlinkID not found, skipping duplicate assay check")
        return _no_duplicates(corr_threshold)

    assay_info = (
        qc_instance.analyte_metadata.filter(pl.col("AssayType") == "assay")
        .select(["Assay", "OlinkID"])
        .unique()
        .sort(["Assay", "OlinkID"])
        .collect()
    )
    dup_assays = assay_info.filter(pl.col("Assay").is_duplicated())

    if len(dup_assays) == 0:
        logger.info("No duplicate assays found")
        return _no_duplicates(corr_threshold)

    assay_olink_map = {
        row["Assay"]: row["OlinkID"]
        for row in dup_assays.group_by("Assay", maintain_order=True)
        .agg(pl.col("OlinkID"))
        .iter_rows(named=True)
    }
    logger.info(f"Found {len(assay_olink_map)} assays with duplicates")

    # Filtered to the duplicate OlinkIDs before collecting, so the whole sample matrix
    # is never materialized.
    dup_olink_ids = dup_assays["OlinkID"].to_list()
    sample_data = (
        qc_instance.samples.filter(pl.col("OlinkID").is_in(dup_olink_ids))
        .join(
            qc_instance.analyte_metadata.filter(pl.col("AssayType") == "assay").select(
                ["OlinkID", "Assay"]
            ),
            on="OlinkID",
            how="inner",
        )
        .select(["SampleID", "Assay", "OlinkID", data_col])
        .collect()
    )
    corr_plots, corr_stats, low_corr = {}, [], []

    olink_ids_to_remove = []
    olink_metrics = []

    if qc_instance.cv_data is not None and len(qc_instance.cv_data) > 0:
        cv_available = qc_instance.cv_data.join(
            qc_instance.analyte_metadata.select(["OlinkID", "Assay"]).collect(),
            on="OlinkID",
            how="left",
        ).select(["OlinkID", "Assay", "Intra_CV", "Inter_CV"])
    else:
        cv_available = None

    # The detection rates decide which copy stays, so they are required.
    det_results_dict = getattr(qc_instance, "detection_results", None)
    det_stats = None if det_results_dict is None else det_results_dict.get("det_stats")
    if det_stats is None:
        raise ValueError(
            "duplicate assays need the detection statistics of filter_assay_detection (qc.detection_results['det_stats']); none found"
        )
    protein_id_col = qc_instance.protein_id_col
    protein_name_col = qc_instance.protein_name_col

    # Overall_Det_Rate (all samples, no bridge) wins over a per-group Det_Rate.
    if "Overall_Det_Rate" in det_stats.columns:
        det_rates = (
            det_stats.select([protein_id_col, protein_name_col, "Overall_Det_Rate"])
            .unique(maintain_order=True)
            .rename({protein_name_col: "Assay", "Overall_Det_Rate": "Det_Rate"})
        )
    elif "Det_Rate" in det_stats.columns:
        det_rates = det_stats.select([protein_id_col, protein_name_col, "Det_Rate"]).rename(
            {protein_name_col: "Assay"}
        )
    else:
        raise ValueError(
            "duplicate assays: the detection statistics carry no Det_Rate or Overall_Det_Rate column"
        )

    for assay, olink_ids in assay_olink_map.items():
        assay_data = sample_data.filter(pl.col("Assay") == assay)
        corrs, pair_data = [], []

        for i in range(len(olink_ids)):
            for j in range(i + 1, len(olink_ids)):
                paired = (
                    assay_data.filter(pl.col("OlinkID") == olink_ids[i])
                    .select(["SampleID", pl.col(data_col).alias("x")])
                    .join(
                        assay_data.filter(pl.col("OlinkID") == olink_ids[j]).select(
                            ["SampleID", pl.col(data_col).alias("y")]
                        ),
                        on="SampleID",
                        how="inner",
                    )
                    .drop_nulls()
                )
                if len(paired) > 10:
                    x, y = paired["x"].to_numpy(), paired["y"].to_numpy()
                    valid_mask = ~(np.isnan(x) | np.isnan(y))
                    if valid_mask.sum() > 10:
                        r, p = spearmanr(x[valid_mask], y[valid_mask])
                        corrs.append(r)
                        pair_data.append((olink_ids[i], olink_ids[j], paired, r, p))

        if corrs:
            mean_r = np.mean(corrs)
            corr_stats.append({"Assay": assay, "Mean_r": mean_r, "n_pairs": len(corrs)})
            if mean_r < corr_threshold:
                low_corr.append(assay)

            corr_plots[assay] = {
                "assay": assay,
                "olink_ids": olink_ids,
                "pair_data": pair_data,
            }

        for oid in olink_ids:
            intra_cv_val = None
            inter_cv_val = None
            det_val = None

            if cv_available is not None:
                cv_row = cv_available.filter(pl.col("OlinkID") == oid)
                if len(cv_row) > 0:
                    intra_cv_val = cv_row["Intra_CV"][0]
                    inter_cv_val = cv_row["Inter_CV"][0]

            det_row = det_rates.filter(pl.col("OlinkID") == oid)
            if len(det_row) > 0:
                det_val = det_row["Det_Rate"][0]

            olink_metrics.append(
                {
                    "Assay": assay,
                    "OlinkID": oid,
                    "Intra_CV": intra_cv_val,
                    "Inter_CV": inter_cv_val,
                    "Det_Rate": det_val,
                    "Kept": False,  # Set below.
                }
            )

        assay_dets = det_rates.filter(pl.col("Assay") == assay)
        if len(assay_dets) == 0:
            raise ValueError(
                f"duplicate assay {assay}: none of its OlinkIDs has detection statistics"
            )
        # Ties go to the lexicographically first OlinkID so the pick does not depend on
        # row order.
        sorted_dets = assay_dets.sort(
            ["Det_Rate", "OlinkID"], descending=[True, False], nulls_last=True
        )
        keep_olink = sorted_dets["OlinkID"][0]
        remove_olinks = [oid for oid in olink_ids if oid != keep_olink]
        olink_ids_to_remove.extend(remove_olinks)

        for metric in olink_metrics:
            if metric["Assay"] == assay and metric["OlinkID"] == keep_olink:
                metric["Kept"] = True

        logger.info(
            f"{assay}: keeping {keep_olink} (highest detection rate), removing {len(remove_olinks)} duplicate(s)"
        )

    if olink_ids_to_remove:
        logger.info(
            f"Marking {len(olink_ids_to_remove)} duplicate OlinkIDs as FAIL based on detection rate"
        )
        qc_instance._flag_proteins(olink_ids_to_remove, FAIL_DUPLICATE_ASSAY)

    if corr_stats:
        stats_df = pl.DataFrame(corr_stats)
        med_corr = stats_df["Mean_r"].median()
        logger.info(f"Median r={med_corr:.3f} | Low (<{corr_threshold}): {len(low_corr)}")
    else:
        med_corr = None
        logger.warning(
            "No correlations calculated (insufficient overlapping samples with non-null values)"
        )
        logger.info(f"Low (<{corr_threshold}): 0")

    qc_instance.qc_results.update(
        {
            "duplicate_assays": len(assay_olink_map),
            "median_duplicate_corr": float(med_corr) if med_corr is not None else None,
            "low_corr_duplicates": len(low_corr),
            "duplicate_olinkids_removed": len(olink_ids_to_remove),
        }
    )

    return {
        "assay_olink_map": assay_olink_map,
        "olink_ids_to_remove": olink_ids_to_remove,
        "olink_metrics": olink_metrics,
        "corr_plots": corr_plots,
        "corr_threshold": corr_threshold,
        "low_corr": low_corr,
    }
