"""Quality control flag checking for proteomics data."""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


def sample_qc_olink(
    qc_instance,
    remove_fail: bool = True,
    flag_warn: bool = True,
    missing_threshold: float = 0.2,
) -> pl.DataFrame:
    """
    Check and handle SampleQC flags.

    Args:
        qc_instance: OlinkQC instance
        remove_fail: Whether to mark failing samples
        flag_warn: Whether to mark warned samples
        missing_threshold: Maximum allowed missing fraction (0-1) before failing a sample

    Returns:
        DataFrame with sample QC status summary
    """
    qc_instance._print_section("SAMPLE & ASSAY QC FLAGS")

    # Get column names from instance
    sample_id_col = qc_instance.sample_id_col
    data_col = qc_instance.data_col

    # Sample QC analysis - one row per sample with missingness and warnings
    # Need to join with metadata to get SampleQC and AssayQC flags
    samples_with_qc = qc_instance.samples.join(
        qc_instance.sample_metadata.select([sample_id_col, "SampleQC"]),
        on=sample_id_col,
        how="left",
    ).join(
        qc_instance.analyte_metadata.select(
            [qc_instance.protein_id_col, "AssayQC", "AssayType"]
        ),
        on=qc_instance.protein_id_col,
        how="left",
    )

    qc_status = (
        samples_with_qc.filter(pl.col("AssayType") == "assay")
        .group_by(sample_id_col)
        .agg(
            [
                pl.col("SampleQC").first().alias("SampleQC"),
                (pl.col(data_col).is_null() | pl.col(data_col).is_nan())
                .sum()
                .alias("Missing_Count"),
                (pl.col("AssayQC") != "PASS").sum().alias("Warn_Count"),
                pl.len().alias("Total_Assays"),
            ]
        )
        .with_columns(
            [
                (pl.col("Missing_Count") / pl.col("Total_Assays")).alias(
                    "Missing_Frac"
                ),
                (pl.col("Warn_Count") / pl.col("Total_Assays")).alias("Warn_Frac"),
            ]
        )
        .sort(sample_id_col)
        .collect()
    )

    # Create summary for logging
    qc_summary = qc_status.group_by("SampleQC").agg(pl.len().alias("count"))
    qc_dict = {r["SampleQC"]: r["count"] for r in qc_summary.iter_rows(named=True)}
    logger.info(
        f"Sample QC - PASS: {qc_dict.get('PASS', 0)} | WARN: {qc_dict.get('WARN', 0)} | FAIL: {qc_dict.get('FAIL', 0)}"
    )

    fail_samples, warn_samples = [], []
    if remove_fail:
        fail_samples = qc_status.filter(pl.col("SampleQC") == "FAIL")[
            sample_id_col
        ].to_list()
        if fail_samples:
            logger.warning(f"Marking {len(fail_samples)} samples as FAIL")
            qc_instance.failed_samples.extend(fail_samples)
            qc_instance._track_sample_qc(fail_samples, "FAIL_Sample_QC")
            qc_instance.qc_results["samples_failed"] = len(fail_samples)

        # Additional check: fail samples with high missing rate
        high_missing = qc_status.filter(pl.col("Missing_Frac") > missing_threshold)[
            sample_id_col
        ].to_list()
        # Exclude samples already marked as FAIL to avoid double-counting
        high_missing = [s for s in high_missing if s not in fail_samples]
        if high_missing:
            logger.warning(
                f"marking {len(high_missing)} samples with >{missing_threshold*100:.0f}% missing data as FAIL"
            )
            qc_instance.failed_samples.extend(high_missing)
            qc_instance._track_sample_qc(high_missing, "FAIL_Missing_Rate")
            fail_samples.extend(high_missing)
            qc_instance.qc_results["samples_failed"] = len(fail_samples)

    if flag_warn:
        warn_samples = qc_status.filter(pl.col("SampleQC") == "WARN")[
            sample_id_col
        ].to_list()
        if warn_samples:
            logger.warning(f"Marking {len(warn_samples)} samples as WARN")
            qc_instance.warned_samples.extend(warn_samples)
            qc_instance._track_sample_qc(warn_samples, "WARN_Sample_QC")
            qc_instance.qc_results["samples_warned"] = len(warn_samples)

    # Update qc_dict to reflect actual FAIL/WARN counts including high missing rate
    total_fail = len(fail_samples)
    total_warn = len(warn_samples)
    total_pass = (
        qc_dict.get("PASS", 0)
        + qc_dict.get("WARN", 0)
        + qc_dict.get("FAIL", 0)
        - total_fail
        - total_warn
    )

    qc_dict = {
        "PASS": total_pass,
        "WARN": total_warn,
        "FAIL": total_fail,
    }

    # Store sample QC data for combined plot and summary
    qc_instance._sample_qc_data = {
        "qc_dict": qc_dict,
        "fail_samples": fail_samples,
        "warn_samples": warn_samples,
        "missing_threshold": missing_threshold,
    }

    # Store sample QC status for inclusion in summary
    qc_instance._sample_qc_status = qc_status.select(
        [sample_id_col, "Warn_Frac", "Missing_Frac"]
    )

    return qc_status


def sample_qc_soma(
    qc_instance,
    remove_fail: bool = True,
    flag_warn: bool = True,
    mad_threshold: float = 6.0,
    fc_threshold: float = 5.0,
    outlier_fraction: float = 0.05,
) -> pl.DataFrame:
    """Check and handle RowCheck flags (sample QC) in SomaScan data.

    Performs multiple sample QC checks:
    - RowCheck: SomaLogic's sample QC flag (pass/flag samples with issues)
    - RFU outliers: Samples where >= 5% of RFU measurements exceed both 6 MADs and 5x fold-change from median
    - Low volume: Samples flagged with "Low Volume" in SampleNotes column (if available)

    Samples are categorized as:
    - FAIL: Failed both RowCheck AND RFU outlier checks
    - WARN: Failed only one check, OR flagged as low volume
    - PASS: Passed all checks

    Args:
        qc_instance: SomaQC instance
        remove_fail: Whether to mark failing samples
        flag_warn: Whether to mark warned samples
        mad_threshold: MAD threshold for outlier detection
        fc_threshold: Fold-change threshold for outlier detection
        outlier_fraction: Fraction of analytes that must be outliers to flag sample

    Returns:
        DataFrame with sample QC status summary
    """
    qc_instance._print_section("SAMPLE QC FLAGS")
    logger.info(
        "checking RowCheck flags, sample-level outliers, and low volume samples (sample QC)..."
    )

    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col

    # Use samples (biological samples only)
    measurements = qc_instance.samples

    # Calculate per-analyte median and MAD for outlier detection.
    analyte_stats = (
        measurements.group_by(protein_id_col)
        .agg(
            [
                pl.col("RFU").median().alias("median_RFU"),
                (pl.col("RFU") - pl.col("RFU").median())
                .abs()
                .median()
                .alias("MAD_RFU"),
            ]
        )
        .collect(engine="streaming")
    )

    # Calculate outlier fraction per sample.
    outlier_summary = (
        measurements.join(analyte_stats.lazy(), on=protein_id_col, how="left")
        .with_columns(
            [
                (
                    (
                        pl.col("RFU")
                        > (pl.col("median_RFU") + mad_threshold * pl.col("MAD_RFU"))
                    )
                    & (pl.col("RFU") > (fc_threshold * pl.col("median_RFU")))
                ).alias("Is_Outlier_RFU")
            ]
        )
        .group_by(sample_id_col)
        .agg(
            [
                pl.col("Is_Outlier_RFU").sum().alias("Outlier_Count"),
                pl.len().alias("Total_Analytes_Outlier_Check"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("Outlier_Count") / pl.col("Total_Analytes_Outlier_Check")
                ).alias("RFU_Outlier_Frac")
            ]
        )
        .collect(engine="streaming")
    )

    # Get RowCheck status and SampleNotes per sample and join with outlier summary.
    qc_status = (
        qc_instance.sample_metadata.select([sample_id_col, "RowCheck", "SampleNotes"])
        .join(
            measurements.group_by(sample_id_col).agg(pl.len().alias("Total_Analytes")),
            on=sample_id_col,
            how="left",
        )
        .join(outlier_summary.lazy(), on=sample_id_col, how="left")
        .sort(sample_id_col)
        .collect()
    )

    # Identify samples with RowCheck != "PASS"
    rowcheck_failed = qc_status.filter(pl.col("RowCheck") != "PASS")[
        sample_id_col
    ].to_list()

    # Identify sample-level RFU outliers
    rfu_outliers = qc_status.filter(pl.col("RFU_Outlier_Frac") >= outlier_fraction)[
        sample_id_col
    ].to_list()

    # Identify low volume samples
    low_volume_samples = qc_status.filter(
        pl.col("SampleNotes").str.contains("Low Volume")
    )[sample_id_col].to_list()

    # FAIL samples that fail BOTH RowCheck AND RFU outlier checks
    fail_samples = list(set(rowcheck_failed) & set(rfu_outliers))

    # WARN samples that fail only one check
    warn_rowcheck_only = list(set(rowcheck_failed) - set(fail_samples))
    warn_rfu_only = list(set(rfu_outliers) - set(fail_samples))
    warn_samples = list(set(warn_rowcheck_only + warn_rfu_only))

    # Mark failed samples
    if fail_samples and remove_fail:
        logger.warning(
            f"marking {len(fail_samples)} samples as FAIL (failed both RowCheck and RFU outlier)"
        )
        qc_instance.failed_samples.extend(fail_samples)
        qc_instance._track_sample_qc(fail_samples, "FAIL_RowCheck_and_RFU_Outlier")

    # Mark warned samples
    if warn_samples and flag_warn:
        logger.warning(f"marking {len(warn_samples)} samples as WARN")
        if warn_rowcheck_only:
            logger.info(f"  - RowCheck warnings only: {len(warn_rowcheck_only)}")
            qc_instance.warned_samples.extend(warn_rowcheck_only)
            qc_instance._track_sample_qc(warn_rowcheck_only, "WARN_RowCheck")
        if warn_rfu_only:
            logger.info(
                f"  - Sample-level RFU outliers only (>={outlier_fraction*100:.0f}% measurements exceeding {mad_threshold} MADs and {fc_threshold}x FC): {len(warn_rfu_only)}"
            )
            qc_instance.warned_samples.extend(warn_rfu_only)
            qc_instance._track_sample_qc(warn_rfu_only, "WARN_RFU_Outlier")

    # Mark low volume samples as WARN (independent of other flags)
    if low_volume_samples and flag_warn:
        logger.info(f"  - Low volume samples: {len(low_volume_samples)}")
        # Add to warned_samples if not already failed or warned
        existing_warned_or_failed = set(qc_instance.failed_samples) | set(
            qc_instance.warned_samples
        )
        new_warned = [
            s for s in low_volume_samples if s not in existing_warned_or_failed
        ]
        qc_instance.warned_samples.extend(new_warned)
        qc_instance._track_sample_qc(low_volume_samples, "WARN_Low_Volume")

    # Summary - recalculate after all warnings are added
    total_warned = len(set(qc_instance.warned_samples))
    total_failed = len(set(qc_instance.failed_samples))
    n_pass = len(qc_status) - total_failed - total_warned
    n_warn = total_warned
    n_fail = total_failed

    logger.info(f"sample QC - PASS: {n_pass} | WARN: {n_warn} | FAIL: {n_fail}")

    # Store sample QC data for combined plot (used by assay_qc and reporting)
    qc_dict = {"PASS": n_pass, "WARN": n_warn, "FAIL": n_fail}
    qc_instance._sample_qc_data = {
        "qc_dict": qc_dict,
        "fail_samples": fail_samples,  # Required by reporting
        "warn_samples": warn_samples,
        "mad_threshold": mad_threshold,  # Store thresholds for reporting
        "fc_threshold": fc_threshold,
        "outlier_fraction": outlier_fraction,
    }

    # Store sample QC status for inclusion in summary (RowCheck, outlier metrics)
    qc_instance._sample_qc_status = qc_status.select(
        [sample_id_col, "RowCheck", "RFU_Outlier_Frac"]
    )

    return qc_status


def assay_qc_olink(
    qc_instance,
    warn_threshold: float = 0.1,
) -> pl.DataFrame:
    """
    Check AssayQC for problematic proteins.

    Args:
        qc_instance: OlinkQC instance
        warn_threshold: Proportion threshold for marking proteins as failed

    Returns:
        DataFrame with assay QC summary
    """
    logger.info("Assay QC flags")

    # Get column names from instance
    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    protein_name_col = qc_instance.protein_name_col
    data_col = qc_instance.data_col

    # Exclude failed samples from assay QC calculation
    # Need to join with analyte_metadata to get AssayQC and AssayType
    protein_data = qc_instance.protein_assays.join(
        qc_instance.analyte_metadata.select(
            [protein_id_col, protein_name_col, "AssayQC", "AssayType"]
        ),
        on=protein_id_col,
        how="left",
    ).filter(pl.col("AssayType") == "assay")

    if qc_instance.failed_samples:
        logger.info(
            f"Excluding {len(qc_instance.failed_samples)} failed samples from Assay QC"
        )
        protein_data = protein_data.filter(
            ~pl.col(sample_id_col).is_in(qc_instance.failed_samples)
        )

    assay_qc = (
        protein_data.group_by(protein_id_col)
        .agg(
            [
                pl.col(protein_name_col).first().alias(protein_name_col),
                (pl.col("AssayQC") == "PASS").sum().alias("PASS"),
                (pl.col("AssayQC") != "PASS").sum().alias("WARN_FAIL"),
                pl.len().alias("Total"),
                (pl.col(data_col).is_null() | pl.col(data_col).is_nan())
                .sum()
                .alias("Missing_Count"),
            ]
        )
        .with_columns(
            [
                (pl.col("WARN_FAIL") / pl.col("Total")).alias("Warn_Frac"),
                (pl.col("Missing_Count") / pl.col("Total")).alias("Missing_Frac"),
            ]
        )
        .collect()
    )

    to_remove = assay_qc.filter(pl.col("Warn_Frac") > warn_threshold)[
        protein_name_col
    ].to_list()
    n_any_warn = (assay_qc["Warn_Frac"] > 0).sum()

    # Flag proteins with warnings but below removal threshold
    to_flag = assay_qc.filter(
        (pl.col("WARN_FAIL") > 0) & (pl.col("Warn_Frac") <= warn_threshold)
    )[protein_name_col].to_list()

    logger.info(
        f"proteins with warnings: {n_any_warn} | >{warn_threshold*100:.0f}% warned: {len(to_remove)}"
    )

    if to_remove:
        logger.warning(f"Marking {len(to_remove)} proteins as FAIL")
        # Get the protein IDs (OlinkID) for the failed assays
        failed_protein_ids = (
            qc_instance.analyte_metadata.filter(
                pl.col(protein_name_col).is_in(to_remove)
            )
            .select(protein_id_col)
            .unique()
            .collect()[protein_id_col]
            .to_list()
        )
        qc_instance.failed_proteins.extend(failed_protein_ids)
        qc_instance._track_protein_qc(failed_protein_ids, "FAIL_Assay_QC")

    if to_flag:
        logger.warning(f"Marking {len(to_flag)} proteins as WARN")
        # Get the protein IDs (OlinkID) for the warned assays
        warned_protein_ids = (
            qc_instance.analyte_metadata.filter(pl.col(protein_name_col).is_in(to_flag))
            .select(protein_id_col)
            .unique()
            .collect()[protein_id_col]
            .to_list()
        )
        qc_instance.warned_proteins.extend(warned_protein_ids)
        qc_instance._track_protein_qc(warned_protein_ids, "WARN_Assay_QC")

    qc_instance.qc_results["proteins_failed"] = len(to_remove)
    qc_instance.qc_results["proteins_warned"] = len(to_flag)

    # Calculate PASS, WARN, FAIL for proteins
    n_pass = len(assay_qc.filter(pl.col("Warn_Frac") == 0))
    n_fail = len(assay_qc.filter(pl.col("Warn_Frac") > warn_threshold))
    n_warn = len(
        assay_qc.filter(
            (pl.col("Warn_Frac") > 0) & (pl.col("Warn_Frac") <= warn_threshold)
        )
    )

    # Store assay QC status for inclusion in summary (protein_name_col already in analyte_metadata)
    qc_instance._assay_qc_status = assay_qc.select(
        [protein_id_col, "Warn_Frac", "Missing_Frac"]
    )

    qc_dict = qc_instance._sample_qc_data["qc_dict"]
    sample_counts = {s: qc_dict.get(s, 0) for s in ["PASS", "WARN", "FAIL"]}
    assay_counts = {"PASS": n_pass, "WARN": n_warn, "FAIL": n_fail}

    return {
        "assay_qc": assay_qc,
        "sample_counts": sample_counts,
        "assay_counts": assay_counts,
        "warn_threshold": warn_threshold,
        "n_any_warn": n_any_warn,
        "to_remove": to_remove,
        "to_flag": to_flag,
    }


def assay_qc_soma(
    qc_instance,
) -> dict:
    """Check ColCheck flags (assay QC) in SomaScan data.

    ColCheck is a per-analyte flag, so any analyte with ColCheck != "PASS" is marked as WARN.

    Args:
        qc_instance: SomaQC instance

    Returns:
        Dictionary with keys: assay_qc, sample_counts, assay_counts, n_any_warn, to_remove, to_flag
    """
    logger.info("checking ColCheck flags (assay QC)...")

    # Get column names from instance
    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col

    # Use qc_instance.samples (biological samples only) instead of get_data() (all data)
    # This matches the Olink pattern
    analyte_data = qc_instance.samples.join(
        qc_instance.analyte_metadata.select([protein_id_col, "ColCheck"]),
        on=protein_id_col,
        how="left",
    )

    if qc_instance.failed_samples:
        logger.info(
            f"Excluding {len(qc_instance.failed_samples)} failed samples from analyte QC"
        )
        analyte_data = analyte_data.filter(
            ~pl.col(sample_id_col).is_in(qc_instance.failed_samples)
        )

    # Analyte QC analysis - one row per analyte
    # ColCheck should be constant per analyte, so we just take first value
    analyte_qc = (
        analyte_data.group_by(protein_id_col)
        .agg(
            [
                pl.col("ColCheck").first().alias("ColCheck"),
                pl.len().alias("Total_Samples"),
                (pl.col("ColCheck") != "PASS").sum().alias("WARN_Count"),
            ]
        )
        .with_columns(
            [
                (pl.col("WARN_Count") / pl.col("Total_Samples")).alias("Warn_Frac"),
            ]
        )
        .collect(engine="streaming")
    )

    # Identify analytes with ColCheck warnings (all go to WARN, not FAIL)
    to_warn = analyte_qc.filter(pl.col("ColCheck") != "PASS")[protein_id_col].to_list()

    n_pass = len(analyte_qc.filter(pl.col("ColCheck") == "PASS"))
    n_warn = len(to_warn)
    n_fail = 0  # ColCheck only causes WARN, not FAIL

    logger.info(f"analyte ColCheck - PASS: {n_pass} | WARN: {n_warn}")

    # Store set for use by finalize_assay_flags_soma
    qc_instance._colcheck_warn_set = set(to_warn)

    if to_warn:
        logger.warning(f"marking {len(to_warn)} analytes as WARN (ColCheck)")
        qc_instance.warned_proteins.extend(to_warn)
        qc_instance._track_protein_qc(to_warn, "WARN_ColCheck")

    # Store assay QC status for summary export.
    qc_instance._assay_qc_status = analyte_qc.select(
        [protein_id_col, "ColCheck", "Warn_Frac"]
    )

    # Calculate sample counts from sample_qc_data (set by sample_qc function)
    qc_dict = qc_instance._sample_qc_data.get("qc_dict", {})
    sample_counts = {s: qc_dict.get(s, 0) for s in ["PASS", "WARN", "FAIL"]}
    assay_counts = {"PASS": n_pass, "WARN": n_warn, "FAIL": n_fail}

    # Return dictionary matching Olink format for compatibility with reporting
    return {
        "assay_qc": analyte_qc,
        "sample_counts": sample_counts,
        "assay_counts": assay_counts,
        "n_any_warn": n_warn,
        "to_remove": [],  # No failed analytes
        "to_flag": to_warn,
    }


def assay_saturation_soma(
    qc_instance,
    rfu_threshold: float = 80_000,
    sample_frac: float = 0.50,
) -> dict:
    """Flag analytes where more than sample_frac of biological samples exceed rfu_threshold RFU.

    High saturation indicates detector ceiling effects — measurements are unreliable at or
    above the instrument's dynamic range limit (~65,535–100,000 RFU depending on scanner).

    Results are stored on the instance:
        qc_instance._saturation_flagged  — set of SeqIds flagged
        qc_instance._saturation_stats    — per-analyte DataFrame with n_samples/n_saturated/saturated_frac

    Args:
        qc_instance: SomaQC instance
        rfu_threshold: RFU ceiling value above which a measurement is considered saturated
        sample_frac: Fraction of samples that must exceed the threshold to flag the analyte

    Returns:
        Dictionary with keys: saturation_stats, flagged, rfu_threshold, sample_frac
    """
    qc_instance._print_section("SATURATION CHECK")
    logger.info(
        f"checking saturation: >{rfu_threshold:,.0f} RFU in >{sample_frac*100:.0f}% samples..."
    )

    protein_id_col = qc_instance.protein_id_col
    sample_id_col = qc_instance.sample_id_col

    samples = qc_instance.samples
    if qc_instance.failed_samples:
        samples = samples.filter(
            ~pl.col(sample_id_col).is_in(qc_instance.failed_samples)
        )

    saturation_stats = (
        samples.group_by(protein_id_col)
        .agg(
            pl.len().alias("n_samples"),
            (pl.col("RFU") > rfu_threshold).sum().alias("n_saturated"),
        )
        .with_columns(
            (pl.col("n_saturated") / pl.col("n_samples")).alias("saturated_frac"),
        )
        .collect()
    )

    flagged = saturation_stats.filter(pl.col("saturated_frac") > sample_frac)[
        protein_id_col
    ].to_list()

    logger.info(
        f"saturation check: {len(flagged)} analytes flagged "
        f"(>{sample_frac*100:.0f}% samples >{rfu_threshold:,.0f} RFU)"
    )

    qc_instance._saturation_flagged = set(flagged)
    qc_instance._saturation_stats = saturation_stats

    return {
        "saturation_stats": saturation_stats,
        "flagged": flagged,
        "rfu_threshold": rfu_threshold,
        "sample_frac": sample_frac,
    }


def assay_sb_ratio_soma(
    qc_instance,
    sb_threshold: float = 2.0,
    buffer_sample_type: str = "Buffer",
) -> dict:
    """Flag analytes where the median sample RFU / median buffer RFU falls below sb_threshold.

    A low signal-to-background ratio indicates the biological signal cannot be reliably
    distinguished from background hybridization noise.

    Results are stored on the instance:
        qc_instance._sb_ratio_flagged — set of SeqIds flagged
        qc_instance._sb_ratio_stats   — per-analyte DataFrame with median_sample_rfu/median_buffer_rfu/sb_ratio

    Args:
        qc_instance: SomaQC instance
        sb_threshold: Minimum acceptable signal-to-background ratio
        buffer_sample_type: SampleType label for buffer/blank samples in metadata

    Returns:
        Dictionary with keys: sb_stats, flagged, sb_threshold
    """
    qc_instance._print_section("SIGNAL-TO-BACKGROUND RATIO")
    logger.info(
        f"checking signal-to-background ratio (threshold: sample/buffer < {sb_threshold})..."
    )

    protein_id_col = qc_instance.protein_id_col
    sample_id_col = qc_instance.sample_id_col

    samples = qc_instance.samples
    if qc_instance.failed_samples:
        samples = samples.filter(
            ~pl.col(sample_id_col).is_in(qc_instance.failed_samples)
        )

    sample_medians = (
        samples.group_by(protein_id_col)
        .agg(pl.col("RFU").median().alias("median_sample_rfu"))
        .collect()
    )

    buffer_ids = qc_instance.sample_metadata.filter(
        pl.col("SampleType") == buffer_sample_type
    ).select("_sample_idx")
    n_buffer = buffer_ids.select(pl.len()).collect().item()
    if n_buffer == 0:
        logger.warning(
            f"No buffer samples found (SampleType='{buffer_sample_type}') — "
            "skipping signal-to-background check"
        )
        qc_instance._sb_ratio_flagged = set()
        qc_instance._sb_ratio_stats = sample_medians.with_columns(
            pl.lit(None).cast(pl.Float64).alias("median_buffer_rfu"),
            pl.lit(None).cast(pl.Float64).alias("sb_ratio"),
        )
        return {
            "sb_stats": qc_instance._sb_ratio_stats,
            "flagged": [],
            "sb_threshold": sb_threshold,
        }

    buffer_medians = (
        qc_instance.get_data()
        .join(buffer_ids, on="_sample_idx", how="semi")
        .group_by(protein_id_col)
        .agg(pl.col("RFU").median().alias("median_buffer_rfu"))
        .collect()
    )

    sb_stats = sample_medians.join(
        buffer_medians, on=protein_id_col, how="left"
    ).with_columns(
        (pl.col("median_sample_rfu") / pl.col("median_buffer_rfu")).alias("sb_ratio"),
    )

    flagged = sb_stats.filter(pl.col("sb_ratio") < sb_threshold)[
        protein_id_col
    ].to_list()

    logger.info(
        f"signal-to-background: {len(flagged)} analytes flagged (ratio < {sb_threshold})"
    )

    qc_instance._sb_ratio_flagged = set(flagged)
    qc_instance._sb_ratio_stats = sb_stats

    return {
        "sb_stats": sb_stats,
        "flagged": flagged,
        "sb_threshold": sb_threshold,
    }


def finalize_assay_flags_soma(qc_instance) -> dict:
    """Consolidate all per-analyte flags and apply unified WARN/FAIL logic.

    After all individual checks have run (ColCheck, Intra-CV, Detection Rate, Saturation,
    Signal-to-Background Ratio), this function applies the unified rule:
        - ≥1 flag  → WARN  (kept in downstream analysis but flagged)
        - ≥2 flags → FAIL  (excluded from downstream analysis)

    Requires that preceding checks have stored their flag sets on the instance:
        _colcheck_warn_set       (filter_assay_qc)
        _cv_fail_proteins        (filter_assay_cv)
        _detection_fail_proteins (filter_assay_detection)
        _saturation_flagged      (filter_assay_saturation)
        _sb_ratio_flagged        (filter_assay_sb_ratio)

    Resets qc_instance.warned_proteins, qc_instance.failed_proteins, and
    qc_instance.protein_qc_actions, then rebuilds them based on consolidated logic.

    Args:
        qc_instance: SomaQC instance after all individual checks have been run

    Returns:
        Dictionary with n_pass, n_warn, n_fail, per_analyte_flags, flag_counts
    """
    qc_instance._print_section("CONSOLIDATING ASSAY FLAGS")

    protein_id_col = qc_instance.protein_id_col

    flag_map = {
        "ColCheck": getattr(qc_instance, "_colcheck_warn_set", set()),
        "Intra_CV": set(getattr(qc_instance, "_cv_fail_proteins", [])),
        "Detection_Rate": getattr(qc_instance, "_detection_fail_proteins", set()),
        "Saturation": getattr(qc_instance, "_saturation_flagged", set()),
        "SB_Ratio": getattr(qc_instance, "_sb_ratio_flagged", set()),
    }

    # Universe of all analytes that were evaluated
    if hasattr(qc_instance, "_cv_stats") and qc_instance._cv_stats is not None:
        all_analytes = set(qc_instance._cv_stats[protein_id_col].to_list())
    else:
        all_analytes = set(
            qc_instance.analyte_metadata.select(protein_id_col)
            .collect()[protein_id_col]
            .to_list()
        )

    # Build per-analyte flag list
    per_analyte_flags: dict[str, list[str]] = {}
    for analyte in all_analytes:
        flags = [
            name for name, flagged_set in flag_map.items() if analyte in flagged_set
        ]
        if flags:
            per_analyte_flags[analyte] = flags

    multi_fail = [a for a, f in per_analyte_flags.items() if len(f) >= 2]
    single_warn = [a for a, f in per_analyte_flags.items() if len(f) == 1]

    # Reset protein WARN/FAIL state and rebuild from scratch
    qc_instance.failed_proteins = multi_fail
    qc_instance.warned_proteins = single_warn
    qc_instance.protein_qc_actions = []

    # Re-track individual check flags (used for per-check reporting detail)
    for flag_name, flagged_set in flag_map.items():
        if flagged_set:
            qc_instance._track_protein_qc(list(flagged_set), f"WARN_{flag_name}")

    # Track the consolidated FAIL classification (drives the waterfall summary table)
    if multi_fail:
        qc_instance._track_protein_qc(multi_fail, "FAIL_Multiple_Flags")

    n_all = len(all_analytes)
    n_any = len(per_analyte_flags)

    logger.info(
        f"Consolidated assay flags: {n_all - n_any} PASS | "
        f"{len(single_warn)} WARN (1 flag) | {len(multi_fail)} FAIL (≥2 flags)"
    )

    flag_counts = {name: len(flagged_set) for name, flagged_set in flag_map.items()}

    return {
        "n_pass": n_all - n_any,
        "n_warn": len(single_warn),
        "n_fail": len(multi_fail),
        "n_total": n_all,
        "per_analyte_flags": per_analyte_flags,
        "flag_counts": flag_counts,
    }
