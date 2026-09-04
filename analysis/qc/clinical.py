"""Clinical filtering for proteomics sample exclusion."""

import logging
import polars as pl

from .utils import ensure_lazy, ensure_date_column

logger = logging.getLogger(__name__)


def sample_unsuitable(
    qc_instance,
    discontinuation_status_col="EOTSTT",
    last_treatment_date_col="TRTEDT",
    rescue_date_col="LTRESCDT",
    sample_date_col="ADT",
    grace_days=7,
    adds=None,
    adds_paramcd="EOTSTT",
):
    """Identify and mark samples unsuitable for longitudinal analysis.

    Filters samples based on:
        - DISCONTINUED patients with samples after treatment end (per-sample)
        - Samples >grace_days after rescue drug initiation
        - Samples >grace_days after last treatment/rescue date
        - Participants with incomplete baseline measurements

    Visit detection:
        - Baseline visit: automatically detected as minimum visit number for
          baseline filtering

    Args:
        qc_instance: SomaQC instance with samples, ADSL, and manifest data.
        discontinuation_status_col: Column in ADSL indicating treatment discontinuation
            status. Defaults to "EOTSTT". Only used if adds is not provided.
        last_treatment_date_col: Column in ADSL with treatment end date for
            discontinued patients. Defaults to "TRTEDT".
        rescue_date_col: Column in ADSL with last treatment/rescue date.
            Defaults to "LTRESCDT".
        sample_date_col: Column in manifest with sample collection date. Defaults to "ADT".
        grace_days: Grace period in days after rescue drug or treatment end.
            Defaults to 7.
        adds: Optional ADDS (Analysis Dataset for Disposition) DataFrame or LazyFrame.
            If provided, discontinuation status is extracted from this dataset using
            the adds_paramcd parameter. Expected columns: USUBJID, PARAMCD, AVALC.
        adds_paramcd: PARAMCD value to filter for in ADDS to get discontinuation status.
            Defaults to "EOTSTT". The corresponding AVALC value is used as the status.

    Returns:
        dict: Dictionary with counts of samples filtered by each criterion:
            - discontinued: DISCONTINUED patients with samples after
              treatment end
            - after_treatment: Samples >grace_days after last treatment/rescue date
            - no_baseline: Participants without baseline measurements
            - only_baseline: Participants with only baseline measurements
    """

    if qc_instance.adsl is None:
        logger.warning("No ADSL data available - skipping clinical filtering")
        return {}

    if qc_instance.manifest is None:
        logger.warning("No manifest data available - skipping clinical filtering")
        return {}

    logger.info("starting clinical filtering for longitudinal analysis")

    sample_id_col = qc_instance.sample_id_col

    # Get the mapping from SampleID to USUBJID.
    if qc_instance.adsl_merged is not None:
        sample_mapping = (
            qc_instance.adsl_merged.select([sample_id_col, "USUBJID"])
            .unique(subset=[sample_id_col])
            .collect()
        )
    else:
        logger.error("no ADSL merged data - cannot perform clinical filtering")
        return {}

    adsl = ensure_lazy(qc_instance.adsl)
    manifest = ensure_lazy(qc_instance.manifest)

    # Get current samples excluding already failed samples.
    # Use sample_metadata directly to get unique samples with SampleType.
    current_samples = (
        qc_instance.sample_metadata.select([sample_id_col, "SampleType"])
        .filter(
            ~pl.col(sample_id_col).is_in(qc_instance.failed_samples),
            pl.col("SampleType") == qc_instance.sample_type,
        )
        .unique(subset=[sample_id_col])
        .collect()
    )

    # Join samples with USUBJID
    samples_with_usubjid = current_samples.select([sample_id_col]).join(
        sample_mapping, on=sample_id_col, how="left"
    )

    # Exclude bridge/control samples from clinical filtering.
    # Bridge samples (USUBJID starting with "BRIDGE_") and null USUBJID samples
    # are controls, not real study participants.
    bridge_or_null_samples = samples_with_usubjid.filter(
        pl.col("USUBJID").is_null() | pl.col("USUBJID").str.starts_with("BRIDGE_")
    )[sample_id_col].to_list()

    if bridge_or_null_samples:
        # Mark bridge samples as FAIL
        qc_instance._track_sample_qc(bridge_or_null_samples, "FAIL_Bridge_Sample")
        qc_instance.failed_samples.extend(bridge_or_null_samples)
        logger.info(
            f"Marked {len(bridge_or_null_samples)} bridge/control samples with FAIL_Bridge_Sample"
        )

        # Exclude from subsequent clinical filtering
        samples_with_usubjid = samples_with_usubjid.filter(
            pl.col("USUBJID").is_not_null()
            & ~pl.col("USUBJID").str.starts_with("BRIDGE_")
        )

    # Get VISITNUM information from adsl_merged (already includes manifest data)
    if qc_instance.adsl_merged is not None:
        adsl_cols = qc_instance.adsl_merged.collect_schema().names()

        if "VISITNUM" in adsl_cols:
            samples_with_usubjid = samples_with_usubjid.join(
                qc_instance.adsl_merged.select([sample_id_col, "VISITNUM"]).collect(),
                on=sample_id_col,
                how="left",
            )
            logger.info("VISITNUM information obtained from adsl_merged")
        else:
            logger.warning(
                "VISITNUM information not available - date-based filtering may be inaccurate"
            )
            samples_with_usubjid = samples_with_usubjid.with_columns(
                pl.lit(None).cast(pl.Int64).alias("VISITNUM")
            )
    else:
        logger.warning(
            "VISITNUM information not available - date-based filtering may be inaccurate"
        )
        samples_with_usubjid = samples_with_usubjid.with_columns(
            pl.lit(None).cast(pl.Int64).alias("VISITNUM")
        )

    # Determine baseline visit, last visit, and early visits.
    all_visits = (
        samples_with_usubjid.filter(pl.col("VISITNUM").is_not_null())
        .select("VISITNUM")
        .unique()
        .sort("VISITNUM")
        .to_series()
        .to_list()
    )

    if all_visits:
        baseline_visit = min(all_visits)
        last_visit = max(all_visits)
        early_visits = [v for v in all_visits if v != last_visit]
        logger.info(
            f"Detected visits: {all_visits} (baseline: {baseline_visit}, last visit: {last_visit}, early visits: {early_visits})"
        )
    else:
        logger.warning(
            "No visit information found in samples - visit-based filtering will be skipped"
        )
        baseline_visit = None
        last_visit = None
        early_visits = []

    failed_counts = {
        "discontinued": 0,  # DISCONTINUED patients with sample after treatment end
        "after_treatment": 0,  # Samples >grace_days after last treatment/rescue date
        "no_baseline": 0,
        "only_baseline": 0,
    }

    # Helper function to ensure date column is properly typed
    def ensure_date_col(df, col_name):
        """Wrapper for imported ensure_date_column."""
        return ensure_date_column(df, col_name)

    # Get sample dates from manifest ADT column directly
    # Parse as date to ensure proper type for comparisons
    sample_dates_all = manifest.select(
        [
            sample_id_col,
            qc_instance.usubjid_col,
            ensure_date_col(manifest, sample_date_col).alias(sample_date_col),
        ]
    ).collect()

    # Get clinical dates for all subjects
    # Parse date columns to ensure proper types
    adsl_columns = adsl.collect_schema().names()

    # Determine discontinuation status source: ADSL column (preferred) or ADDS table
    if discontinuation_status_col in adsl_columns:
        # Primary method: get discontinuation status from ADSL column (e.g., EOTSTT)
        logger.info(
            f"Using '{discontinuation_status_col}' column from ADSL for discontinuation status"
        )
        clinical_dates = adsl.select(
            [
                "USUBJID",
                ensure_date_col(adsl, last_treatment_date_col).alias(
                    last_treatment_date_col
                ),
                discontinuation_status_col,
            ]
        ).collect()
    elif adds is not None:
        # Fallback method: extract discontinuation status from ADDS dataset
        logger.info(
            f"Column '{discontinuation_status_col}' not found in ADSL, using ADDS table"
        )
        adds_lazy = ensure_lazy(adds)
        adds_cols = adds_lazy.collect_schema().names()

        required_adds_cols = ["USUBJID", "AVALC", "ADT"]
        missing_cols = [c for c in required_adds_cols if c not in adds_cols]
        if missing_cols:
            logger.error(
                f"ADDS dataset missing required columns: {missing_cols}. "
                f"Available columns: {adds_cols}"
            )
            raise ValueError(f"ADDS dataset missing required columns: {missing_cols}")

        # Extract discontinuation status from ADDS by checking AVALC for "Discontinued"
        # Only consider discontinuations on or before each patient's last visit date

        # Get max ADT (sample date) per USUBJID from manifest
        max_sample_dates = (
            sample_dates_all.select(["USUBJID", sample_date_col])
            .group_by("USUBJID")
            .agg(pl.col(sample_date_col).max().alias("max_manifest_date"))
        )

        # Prepare ADDS data with dates parsed
        adds_with_dates = (
            adds_lazy.select(
                ["USUBJID", "AVALC", ensure_date_col(adds_lazy, "ADT").alias("ADT")]
            )
            .collect()
            .join(max_sample_dates, on="USUBJID", how="left")
        )

        # Extract discontinuation: patient is discontinued if ANY record has
        # "Discontinued" in AVALC and ADT is on or before max manifest date
        # (or ADT is null, which we treat as discontinued)
        discontinuation_from_adds = adds_with_dates.group_by("USUBJID").agg(
            pl.when(
                (
                    (pl.col("AVALC").str.to_uppercase().str.contains("DISCONTINUED"))
                    & (
                        pl.col("ADT").is_null()
                        | (pl.col("ADT") <= pl.col("max_manifest_date"))
                    )
                ).any()
            )
            .then(pl.lit("DISCONTINUED"))
            .otherwise(pl.lit(None))
            .alias(discontinuation_status_col)
        )

        if discontinuation_from_adds.height == 0:
            logger.warning(
                f"No discontinuation records found in ADDS. "
                "Discontinuation filtering will be skipped."
            )
            # Create empty discontinuation status
            discontinuation_from_adds = (
                adsl.select("USUBJID")
                .collect()
                .with_columns(
                    pl.lit(None).cast(pl.Utf8).alias(discontinuation_status_col)
                )
            )
        else:
            logger.info(
                f"Extracted discontinuation status from ADDS "
                f"for {discontinuation_from_adds.height} subjects"
            )

        # Build clinical_dates by joining ADSL with discontinuation from ADDS
        clinical_dates = (
            adsl.select(
                [
                    "USUBJID",
                    ensure_date_col(adsl, last_treatment_date_col).alias(
                        last_treatment_date_col
                    ),
                ]
            )
            .collect()
            .join(discontinuation_from_adds, on="USUBJID", how="left")
        )
    else:
        # Column not in ADSL and no ADDS provided - skip discontinuation filtering
        logger.warning(
            f"Column '{discontinuation_status_col}' not found in ADSL and no ADDS provided. "
            "Discontinuation filtering will be skipped."
        )
        clinical_dates = (
            adsl.select(
                [
                    "USUBJID",
                    ensure_date_col(adsl, last_treatment_date_col).alias(
                        last_treatment_date_col
                    ),
                ]
            )
            .collect()
            .with_columns(pl.lit(None).cast(pl.Utf8).alias(discontinuation_status_col))
        )

    # Get rescue dates - handle missing column gracefully
    if rescue_date_col in adsl_columns:
        rescue_dates = adsl.select(
            ["USUBJID", ensure_date_col(adsl, rescue_date_col).alias(rescue_date_col)]
        ).collect()
    else:
        logger.warning(
            f"Column '{rescue_date_col}' not found in ADSL - "
            "rescue date filtering will be skipped"
        )
        # Create a DataFrame with null rescue dates for all subjects
        rescue_dates = (
            adsl.select("USUBJID")
            .collect()
            .with_columns(pl.lit(None).cast(pl.Date).alias(rescue_date_col))
        )

    # Pre-compute complete sample information with all clinical data in one join
    # This avoids re-joining for each filter
    samples_full = (
        samples_with_usubjid.join(clinical_dates, on="USUBJID", how="left")
        .join(rescue_dates, on="USUBJID", how="left")
        .join(sample_dates_all, on=sample_id_col, how="left")
    )

    # 1. Filter samples from DISCONTINUED patients collected after treatment end
    # This is per-sample, not per-patient
    discontinued_samples = samples_full.filter(
        ~pl.col(sample_id_col).is_in(qc_instance.failed_samples),
        pl.col(discontinuation_status_col).is_not_null(),
        (pl.col(discontinuation_status_col).str.to_uppercase() == "DISCONTINUED"),
        (pl.col(sample_date_col) > pl.col(last_treatment_date_col)),
    )

    failed_discontinued = discontinued_samples[sample_id_col].unique().to_list()

    if failed_discontinued:
        qc_instance._track_sample_qc(failed_discontinued, "FAIL_Discontinued")
        qc_instance.failed_samples.extend(failed_discontinued)
    failed_counts["discontinued"] = len(failed_discontinued)
    logger.info(
        f"filtered {len(failed_discontinued)} samples from DISCONTINUED patients collected after treatment end"
    )

    # 2. Filter samples >grace_days after last treatment/rescue date
    after_treatment_samples = samples_full.filter(
        ~pl.col(sample_id_col).is_in(qc_instance.failed_samples),
        pl.col(rescue_date_col).is_not_null(),
        pl.col(sample_date_col).is_not_null(),
        (pl.col(sample_date_col) - pl.col(rescue_date_col)).dt.total_days()
        > grace_days,
    )

    failed_after_treatment = after_treatment_samples[sample_id_col].unique().to_list()

    if failed_after_treatment:
        qc_instance._track_sample_qc(failed_after_treatment, "FAIL_After_Treatment")
        qc_instance.failed_samples.extend(failed_after_treatment)
    failed_counts["after_treatment"] = len(failed_after_treatment)
    logger.info(
        f"filtered {len(failed_after_treatment)} samples collected >{grace_days} days after last treatment/rescue date"
    )

    # 3. Filter participants without baseline measurements (excluding bridge samples)
    current_valid_samples = samples_with_usubjid.filter(
        ~pl.col(sample_id_col).is_in(qc_instance.failed_samples)
    )

    # VISITNUM already in samples_with_usubjid
    if "VISITNUM" in current_valid_samples.columns:
        # Count samples per participant by baseline status
        participant_baseline_counts = (
            current_valid_samples.with_columns(
                pl.when(pl.col("VISITNUM") == baseline_visit)
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
                .alias("is_baseline")
            )
            .group_by("USUBJID")
            .agg(
                [
                    pl.col("is_baseline").sum().alias("baseline_count"),
                    pl.col(sample_id_col).count().alias("total_count"),
                    pl.col(sample_id_col).alias("sample_ids"),
                ]
            )
        )

        # Participants with NO baseline samples
        no_baseline = participant_baseline_counts.filter(pl.col("baseline_count") == 0)

        failed_no_baseline = []
        if no_baseline.height > 0:
            failed_no_baseline.extend(
                no_baseline.select(pl.col("sample_ids").flatten()).to_series().to_list()
            )

        failed_no_baseline = list(set(failed_no_baseline))

        if failed_no_baseline:
            qc_instance._track_sample_qc(failed_no_baseline, "FAIL_No_Baseline")
            qc_instance.failed_samples.extend(failed_no_baseline)
        failed_counts["no_baseline"] = len(failed_no_baseline)

        logger.info(
            f"Filtered {len(failed_no_baseline)} samples from {no_baseline.height} participants without baseline"
        )

    # 4. Filter participants with only baseline measurements
    current_valid_samples = samples_with_usubjid.filter(
        ~pl.col(sample_id_col).is_in(qc_instance.failed_samples)
    )

    if "VISITNUM" in current_valid_samples.columns:
        # Recalculate baseline counts after no_baseline filter
        participant_baseline_counts = (
            current_valid_samples.with_columns(
                pl.when(pl.col("VISITNUM") == baseline_visit)
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
                .alias("is_baseline")
            )
            .group_by("USUBJID")
            .agg(
                [
                    pl.col("is_baseline").sum().alias("baseline_count"),
                    pl.col(sample_id_col).count().alias("total_count"),
                    pl.col(sample_id_col).alias("sample_ids"),
                ]
            )
        )

        # Participants with ONLY baseline samples
        only_baseline = participant_baseline_counts.filter(
            (pl.col("baseline_count") > 0)
            & (pl.col("baseline_count") == pl.col("total_count"))
        )

        failed_only_baseline = []
        if only_baseline.height > 0:
            failed_only_baseline.extend(
                only_baseline.select(pl.col("sample_ids").flatten())
                .to_series()
                .to_list()
            )

        failed_only_baseline = list(set(failed_only_baseline))

        if failed_only_baseline:
            qc_instance._track_sample_qc(failed_only_baseline, "FAIL_Only_Baseline")
            qc_instance.failed_samples.extend(failed_only_baseline)
        failed_counts["only_baseline"] = len(failed_only_baseline)

        logger.info(
            f"filtered {len(failed_only_baseline)} samples from {only_baseline.height} participants with only baseline"
        )

    # Bridge samples were already excluded earlier, so count is stored in bridge_or_null_samples
    failed_counts["bridge_samples"] = (
        len(bridge_or_null_samples) if "bridge_or_null_samples" in locals() else 0
    )

    # Summary
    total_failed = sum(failed_counts.values())
    logger.info(f"clinical filtering complete: {total_failed} total samples failed")

    return failed_counts
