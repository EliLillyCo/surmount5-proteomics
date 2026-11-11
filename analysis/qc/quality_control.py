"""Vendor sample and assay QC flags of Olink Explore HT and SomaScan data."""

from __future__ import annotations

import logging

import polars as pl

from analysis.qc.flags import (
    FAIL_ASSAY_QC,
    FAIL_MULTIPLE_FLAGS,
    FAIL_ROWCHECK,
    FAIL_SAMPLE_QC,
    WARN_ASSAY_QC,
    WARN_COLCHECK,
    WARN_LOW_VOLUME,
    WARN_MISSING_RATE,
    WARN_RFU_OUTLIER,
    WARN_SAMPLE_QC,
)
from analysis.qc.platform import OLINK_NPX, OLINK_PC_NPX

logger = logging.getLogger(__name__)

# The RFU as the ADAT delivered it (vendor normalization applied); the eLOD and detection
# read it.
DELIVERED_RFU_COL = "RFU_delivered"


def _require_row_flags(frame: pl.LazyFrame, cols: list[str]) -> None:
    """Raises ``ValueError`` unless the per-row Olink QC flag columns are present."""
    missing = [c for c in cols if c not in frame.collect_schema().names()]
    if missing:
        raise ValueError(
            f"Olink QC needs per-row {missing} in data_full; OlinkQC must keep "
            "these columns when it builds data_full from the NPX file"
        )


VENDOR_VERDICTS = ("PASS", "WARN", "FAIL")


def _require_verdicts(frame: pl.LazyFrame, col: str) -> None:
    """Raises ``ValueError`` when ``col`` holds a null or a value other than PASS, WARN or FAIL."""
    other = (
        frame.filter(pl.col(col).is_null() | ~pl.col(col).is_in(VENDOR_VERDICTS))
        .group_by(col)
        .len()
        .sort(col, nulls_last=True)
        .collect()
    )
    if other.height:
        raise ValueError(
            f"{col} has values other than {', '.join(VENDOR_VERDICTS)} on study-sample assay rows: "
            f"{dict(other.iter_rows())}"
        )


def _drop_failed_runs(qc_instance, failed_runs: pl.DataFrame) -> None:
    """Removes the study-sample rows of the runs in ``failed_runs``.

    A run is one sample on one plate, keyed by sample ID and ``PlateID``. Applied to
    every frame later steps read, like ``_null_failed_blocks``.
    """
    keys = [qc_instance.sample_id_col, "PlateID"]
    flagged = failed_runs.select(keys).with_columns(pl.lit(True).alias("_failed_run")).lazy()

    def drop_rows(frame: pl.LazyFrame) -> pl.LazyFrame:
        hit = pl.col("_failed_run").fill_null(False) & (
            pl.col("SampleType") == qc_instance.sample_type
        )
        return (
            frame.join(flagged, on=keys, how="left", validate="m:1", maintain_order="left")
            .filter(~hit)
            .drop("_failed_run")
        )

    qc_instance.data_full = drop_rows(qc_instance.data_full)
    qc_instance.samples = drop_rows(qc_instance.samples)
    qc_instance.protein_assays = drop_rows(qc_instance.protein_assays)


def _null_failed_blocks(qc_instance, failed_blocks: pl.DataFrame) -> None:
    """Nulls the measurement columns of study-sample assay rows in ``failed_blocks``.

    ``failed_blocks`` has the sample ID, ``PlateID`` and ``Block`` of each failed run
    block. Applied to every frame later steps read (``data_full``, ``samples`` and
    ``protein_assays``), so LOD, detection, CV, outliers and the long export all see
    the failed blocks as missing.
    """
    sample_id_col = qc_instance.sample_id_col
    keys = [sample_id_col, "PlateID", "Block"]
    flagged = failed_blocks.select(keys).with_columns(pl.lit(True).alias("_failed_block")).lazy()

    def null_rows(frame: pl.LazyFrame) -> pl.LazyFrame:
        names = frame.collect_schema().names()
        cols = [
            c
            for c in dict.fromkeys(
                [qc_instance.data_col, OLINK_NPX, OLINK_PC_NPX, "Count", "ExtNPX"]
            )
            if c in names
        ]
        hit = (
            pl.col("_failed_block").fill_null(False)
            & (pl.col("SampleType") == qc_instance.sample_type)
            & (pl.col("AssayType") == "assay")
        )
        return (
            frame.join(flagged, on=keys, how="left", validate="m:1", maintain_order="left")
            .with_columns([pl.when(hit).then(None).otherwise(pl.col(c)).alias(c) for c in cols])
            .drop("_failed_block")
        )

    qc_instance.data_full = null_rows(qc_instance.data_full)
    qc_instance.samples = null_rows(qc_instance.samples)
    qc_instance.protein_assays = null_rows(qc_instance.protein_assays)


def _warn_only(qc_instance, samples: list[str], reason: str) -> list[str]:
    """Flags ``samples`` with the WARN-only flag ``reason``.

    Returns the samples newly listed as warned. The flag is tracked on every sample,
    a failed one included; a sample joins
    ``warned_samples`` only when it is not already failed or warned, so Status stays
    FAIL for a failed sample and a warned sample is listed once.
    """
    already = set(qc_instance.failed_samples) | set(qc_instance.warned_samples)
    new_warned = [s for s in samples if s not in already]
    qc_instance.warned_samples.extend(new_warned)
    qc_instance._track_sample_qc(samples, reason)
    return new_warned


def _warn_low_volume(qc_instance, low_volume_samples: list[str]) -> list[str]:
    """Flags ``low_volume_samples`` ``WARN_Low_Volume``.

    Returns the samples newly listed as warned.
    """
    return _warn_only(qc_instance, low_volume_samples, WARN_LOW_VOLUME)


def _olink_low_volume_samples(qc_instance, study_ids: list[str]) -> list[str] | None:
    """Lists the study samples whose manifest ``VOLUME_NOTE`` matches the config.

    The match is ``config.qc_low_volume_match``: a case-insensitive literal substring
    of any listed text; with no list, any non-blank note. A manifest without
    ``VOLUME_NOTE`` (or none at all) has no rule: None. The column is how a trial opts
    in (``[manifest.columns] volume_note``).
    """
    if qc_instance.manifest is None:
        return None
    manifest = qc_instance._ensure_lazy(qc_instance.manifest)
    if "VOLUME_NOTE" not in manifest.collect_schema().names():
        return None
    note = pl.col("VOLUME_NOTE").cast(pl.Utf8).str.strip_chars()
    match = qc_instance.config.qc_low_volume_match
    if match is None:
        noted = note.str.len_chars() > 0
    else:
        noted = pl.any_horizontal(
            [note.str.to_lowercase().str.contains(t.lower(), literal=True) for t in match]
        )
    sample_id_col = qc_instance.sample_id_col
    ids = (
        manifest.filter(noted.fill_null(False))
        .select(pl.col(sample_id_col).cast(pl.Utf8))
        .unique()
        .collect()[sample_id_col]
    )
    return sorted(set(ids) & set(study_ids))


def sample_qc_olink(
    qc_instance,
    remove_fail: bool = True,
    flag_warn: bool = True,
    missing_threshold: float = 0.2,
) -> pl.DataFrame:
    """Checks and handles the Olink SampleQC flags.

    SampleQC is judged per run: one sample on one plate. A run FAILs when all its
    blocks FAIL, and a sample fails only when every one of its runs FAILs. For a sample with
    a passing run, a failed run's rows are dropped. A FAIL in some blocks of a run nulls the assays of
    those blocks on that plate and the sample is WARN; so is a sample with a
    dropped run. The missing rate is taken after that, per sample and assay across the
    kept runs; above ``missing_threshold`` the sample gets ``WARN_Missing_Rate``, never
    a FAIL. Samples with an assay left missing by nulled blocks, or above the threshold,
    are kept out of the outlier fits (``_sample_qc_data["outlier_fit_exclude"]``).

    Args:
        qc_instance: OlinkQC instance.
        remove_fail: Whether to mark failing samples and null failed blocks.
        flag_warn: Whether to mark warned samples.
        missing_threshold: Missing fraction (0-1) above which a sample is WARN.

    Returns:
        One row per sample with its QC status and missing and warn fractions.
    """
    sample_id_col = qc_instance.sample_id_col
    data_col = qc_instance.data_col

    # SampleQC is set per sample and block, AssayQC per assay and plate, so both
    # are read from the measurement rows.
    _require_row_flags(
        qc_instance.samples, ["SampleQC", "AssayQC", "AssayType", "Block", "PlateID"]
    )

    def assay_rows() -> pl.LazyFrame:
        return qc_instance.samples.filter(pl.col("AssayType") == "assay")

    _require_verdicts(assay_rows(), "SampleQC")

    blocks = (
        assay_rows()
        .group_by([sample_id_col, "PlateID", "Block"])
        .agg(
            [
                (pl.col("SampleQC") == "FAIL").any().alias("Block_FAIL"),
                (pl.col("SampleQC") == "WARN").any().alias("Block_WARN"),
                (pl.col("SampleQC") == "PASS").any().alias("Block_PASS"),
                pl.len().alias("Block_Assays"),
            ]
        )
        .sort([sample_id_col, "PlateID", "Block"])
        .collect()
    )
    # A run (sample on one plate) fails when all its blocks fail; the sample fails when
    # all its runs do.
    runs = (
        blocks.group_by([sample_id_col, "PlateID"])
        .agg((pl.col("Block_FAIL").all()).alias("Run_FAIL"))
        .sort([sample_id_col, "PlateID"])
    )
    run_totals = runs.group_by(sample_id_col).agg(
        pl.len().alias("N_Runs"), pl.col("Run_FAIL").sum().alias("N_Failed_Runs")
    )
    blocks = (
        blocks.join(runs, on=[sample_id_col, "PlateID"], how="left", validate="m:1")
        .join(run_totals, on=sample_id_col, how="left", validate="m:1")
        .with_columns((pl.col("N_Failed_Runs") == pl.col("N_Runs")).alias("Sample_FAIL"))
    )
    # Block counts are over the kept runs: the failed runs of a sample with a passing
    # run are dropped.
    kept_blocks = blocks.filter(~pl.col("Run_FAIL") | pl.col("Sample_FAIL"))
    block_status = (
        blocks.group_by(sample_id_col)
        .agg(
            [
                pl.col("Sample_FAIL").first(),
                pl.col("Block_WARN").any().alias("Any_WARN"),
                pl.col("Block_FAIL").any().alias("Any_FAIL"),
                pl.col("Block_PASS").any().alias("Any_PASS"),
            ]
        )
        .join(
            kept_blocks.group_by(sample_id_col).agg(
                pl.len().alias("N_Blocks"),
                pl.col("Block_FAIL").sum().alias("N_Failed_Blocks"),
            ),
            on=sample_id_col,
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.when(pl.col("Sample_FAIL"))
            .then(pl.lit("FAIL"))
            .when(pl.col("Any_WARN") | pl.col("Any_FAIL"))
            .then(pl.lit("WARN"))
            .when(pl.col("Any_PASS"))
            .then(pl.lit("PASS"))
            .otherwise(None)
            .alias("SampleQC")
        )
        .sort(sample_id_col)
    )

    qc_summary = block_status.group_by("SampleQC").agg(pl.len().alias("count"))
    qc_dict = {r["SampleQC"]: r["count"] for r in qc_summary.iter_rows(named=True)}
    logger.info(
        f"Sample QC - PASS: {qc_dict.get('PASS', 0)} | WARN: {qc_dict.get('WARN', 0)} | FAIL: {qc_dict.get('FAIL', 0)}"
    )

    fail_samples, warn_samples, nulled_samples = [], [], []
    if remove_fail:
        fail_samples = block_status.filter(pl.col("SampleQC") == "FAIL")[sample_id_col].to_list()
        if fail_samples:
            logger.warning(
                f"Marking {len(fail_samples)} samples as FAIL (SampleQC FAIL in all blocks)"
            )
            qc_instance._flag_samples(fail_samples, FAIL_SAMPLE_QC)
            qc_instance.qc_results["samples_failed"] = len(fail_samples)

        # A failed run of a sample with a passing run is dropped; the passing run
        # stands.
        dropped_runs = (
            blocks.filter(pl.col("Run_FAIL") & ~pl.col("Sample_FAIL"))
            .select([sample_id_col, "PlateID"])
            .unique()
        )
        if dropped_runs.height:
            logger.info(
                f"{dropped_runs[sample_id_col].n_unique()} samples have a failed run next to a "
                f"passing one; dropped {dropped_runs.height} failed runs and kept the passing runs"
            )
            qc_instance.qc_results["sample_runs_discarded"] = dropped_runs[sample_id_col].n_unique()
            _drop_failed_runs(qc_instance, dropped_runs)

        # Failed blocks of a kept run: null that run's assays in those blocks.
        failed_blocks = blocks.filter(
            pl.col("Block_FAIL") & ~pl.col("Run_FAIL") & ~pl.col("Sample_FAIL")
        )
        nulled_samples = failed_blocks[sample_id_col].unique().sort().to_list()
        logger.info(
            f"{len(nulled_samples)} samples have SampleQC FAIL in some blocks only; "
            f"nulled {failed_blocks['Block_Assays'].sum()} sample x assay values "
            "in the failed blocks"
        )
        if failed_blocks.height:
            _null_failed_blocks(
                qc_instance, failed_blocks.select([sample_id_col, "PlateID", "Block"])
            )

    # Missingness is read after block nulling, so nulled blocks count as missing. An
    # assay is missing for a sample when no kept run has a value for it.
    qc_status = (
        assay_rows()
        .group_by([sample_id_col, qc_instance.protein_id_col])
        .agg(
            [
                (pl.col(data_col).is_null() | pl.col(data_col).is_nan()).all().alias("_missing"),
                (pl.col("AssayQC") != "PASS").any().alias("_warn"),
            ]
        )
        .group_by(sample_id_col)
        .agg(
            [
                pl.col("_missing").sum().alias("Missing_Count"),
                pl.col("_warn").sum().alias("Warn_Count"),
                pl.len().alias("Total_Assays"),
            ]
        )
        .collect()
        .join(block_status, on=sample_id_col, how="left", validate="1:1")
        .with_columns(
            [
                (pl.col("Missing_Count") / pl.col("Total_Assays")).alias("Missing_Frac"),
                (pl.col("Warn_Count") / pl.col("Total_Assays")).alias("Warn_Frac"),
            ]
        )
        .select(
            [
                sample_id_col,
                "SampleQC",
                "Missing_Count",
                "Warn_Count",
                "Total_Assays",
                "Missing_Frac",
                "Warn_Frac",
                "N_Blocks",
                "N_Failed_Blocks",
            ]
        )
        .sort(sample_id_col)
    )

    # Missing rate is a warning only; failed samples are not re-flagged.
    high_missing = qc_status.filter(
        (pl.col("Missing_Frac") > missing_threshold) & ~pl.col(sample_id_col).is_in(fail_samples)
    )[sample_id_col].to_list()

    # The vendor key's volume note: None when the manifest carries none (no rule).
    low_volume = _olink_low_volume_samples(qc_instance, qc_status[sample_id_col].to_list())

    if flag_warn:
        warn_sample_qc = [
            s
            for s in qc_status.filter(pl.col("SampleQC") == "WARN")[sample_id_col]
            if s not in fail_samples
        ]
        if warn_sample_qc:
            qc_instance._flag_samples(warn_sample_qc, WARN_SAMPLE_QC)
        if high_missing:
            logger.warning(
                f"marking {len(high_missing)} samples with "
                f">{missing_threshold * 100:.0f}% missing data as WARN"
            )
            # Track every high-missing sample; list each warned sample once.
            already = set(warn_sample_qc)
            qc_instance.warned_samples.extend(s for s in high_missing if s not in already)
            qc_instance._track_sample_qc(high_missing, WARN_MISSING_RATE)
        # WARN_Low_Volume: WARN only, as SomaScan's; the tube stays in the outlier fits.
        new_low_volume = [] if low_volume is None else _warn_low_volume(qc_instance, low_volume)
        warn_samples = sorted(set(warn_sample_qc) | set(high_missing) | set(new_low_volume))
        if warn_samples:
            logger.warning(f"Marking {len(warn_samples)} samples as WARN")
            qc_instance.qc_results["samples_warned"] = len(warn_samples)

    # An assay left missing by nulled blocks, and high missingness, keep a sample out of
    # the outlier fits.
    left_missing = set(qc_status.filter(pl.col("Missing_Count") > 0)[sample_id_col])
    outlier_fit_exclude = sorted((set(nulled_samples) & left_missing) | set(high_missing))
    logger.info(
        f"{len(outlier_fit_exclude)} samples kept out of the outlier fits "
        f"(nulled blocks or >{missing_threshold * 100:.0f}% missing); they are still scored"
    )

    n_samples = sum(qc_dict.get(k, 0) for k in ("PASS", "WARN", "FAIL"))
    qc_dict = {
        "PASS": n_samples - len(fail_samples) - len(warn_samples),
        "WARN": len(warn_samples),
        "FAIL": len(fail_samples),
    }

    qc_instance._sample_qc_data = {
        "qc_dict": qc_dict,
        "fail_samples": fail_samples,
        "warn_samples": warn_samples,
        "missing_threshold": missing_threshold,
        "outlier_fit_exclude": outlier_fit_exclude,
        # Samples whose manifest volume note matched (failed or warned ones included);
        # None: no rule.
        "low_volume_samples": low_volume,
    }

    qc_instance._sample_qc_status = qc_status.select([sample_id_col, "Warn_Frac", "Missing_Frac"])

    return qc_status


def sample_qc_soma(
    qc_instance,
    remove_fail: bool = True,
    flag_warn: bool = True,
    mad_threshold: float = 6.0,
    fc_threshold: float = 5.0,
    outlier_fraction: float = 0.05,
) -> pl.DataFrame:
    """Checks and handles the SomaScan sample QC flags.

    The checks are RowCheck (SomaLogic's sample QC flag), RFU outliers (samples where
    at least ``outlier_fraction`` of the RFU measurements exceed both
    ``mad_threshold`` MADs and ``fc_threshold`` times the analyte median), low volume
    (``SampleNotes`` containing "Low Volume") and scale factors outside the band.

    A sample is FAIL when RowCheck is not PASS (``FAIL_RowCheck``); WARN for an
    RFU outlier only, low volume or an out-of-band scale factor; otherwise PASS.

    Args:
        qc_instance: SomaQC instance.
        remove_fail: Whether to mark failing samples.
        flag_warn: Whether to mark warned samples.
        mad_threshold: MAD threshold for outlier detection.
        fc_threshold: Fold-change threshold for outlier detection.
        outlier_fraction: Fraction of analytes that must be outliers to flag a sample.

    Returns:
        One row per sample with RowCheck, outlier metrics and scale-factor groups.
    """
    logger.info(
        "checking RowCheck flags, sample-level outliers, and low volume samples (sample QC)..."
    )

    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col

    measurements = qc_instance.samples

    analyte_stats = (
        measurements.group_by(protein_id_col)
        .agg(
            [
                pl.col("RFU").median().alias("median_RFU"),
                (pl.col("RFU") - pl.col("RFU").median()).abs().median().alias("MAD_RFU"),
            ]
        )
        .collect(engine="streaming")
    )

    outlier_summary = (
        measurements.join(analyte_stats.lazy(), on=protein_id_col, how="left")
        .with_columns(
            [
                (
                    (pl.col("RFU") > (pl.col("median_RFU") + mad_threshold * pl.col("MAD_RFU")))
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
                (pl.col("Outlier_Count") / pl.col("Total_Analytes_Outlier_Check")).alias(
                    "RFU_Outlier_Frac"
                )
            ]
        )
        .collect(engine="streaming")
    )

    # The inner join with the study measurements drops control wells.
    qc_status = (
        qc_instance.sample_metadata.select(
            [
                sample_id_col,
                "RowCheck",
                "SampleNotes",
            ]
        )
        .join(
            measurements.group_by(sample_id_col).agg(pl.len().alias("Total_Analytes")),
            on=sample_id_col,
            how="inner",
        )
        .join(outlier_summary.lazy(), on=sample_id_col, how="left")
        .sort(sample_id_col)
        .collect()
    )

    n_null_rowcheck = qc_status["RowCheck"].null_count()
    if n_null_rowcheck:
        raise ValueError(f"{n_null_rowcheck} samples have a null RowCheck verdict")
    rowcheck_failed = qc_status.filter(pl.col("RowCheck") != "PASS")[sample_id_col].to_list()

    rfu_outliers = qc_status.filter(pl.col("RFU_Outlier_Frac") >= outlier_fraction)[
        sample_id_col
    ].to_list()

    low_volume_samples = qc_status.filter(pl.col("SampleNotes").str.contains("Low Volume"))[
        sample_id_col
    ].to_list()

    # RowCheck alone FAILs a sample.
    fail_samples = sorted(rowcheck_failed)

    # RFU-only outliers stay WARN.
    warn_rfu_only = sorted(set(rfu_outliers) - set(fail_samples))
    warn_samples = warn_rfu_only

    if fail_samples and remove_fail:
        logger.warning(f"marking {len(fail_samples)} samples as FAIL (RowCheck)")
        qc_instance._flag_samples(fail_samples, FAIL_ROWCHECK)

    if warn_samples and flag_warn:
        logger.warning(f"marking {len(warn_samples)} samples as WARN")
        if warn_rfu_only:
            logger.info(
                f"  - Sample-level RFU outliers only (>={outlier_fraction * 100:.0f}% measurements exceeding {mad_threshold} MADs and {fc_threshold}x FC): {len(warn_rfu_only)}"
            )
            qc_instance._flag_samples(warn_rfu_only, WARN_RFU_OUTLIER)

    # Low volume is WARN independent of other flags.
    if low_volume_samples and flag_warn:
        logger.info(f"  - Low volume samples: {len(low_volume_samples)}")
        _warn_low_volume(qc_instance, low_volume_samples)

    # Recount after all warnings are added.
    total_warned = len(set(qc_instance.warned_samples))
    total_failed = len(set(qc_instance.failed_samples))
    n_pass = len(qc_status) - total_failed - total_warned
    n_warn = total_warned
    n_fail = total_failed

    logger.info(f"sample QC - PASS: {n_pass} | WARN: {n_warn} | FAIL: {n_fail}")

    qc_dict = {"PASS": n_pass, "WARN": n_warn, "FAIL": n_fail}
    qc_instance._sample_qc_data = {
        "qc_dict": qc_dict,
        "fail_samples": fail_samples,
        "warn_samples": warn_samples,
        "mad_threshold": mad_threshold,
        "fc_threshold": fc_threshold,
        "outlier_fraction": outlier_fraction,
        # No blocks and no missing-rate rule on SomaScan: every sample is fitted.
        "outlier_fit_exclude": [],
    }

    qc_instance._sample_qc_status = qc_status.select(
        [sample_id_col, "RowCheck", "RFU_Outlier_Frac"]
    )

    return qc_status


def assay_qc_olink(
    qc_instance,
    warn_threshold: float = 0.1,
) -> pl.DataFrame:
    """Checks AssayQC for problematic proteins.

    Args:
        qc_instance: OlinkQC instance.
        warn_threshold: Proportion of warned rows above which a protein is failed.

    Returns:
        Dictionary with keys: assay_qc, sample_counts, assay_counts, warn_threshold,
            n_any_warn, to_remove, to_flag.
    """
    logger.info("Assay QC flags")

    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    protein_name_col = qc_instance.protein_name_col
    data_col = qc_instance.data_col

    # AssayQC is set per assay and plate, so aggregate it over the study-sample rows
    # of each OlinkID (control wells would weight plates by their controls). The Assay
    # name only labels the output: OlinkIDs that share one are flagged independently.
    protein_data = qc_instance.protein_assays
    _require_row_flags(protein_data, ["AssayQC", "AssayType", "SampleType"])
    protein_data = protein_data.filter(
        (pl.col("AssayType") == "assay") & (pl.col("SampleType") == qc_instance.sample_type)
    )
    _require_verdicts(protein_data, "AssayQC")

    if qc_instance.failed_samples:
        logger.info(f"Excluding {len(qc_instance.failed_samples)} failed samples from Assay QC")
        protein_data = protein_data.filter(~pl.col(sample_id_col).is_in(qc_instance.failed_samples))

    assay_names = qc_instance.analyte_metadata.select([protein_id_col, protein_name_col])
    assay_qc = (
        protein_data.group_by(protein_id_col)
        .agg(
            [
                (pl.col("AssayQC") == "PASS").sum().alias("PASS"),
                (pl.col("AssayQC") != "PASS").sum().alias("WARN_FAIL"),
                pl.len().alias("Total"),
                (pl.col(data_col).is_null() | pl.col(data_col).is_nan())
                .sum()
                .alias("Missing_Count"),
            ]
        )
        .join(assay_names, on=protein_id_col, how="left", validate="1:1")
        .with_columns(
            [
                (pl.col("WARN_FAIL") / pl.col("Total")).alias("Warn_Frac"),
                (pl.col("Missing_Count") / pl.col("Total")).alias("Missing_Frac"),
            ]
        )
        .select(
            [
                protein_id_col,
                protein_name_col,
                "PASS",
                "WARN_FAIL",
                "Total",
                "Missing_Count",
                "Warn_Frac",
                "Missing_Frac",
            ]
        )
        .sort(protein_id_col)
        .collect()
    )

    to_remove = assay_qc.filter(pl.col("Warn_Frac") > warn_threshold)[protein_id_col].to_list()
    n_any_warn = (assay_qc["Warn_Frac"] > 0).sum()

    # Proteins with warnings but below the removal threshold.
    to_flag = assay_qc.filter((pl.col("WARN_FAIL") > 0) & (pl.col("Warn_Frac") <= warn_threshold))[
        protein_id_col
    ].to_list()

    logger.info(
        f"proteins with warnings: {n_any_warn} | >{warn_threshold * 100:.0f}% warned: {len(to_remove)}"
    )

    if to_remove:
        logger.warning(f"Marking {len(to_remove)} proteins as FAIL")
        qc_instance._flag_proteins(to_remove, FAIL_ASSAY_QC)

    if to_flag:
        logger.warning(f"Marking {len(to_flag)} proteins as WARN")
        qc_instance._flag_proteins(to_flag, WARN_ASSAY_QC)

    qc_instance.qc_results["proteins_failed"] = len(to_remove)
    qc_instance.qc_results["proteins_warned"] = len(to_flag)

    n_pass = len(assay_qc.filter(pl.col("Warn_Frac") == 0))
    n_fail = len(assay_qc.filter(pl.col("Warn_Frac") > warn_threshold))
    n_warn = len(
        assay_qc.filter((pl.col("Warn_Frac") > 0) & (pl.col("Warn_Frac") <= warn_threshold))
    )

    qc_instance._assay_qc_status = assay_qc.select([protein_id_col, "Warn_Frac", "Missing_Frac"])

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
    """Checks the ColCheck flags (assay QC) in SomaScan data.

    ColCheck is a per-analyte flag, so any analyte with ColCheck != "PASS" is marked as
    WARN, never FAIL. The result has the same shape as the Olink assay QC.

    Args:
        qc_instance: SomaQC instance.

    Returns:
        Dictionary with keys: assay_qc, sample_counts, assay_counts, n_any_warn,
            to_remove, to_flag.
    """
    logger.info("checking ColCheck flags (assay QC)...")

    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col

    analyte_data = qc_instance.samples.join(
        qc_instance.analyte_metadata.select([protein_id_col, "ColCheck"]),
        on=protein_id_col,
        how="left",
    )

    if qc_instance.failed_samples:
        logger.info(f"Excluding {len(qc_instance.failed_samples)} failed samples from analyte QC")
        analyte_data = analyte_data.filter(~pl.col(sample_id_col).is_in(qc_instance.failed_samples))

    # One row per analyte; ColCheck is constant per analyte, so take the first value.
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

    n_null_colcheck = analyte_qc["ColCheck"].null_count()
    if n_null_colcheck:
        raise ValueError(f"{n_null_colcheck} analytes have a null ColCheck verdict")

    # All ColCheck warnings go to WARN, not FAIL.
    to_warn = analyte_qc.filter(pl.col("ColCheck") != "PASS")[protein_id_col].to_list()

    n_pass = len(analyte_qc.filter(pl.col("ColCheck") == "PASS"))
    n_warn = len(to_warn)
    n_fail = 0

    logger.info(f"analyte ColCheck - PASS: {n_pass} | WARN: {n_warn}")

    # Read by finalize_assay_flags_soma.
    qc_instance._colcheck_warn_set = set(to_warn)

    if to_warn:
        logger.warning(f"marking {len(to_warn)} analytes as WARN (ColCheck)")
        qc_instance._flag_proteins(to_warn, WARN_COLCHECK)

    qc_instance._assay_qc_status = analyte_qc.select([protein_id_col, "ColCheck", "Warn_Frac"])

    qc_dict = qc_instance._sample_qc_data.get("qc_dict", {})
    sample_counts = {s: qc_dict.get(s, 0) for s in ["PASS", "WARN", "FAIL"]}
    assay_counts = {"PASS": n_pass, "WARN": n_warn, "FAIL": n_fail}

    return {
        "assay_qc": analyte_qc,
        "sample_counts": sample_counts,
        "assay_counts": assay_counts,
        "n_any_warn": n_warn,
        "to_remove": [],
        "to_flag": to_warn,
    }


def assay_saturation_soma(
    qc_instance,
    rfu_threshold: float = 80_000,
    sample_frac: float = 0.50,
) -> dict:
    """Flags analytes saturated in more than ``sample_frac`` of the study samples.

    An analyte is saturated in a sample when its RFU exceeds ``rfu_threshold``. High
    saturation indicates detector ceiling effects: measurements are unreliable at or
    above the instrument's dynamic range limit (about 65,535 to 100,000 RFU depending
    on the scanner).

    Stores ``_saturation_flagged`` (the set of flagged SeqIds) on the instance.

    Args:
        qc_instance: SomaQC instance.
        rfu_threshold: RFU ceiling above which a measurement is saturated.
        sample_frac: Fraction of samples that must exceed the threshold to flag the
            analyte.

    Returns:
        Dictionary with keys: saturation_stats, flagged, rfu_threshold, sample_frac.
    """
    logger.info(
        f"checking saturation: >{rfu_threshold:,.0f} RFU in >{sample_frac * 100:.0f}% samples..."
    )

    protein_id_col = qc_instance.protein_id_col
    sample_id_col = qc_instance.sample_id_col

    samples = qc_instance.samples
    if qc_instance.failed_samples:
        samples = samples.filter(~pl.col(sample_id_col).is_in(qc_instance.failed_samples))

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
        f"(>{sample_frac * 100:.0f}% samples >{rfu_threshold:,.0f} RFU)"
    )

    qc_instance._saturation_flagged = set(flagged)

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
    """Flags analytes whose sample-to-buffer median RFU ratio is below ``sb_threshold``.

    A low signal-to-background ratio means the biological signal cannot be reliably
    distinguished from background hybridization noise.

    Stores ``_sb_ratio_flagged`` (the set of flagged SeqIds) on the instance.

    Args:
        qc_instance: SomaQC instance.
        sb_threshold: Minimum acceptable signal-to-background ratio.
        buffer_sample_type: SampleType label for buffer/blank samples in metadata.

    Returns:
        Dictionary with keys: sb_stats, flagged, sb_threshold.

    Raises:
        ValueError: If no buffer samples are found.
    """
    logger.info(
        f"checking signal-to-background ratio (threshold: sample/buffer < {sb_threshold})..."
    )

    protein_id_col = qc_instance.protein_id_col
    sample_id_col = qc_instance.sample_id_col

    samples = qc_instance.samples
    if qc_instance.failed_samples:
        samples = samples.filter(~pl.col(sample_id_col).is_in(qc_instance.failed_samples))

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
        raise ValueError(
            f"No buffer samples found (SampleType='{buffer_sample_type}'): the signal-to-background check needs buffer wells"
        )

    buffer_medians = (
        qc_instance.get_data()
        .join(buffer_ids, on="_sample_idx", how="semi")
        .group_by(protein_id_col)
        .agg(pl.col("RFU").median().alias("median_buffer_rfu"))
        .collect()
    )

    sb_stats = sample_medians.join(buffer_medians, on=protein_id_col, how="left").with_columns(
        (pl.col("median_sample_rfu") / pl.col("median_buffer_rfu")).alias("sb_ratio"),
    )

    flagged = sb_stats.filter(pl.col("sb_ratio") < sb_threshold)[protein_id_col].to_list()

    logger.info(f"signal-to-background: {len(flagged)} analytes flagged (ratio < {sb_threshold})")

    qc_instance._sb_ratio_flagged = set(flagged)

    return {
        "sb_stats": sb_stats,
        "flagged": flagged,
        "sb_threshold": sb_threshold,
    }


def finalize_assay_flags_soma(qc_instance) -> dict:
    """Consolidates individual assay-check results on the QC instance.

    Run after the individual checks (ColCheck, Intra-CV, Detection Rate, Saturation,
    Signal-to-Background Ratio).

    Each check must have stored its flag set on the instance: ``_colcheck_warn_set``
    (``filter_assay_qc``), ``_cv_fail_proteins`` (``filter_assay_cv``),
    ``_detection_fail_proteins`` (``filter_assay_detection``), ``_saturation_flagged``
    (``filter_assay_saturation``) and ``_sb_ratio_flagged``
    (``filter_assay_sb_ratio``).

    Resets ``warned_proteins``, ``failed_proteins`` and ``protein_qc_actions`` on the
    instance, then rebuilds them from the consolidated flags.

    Args:
        qc_instance: SomaQC instance after all individual checks have been run.

    Returns:
        Dictionary with n_pass, n_warn, n_fail, n_total, per_analyte_flags and
            flag_counts.

    Raises:
        ValueError: If a check has not stored its flag set.
    """

    protein_id_col = qc_instance.protein_id_col

    # Each input is stored by its own step; a step that did not run raises here instead
    # of passing as empty.
    inputs = {
        "ColCheck": ("_colcheck_warn_set", "filter_assay_qc"),
        "Intra_CV": ("_cv_fail_proteins", "filter_assay_cv"),
        "Detection_Rate": ("_detection_fail_proteins", "filter_assay_detection"),
        "Saturation": ("_saturation_flagged", "filter_assay_saturation"),
        "SB_Ratio": ("_sb_ratio_flagged", "filter_assay_sb_ratio"),
    }
    missing = [
        f"{attr} (set by {step})"
        for attr, step in inputs.values()
        if not hasattr(qc_instance, attr)
    ]
    if missing:
        raise ValueError("finalize_assay_flags_soma is missing its inputs: " + ", ".join(missing))
    flag_map = {name: set(getattr(qc_instance, attr)) for name, (attr, _) in inputs.items()}

    # Every analyte of the assay, not only those with a CV row.
    all_analytes = set(
        qc_instance.analyte_metadata.select(protein_id_col).collect()[protein_id_col].to_list()
    )

    per_analyte_flags: dict[str, list[str]] = {}
    for analyte in sorted(all_analytes):
        flags = [name for name, flagged_set in flag_map.items() if analyte in flagged_set]
        if flags:
            per_analyte_flags[analyte] = flags

    multi_fail = [a for a, f in per_analyte_flags.items() if len(f) >= 2]
    single_warn = [a for a, f in per_analyte_flags.items() if len(f) == 1]

    # Reset protein WARN/FAIL state and rebuild from scratch.
    qc_instance.failed_proteins = multi_fail
    qc_instance.warned_proteins = single_warn
    qc_instance.protein_qc_actions = []

    # Re-track individual check flags (per-check reporting detail).
    for flag_name, flagged_set in flag_map.items():
        if flagged_set:
            qc_instance._track_protein_qc(list(flagged_set), f"WARN_{flag_name}")

    # The consolidated FAIL classification drives the waterfall summary table.
    if multi_fail:
        qc_instance._track_protein_qc(multi_fail, FAIL_MULTIPLE_FLAGS)

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
