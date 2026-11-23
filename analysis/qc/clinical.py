"""Clinical exclusion rules and the sample-flag names."""

from __future__ import annotations

import logging

import polars as pl

from analysis.qc.exclusion_table import (
    exclusion_table,
    first_reason,
    non_analysis_visit_counts,
)
from analysis.qc.flags import (
    CORRECTION_EXCLUDE_FLAG,
    FAIL_AFTER_TREATMENT,
    FAIL_ASSAY_QC,
    FAIL_BASELINE_AFTER_DOSE,
    FAIL_BRIDGE,
    FAIL_DUPLICATE_ASSAY,
    FAIL_DUPLICATE_VISIT,
    FAIL_IDENTITY_FOREIGN,
    FAIL_IDENTITY_UNRESOLVED,
    FAIL_INTRA_CV_AND_DETECTION_RATE,
    FAIL_LOW_DETECTION,
    FAIL_MEDIAN_IQR_OUTLIER,
    FAIL_MULTIPLE_FLAGS,
    FAIL_NOT_ANALYSIS_VISIT,
    FAIL_NOT_ASSAYED,
    FAIL_NO_ADSL_MATCH,
    FAIL_NO_ADT,
    FAIL_NO_BASELINE,
    FAIL_NO_MANIFEST_MATCH,
    FAIL_NO_TREATMENT_ARM,
    FAIL_NO_TREATMENT_WINDOW,
    FAIL_ONLY_BASELINE,
    FAIL_PCA_OUTLIER,
    FAIL_ROWCHECK,
    FAIL_SAMPLE_QC,
    WARN_AGE_CONCORDANCE,
    WARN_AGE_CONSISTENCY,
    WARN_AGE_PREDICTION,
    WARN_ASSAY_QC,
    WARN_COLCHECK,
    WARN_DETECTION_RATE,
    WARN_IDENTITY_AMBIGUOUS,
    WARN_IDENTITY_PQTL,
    WARN_IDENTITY_VETOED,
    WARN_INTRA_CV,
    WARN_LOW_VOLUME,
    WARN_MISSING_RATE,
    WARN_RFU_OUTLIER,
    WARN_SAMPLE_QC,
    WARN_SATURATION,
    WARN_SB_RATIO,
    WARN_SEX_CONCORDANCE,
)
from analysis.qc.utils import ensure_date_column, ensure_lazy

logger = logging.getLogger(__name__)

# The flag names are in flags.py. The manifest and clinical exclusion flags have their
# precedence in the exclusion table in EXCLUSION_REASON_FLAGS below.
# ── Identity flags ──
# The FAIL flags cascade through the clinical rules and are exclusion reasons; the WARN
# flags change Status only.
IDENTITY_FAIL_FLAGS = (FAIL_IDENTITY_FOREIGN, FAIL_IDENTITY_UNRESOLVED)
IDENTITY_WARN_FLAGS = (
    WARN_IDENTITY_VETOED,
    WARN_IDENTITY_AMBIGUOUS,
    WARN_IDENTITY_PQTL,
)
# Technical failures rank after the manifest group and before identity. The
# strings are the ones quality_control.py and outliers.py emit.
TECHNICAL_REASON_FLAGS = (
    FAIL_SAMPLE_QC,
    FAIL_PCA_OUTLIER,
    FAIL_MEDIAN_IQR_OUTLIER,
    FAIL_ROWCHECK,
)

# The treatment-disposition values the on-treatment window recognizes.
STATUS_COMPLETED = "COMPLETED"
STATUS_DISCONTINUED = "DISCONTINUED"

# Precedence: correction label, manifest, technical, identity, clinical. A
# sample with several flags is counted once, under the first one listed.
EXCLUSION_REASON_FLAGS = (
    FAIL_BRIDGE,
    FAIL_NO_MANIFEST_MATCH,
    FAIL_NOT_ASSAYED,
    *TECHNICAL_REASON_FLAGS,
    *IDENTITY_FAIL_FLAGS,
    FAIL_NO_TREATMENT_ARM,
    FAIL_NO_ADSL_MATCH,
    FAIL_NO_TREATMENT_WINDOW,
    FAIL_NO_ADT,
    FAIL_AFTER_TREATMENT,
    FAIL_NOT_ANALYSIS_VISIT,
    FAIL_DUPLICATE_VISIT,
    FAIL_BASELINE_AFTER_DOSE,
    FAIL_NO_BASELINE,
    FAIL_ONLY_BASELINE,
)


# Flag columns each QC summary always carries, all-False when nothing fired.
# They must match every flag string QC can emit. FAIL_Not_Assayed is table-only and is not here.
_SAMPLE_FLAGS_COMMON = (
    FAIL_NO_MANIFEST_MATCH,
    CORRECTION_EXCLUDE_FLAG,
    FAIL_NO_TREATMENT_ARM,
    FAIL_PCA_OUTLIER,
    FAIL_MEDIAN_IQR_OUTLIER,
    WARN_SEX_CONCORDANCE,
    WARN_AGE_CONCORDANCE,
    WARN_AGE_PREDICTION,
    WARN_AGE_CONSISTENCY,
    FAIL_BRIDGE,
    FAIL_NO_ADSL_MATCH,
    FAIL_NO_TREATMENT_WINDOW,
    FAIL_NO_ADT,
    FAIL_AFTER_TREATMENT,
    FAIL_NOT_ANALYSIS_VISIT,
    FAIL_DUPLICATE_VISIT,
    FAIL_BASELINE_AFTER_DOSE,
    FAIL_NO_BASELINE,
    FAIL_ONLY_BASELINE,
    *IDENTITY_FAIL_FLAGS,
    *IDENTITY_WARN_FLAGS,
)
_PROTEIN_FLAGS_COMMON = (
    WARN_DETECTION_RATE,
    WARN_INTRA_CV,
)
# Olink only: the SomaScan consolidation (``finalize_assay_flags_soma``) rebuilds every
# protein flag as WARN_<check> and FAIL_Multiple_Flags, so these never fire there.
OLINK_ONLY_PROTEIN_FLAGS = (FAIL_INTRA_CV_AND_DETECTION_RATE, FAIL_LOW_DETECTION)
SAMPLE_QC_FLAGS = {
    "olink": (
        *_SAMPLE_FLAGS_COMMON,
        FAIL_SAMPLE_QC,
        WARN_SAMPLE_QC,
        WARN_MISSING_RATE,
        WARN_LOW_VOLUME,
    ),
    "soma": (
        *_SAMPLE_FLAGS_COMMON,
        FAIL_ROWCHECK,
        WARN_RFU_OUTLIER,
        WARN_LOW_VOLUME,
    ),
}
# Written by ``SomaQC.finalize_assay_flags``, called after the saturation and
# signal-to-background checks.
SOMA_FINALIZE_FLAGS = (FAIL_MULTIPLE_FLAGS, WARN_SATURATION, WARN_SB_RATIO)
PROTEIN_QC_FLAGS = {
    "olink": (
        *OLINK_ONLY_PROTEIN_FLAGS,
        *_PROTEIN_FLAGS_COMMON,
        FAIL_ASSAY_QC,
        WARN_ASSAY_QC,
        FAIL_DUPLICATE_ASSAY,
    ),
    "soma": (*_PROTEIN_FLAGS_COMMON, WARN_COLCHECK, *SOMA_FINALIZE_FLAGS),
}


def _adsl_dates(adsl_df: pl.DataFrame, cols: list[str], date_format: str) -> list[pl.Expr]:
    """Returns expressions casting the ADSL columns ``cols`` to Date.

    Raises:
        ValueError: If a column that has values is not a date.
    """

    def as_date(col):
        dtype = adsl_df.schema[col]
        if dtype == pl.Null or (dtype.is_numeric() and adsl_df[col].null_count() == adsl_df.height):
            # An all-null date column is read as null dates.
            return pl.lit(None, dtype=pl.Date).alias(col)
        expr = ensure_date_column(adsl_df, col, date_format).alias(col)
        if adsl_df.select(expr).schema[col] != pl.Date:
            raise ValueError(f"ADSL column {col} has type {dtype}, not a date")
        return expr

    return [as_date(c) for c in cols]


def _first_present(adsl_cols: list[str], candidates: list[str], key: str) -> str:
    present = [c for c in candidates if c in adsl_cols]
    if not present:
        raise ValueError(f"ADSL has none of the columns {', '.join(candidates)} ([clinical] {key})")
    return present[0]


def _window_end(adsl: pl.LazyFrame, config, date_format: str) -> tuple[pl.DataFrame, str]:
    """Builds the per-subject on-treatment window.

    Returns the window frame and the name of the last-dose ADSL column used. The frame
    has USUBJID, ``completion_status`` (EOTSTT), ``last_dose`` (the first
    ``config.clinical_last_dose`` column present in ADSL) and ``window_end``:

    - DISCONTINUED: ``last_dose`` plus
      ``config.clinical_discontinued_days_after_last_dose`` days. Null when
      ``last_dose`` is null; the clinical rules fail such a subject's samples first.
    - COMPLETED: null, so no date cut from dosing.
    - Any other value, including null: null here. ``_apply_rules`` raises for it when
      the subject still has study samples; ``dose_timing`` only reports it.

    ``config.clinical_window_stop_events`` then move ``window_end`` earlier, for
    completers and discontinuers alike, to the earliest non-null stop date. A stop event
    never defines a window for a discontinuer with no last dose.

    Raises:
        ValueError: If ADSL lacks the completion-status column, every last-dose
            column, or a stop-event column, or a date column is not a date.
    """
    adsl_cols = adsl.collect_schema().names()
    status_col = config.clinical_completion_status
    if status_col not in adsl_cols:
        raise ValueError(f"ADSL has no column {status_col} ([clinical] completion_status)")
    last_col = _first_present(adsl_cols, config.clinical_last_dose, "last_dose")
    stop_cols = config.clinical_window_stop_events
    missing = [c for c in stop_cols if c not in adsl_cols]
    if missing:
        raise ValueError(
            f"ADSL lacks window_stop_events column(s) {', '.join(missing)} "
            "([clinical] window_stop_events)"
        )
    adsl_df = adsl.select(
        "USUBJID", pl.col(status_col).cast(pl.Utf8), last_col, *stop_cols
    ).collect()
    dates = adsl_df.select(
        "USUBJID",
        status_col,
        *_adsl_dates(adsl_df, [last_col, *stop_cols], date_format),
    )
    days = config.clinical_discontinued_days_after_last_dose
    discontinued = pl.col(status_col) == STATUS_DISCONTINUED
    base_end = pl.when(discontinued).then(pl.col(last_col) + pl.duration(days=days))
    earliest_stop = pl.min_horizontal(base_end, *stop_cols)
    window = dates.select(
        "USUBJID",
        pl.col(status_col).alias("completion_status"),
        pl.col(last_col).alias("last_dose"),
        pl.when(discontinued & pl.col(last_col).is_null())
        .then(None)
        .otherwise(earliest_stop)
        .alias("window_end"),
    )
    if stop_cols:
        n_stop = dates.select(pl.min_horizontal(*stop_cols).is_not_null().sum()).item()
        logger.info(
            f"window_stop_events {', '.join(stop_cols)} give {n_stop} subjects a stop date "
            "(the window ends there if that is earlier than last dose + "
            f"{days} days, or if the subject completed)"
        )
    return window, last_col


def _first_dose(adsl: pl.LazyFrame, config, date_format: str) -> tuple[pl.DataFrame, str]:
    """Returns ``first_dose`` per subject and the ADSL column it came from.

    The column is the first ``clinical_first_dose`` column present in ADSL.
    """
    col = _first_present(adsl.collect_schema().names(), config.clinical_first_dose, "first_dose")
    adsl_df = adsl.select("USUBJID", col).collect()
    return (
        adsl_df.select("USUBJID", *_adsl_dates(adsl_df, [col], date_format)).rename(
            {col: "first_dose"}
        ),
        col,
    )


def dose_timing(
    qc_instance, sample_date_col: str = "ADT", date_format: str = "%Y-%m-%d"
) -> pl.DataFrame:
    """Returns dose dates and draw timing for every sample with a subject.

    Informational only: no rule reads it. The subject is the one in ``adsl_merged``, so
    a reassigned tube is timed against its new owner.

    Args:
        qc_instance: QC instance with ADSL, manifest, ``adsl_merged`` and ``config``.
        sample_date_col: Manifest column with the sample collection date.
        date_format: strftime format for string date columns.

    Returns:
        One row per SampleID with ``TRTEDT`` (the last dose), the completion status
        (a column named after the ADSL column, ``EOTSTT`` by default),
        ``TRT_WINDOW_END`` (the window end the rules used; null for a completer without
        a stop event), and ``DAYS_SINCE_FIRST_DOSE`` and ``DAYS_SINCE_LAST_DOSE`` (the
        manifest date minus first dose and minus ``TRTEDT``; negative before the dose;
        the latter null when ``TRTEDT`` is null).
    """
    sid = qc_instance.sample_id_col
    config = qc_instance.config
    adsl = ensure_lazy(qc_instance.adsl)
    manifest = ensure_lazy(qc_instance.manifest)
    window, col = _window_end(adsl, config, date_format)
    first, _ = _first_dose(adsl, config, date_format)
    status_name = config.clinical_completion_status
    last = window.select("USUBJID", pl.col("last_dose").alias("TRTEDT"))
    logger.info(f"TRTEDT in the sample summary comes from ADSL {col}")
    dates = (
        manifest.select(
            pl.col(sid).cast(pl.Utf8),
            ensure_date_column(manifest, sample_date_col, date_format).alias("_adt"),
        )
        .unique(subset=[sid], keep="first", maintain_order=True)
        .collect()
    )
    mapping = _one_row_per(
        qc_instance.adsl_merged.select(pl.col(sid).cast(pl.Utf8), "USUBJID"),
        sid,
        "merged ADSL sample to subject",
    )
    return (
        mapping.join(dates, on=sid, how="left")
        .join(
            window.select(
                "USUBJID",
                pl.col("completion_status").alias(status_name),
                pl.col("window_end").alias("TRT_WINDOW_END"),
            ),
            on="USUBJID",
            how="left",
        )
        .join(first, on="USUBJID", how="left")
        .join(last, on="USUBJID", how="left")
        .select(
            sid,
            "TRTEDT",
            status_name,
            "TRT_WINDOW_END",
            (pl.col("_adt") - pl.col("first_dose")).dt.total_days().alias("DAYS_SINCE_FIRST_DOSE"),
            (pl.col("_adt") - pl.col("TRTEDT")).dt.total_days().alias("DAYS_SINCE_LAST_DOSE"),
        )
        .sort(sid)
    )


def sample_unsuitable(
    qc_instance,
    sample_date_col="ADT",
    date_format="%Y-%m-%d",
    screening_dates: pl.DataFrame | None = None,
):
    """Flags samples unsuitable for longitudinal analysis and counts them per rule.

    Rules, in order; each skips samples an earlier rule failed:

    1. Bridge samples (FAIL_Bridge_Sample) and samples without an ADSL subject
       (FAIL_No_ADSL_Match).
    2. Study samples of a DISCONTINUED subject with a null last dose
       (FAIL_No_Treatment_Window). A stop-event date does not stand in for it.
    3. Study samples with a null sample date (FAIL_No_ADT).
    4. On-treatment window, by completion status
       (``clinical_completion_status``). DISCONTINUED: a sample fails
       (FAIL_After_Treatment) when ADT > last dose +
       ``clinical_discontinued_days_after_last_dose``; the last dose is the first
       ``clinical_last_dose`` column present in ADSL. COMPLETED: no date cut from
       dosing. Either window moves earlier to the earliest non-null
       ``clinical_window_stop_events`` date.
    5. Visit not in ``visits_analysis`` (FAIL_Not_Analysis_Visit): other
       visits outside ``visits_analysis`` and null VISITNUM.
       Exception: a subject's last draw with ADT on or before first dose (the first
       ``clinical_first_dose`` column in ADSL) and after its last screening date
       counts as baseline.
    6. Two or more remaining samples of one subject at one analysis visit all fail
       (FAIL_Duplicate_Visit).
    7. A sample at the baseline visit with ADT after first dose
       (FAIL_Baseline_After_Dose). When a subject has a pre-dose draw, that
       draw is the baseline and any other baseline-coded sample fails rule 5.
    8. Participants without a sample at ``visits_baseline`` (FAIL_No_Baseline), then
       participants whose remaining samples are all at baseline
       (FAIL_Only_Baseline).

    VISITNUM comes from ``adsl_merged``, so a reassigned sample is judged at its
    inferred visit. ADT comes from the manifest (the tube's own date).

    Sets ``qc_instance.analysis_visits``: SampleID and Float64 ``VISITNUM_ANALYSIS``
    for each sample the visit rule gave an analysis visit (the baseline code for a
    pre-dose exception, else VISITNUM). Samples failed before the visit rule get none.
    VISITNUM itself stays the tube's visit.

    Also sets ``qc_instance.dose_timing`` (see ``dose_timing``), which the sample
    summary exports. No rule reads it.

    Args:
        qc_instance: QC instance with samples, ADSL, manifest and ``config``.
        sample_date_col: Manifest column with the sample collection date.
        date_format: strftime format for string date columns.
        screening_dates: USUBJID and SCREENING_DT (see ``screening_dates``). None, or
            a subject absent from it, means no screening date is known and the
            pre-dose exception needs only ADT on or before first dose.

    Returns:
        Sample counts per rule (no_treatment_window, no_adt, after_treatment,
        not_analysis_visit, duplicate_visit, baseline_after_dose, no_baseline,
        only_baseline, bridge_samples).

    Raises:
        ValueError: If ADSL, the manifest or the merged ADSL is missing, the config has
            no ``[visits]``, ADSL lacks a column the window needs, or a subject with
            study samples at the window step has a status other than COMPLETED or
            DISCONTINUED.
    """
    counts, qc_instance.analysis_visits = _apply_rules(
        qc_instance,
        sample_date_col,
        date_format,
        screening_dates,
        failed=set(qc_instance.failed_samples),
        emit=qc_instance._flag_samples,
    )
    qc_instance.dose_timing = dose_timing(qc_instance, sample_date_col, date_format)
    return counts


def _one_row_per(frame, key: str, what: str) -> pl.DataFrame:
    """Returns the distinct rows of ``frame``, in order of first appearance.

    Raises ``ValueError`` when a ``key`` still has more than one row, so two rows of one
    key that disagree stop the run and neither is picked.
    """
    rows = ensure_lazy(frame).unique(maintain_order=True).collect()
    n_conflicting = rows.height - rows[key].n_unique()
    if n_conflicting:
        raise ValueError(f"{what}: {n_conflicting} rows repeat a {key} with different values")
    return rows


def _require_clinical_inputs(qc_instance) -> None:
    """Raises ``ValueError`` naming each of ADSL, manifest and merged ADSL that is missing."""
    missing = [
        name for name in ("adsl", "manifest", "adsl_merged") if getattr(qc_instance, name) is None
    ]
    if missing:
        raise ValueError(
            f"The clinical rules need ADSL, the manifest and the merged ADSL; missing: {missing}"
        )


def _apply_rules(qc_instance, sample_date_col, date_format, screening_dates, failed: set, emit):
    """Applies the rules of ``sample_unsuitable``, leaving the flagging to ``emit``.

    ``failed`` holds the samples the rules skip and grows as they flag; ``emit(ids,
    reason)`` records a flag. ``sample_unsuitable`` passes the QC instance's failed
    samples and ``_flag_samples``; ``exclusion_frame`` passes only the upstream
    failures and a collector, so no QC state changes.

    Returns:
        The counts per rule and the analysis visits (sample ID and Float64
        ``VISITNUM_ANALYSIS``).

    Raises:
        ValueError: If ADSL, the manifest or the merged ADSL is missing.
    """
    record = emit

    def emit(ids, reason):
        failed.update(ids)
        record(ids, reason)

    _require_clinical_inputs(qc_instance)

    logger.info("starting clinical filtering for longitudinal analysis")

    sample_id_col = qc_instance.sample_id_col

    sample_mapping = _one_row_per(
        qc_instance.adsl_merged.select([sample_id_col, "USUBJID"]),
        sample_id_col,
        "merged ADSL sample to subject",
    )

    adsl = ensure_lazy(qc_instance.adsl)
    manifest = ensure_lazy(qc_instance.manifest)

    # Every study sample, failed or not: a bridge carries FAIL_Bridge_Sample even when
    # an earlier step failed it.
    study_samples = _one_row_per(
        qc_instance.sample_metadata.select([sample_id_col, "SampleType"]).filter(
            pl.col("SampleType") == qc_instance.sample_type
        ),
        sample_id_col,
        "study samples",
    )
    # The other rules skip samples already failed.
    current_samples = study_samples.filter(~pl.col(sample_id_col).is_in(list(failed)))

    samples_with_usubjid = current_samples.select([sample_id_col]).join(
        sample_mapping, on=sample_id_col, how="left"
    )

    # Bridge samples: the manifest BridgeSample flag plus the BRIDGE_ USUBJID prefix.
    # A null USUBJID is not a bridge indicator; those samples are handled below.
    manifest_schema = manifest.collect_schema().names()
    if "BridgeSample" not in manifest_schema:
        logger.warning(
            "manifest has no 'BridgeSample' column; falling back to USUBJID "
            "'BRIDGE_' prefix to identify bridge samples for clinical filtering"
        )

    bridge_samples = study_samples.filter(
        pl.col(sample_id_col).cast(pl.Utf8).is_in(qc_instance.bridge_sample_ids)
    )[sample_id_col].to_list()

    if bridge_samples:
        emit(bridge_samples, FAIL_BRIDGE)
        logger.info(f"Marked {len(bridge_samples)} bridge samples with FAIL_Bridge_Sample")

        samples_with_usubjid = samples_with_usubjid.filter(
            ~pl.col(sample_id_col).cast(pl.Utf8).is_in(set(bridge_samples))
        )

    # Samples with no USUBJID cannot be date-filtered without a subject's dates.
    no_adsl_match_samples = samples_with_usubjid.filter(pl.col("USUBJID").is_null())[
        sample_id_col
    ].to_list()

    if no_adsl_match_samples:
        emit(no_adsl_match_samples, FAIL_NO_ADSL_MATCH)
        logger.info(
            f"Marked {len(no_adsl_match_samples)} samples with no ADSL match with FAIL_No_ADSL_Match"
        )

        samples_with_usubjid = samples_with_usubjid.filter(pl.col("USUBJID").is_not_null())

    # Subjects absent from ADSL cannot be date-filtered. The subject is the one in
    # adsl_merged, so a reassigned sample is judged under its new owner.
    adsl_subjects = adsl.select("USUBJID").unique().collect()
    not_in_adsl = (
        samples_with_usubjid.join(adsl_subjects, on="USUBJID", how="anti")[sample_id_col]
        .unique(maintain_order=True)
        .to_list()
    )
    if not_in_adsl:
        logger.warning(
            f"{len(not_in_adsl)} samples have a USUBJID missing from ADSL; "
            "flagging with FAIL_No_ADSL_Match"
        )
        emit(not_in_adsl, FAIL_NO_ADSL_MATCH)
        samples_with_usubjid = samples_with_usubjid.filter(
            ~pl.col(sample_id_col).is_in(not_in_adsl)
        )

    # VISITNUM from adsl_merged: a reassigned sample is judged at its inferred visit.
    samples_with_usubjid = samples_with_usubjid.join(
        qc_instance.adsl_merged.select(
            sample_id_col, pl.col("VISITNUM").cast(pl.Float64)
        ).collect(),
        on=sample_id_col,
        how="left",
    )

    config = qc_instance.config
    if config.visits_baseline is None:
        raise ValueError(
            "The clinical step needs [visits] baseline and analysis in the trial config"
        )
    baseline = config.visits_baseline
    analysis = list(config.visits_analysis)
    logger.info(f"analysis visits {analysis}, baseline {baseline}")

    failed_counts = {
        "no_treatment_window": 0,
        "no_adt": 0,
        "after_treatment": 0,
        "not_analysis_visit": 0,
        "duplicate_visit": 0,
        "baseline_after_dose": 0,
        "no_baseline": 0,
        "only_baseline": 0,
    }

    window_end, last_dose_col = _window_end(adsl, config, date_format)
    first_dose, dose_col = _first_dose(adsl, config, date_format)
    logger.info(
        f"on-treatment window: {config.clinical_completion_status} {STATUS_DISCONTINUED} ends at "
        f"{last_dose_col} + {config.clinical_discontinued_days_after_last_dose} days, "
        f"{STATUS_COMPLETED} has no dose-date cut"
    )

    sample_dates = (
        manifest.select(
            sample_id_col,
            ensure_date_column(manifest, sample_date_col, date_format).alias(sample_date_col),
        )
        .unique(subset=[sample_id_col], keep="first", maintain_order=True)
        .collect()
    )

    samples_full = (
        samples_with_usubjid.join(window_end, on="USUBJID", how="left")
        .join(first_dose, on="USUBJID", how="left")
        .join(sample_dates, on=sample_id_col, how="left")
    )

    def remaining(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.filter(~pl.col(sample_id_col).is_in(list(failed)))

    def flag(ids, reason: str, key: str) -> list:
        ids = list(dict.fromkeys(ids))
        if ids:
            emit(ids, reason)
        failed_counts[key] = len(ids)
        return ids

    # Every remaining subject must be COMPLETED or DISCONTINUED.
    status_col = config.clinical_completion_status
    other = (
        remaining(samples_full)
        .filter(
            pl.col("completion_status").is_null()
            | ~pl.col("completion_status").is_in([STATUS_COMPLETED, STATUS_DISCONTINUED])
        )
        .group_by("completion_status")
        .agg(
            pl.len().alias("n_samples"),
            pl.col("USUBJID").n_unique().alias("n_subjects"),
        )
        .sort("completion_status", nulls_last=True)
    )
    if other.height:
        found = "; ".join(
            f"{'null' if value is None else repr(value)}: {n} samples, {k} subjects"
            for value, n, k in other.rows()
        )
        raise ValueError(
            f"ADSL {status_col} must be {STATUS_COMPLETED} or {STATUS_DISCONTINUED} for every "
            f"subject with study samples at the on-treatment window step; found {found}"
        )

    # Discontinued subjects without a last dose (stop events alone do not define a
    # window) are flagged here. A completer's window does not use the last dose.
    no_window = remaining(samples_full).filter(
        (pl.col("completion_status") == STATUS_DISCONTINUED) & pl.col("last_dose").is_null()
    )
    flag(no_window[sample_id_col], FAIL_NO_TREATMENT_WINDOW, "no_treatment_window")
    logger.info(
        f"filtered {failed_counts['no_treatment_window']} samples from "
        f"{no_window['USUBJID'].n_unique()} discontinued subjects with null {last_dose_col}"
    )

    # Samples without a date cannot be placed in the window.
    no_adt = remaining(samples_full).filter(pl.col(sample_date_col).is_null())
    flag(no_adt[sample_id_col], FAIL_NO_ADT, "no_adt")
    logger.info(f"filtered {failed_counts['no_adt']} samples with null {sample_date_col}")

    # Samples drawn after the window end. A null window end (a completer without a
    # stop event) is never after it.
    after = remaining(samples_full).filter(pl.col(sample_date_col) > pl.col("window_end"))
    flag(after[sample_id_col], FAIL_AFTER_TREATMENT, "after_treatment")
    logger.info(
        f"filtered {failed_counts['after_treatment']} samples collected after the "
        "on-treatment window end"
    )

    # Visit not in [visits] analysis, except the pre-dose baseline.
    current = _with_analysis_visit(
        remaining(samples_full),
        analysis,
        baseline,
        sample_date_col,
        screening_dates,
    )
    n_exception = current.filter(pl.col("PREDOSE_BASELINE")).height
    logger.info(f"pre-dose baseline exception: {n_exception} samples counted as baseline")
    not_analysis = current.filter(pl.col("VISITNUM_ANALYSIS").is_null())
    flag(not_analysis[sample_id_col], FAIL_NOT_ANALYSIS_VISIT, "not_analysis_visit")
    codes = not_analysis.group_by("VISITNUM").len().sort("VISITNUM", nulls_last=True).rows()
    logger.info(
        f"filtered {failed_counts['not_analysis_visit']} samples outside the analysis "
        f"visits; by VISITNUM: {codes}"
    )

    # Duplicate (USUBJID, analysis visit) among the remaining samples: all fail.
    current = current.filter(pl.col("VISITNUM_ANALYSIS").is_not_null())
    analysis_visits = current.select(
        sample_id_col, pl.col("VISITNUM_ANALYSIS").cast(pl.Float64)
    ).sort(sample_id_col)
    dup_keys = current.group_by("USUBJID", "VISITNUM_ANALYSIS").len().filter(pl.col("len") > 1)
    duplicates = current.join(dup_keys, on=["USUBJID", "VISITNUM_ANALYSIS"], how="semi")
    flag(duplicates[sample_id_col], FAIL_DUPLICATE_VISIT, "duplicate_visit")
    logger.info(
        f"filtered {failed_counts['duplicate_visit']} samples sharing an analysis visit "
        f"with another sample of the same subject ({dup_keys.height} subject-visits)"
    )

    # A tube at the baseline visit dated after first dose is not a baseline;
    # the cascade below then judges the subject's other tubes.
    current = remaining(current)
    post_dose = current.filter(pl.col("BASELINE_AFTER_DOSE"))
    flag(post_dose[sample_id_col], FAIL_BASELINE_AFTER_DOSE, "baseline_after_dose")
    logger.info(
        f"filtered {failed_counts['baseline_after_dose']} baseline samples drawn after "
        f"first dose ({dose_col})"
    )

    # Subjects with no sample at the baseline visit, then subjects whose remaining
    # samples are all at baseline.
    current = remaining(current)
    counts = current.group_by("USUBJID").agg(
        (pl.col("VISITNUM_ANALYSIS") == baseline).sum().alias("n_baseline"),
        pl.len().alias("n_total"),
    )
    no_base = current.join(counts.filter(pl.col("n_baseline") == 0), on="USUBJID", how="semi")
    flag(no_base[sample_id_col], FAIL_NO_BASELINE, "no_baseline")
    logger.info(
        f"filtered {failed_counts['no_baseline']} samples from "
        f"{no_base['USUBJID'].n_unique()} participants without baseline"
    )
    current = remaining(current)
    only_base = current.join(
        counts.filter((pl.col("n_baseline") > 0) & (pl.col("n_baseline") == pl.col("n_total"))),
        on="USUBJID",
        how="semi",
    )
    flag(only_base[sample_id_col], FAIL_ONLY_BASELINE, "only_baseline")
    logger.info(
        f"filtered {failed_counts['only_baseline']} samples from "
        f"{only_base['USUBJID'].n_unique()} participants with only baseline"
    )

    failed_counts["bridge_samples"] = len(bridge_samples)

    total_failed = sum(failed_counts.values())
    logger.info(f"clinical filtering complete: {total_failed} total samples failed")

    return failed_counts, analysis_visits


def _with_analysis_visit(
    samples: pl.DataFrame,
    analysis: list[float],
    baseline: float,
    sample_date_col: str,
    screening_dates: pl.DataFrame | None,
) -> pl.DataFrame:
    """Adds ``VISITNUM_ANALYSIS``, ``PREDOSE_BASELINE`` and ``BASELINE_AFTER_DOSE``.

    ``VISITNUM_ANALYSIS`` is VISITNUM when it is an analysis visit, else the baseline
    for the subject's last pre-dose draw (``ADT <= first dose`` and after the
    subject's last screening date, when one is known), else null. Draws tied on the
    last pre-dose date all count, so the duplicate-visit rule fails them. When a
    subject has such a pre-dose draw, it is the baseline even if another draw carries
    the baseline code: that draw gets no analysis visit.
    ``BASELINE_AFTER_DOSE`` marks a baseline-coded draw dated after first dose that no
    pre-dose draw replaced.
    """
    if screening_dates is None:
        samples = samples.with_columns(pl.lit(None, dtype=pl.Date).alias("SCREENING_DT"))
    else:
        samples = samples.join(screening_dates, on="USUBJID", how="left")
    adt = pl.col(sample_date_col)
    pre = (
        (adt <= pl.col("first_dose"))
        & (pl.col("SCREENING_DT").is_null() | (adt > pl.col("SCREENING_DT")))
    ).fill_null(False)
    in_analysis = pl.col("VISITNUM").is_in(analysis).fill_null(False)
    coded_baseline = in_analysis & (pl.col("VISITNUM") == baseline).fill_null(False)
    other_analysis = in_analysis & ~coded_baseline
    # The last pre-dose draw is the subject's baseline unless it sits at another
    # analysis visit.
    candidate = (
        pre & ~other_analysis & (adt == adt.filter(pre & ~other_analysis).max().over("USUBJID"))
    )
    replaced = coded_baseline & ~candidate & candidate.any().over("USUBJID")
    return samples.with_columns(
        pl.when(other_analysis)
        .then(pl.col("VISITNUM"))
        .when(candidate)
        .then(pl.lit(baseline))
        .when(coded_baseline & ~replaced)
        .then(pl.col("VISITNUM"))
        .otherwise(None)
        .alias("VISITNUM_ANALYSIS"),
        (candidate & ~coded_baseline).alias("PREDOSE_BASELINE"),
        (
            coded_baseline & ~candidate & ~replaced & (adt > pl.col("first_dose")).fill_null(False)
        ).alias("BASELINE_AFTER_DOSE"),
    ).drop("SCREENING_DT")


def screening_dates(
    frames: list[pl.LazyFrame],
    screening: tuple[float, ...],
    date_format: str = "%Y-%m-%d",
) -> pl.DataFrame:
    """Returns each subject's last screening date: the latest ADT at a screening visit.

    Args:
        frames: ADaM BDS tables (ADLB, ADVS) with USUBJID, VISITNUM and ADT. Derived
            rows (non-null DTYPE) are dropped.
        screening: VISITNUM codes of the screening visits.
        date_format: strftime format for string date columns.

    Returns:
        USUBJID and SCREENING_DT.
    """
    parts = []
    for frame in frames:
        if "DTYPE" in frame.collect_schema().names():
            frame = frame.filter(pl.col("DTYPE").is_null() | (pl.col("DTYPE").cast(pl.Utf8) == ""))
        parts.append(
            frame.select(
                pl.col("USUBJID").cast(pl.Utf8),
                pl.col("VISITNUM").cast(pl.Float64),
                ensure_date_column(frame, "ADT", date_format).alias("SCREENING_DT"),
            )
        )
    return (
        pl.concat(parts)
        .filter(
            pl.col("VISITNUM").is_in(list(screening)),
            pl.col("SCREENING_DT").is_not_null(),
        )
        .group_by("USUBJID")
        .agg(pl.col("SCREENING_DT").max())
        .sort("USUBJID")
        .collect()
    )


# Flags set at construction, before any assay QC.
PRE_QC_FLAGS = (FAIL_NO_MANIFEST_MATCH, FAIL_NO_TREATMENT_ARM, CORRECTION_EXCLUDE_FLAG)
# The table pass starts from every FAIL set before the clinical rules: construction-
# time checks, technical QC and identity. The rules skip those samples.
TABLE_START_FLAGS = (*PRE_QC_FLAGS, *TECHNICAL_REASON_FLAGS, *IDENTITY_FAIL_FLAGS)
# Cascade reasons whose root_reason names the upstream cause.
CASCADE_FLAGS = (FAIL_NO_BASELINE, FAIL_ONLY_BASELINE)
# root_reason values that are not flags: the subject had no tube where the cascade
# looks (none at baseline; none outside baseline and screening).
ROOT_NO_BASELINE_TUBE = "No_Baseline_Tube"
ROOT_NO_LATER_TUBE = "No_Later_Tube"


def exclusion_frame(
    qc_instance, sample_date_col="ADT", date_format="%Y-%m-%d", screening_dates=None
) -> pl.DataFrame:
    """Builds one row per study sample with the columns ``exclusion_table`` reads.

    Every FAIL is a reason. Samples carrying a ``TABLE_START_FLAGS`` flag keep
    those flags and the clinical rules are applied to the rest, collecting flags
    without writing QC flags or changing ``failed_samples``. A tube failed upstream is
    therefore absent for the rules, so its subject's other tubes cascade
    (``FAIL_No_Baseline``, ``FAIL_Only_Baseline``). Bridge samples are left out. Manifest study tubes with no assay data are added
    with ``FAIL_Not_Assayed``; USUBJID and VISITNUM come from the manifest for them.

    Args:
        qc_instance: QC instance with ADSL, manifest, ``adsl_merged`` and ``config``.
        sample_date_col: Manifest column with the sample collection date.
        date_format: strftime format for string date columns.
        screening_dates: Output of ``screening_dates``, or None.

    Returns:
        USUBJID and VISITNUM (``adsl_merged``; a reassigned tube sits under its new
        subject), ARM (ADSL), MANIFEST_EXCLUDE (when the manifest has it), one boolean
        per ``EXCLUSION_REASON_FLAGS`` and ``root_reason``.

    Raises:
        ValueError: If ADSL, the manifest or the merged ADSL is missing, or ADSL has no
            ARM column.
    """
    sid = qc_instance.sample_id_col
    _require_clinical_inputs(qc_instance)
    adsl = ensure_lazy(qc_instance.adsl)
    if "ARM" not in adsl.collect_schema().names():
        raise ValueError("The exclusion table needs the randomized arm, but ADSL has no ARM column")

    pre = {}
    for action in qc_instance.sample_qc_actions:
        for c in action.columns:
            if c in TABLE_START_FLAGS:
                pre.setdefault(c, set()).update(action.filter(pl.col(c))[sid].cast(pl.Utf8))
    collected = {c: set(ids) for c, ids in pre.items()}

    def collect(ids, reason):
        collected.setdefault(reason, set()).update(str(i) for i in ids)

    failed = set().union(*pre.values()) if pre else set()
    _apply_rules(qc_instance, sample_date_col, date_format, screening_dates, failed, collect)

    study = (
        qc_instance.sample_metadata.filter(pl.col("SampleType") == qc_instance.sample_type)
        .select(pl.col(sid).cast(pl.Utf8))
        .unique()
        .collect()
    )
    mapping = _one_row_per(
        qc_instance.adsl_merged.select(
            pl.col(sid).cast(pl.Utf8), "USUBJID", pl.col("VISITNUM").cast(pl.Float64)
        ),
        sid,
        "merged ADSL sample to subject and visit",
    )
    arms = _one_row_per(adsl.select("USUBJID", "ARM"), "USUBJID", "ADSL randomized arm")
    manifest = ensure_lazy(qc_instance.manifest)
    assayed = qc_instance.sample_metadata.select(pl.col(sid).cast(pl.Utf8)).unique()
    not_assayed = _one_row_per(
        manifest.filter(
            pl.col(qc_instance.usubjid_col).is_not_null()
            & ~pl.col(sid).cast(pl.Utf8).is_in(list(qc_instance.bridge_sample_ids))
            & ~pl.col(qc_instance.usubjid_col).str.starts_with("BRIDGE_")
        ).select(
            pl.col(sid).cast(pl.Utf8),
            pl.col(qc_instance.usubjid_col).alias("USUBJID"),
            pl.col("VISITNUM").cast(pl.Float64),
        ),
        sid,
        "manifest study tubes",
    ).join(assayed.collect(), on=sid, how="anti")
    frame = pl.concat(
        [study.join(mapping, on=sid, how="left"), not_assayed], how="diagonal_relaxed"
    ).join(arms, on="USUBJID", how="left")
    if "MANIFEST_EXCLUDE" in manifest.collect_schema().names():
        frame = frame.join(
            _one_row_per(
                manifest.select(pl.col(sid).cast(pl.Utf8), "MANIFEST_EXCLUDE").filter(
                    pl.col("MANIFEST_EXCLUDE").is_not_null()
                ),
                sid,
                "manifest exclusions",
            ),
            on=sid,
            how="left",
        )
    collected[FAIL_NOT_ASSAYED] = set(not_assayed[sid])
    frame = frame.with_columns(
        [pl.col(sid).is_in(list(collected.get(c, ()))).alias(c) for c in EXCLUSION_REASON_FLAGS]
    ).filter(~pl.col(FAIL_BRIDGE))
    ambiguous = {
        str(s)
        for action in qc_instance.sample_qc_actions
        if WARN_IDENTITY_AMBIGUOUS in action.columns
        for s in action.filter(pl.col(WARN_IDENTITY_AMBIGUOUS))[sid].to_list()
    }
    return _root_reason(frame, sid, qc_instance.config, ambiguous)


def _root_reason(frame: pl.DataFrame, sid: str, config, ambiguous: set[str]) -> pl.DataFrame:
    """Adds ``root_reason``, the upstream cause of a cascade row.

    Null on every row not counted under ``FAIL_No_Baseline`` or ``FAIL_Only_Baseline``.
    On a cascade row it is the reason of the subject's other tubes counted under a
    non-cascade reason (manifest, not assayed, technical, identity, clinical, or an
    ``exclude`` correction's label): for ``FAIL_No_Baseline``, those at the baseline
    visit; for ``FAIL_Only_Baseline``, those outside the baseline and screening visits.
    Several such tubes give the first reason in ``EXCLUSION_REASON_FLAGS`` order, a
    correction label first (two labels: the first in sort order). Without one, a row
    whose own tube is ``WARN_Identity_Ambiguous`` gets that flag; otherwise the
    subject never had such a tube: ``ROOT_NO_BASELINE_TUBE`` or ``ROOT_NO_LATER_TUBE``.
    A baseline that only the pre-dose exception would have supplied is not traced.
    """
    exclude = "MANIFEST_EXCLUDE" if "MANIFEST_EXCLUDE" in frame.columns else None
    rank = {flag: i for i, flag in enumerate(EXCLUSION_REASON_FLAGS)}
    reasons = frame.with_columns(first_reason(EXCLUSION_REASON_FLAGS, exclude).alias("_reason"))
    baseline, screening = config.visits_baseline, list(config.visits_screening)
    visit = pl.col("VISITNUM")
    reason = pl.col("_reason")
    # A correction label is not a flag: it outranks every flag, as in the table.
    lost = reasons.filter(reason.is_not_null() & ~reason.is_in(CASCADE_FLAGS)).with_columns(
        reason.replace_strict(rank, default=-1, return_dtype=pl.Int64).alias("_rank")
    )

    def first_loss(where: pl.Expr, name: str) -> pl.DataFrame:
        # Every correction label ranks -1, so the reason text breaks ties: the result
        # must not depend on row order.
        return (
            lost.filter(where.fill_null(False))
            .group_by("USUBJID")
            .agg(reason.sort_by(["_rank", "_reason"]).first().alias(name))
        )

    lost_baseline = first_loss(visit == baseline, "_lost_baseline")
    lost_later = first_loss(
        visit.is_null() | ((visit != baseline) & ~visit.is_in(screening)), "_lost_later"
    )
    return (
        reasons.join(lost_baseline, on="USUBJID", how="left")
        .join(lost_later, on="USUBJID", how="left")
        .with_columns(
            pl.when(reason == FAIL_NO_BASELINE)
            .then(pl.col("_lost_baseline"))
            .when(reason == FAIL_ONLY_BASELINE)
            .then(pl.col("_lost_later"))
            .alias("root_reason")
        )
        .with_columns(
            pl.when(
                pl.col("root_reason").is_null()
                & reason.is_in(CASCADE_FLAGS)
                & pl.col(sid).is_in(list(ambiguous))
            )
            .then(pl.lit(WARN_IDENTITY_AMBIGUOUS))
            .otherwise(pl.col("root_reason"))
            .alias("root_reason")
        )
        .with_columns(
            pl.when(pl.col("root_reason").is_null() & (reason == FAIL_NO_BASELINE))
            .then(pl.lit(ROOT_NO_BASELINE_TUBE))
            .when(pl.col("root_reason").is_null() & (reason == FAIL_ONLY_BASELINE))
            .then(pl.lit(ROOT_NO_LATER_TUBE))
            .otherwise(pl.col("root_reason"))
            .alias("root_reason")
        )
        .drop("_reason", "_lost_baseline", "_lost_later")
    )


def exclusion_summary(
    qc_instance,
    save_prefix: str | None = None,
    sample_date_col="ADT",
    date_format="%Y-%m-%d",
    screening_dates=None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Writes the exclusion table and the non-analysis visit counts.

    Run after ``sample_unsuitable``, with the same date column, date format and
    screening dates. Every FAIL is counted once, under the first reason in
    ``EXCLUSION_REASON_FLAGS`` order; WARN flags add no row. Writes
    ``{save_prefix}.exclusion_table.tsv`` and ``{save_prefix}.non_analysis_visits.tsv``
    when ``save_prefix`` is given, each led by a STUDY column (``config.study_name``).
    Counts only; no IDs. Changes no QC flag and no sample's status.

    Args:
        qc_instance: QC instance with ADSL, manifest, ``adsl_merged`` and ``config``.
        save_prefix: Path prefix for the two TSV files, or None to skip writing.
        sample_date_col: Manifest column with the sample collection date.
        date_format: strftime format for string date columns.
        screening_dates: Output of ``screening_dates``, or None.

    Returns:
        The exclusion table (arm, status, reason, root_reason, visit, n_samples,
        n_subjects) and the non-analysis visit counts (visit, n_samples, n_subjects).

    Raises:
        ValueError: If ADSL, the manifest or the merged ADSL is missing, or ADSL has no
            ARM column.
    """
    frame = exclusion_frame(qc_instance, sample_date_col, date_format, screening_dates)
    table = exclusion_table(
        frame,
        EXCLUSION_REASON_FLAGS,
        manifest_exclude_col="MANIFEST_EXCLUDE" if "MANIFEST_EXCLUDE" in frame.columns else None,
        root_reason_col="root_reason",
    )
    counts = non_analysis_visit_counts(frame, flag_col=FAIL_NOT_ANALYSIS_VISIT)
    if save_prefix:
        # STUDY is the first column of both files.
        study = pl.lit(qc_instance.config.study_name).alias("STUDY")
        table.select(study, pl.all()).write_csv(
            f"{save_prefix}.exclusion_table.tsv", separator="\t"
        )
        counts.select(study, pl.all()).write_csv(
            f"{save_prefix}.non_analysis_visits.tsv", separator="\t"
        )
        logger.info(
            f"saved: {save_prefix}.exclusion_table.tsv, {save_prefix}.non_analysis_visits.tsv"
        )
    return table, counts
