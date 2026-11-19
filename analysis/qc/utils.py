"""Shared helpers for the QC pipeline: run output, dates, QC summaries, matrices."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.impute import KNNImputer

from analysis.qc.platform import (
    OLINK_LOD_IN_FILES,
    OLINK_NPX,
    OLINK_NPX_LOD,
    OLINK_PC_LOD,
    OLINK_PC_NPX,
)
from analysis.qc.schemas import IDENTITY_SCORE_COLUMNS, UNFIT_IDENTITY_SUMMARY_COLUMN

logger = logging.getLogger(__name__)

# Nullable, report-only columns in the per-tube sample QC summary.
VENDOR_NOTE_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "SampleNotes": pl.Utf8,
    "AssayNotes": pl.Utf8,
    "AliquotingNotes": pl.Utf8,
    "VOLUME_NOTE": pl.Utf8,
}


def ensure_lazy(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Converts a DataFrame to a LazyFrame, passing a LazyFrame through.

    Args:
        df: Polars DataFrame or LazyFrame.

    Returns:
        The LazyFrame.
    """
    if isinstance(df, pl.DataFrame):
        return df.lazy()
    return df


# Run outputs without the study-name prefix (manifest.build_manifest writes them).
UNPREFIXED_RUN_OUTPUTS = ("manifest.parquet", "manifest_build_report.json")


def _platform_marker(study_name: str) -> str:
    return f"{study_name}.platform"


def _runs_own(name: str, study_name: str, other_studies: set[str]) -> bool:
    """Checks whether a file in an output directory belongs to ``study_name``'s run.

    A rerun replaces the run's files. The unprefixed manifest files are the run's
    unless another run shares the directory.
    """
    if name.startswith(f"{study_name}."):
        return True
    return name in UNPREFIXED_RUN_OUTPUTS and not other_studies


def _other_studies(directory: Path, study_name: str) -> set[str]:
    return {p.name[: -len(".platform")] for p in directory.glob("*.platform")} - {study_name}


def stage_output_dir(output_dir: str | Path, study_name: str, platform: str) -> Path:
    """Checks that a run may write to ``output_dir`` and creates its staging directory.

    The run writes every output into the staging directory, and ``publish_output_dir``
    renames it into place when the run has finished: the staging directory is a
    sibling, so it is on the same filesystem and the rename is atomic. A run that
    fails leaves ``output_dir`` as it was. Output files are prefixed with the study
    name only, and ``manifest.parquet`` has no prefix, so an Olink and a SomaScan run
    of one trial cannot share a directory: ``{study_name}.platform`` holds the
    platform, and a run of the other platform raises.

    Args:
        output_dir: The run's ``--output-dir``; it need not exist.
        study_name: The run's study name (the files' prefix).
        platform: ``olink`` or ``soma``.

    Returns:
        The new, empty staging directory, holding only the platform marker.

    Raises:
        FileExistsError: ``output_dir`` holds a run of this study on the other
            platform.
        NotADirectoryError: ``output_dir`` exists and is not a directory.
        ValueError: ``output_dir`` holds a subdirectory (the rename would have to copy
            it) or contains the working directory (the rename would move it).
    """
    final = Path(output_dir).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if cwd == final or final in cwd.parents:
        raise ValueError(
            f"--output-dir {final} contains the working directory; a run replaces it whole, so run elsewhere"
        )
    if final.exists():
        if not final.is_dir():
            raise NotADirectoryError(f"--output-dir {final} is not a directory")
        marker = final / _platform_marker(study_name)
        if marker.exists() and (existing := marker.read_text().strip()) != platform:
            raise FileExistsError(
                f"{final} already holds a {existing} run of {study_name}; this run is "
                f"{platform}. Their outputs share file names, so give each platform its own --output-dir."
            )
        subdirs = sorted(p.name for p in final.iterdir() if p.is_dir() and not p.is_symlink())
        if subdirs:
            raise ValueError(
                f"--output-dir {final} holds the subdirectory {subdirs[0]}; a run replaces the directory whole and "
                "keeps only files, so give it a directory without subdirectories"
            )
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{final.name}.staging-{study_name}-", dir=final.parent)
    )
    (staging / _platform_marker(study_name)).write_text(platform + "\n")
    return staging


def publish_output_dir(staging: str | Path, output_dir: str | Path, study_name: str) -> None:
    """Renames a finished run's staging directory into place as ``output_dir``.

    Into a missing or empty ``output_dir`` this is one atomic rename. Otherwise the
    files that belong to something else (another run, or a log written by a wrapper)
    are hard-linked into the staging directory first, so they are kept as the same
    files, and the two directories swap by two renames; the previous run's files of
    this study go with the old directory, so a rerun leaves none behind.

    Args:
        staging: The directory ``stage_output_dir`` returned, holding every output of
            the run.
        output_dir: The run's ``--output-dir``.
        study_name: The run's study name (the files' prefix).

    Raises:
        OSError: A kept file cannot be linked (another filesystem, or a subdirectory
            created during the run).
    """
    staging = Path(staging)
    final = Path(output_dir).expanduser().resolve()
    if not final.exists() or not any(final.iterdir()):
        os.replace(staging, final)
        return
    others = _other_studies(final, study_name)
    for entry in sorted(final.iterdir()):
        if not _runs_own(entry.name, study_name, others) and not (staging / entry.name).exists():
            os.link(entry, staging / entry.name, follow_symlinks=False)
    previous = Path(
        tempfile.mkdtemp(prefix=f".{final.name}.previous-{study_name}-", dir=final.parent)
    )
    os.replace(final, previous)
    os.replace(staging, final)
    shutil.rmtree(previous)


MIN_FOUR_DIGIT_YEAR = 1000


def reject_short_years(dates: pl.Expr) -> pl.Expr:
    """Rejects date values whose year falls below ``MIN_FOUR_DIGIT_YEAR``.

    Date inputs require four-digit years at or above the minimum supported year.
    """

    def check(s: pl.Series) -> pl.Series:
        bad = s.filter(s.dt.year() < MIN_FOUR_DIGIT_YEAR)
        if len(bad):
            raise ValueError(
                f"{len(bad)} date(s) have a year before {MIN_FOUR_DIGIT_YEAR} "
                f"(e.g. {bad[0]}). "
                "Use a four-digit year at or above the minimum supported year."
            )
        return s

    return dates.map_batches(check, return_dtype=pl.Date, is_elementwise=True)


def ensure_date_column(
    df: pl.DataFrame | pl.LazyFrame, col_name: str, date_format: str = "%Y-%m-%d"
) -> pl.Expr:
    """Builds an expression yielding the column as Date.

    Strings are parsed with the explicit ``date_format``; Datetime values are cast to
    Date; Date columns pass through. A column absent from ``df`` is returned as a bare
    ``pl.col``.

    Args:
        df: DataFrame or LazyFrame.
        col_name: Name of the date column.
        date_format: strftime format for string columns (default ISO).

    Returns:
        Polars expression for the date column.
    """
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    if col_name not in schema.names():
        return pl.col(col_name)

    col_type = schema[col_name]
    if col_type in [pl.String, pl.Utf8]:
        return reject_short_years(pl.col(col_name).str.to_date(date_format))
    if isinstance(col_type, pl.Datetime):
        return pl.col(col_name).cast(pl.Date)
    return pl.col(col_name)


def _soma_analyte_lod(qc_instance) -> pl.DataFrame:
    """Returns the eLOD of each SomaScan analyte, from the in-memory long data.

    The eLOD comes from the buffer wells pooled over every plate, so it is one value
    per analyte; the long data repeats it on every row of the analyte.

    Raises:
        ValueError: If no eLOD was computed or an analyte has more than one value.
    """
    protein_id_col = qc_instance.protein_id_col
    data_full = qc_instance.data_full
    if data_full is None or "LOD" not in data_full.collect_schema().names():
        raise ValueError(
            "the SomaScan eLOD was not calculated: the assay summary needs the detection step's LOD column"
        )
    per_analyte = (
        data_full.group_by(protein_id_col)
        .agg(pl.col("LOD").n_unique().alias("_n_lod"), pl.col("LOD").first())
        .collect(engine="streaming")
    )
    n_many = per_analyte.filter(pl.col("_n_lod") != 1).height
    if n_many:
        raise ValueError(
            f"{n_many} {protein_id_col}s carry more than one distinct LOD; the eLOD is one value per analyte"
        )
    return per_analyte.select(protein_id_col, "LOD")


def _assert_one_frame_per_reason(actions: list[pl.DataFrame], key_col: str) -> None:
    """Asserts that each QC reason is tracked in one frame.

    A second frame for a reason would join as ``<reason>_right``.
    """
    seen: set[str] = set()
    for df in actions:
        for reason in (c for c in df.columns if c != key_col):
            if reason in seen:
                raise AssertionError(f"QC reason {reason} is tracked in more than one frame")
            seen.add(reason)


_SOMA_NOTE_COLUMNS = ("SampleNotes", "AssayNotes", "AliquotingNotes")


def _attach_vendor_notes(qc_instance, summary: pl.DataFrame) -> pl.DataFrame:
    """Join the verbatim vendor note columns to the sample summary by tube."""
    key = qc_instance.sample_id_col
    if qc_instance.platform.name == "soma":
        source = getattr(qc_instance, "sample_metadata", None)
        note_cols = _SOMA_NOTE_COLUMNS
    else:
        source = getattr(qc_instance, "manifest", None)
        note_cols = ("VOLUME_NOTE",)
        # Olink joins VOLUME_NOTE only.
    available = set(source.collect_schema().names()) if source is not None else set()
    if source is None or key not in available:
        note_rows = pl.DataFrame(schema={key: summary.schema[key]})
    else:
        # Only study wells (``SampleType == sample_type``) join the summary.
        if qc_instance.platform.name == "soma" and "SampleType" in available:
            source = source.filter(pl.col("SampleType") == qc_instance.sample_type)
        note_rows = source.select(
            pl.col(key).cast(summary.schema[key]),
            *(pl.col(c).cast(pl.Utf8) for c in note_cols if c in available),
        ).collect()
    note_rows = note_rows.with_columns(
        pl.when(pl.col(c).str.strip_chars() == "")
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in note_cols
        if c in note_rows.columns
    )
    note_rows = note_rows.with_columns(
        pl.lit(None, dtype=pl.Utf8).alias(c)
        for c in note_cols
        if c not in note_rows.columns
    )
    note_rows = note_rows.unique()
    if note_rows[key].n_unique() != note_rows.height:
        raise ValueError(f"duplicate vendor note join key {key}")
    summary = summary.join(note_rows, on=key, how="left", validate="m:1")
    for c, dtype in VENDOR_NOTE_SUMMARY_SCHEMA.items():
        if c not in summary.columns:
            summary = summary.with_columns(pl.lit(None, dtype=dtype).alias(c))
    return summary


def _one_row_per_assay(frame: pl.DataFrame, protein_id_col: str, name: str) -> pl.DataFrame:
    """Returns ``frame``, raising when an assay has more than one row in it."""
    n_dups = frame.height - frame[protein_id_col].n_unique()
    if n_dups > 0:
        raise ValueError(f"{name} contains {n_dups} duplicate rows per assay")
    return frame


def save_qc_results(
    qc_instance,
    save_prefix: str | None = None,
    usubjid_col: str = "USUBJID",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Builds the sample and protein QC summaries and optionally saves them.

    With ``save_prefix`` it writes the sample summary TSV (with the sample ID,
    USUBJID, STUDY and VISITNUM), the protein/assay summary TSV, the long-format
    parquet with QC status, and the identity and visit-order outputs.

    Args:
        qc_instance: SomaQC or OlinkQC instance.
        save_prefix: Prefix for the files to save; nothing is written if None.
        usubjid_col: Column name for the unique participant ID.

    Returns:
        The sample and protein QC summaries.
    """
    from analysis.qc.clinical import (  # clinical imports this module
        PROTEIN_QC_FLAGS,
        SAMPLE_QC_FLAGS,
    )

    def _add_qc_status(df, id_col, failed_list, warned_list):
        """Adds a FAIL/WARN/PASS ``Status`` column."""
        return df.with_columns(
            pl.when(pl.col(id_col).is_in(failed_list))
            .then(pl.lit("FAIL"))
            .when(pl.col(id_col).is_in(warned_list))
            .then(pl.lit("WARN"))
            .otherwise(pl.lit("PASS"))
            .alias("Status")
        )

    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    protein_name_col = qc_instance.protein_name_col
    uniprot_col = qc_instance.uniprot_col

    # Biological samples only (no QC controls).
    sample_summary = qc_instance.samples.select(sample_id_col).unique().collect()
    logger.debug(f"sample_summary initial: {sample_summary.height} rows")

    # Protein-level columns come from analyte_metadata; ColCheck and RowCheck are
    # tracked via the QC actions instead.
    analyte_schema = qc_instance.analyte_metadata.collect_schema().names()
    protein_cols = [protein_id_col]
    if protein_name_col and protein_name_col in analyte_schema:
        protein_cols.append(protein_name_col)
    if uniprot_col and uniprot_col in analyte_schema:
        protein_cols.append(uniprot_col)
    for col in analyte_schema:
        if col not in protein_cols and col not in [
            protein_id_col,
            "ColCheck",
            "RowCheck",
        ]:
            protein_cols.append(col)

    protein_summary = qc_instance.analyte_metadata.select(protein_cols).unique().collect()

    # Real assays only (AssayType "assay"), not the internal controls.
    if "AssayType" in protein_summary.columns:
        n_before = protein_summary.height
        protein_summary = protein_summary.filter(pl.col("AssayType") == "assay")
        n_after = protein_summary.height
        logger.info(
            f"Filtered protein summary to AssayType == 'assay': {n_before} -> {n_after} proteins"
        )

    n_proteins = protein_summary.height
    n_unique_proteins = protein_summary[protein_id_col].n_unique()
    if n_proteins != n_unique_proteins:
        raise ValueError(
            f"Protein summary has duplicates: {n_proteins} rows but {n_unique_proteins} "
            f"unique {protein_id_col}s"
        )

    # USUBJID, STUDY, VISITNUM and the draw date come from adsl_merged, which already
    # maps sample ID to USUBJID.
    if qc_instance.adsl_merged is not None:
        merged_cols = qc_instance.adsl_merged.collect_schema().names()
        id_cols = [sample_id_col, usubjid_col]
        if "STUDY" in merged_cols:
            id_cols.append("STUDY")
        if "VISITNUM" in merged_cols:
            id_cols.append("VISITNUM")
        # The labeled subject and visit, and the tube's draw date and source.
        for extra_col in ["USUBJID_LABELED", "VISITNUM_LABELED", "ADT", "ADT_SOURCE"]:
            if extra_col in merged_cols:
                id_cols.append(extra_col)

        adsl_id_map = qc_instance.adsl_merged.select(id_cols).collect()
        # The labeled columns exist only after a move (core.apply_reassignments); write
        # them null otherwise, so every trial's summary has the same columns.
        if "USUBJID_LABELED" not in adsl_id_map.columns:
            visit_dtype = adsl_id_map.schema.get("VISITNUM", pl.Float64)
            adsl_id_map = adsl_id_map.with_columns(
                pl.lit(None, dtype=pl.Utf8).alias("USUBJID_LABELED"),
                pl.lit(None, dtype=visit_dtype).alias("VISITNUM_LABELED"),
            )
            id_cols += ["USUBJID_LABELED", "VISITNUM_LABELED"]

        conflict_cols = [col for col in id_cols if col != sample_id_col]
        conflicting_mappings = (
            adsl_id_map.group_by(sample_id_col)
            .agg(pl.struct(conflict_cols).n_unique().alias("_mapping_count"))
            .filter(pl.col("_mapping_count") > 1)
        )
        if conflicting_mappings.height > 0:
            conflicting_samples = conflicting_mappings[sample_id_col].to_list()
            sample_conflicts = adsl_id_map.filter(
                pl.col(sample_id_col).is_in(conflicting_samples)
            ).sort([sample_id_col] + [col for col in conflict_cols if col in adsl_id_map.columns])
            raise ValueError(
                "Conflicting sample-to-subject mappings found in adsl_merged for "
                f"{conflicting_mappings.height} {sample_id_col} value(s); examples: "
                f"{sample_conflicts.head(10).to_dicts()}"
            )

        sort_by: list[str] = ["_has_usubjid"]
        sort_descending = [True]
        sort_exprs: list[pl.Expr] = [pl.col(usubjid_col).is_not_null().alias("_has_usubjid")]
        if "VISITNUM" in adsl_id_map.columns:
            sort_by.append("VISITNUM")
            sort_descending.append(False)
        if "ADT" in adsl_id_map.columns:
            sort_exprs.append(ensure_date_column(adsl_id_map, "ADT").alias("_adt_sort"))
            sort_by.append("_adt_sort")
            sort_descending.append(False)

        adsl_id_map = adsl_id_map.with_columns(sort_exprs)
        temp_cols = [c for c in sort_by if c.startswith("_")]
        id_map = (
            adsl_id_map.sort(
                by=sort_by + [sample_id_col],
                descending=sort_descending + [False],
                nulls_last=True,
            )
            .unique(subset=[sample_id_col], keep="first")
            .drop(temp_cols)
        )
        logger.debug(
            f"id_map: {id_map.height} rows, {id_map[sample_id_col].n_unique()} unique {sample_id_col}"
        )

        # Visit codes are floats.
        if "VISITNUM" in id_map.columns:
            id_map = id_map.with_columns(pl.col("VISITNUM").cast(pl.Float64))
        if "ADT" in id_map.columns:
            id_map = id_map.with_columns(ensure_date_column(id_map, "ADT"))

        sample_summary = sample_summary.join(id_map, on=sample_id_col, how="left")

        # The analysis visit the clinical rules gave each sample (null when none): the
        # pre-dose baselines keep their tube VISITNUM but carry the baseline here.
        analysis_visits = getattr(qc_instance, "analysis_visits", None)
        if analysis_visits is not None:
            sample_summary = sample_summary.join(
                analysis_visits.select(
                    pl.col(sample_id_col).cast(sample_summary.schema[sample_id_col]),
                    pl.col("VISITNUM_ANALYSIS").cast(pl.Float64),
                ),
                on=sample_id_col,
                how="left",
            )
        # Dose timing: the dates and day counts the clinical rules worked from.
        dose_timing = getattr(qc_instance, "dose_timing", None)
        if dose_timing is not None:
            sample_summary = sample_summary.join(
                dose_timing.with_columns(
                    pl.col(sample_id_col).cast(sample_summary.schema[sample_id_col])
                ),
                on=sample_id_col,
                how="left",
            )
        logger.debug(f"sample_summary after id_map join: {sample_summary.height} rows")

        # Drop bridge samples.
        bridge_ids = set(qc_instance.bridge_sample_ids) | set(
            qc_instance.manifest_bridge_sample_ids
        )
        is_bridge = pl.col(usubjid_col).str.starts_with("BRIDGE_").fill_null(False) | pl.col(
            sample_id_col
        ).cast(pl.Utf8).is_in(bridge_ids)
        n_bridge = sample_summary.filter(is_bridge).height
        if n_bridge > 0:
            logger.info(f"removing {n_bridge} bridge samples from QC summary")
            sample_summary = sample_summary.filter(~is_bridge)

    _assert_one_frame_per_reason(qc_instance.sample_qc_actions, sample_id_col)
    _assert_one_frame_per_reason(qc_instance.protein_qc_actions, protein_id_col)

    for qc_df in qc_instance.sample_qc_actions:
        sample_summary = sample_summary.join(qc_df, on=sample_id_col, how="left", coalesce=True)
        reason_col = [c for c in qc_df.columns if c != sample_id_col][0]
        logger.debug(
            f"sample_summary after '{reason_col}' join: {sample_summary.height} rows "
            f"(qc_df has {qc_df.height} rows, {qc_df[sample_id_col].n_unique()} unique)"
        )

    sample_summary = _add_qc_status(
        sample_summary,
        sample_id_col,
        qc_instance.failed_samples,
        qc_instance.warned_samples,
    )

    for qc_df in qc_instance.protein_qc_actions:
        protein_summary = protein_summary.join(qc_df, on=protein_id_col, how="left", coalesce=True)
    protein_summary = _add_qc_status(
        protein_summary,
        protein_id_col,
        qc_instance.failed_proteins,
        qc_instance.warned_proteins,
    )

    # Sample QC metrics (RowCheck status, outlier fractions).
    if hasattr(qc_instance, "_sample_qc_status") and qc_instance._sample_qc_status is not None:
        sample_summary = sample_summary.join(
            qc_instance._sample_qc_status, on=sample_id_col, how="left", coalesce=True
        )
        logger.debug(f"sample_summary after _sample_qc_status join: {sample_summary.height} rows")

    # PCA results (PC1_std, PC2_std, Z_Median, Z_IQR).
    if hasattr(qc_instance, "_pca_results") and qc_instance._pca_results is not None:
        n_pca_dups = (
            qc_instance._pca_results.height
            - qc_instance._pca_results.unique(subset=[sample_id_col]).height
        )
        if n_pca_dups > 0:
            raise ValueError(
                f"_pca_results contains {n_pca_dups} duplicate rows per sample"
            )
        pca_deduped = qc_instance._pca_results.unique(subset=[sample_id_col], keep="first")
        sample_summary = sample_summary.join(pca_deduped, on=sample_id_col, how="left")

    # Sex concordance results (Reported_Sex, predicted probabilities).
    if (
        hasattr(qc_instance, "_sex_concordance_results")
        and qc_instance._sex_concordance_results is not None
    ):
        n_sex_dups = (
            qc_instance._sex_concordance_results.height
            - qc_instance._sex_concordance_results.unique(subset=[sample_id_col]).height
        )
        if n_sex_dups > 0:
            raise ValueError(
                f"_sex_concordance_results contains {n_sex_dups} duplicate rows per sample"
            )
        sex_deduped = qc_instance._sex_concordance_results.unique(
            subset=[sample_id_col], keep="first"
        )
        sample_summary = sample_summary.join(sex_deduped, on=sample_id_col, how="left")

    # Age concordance results (Visit_Adj_Age, Predicted_Age, Std_Predicted_Age).
    if (
        hasattr(qc_instance, "_age_concordance_results")
        and qc_instance._age_concordance_results is not None
    ):
        n_age_dups = (
            qc_instance._age_concordance_results.height
            - qc_instance._age_concordance_results.unique(subset=[sample_id_col]).height
        )
        if n_age_dups > 0:
            raise ValueError(
                f"_age_concordance_results contains {n_age_dups} duplicate rows per sample"
            )
        age_deduped = qc_instance._age_concordance_results.unique(
            subset=[sample_id_col], keep="first"
        )
        sample_summary = sample_summary.join(age_deduped, on=sample_id_col, how="left")

    # The final Lp scores of every scored tube (null without identity); the columns are
    # always written.
    proposals = getattr(qc_instance, "identity_proposals", None)
    if proposals is not None:
        scores = proposals.select(
            pl.col("SampleID").cast(pl.String),
            pl.col("self_z").alias("identity_self_z"),
            pl.col("best_subject").alias("identity_best_USUBJID"),
            pl.col("best_z").alias("identity_best_z"),
            pl.col("second_z").alias("identity_second_z"),
            pl.col("pass1_best_subject").alias("identity_pass1_best_USUBJID"),
            pl.col("pass1_best_z").alias("identity_pass1_best_z"),
            pl.col("pass1_second_z").alias("identity_pass1_second_z"),
        ).select("SampleID", *[pl.col(c).cast(t) for c, t in IDENTITY_SCORE_COLUMNS.items()])
    else:
        scores = pl.DataFrame(schema={"SampleID": pl.String, **IDENTITY_SCORE_COLUMNS})
    sample_summary = sample_summary.join(
        scores.rename({"SampleID": sample_id_col}).with_columns(
            pl.col(sample_id_col).cast(sample_summary.schema[sample_id_col])
        ),
        on=sample_id_col,
        how="left",
    )
    # The unfit-tube identity annotation's class, null on every tube but the
    # technically failed ones.
    unfit = getattr(qc_instance, "unfit_identity", None)
    unfit = (
        pl.DataFrame(schema={"SampleID": pl.String, UNFIT_IDENTITY_SUMMARY_COLUMN: pl.String})
        if unfit is None
        else unfit.select(
            pl.col("SampleID"),
            pl.col("unfit_identity").alias(UNFIT_IDENTITY_SUMMARY_COLUMN),
        )
    )
    sample_summary = sample_summary.join(
        unfit.rename({"SampleID": sample_id_col}).with_columns(
            pl.col(sample_id_col).cast(sample_summary.schema[sample_id_col])
        ),
        on=sample_id_col,
        how="left",
    )

    n_proteins_initial = protein_summary.height
    logger.debug(f"Protein summary initial size: {n_proteins_initial} rows")

    # Assay QC metrics (ColCheck status, Total_Samples).
    if hasattr(qc_instance, "_assay_qc_status") and qc_instance._assay_qc_status is not None:
        assay_qc = _one_row_per_assay(qc_instance._assay_qc_status, protein_id_col, "_assay_qc_status")
        n_assay_qc = assay_qc.height
        logger.debug(f"Joining assay QC status: {n_assay_qc} unique {protein_id_col}s")

        protein_summary = protein_summary.join(
            assay_qc, on=protein_id_col, how="left", coalesce=True
        )

    # CV stats (Intra_CV, Inter_CV).
    if hasattr(qc_instance, "_cv_stats") and qc_instance._cv_stats is not None:
        cv_stats = _one_row_per_assay(qc_instance._cv_stats, protein_id_col, "_cv_stats")
        n_cv = cv_stats.height
        logger.debug(f"Joining CV stats: {n_cv} unique {protein_id_col}s")

        protein_summary = protein_summary.join(
            cv_stats, on=protein_id_col, how="left", coalesce=True
        )

    # Detection stats (Detection_Rate).
    if hasattr(qc_instance, "_detection_stats") and qc_instance._detection_stats is not None:
        det_stats = _one_row_per_assay(qc_instance._detection_stats, protein_id_col, "_detection_stats")
        n_det = det_stats.height
        logger.debug(f"Joining detection stats: {n_det} unique {protein_id_col}s")

        protein_summary = protein_summary.join(
            det_stats, on=protein_id_col, how="left", coalesce=True
        )

    # SomaScan: the analyte's eLOD, which is not in the long table;
    # Dilution comes with the analyte metadata.
    if qc_instance.platform.name == "soma":
        protein_summary = protein_summary.join(
            _soma_analyte_lod(qc_instance), on=protein_id_col, how="left", coalesce=True
        )

    logger.debug(f"Protein summary final size: {protein_summary.height} rows")

    # Column order: ID, priority, Status, FAIL_*, WARN_*, then the analysis results.
    def _reorder_cols(df, id_cols, priority_cols, expected_flags, is_protein=False):
        base = [c for c in id_cols + priority_cols + ["Status"] if c in df.columns]

        # Every flag of the platform's set is a column, all-False when nothing fired, so
        # the schema does not depend on the data. Other flags present are kept.
        df = df.with_columns(
            [pl.lit(False).alias(c) for c in expected_flags if c not in df.columns]
        )
        fail_cols = sorted([c for c in df.columns if c.startswith("FAIL_")])
        warn_cols = sorted([c for c in df.columns if c.startswith("WARN_")])

        if is_protein:
            analysis_col_order = [
                "EntrezGeneSymbol",
                "Dilution",
                "LOD",
                "Warn_Frac",
                "Missing_Frac",
                "LOD_Method",
            ]
            # CV columns may be grouped by study (Intra_CV_All, Intra_CV_{study}).
            cv_cols = sorted(
                [c for c in df.columns if c.startswith("Intra_CV") or c.startswith("Inter_CV")]
            )
            analysis_col_order.extend(cv_cols)
            # Detection_Rate columns may be one per group.
            detection_cols = sorted([c for c in df.columns if c.startswith("Detection_Rate")])
            analysis_col_order.extend(detection_cols)
        else:
            analysis_col_order = [
                *VENDOR_NOTE_SUMMARY_SCHEMA,
                "RFU_Outlier_Frac",
                "Warn_Frac",
                "Missing_Frac",
                "PC1_std",
                "PC2_std",
                "Z_Median",
                "Z_IQR",
                "Reported_Sex",
                # The sex the checks compared with and whether an expected_sex exception
                # supplied it (always present).
                "Expected_Protein_Sex",
                "Sex_Exception",
                "Predicted_Sex",
                "Predicted_Male_Probability",
                "Visit_Adj_Age",
                "Predicted_Age",
                "Std_Predicted_Age",
                *IDENTITY_SCORE_COLUMNS,
                UNFIT_IDENTITY_SUMMARY_COLUMN,
            ]

        analysis_cols = [c for c in analysis_col_order if c in df.columns]

        return df.with_columns(
            [pl.col(c).fill_null(False) for c in fail_cols + warn_cols if c in df.columns]
        ).select(base + fail_cols + warn_cols + analysis_cols)

    sample_priority = [
        "AccessionNumber",
        usubjid_col,
        "STUDY",
        "VISITNUM",
        "VISITNUM_ANALYSIS",
        "USUBJID_LABELED",
        "VISITNUM_LABELED",
        "ADT",
        "ADT_SOURCE",
    ]
    dose_timing = getattr(qc_instance, "dose_timing", None)
    if dose_timing is not None:
        # TRTEDT, the completion status (named after its ADSL column), the window end
        # and the day counts, in the order clinical.dose_timing gives them.
        sample_priority += [c for c in dose_timing.columns if c != sample_id_col]

    protein_priority = []
    if protein_name_col:
        protein_priority.append(protein_name_col)
    if uniprot_col and uniprot_col in analyte_schema:
        protein_priority.append(uniprot_col)

    sample_summary = _attach_vendor_notes(qc_instance, sample_summary)

    sample_summary = _reorder_cols(
        sample_summary,
        [sample_id_col],
        sample_priority,
        SAMPLE_QC_FLAGS[qc_instance.platform.name],
        is_protein=False,
    ).sort(sample_id_col)

    protein_summary = _reorder_cols(
        protein_summary,
        [protein_id_col] + ([protein_name_col] if protein_name_col else []),
        protein_priority[1:] if protein_name_col else protein_priority,
        PROTEIN_QC_FLAGS[qc_instance.platform.name],
        is_protein=True,
    ).sort([protein_name_col, protein_id_col] if protein_name_col else protein_id_col)
    # The protein ID breaks ties between assays that share a name, so the row order
    # does not depend on the hash order of the analyte table.

    if save_prefix:
        # Spaces in column names become underscores.
        sample_summary = sample_summary.rename(
            {col: col.replace(" ", "_") for col in sample_summary.columns}
        )
        protein_summary = protein_summary.rename(
            {col: col.replace(" ", "_") for col in protein_summary.columns}
        )

        sample_summary.sort(sample_id_col).write_csv(
            f"{save_prefix}.sample_qc_summary.tsv", separator="\t"
        )
        protein_summary.sort(protein_id_col).write_csv(
            f"{save_prefix}.assay_qc_summary.tsv", separator="\t"
        )
        logger.info(
            f"saved: {save_prefix}.sample_qc_summary.tsv, {save_prefix}.assay_qc_summary.tsv"
        )

        # Long format with QC status, biological samples only.
        schema_names = qc_instance.samples.collect_schema().names()

        # SomaScan: RFU is the delivered non-ANML RFU, and the eLOD is the assay summary's
        # LOD column. Olink: NPX and
        # PCNormalizedNPX, each with its own LOD (OlinkAnalyze's names: LOD pairs with
        # NPX, PCNormalizedLOD with PCNormalizedNPX); in memory olink_lod calls the
        # NPX-scale one NPXLOD.
        if "RFU" in schema_names:
            filename_suffix = "rfu_long.parquet"
            measurement_cols = ["RFU"]
            lod_cols = {}
        else:
            filename_suffix = "npx_long.parquet"
            measurement_cols = [OLINK_NPX, OLINK_PC_NPX]
            # In-memory LOD name -> output name, whatever value column QC read.
            lod_cols = {
                OLINK_NPX_LOD: OLINK_LOD_IN_FILES[OLINK_NPX],
                OLINK_PC_LOD: OLINK_LOD_IN_FILES[OLINK_PC_NPX],
            }
            absent = [c for c in measurement_cols if c not in schema_names]
            if absent:
                raise ValueError(
                    f"the Olink data lacks the column(s) {absent}: the long table carries both NPX and PCNormalizedNPX"
                )
        # The vendor's per-row flags on Olink, unchanged and under the vendor's names:
        # SampleQC per sample and block, AssayQC per sample and assay. QC already
        # requires both on input.
        vendor_flags = ["SampleQC", "AssayQC"] if lod_cols else []
        absent = [c for c in vendor_flags if c not in schema_names]
        if absent:
            raise ValueError(
                f"the Olink data lacks the per-row column(s) {absent}: the long table carries the vendor's SampleQC and AssayQC"
            )

        select_cols = [sample_id_col, protein_id_col, *measurement_cols]

        # PlateID goes on every output row.
        plate_col = getattr(qc_instance, "plate_id_col", None)
        if plate_col and plate_col in schema_names:
            select_cols.append(plate_col)
        select_cols += vendor_flags

        sample_status_expr = (
            pl.when(pl.col(sample_id_col).is_in(qc_instance.failed_samples))
            .then(pl.lit("FAIL"))
            .when(pl.col(sample_id_col).is_in(qc_instance.warned_samples))
            .then(pl.lit("WARN"))
            .otherwise(pl.lit("PASS"))
            .alias("SampleQC_Status")
        )
        protein_status_expr = (
            pl.when(pl.col(protein_id_col).is_in(qc_instance.failed_proteins))
            .then(pl.lit("FAIL"))
            .when(pl.col(protein_id_col).is_in(qc_instance.warned_proteins))
            .then(pl.lit("WARN"))
            .otherwise(pl.lit("PASS"))
            .alias("ProteinQC_Status")
        )

        long_plan = qc_instance.samples.select(select_cols).with_columns(
            [sample_status_expr, protein_status_expr]
        )

        # The LODs live in data_full, where olink_lod writes them (the samples
        # frame was captured before it ran).
        if lod_cols:
            data_full_schema = (
                qc_instance.data_full.collect_schema().names()
                if qc_instance.data_full is not None
                else []
            )
            absent = [c for c in lod_cols if c not in data_full_schema]
            if absent:
                raise ValueError(
                    f"the Olink LOD column(s) {absent} were not calculated before the long table was written"
                )
            # Key on the plate so each row keeps its own LOD.
            lod_keys = [sample_id_col, protein_id_col]
            if plate_col and plate_col in data_full_schema and plate_col in select_cols:
                lod_keys.append(plate_col)
            lod_rows = qc_instance.data_full.select([*lod_keys, *lod_cols])
            for lod_col in lod_cols:
                n_conflicting = (
                    lod_rows.group_by(lod_keys)
                    .agg(pl.col(lod_col).n_unique())
                    .filter(pl.col(lod_col) > 1)
                    .select(pl.len())
                    .collect()
                    .item()
                )
                if n_conflicting:
                    raise ValueError(
                        f"{n_conflicting} {tuple(lod_keys)} keys carry more than one distinct LOD"
                    )
            lod_plan = (
                lod_rows.group_by(lod_keys, maintain_order=True)
                .agg([pl.col(c).first() for c in lod_cols])
                .rename(lod_cols)
            )
            long_plan = long_plan.join(lod_plan, on=lod_keys, how="left")
            # The file's column order: values, their LODs, the plate, the statuses.
            long_plan = long_plan.select(
                [sample_id_col, protein_id_col, *measurement_cols, *lod_cols.values()]
                + ([plate_col] if plate_col in select_cols else [])
                + vendor_flags
                + ["SampleQC_Status", "ProteinQC_Status"]
            )

        # Row order must not depend on the input order or the thread count: sort on the
        # keys (sample, protein, plate), then on every other column so rows tied on the
        # keys are ordered too.
        long_columns = long_plan.collect_schema().names()
        long_keys = [c for c in (sample_id_col, protein_id_col, plate_col) if c in long_columns]
        long_plan = long_plan.sort(long_keys + [c for c in long_columns if c not in long_keys])

        output_file = f"{save_prefix}.{filename_suffix}"
        long_plan.sink_parquet(output_file)
        n_rows = pl.scan_parquet(output_file).select(pl.len()).collect().item()
        logger.info(f"saved: {output_file} ({n_rows:,} rows)")

        # The identity events ledger (always) and, when pQTL ran, the pQTL scores and
        # DNA links.
        from analysis.qc.identity import write_identity_outputs

        write_identity_outputs(qc_instance, save_prefix)

    return sample_summary, protein_summary


def _dense_codes(keys: pl.Series) -> tuple[pl.Series, np.ndarray]:
    """Returns the unique keys in first-appearance order and each row's index."""
    if keys.null_count():
        raise ValueError(f"cannot widen on '{keys.name}': {keys.null_count()} null keys")
    uniques = keys.unique(maintain_order=True)
    codes = keys.to_frame().join(
        uniques.to_frame().with_row_index("_code"),
        on=keys.name,
        how="left",
        maintain_order="left",
    )["_code"]
    return uniques, codes.to_numpy()


def _scatter_wide(
    long: pl.DataFrame, index: str, on: str, values: str, sort: bool = False
) -> tuple[pl.Series, list[str], np.ndarray, np.ndarray]:
    """Builds the matrix behind ``wide_matrix`` and which of its cells hold a value.

    Returns:
        Tuple of the row IDs, the column names, the matrix ``X`` and the boolean
        ``present`` mask marking the cells that hold a value.
    """
    dtype = long.schema[values]
    if dtype not in (pl.Float32, pl.Float64):
        raise TypeError(f"wide_matrix needs a float '{values}' column, got {dtype}")
    rows, r = _dense_codes(long[index])
    cols, c = _dense_codes(long[on])
    names = cols.cast(pl.String).to_list()
    if sort:
        row_order = rows.arg_sort().to_numpy()
        col_order = np.array(sorted(range(len(names)), key=names.__getitem__), dtype=np.int64)
        # Position of each code after sorting: index into the permutation's inverse.
        r, c = np.argsort(row_order)[r], np.argsort(col_order)[c]
        rows, names = rows[row_order], [names[j] for j in col_order]
    present = np.zeros((len(rows), len(names)), dtype=bool)
    present[r, c] = True
    if int(present.sum()) != long.height:
        raise ValueError(
            f"more than one value for a ('{index}', '{on}') pair; aggregate before widening"
        )
    series = long[values]
    # Fortran order, the layout of DataFrame.to_numpy(): the QC steps that follow (BLAS,
    # nanmedian, column means) add in a layout-dependent order, so a C-ordered matrix
    # would differ from the pivot's in the last bits.
    X = np.full(
        present.shape,
        np.nan,
        dtype=np.float32 if dtype == pl.Float32 else np.float64,
        order="F",
    )
    X[r, c] = series.to_numpy()
    if series.null_count():
        present[r, c] = series.is_not_null().to_numpy()
    return rows, names, X, present


def wide_matrix(
    long: pl.DataFrame, *, index: str, on: str, values: str, sort: bool = False
) -> tuple[pl.Series, list[str], np.ndarray]:
    """Builds the ``values`` of ``long`` as an ``index`` x ``on`` numpy matrix.

    The numpy form of ``long.pivot(on=on, index=index, values=values)`` followed by
    ``to_numpy``: the rows are the distinct ``index`` values and the columns the
    distinct ``on`` values, both in order of first appearance, and a pair with no row
    (or a null value) is NaN. ``X`` keeps the dtype of ``values`` (Float32 or Float64)
    and, like ``to_numpy`` of a frame, is Fortran-ordered. Polars' ``pivot`` costs more
    than linearly in the number of columns, so every samples x assays matrix of the pipeline comes from
    here.

    Args:
        long: Long-format frame.
        index: Column whose distinct values are the rows.
        on: Column whose distinct values are the columns.
        values: Float column holding the cell values.
        sort: If True, rows are in ``index`` order (Polars' sort) and columns in
            ``sorted()`` order of their names.

    Returns:
        Tuple of the row IDs, the column names and the matrix ``X``.

    Raises:
        ValueError: If a key is null or a pair occurs twice (aggregate first).
        TypeError: If ``values`` is not a float column.
    """
    rows, names, X, _ = _scatter_wide(long, index, on, values, sort)
    return rows, names, X


def wide_frame(long: pl.DataFrame, *, index: str, on: str, values: str) -> pl.DataFrame:
    """Builds ``long.pivot(on=on, index=index, values=values)`` from ``wide_matrix``.

    The result has the same columns, dtypes and nulls as the pivot: a missing pair is
    null, an explicit NaN stays NaN.
    """
    import pyarrow as pa

    rows, cols, X, present = _scatter_wide(long, index, on, values)
    XT, absent = np.ascontiguousarray(X.T), np.ascontiguousarray(~present.T)
    columns = [pa.array(XT[j], mask=absent[j]) for j in range(len(cols))]
    wide = pl.from_arrow(pa.table(columns, names=cols)) if cols else pl.DataFrame()
    return wide.insert_column(0, rows)


def impute_missing_values(
    X: np.ndarray,
    donors: np.ndarray | None = None,
    n_neighbors: int = 5,
) -> np.ndarray:
    """Imputes missing values by K-nearest neighbors (deterministic: no seed).

    Args:
        X: 2D array with potential NaN values (samples x features).
        donors: Boolean row mask; the imputer is fitted on these rows only, so only
            they can serve as neighbors. None fits on every row.
        n_neighbors: Number of neighbors for KNN imputation.

    Returns:
        The array with NaN values imputed; ``X`` itself when every value is finite.

    Raises:
        ValueError: If ``X`` holds an infinite value (``KNNImputer`` rejects it).
    """
    if np.isfinite(X).all():
        return X
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    return imputer.fit(X if donors is None else X[donors]).transform(X)
