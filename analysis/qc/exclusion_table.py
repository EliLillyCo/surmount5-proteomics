"""Counts excluded samples and non-analysis visits for the exclusion tables.

Pure functions on a sample-level frame. They add no rule and change no sample's
status; they only count what the manifest, technical, identity and clinical steps
already flagged.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from analysis.qc.flags import FAIL_NOT_ANALYSIS_VISIT

STATUS_INCLUDED = "Included"
STATUS_EXCLUDED = "Excluded"

VISIT_COLUMNS = ["visit", "n_samples", "n_subjects"]


def _lazy(samples: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    return samples.lazy() if isinstance(samples, pl.DataFrame) else samples


def _require(lf: pl.LazyFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in lf.collect_schema().names()]
    if missing:
        raise ValueError(f"exclusion_table: missing columns {missing}")


def first_reason(reason_cols: Sequence[str], manifest_exclude_col: str | None) -> pl.Expr:
    """Builds the reason a sample is counted under.

    The reason is the sample's ``exclude`` label, else the first true flag of
    ``reason_cols`` (null counts as False), else null.

    Args:
        reason_cols: Boolean FAIL flag columns, in precedence order.
        manifest_exclude_col: Column holding an ``exclude`` correction's label, or
            None.

    Returns:
        A string expression.
    """
    candidates = [pl.when(pl.col(c).fill_null(False)).then(pl.lit(c)) for c in reason_cols]
    if manifest_exclude_col is not None:
        candidates.insert(0, pl.col(manifest_exclude_col))
    return pl.coalesce(candidates) if candidates else pl.lit(None, dtype=pl.Utf8)


def exclusion_table(
    samples: pl.DataFrame | pl.LazyFrame,
    reason_cols: Sequence[str],
    *,
    arm_col: str = "ARM",
    subject_col: str = "USUBJID",
    visit_col: str = "VISITNUM",
    manifest_exclude_col: str | None = "MANIFEST_EXCLUDE",
    root_reason_col: str | None = None,
) -> pl.DataFrame:
    """Counts included and excluded samples by randomized arm, reason and visit.

    Args:
        samples: One row per study sample. The caller removes bridge and control rows,
            or lists their flag in ``reason_cols``; unflagged rows count as included.
        reason_cols: Boolean FAIL flag columns (for example ``FAIL_Sample_QC``,
            ``FAIL_After_Treatment``), in precedence order. Null counts as False. A
            sample with several flags is counted once, under the first one listed.
        arm_col: Randomized arm (ADSL ARM).
        subject_col: Subject identifier.
        visit_col: VISITNUM of the sample, the code the visit rules used.
        manifest_exclude_col: The ``label`` of an ``exclude`` correction, null
            otherwise; ``None`` if the frame has no such column. A correction outranks
            every flag, and its label is the reason label.
        root_reason_col: The upstream cause of a cascade loss, null otherwise;
            ``None`` leaves ``root_reason`` out of the table.

    Returns:
        One row per (arm, status, reason, root_reason when given, visit) with
        ``n_samples`` and ``n_subjects`` (subjects with at least one sample in that
        row). ``reason`` is null for included rows. Counts only; no sample or subject
        ids.

    Raises:
        ValueError: If a needed column is missing or an ``exclude`` label is blank.
    """
    lf = _lazy(samples)
    needed = [arm_col, subject_col, visit_col, *reason_cols]
    if manifest_exclude_col is not None:
        needed.append(manifest_exclude_col)
    if root_reason_col is not None:
        needed.append(root_reason_col)
    _require(lf, needed)
    if manifest_exclude_col is not None:
        n_blank = (
            lf.filter(pl.col(manifest_exclude_col).str.strip_chars() == "")
            .select(pl.len())
            .collect()
            .item()
        )
        if n_blank:
            raise ValueError(
                f"exclusion_table: {n_blank} samples have a blank {manifest_exclude_col}; "
                "an exclude correction needs a reason"
            )

    keys = ["arm", "status", "reason", "visit"]
    root = []
    if root_reason_col is not None:
        keys.insert(3, "root_reason")
        root = [pl.col(root_reason_col).cast(pl.Utf8).alias("root_reason")]
    return (
        lf.select(
            pl.col(arm_col).alias("arm"),
            pl.col(subject_col).alias("_subject"),
            pl.col(visit_col).alias("visit"),
            first_reason(reason_cols, manifest_exclude_col).alias("reason"),
            *root,
        )
        .with_columns(
            pl.when(pl.col("reason").is_null())
            .then(pl.lit(STATUS_INCLUDED))
            .otherwise(pl.lit(STATUS_EXCLUDED))
            .alias("status")
        )
        .group_by(keys)
        .agg(
            pl.len().cast(pl.Int64).alias("n_samples"),
            pl.col("_subject").drop_nulls().n_unique().cast(pl.Int64).alias("n_subjects"),
        )
        .select(*keys, "n_samples", "n_subjects")
        .sort(keys, nulls_last=True)
        .collect()
    )


def non_analysis_visit_counts(
    samples: pl.DataFrame | pl.LazyFrame,
    *,
    flag_col: str = FAIL_NOT_ANALYSIS_VISIT,
    subject_col: str = "USUBJID",
    visit_col: str = "VISITNUM",
) -> pl.DataFrame:
    """Counts samples failing the non-analysis-visit rule by VISITNUM code.

    Every sample with ``flag_col`` true is counted, whatever other flags it carries. A
    null VISITNUM is its own row. The caller labels the result with the trial.

    Args:
        samples: One row per study sample.
        flag_col: Boolean non-analysis-visit flag.
        subject_col: Subject identifier.
        visit_col: VISITNUM of the sample.

    Returns:
        ``visit``, ``n_samples``, ``n_subjects``, sorted by visit (null last).
    """
    lf = _lazy(samples)
    _require(lf, [flag_col, subject_col, visit_col])
    return (
        lf.filter(pl.col(flag_col).fill_null(False))
        .group_by(pl.col(visit_col).alias("visit"))
        .agg(
            pl.len().cast(pl.Int64).alias("n_samples"),
            pl.col(subject_col).drop_nulls().n_unique().cast(pl.Int64).alias("n_subjects"),
        )
        .select(VISIT_COLUMNS)
        .sort("visit", nulls_last=True)
        .collect()
    )
