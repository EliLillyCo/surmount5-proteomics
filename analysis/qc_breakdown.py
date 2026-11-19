"""QC exclusion-flow breakdown for the study-design figure, per platform.

Builds the frame that ``analysis/pipelines/qc_exclusion_breakdown.py`` writes to
``analysis/outputs/qc_exclusion_breakdown.parquet``, the artifact holding the per-reason
sample/subject counts of the Olink and SomaScan QC flows, so that the
study-design figure reads them from data.

Staging (reproduces the net step totals in the figure's CONSORT)
----------------------------------------------------------------
Working from each platform's sample-QC summary (``analysis.qc_contract``)::

    cohort       = analyzed-arm subjects profiled on the platform
    input        = ~BRIDGE_FLAG                            (samples entering QC)
    excluded     = input samples failing an active flag, each charged to the first
                   flag it fails in FLAG_REGISTRY order and so to that flag's
                   category (technical, identity, longitudinal)
    final        = input minus every exclusion              (== Status != FAIL)

A registered flag is active for a platform when its column is in the summary and, for a
run-specific flag (``FlagSpec.study_specific``), it fires on at least one input sample;
the reported reasons are the active flags grouped by reason, in registry order, and an
active reason with no samples keeps its zero-count row.

The final tier keeps WARN samples: the analysis cohort is ``Status != "FAIL"``.
``build_breakdown`` asserts ``final == input - sum(excluded)``, ``final ==
count(Status != FAIL, non-bridge)`` and that each category total equals the QC's own
exclusion table (single attribution in the QC's precedence). Tubes the identity check
moved to the subject their genotype matches are counted (``reassigned``), not excluded.

Artifact schema (tidy long form, one fact per row)
--------------------------------------------------
- platform       — ``olink`` | ``soma``
- stage          — ``randomized`` (repeated per platform) | ``cohort`` | ``input`` | a
                   flag category | ``reassigned`` | ``identity_check`` | ``correction`` |
                   ``genotype_coverage`` | ``final_visit`` (``reason`` ``week_<n>``,
                   ``label`` ``Week <n>``) | ``final``
- order          — bullet order within (platform, stage); 0 for non-bullet rows
- reason         — machine slug (null for non-bullet rows)
- label          — display label used in the figure callouts (null where N/A)
- arm            — ``TREAT_VALUE`` | ``CONTROL_VALUE`` of ``analysis.platform_inputs``
                   for per-arm rows (else null)
- n_samples      — sample count (null where the row carries only subjects)
- n_subjects     — subject count (randomized / cohort / final rows; else null)
"""

from __future__ import annotations

import json

import polars as pl

from analysis.design import CONTROL_VALUE, RAW_CONTROL_VALUE, RAW_TREAT_VALUE, TREAT_VALUE
from analysis.qc.corrections import digest
from analysis.qc.study import trial_config
from analysis.qc_contract import (
    BRIDGE_FLAG,
    COHORT,
    EXCLUDED,
    FLAG_CATEGORIES,
    FLAG_REGISTRY,
    IDENTITY,
    IDENTITY_FAIL_PREFIX,
    IDENTITY_WARN_PREFIX,
    PLATFORM_SPECS,
    PLATFORMS,
    REASSIGNED,
    TABLE_ONLY_REASONS,
    WEEK_COL,
    cohort_samples,
    flag_category,
    genotyped_subjects,
    manifest_build_report_path,
    read_exclusion_table,
    read_sample_summary,
    reassigned_samples,
    study_of,
)
from analysis.qc_flow import (
    BRIDGE_SWAP,
    CORRECTION,
    FINAL_VISIT,
    GENOTYPE_COVERAGE,
    IDENTITY_CHECK,
    IDENTITY_CHECKED,
    NOT_GENOTYPED,
    REASSIGN,
)
from paths import PHENO_DIR

# ADSL TRT01A -> the arm label of the analysis outputs; other arms map to nothing.
ARM_MAP: dict[str, str] = {RAW_TREAT_VALUE: TREAT_VALUE, RAW_CONTROL_VALUE: CONTROL_VALUE}
ARM_ORDER: tuple[str, ...] = (TREAT_VALUE, CONTROL_VALUE)

# Outcomes of the genetic identity check, in the order a sample is assigned to the first
# that applies: (reason slug, display label). Every input sample gets exactly one.
# "corrected": tubes a ``CORRECTED_TYPES`` correction moved in the manifest build. A tube
# the identity check moved carries its genotype's participant in the summary, so it counts
# as confirmed, and the flow reports those as reassigned.
IDENTITY_OUTCOMES: tuple[tuple[str, str], ...] = (
    ("excluded", "Excluded by an identity flag"),
    ("corrected", "Corrected (sample swap)"),
    ("flagged", "Flagged, kept"),
    (NOT_GENOTYPED, "Participant not genotyped"),
    ("confirmed", "Genotype confirmed"),
)
# Correction types that move a tube in the manifest build, each counted as one corrected
# sample.
CORRECTED_TYPES = (REASSIGN, BRIDGE_SWAP)
# The ``correction`` rows count only the corrected samples QC kept (Status != FAIL).
CORRECTION_LABELS = {
    REASSIGN: "Reassigned by adjudication, retained",
    BRIDGE_SWAP: "Well swap corrected, retained",
}

SCHEMA = {
    "platform": pl.String,
    "stage": pl.String,
    "order": pl.Int32,
    "reason": pl.String,
    "label": pl.String,
    "arm": pl.String,
    "n_samples": pl.Int64,
    "n_subjects": pl.Int64,
}

# (reason slug, display label, [FAIL flag columns]) in single-attribution order.
Reason = tuple[str, str, list[str]]


def _row(platform: str, stage: str, order: int = 0, **values) -> dict:
    row = {"platform": platform, "stage": stage, "order": order}
    row |= {k: values.get(k) for k in ("reason", "label", "arm", "n_samples", "n_subjects")}
    return row


def active_reasons(inputs: pl.DataFrame) -> dict[str, list[Reason]]:
    """The reasons each flag category reports for one platform.

    Args:
        inputs: The platform's non-bridge samples from its sample summary.

    Returns:
        Category -> reasons in registry order, each with its active flags: those in
        the summary, and for run-specific flags only those that fire on an input
        sample.
    """
    out: dict[str, list[Reason]] = {category: [] for category in FLAG_CATEGORIES}
    for spec in FLAG_REGISTRY:
        if spec.flag not in inputs.columns:
            continue
        if spec.study_specific and not inputs[spec.flag].any():
            continue
        reasons = out[spec.category]
        for reason, _, flags in reasons:
            if reason == spec.reason:
                flags.append(spec.flag)
                break
        else:
            reasons.append((spec.reason, spec.label, [spec.flag]))
    return out


def charged_reason(reasons: dict[str, list[Reason]]) -> pl.Expr:
    """The reason each sample is charged to: its first active flag in registry order.

    Args:
        reasons: The platform's active reasons, as ``active_reasons`` returns them.

    Returns:
        A string expression, null for a sample that fails no active flag.
    """
    reason_of = {
        flag: reason for specs in reasons.values() for reason, _, flags in specs for flag in flags
    }
    candidates = [
        pl.when(pl.col(spec.flag)).then(pl.lit(reason_of[spec.flag]))
        for spec in FLAG_REGISTRY
        if spec.flag in reason_of
    ]
    return pl.coalesce(candidates) if candidates else pl.lit(None, dtype=pl.String)


def category_totals(charged: pl.DataFrame, reasons: dict[str, list[Reason]]) -> dict[str, int]:
    """Samples charged to each category's reasons.

    Args:
        charged: Samples with their charged reason in ``_reason``.
        reasons: The platform's active reasons.

    Returns:
        Category -> excluded sample count, for every category.
    """
    return {
        category: charged.filter(pl.col("_reason").is_in([r for r, _, _ in specs])).height
        for category, specs in reasons.items()
    }


def table_category_totals(table: pl.DataFrame) -> dict[str, int]:
    """Excluded samples per category in the QC's exclusion table.

    Tubes the table counts but the sample summary has no row for (``TABLE_ONLY_REASONS``)
    are left out. Any other reason that is not a registered flag is an exclude
    correction's label, which the table writes in place of ``FAIL_Correction_Exclude``.

    Args:
        table: The platform's exclusion table (``read_exclusion_table``).

    Returns:
        Category -> excluded sample count, for every category.
    """
    registered = {spec.flag for spec in FLAG_REGISTRY}
    totals = dict.fromkeys(FLAG_CATEGORIES, 0)
    for reason, n in (
        table.filter(pl.col("status") == EXCLUDED, ~pl.col("reason").is_in(TABLE_ONLY_REASONS))
        .group_by("reason")
        .agg(pl.col("n_samples").sum())
        .sort("reason")
        .iter_rows()
    ):
        totals[flag_category(reason) if reason in registered else IDENTITY] += n
    return totals


def reassigned_count(platform: str, summary: pl.DataFrame) -> int:
    """Reassigned tubes of a platform, checked against its sample summary.

    Args:
        platform: ``olink`` or ``soma``.
        summary: The platform's sample summary.

    Returns:
        The number of identity events whose action is ``REASSIGNED``.

    Raises:
        RuntimeError: When a reassigned tube's summary row does not carry the subject
            its genotype matched.
    """
    sid = PLATFORM_SPECS[platform].sample_id
    moved = reassigned_samples(platform)
    rows = moved.join(
        summary.select(pl.col(sid).alias("SampleID"), "USUBJID"), on="SampleID", how="left"
    )
    wrong = rows.filter(pl.col("USUBJID").ne_missing(pl.col("USUBJID_INFERRED"))).height
    if wrong:
        raise RuntimeError(f"{platform}: {wrong} reassigned tubes lack their inferred USUBJID")
    return moved.height


def identity_outcome(
    inputs: pl.DataFrame, sid: str, corrected: pl.Series, genotyped: pl.Series
) -> pl.Expr:
    """The identity-check outcome of each input sample (``IDENTITY_OUTCOMES`` slugs).

    Args:
        inputs: The platform's non-bridge samples with their identity flag columns.
        sid: The platform's sample ID column.
        corrected: Sample IDs a ``CORRECTED_TYPES`` correction moved in the manifest
            build (``corrected_samples``).
        genotyped: Participants with a genotype to compare against.

    Returns:
        A string expression: excluded (any FAIL identity flag), corrected, flagged (any
        WARN identity flag), not genotyped (the sample's participant has no genotype) or
        confirmed, the first that applies.
    """
    fail = [c for c in inputs.columns if c.startswith(IDENTITY_FAIL_PREFIX)]
    warn = [c for c in inputs.columns if c.startswith(IDENTITY_WARN_PREFIX)]
    return (
        pl.when(pl.any_horizontal(fail) if fail else pl.lit(False))
        .then(pl.lit("excluded"))
        .when(pl.col(sid).is_in(corrected.implode()))
        .then(pl.lit("corrected"))
        .when(pl.any_horizontal(warn) if warn else pl.lit(False))
        .then(pl.lit("flagged"))
        .when(~pl.col("USUBJID").is_in(genotyped.implode()))
        .then(pl.lit(NOT_GENOTYPED))
        .otherwise(pl.lit("confirmed"))
    )


def corrected_samples(platform: str, sample_ids: pl.Series) -> pl.DataFrame:
    """The samples a ``CORRECTED_TYPES`` correction moved in the platform's build.

    Args:
        platform: ``olink`` or ``soma``.
        sample_ids: The platform's input sample IDs.

    Returns:
        ``SampleID`` and correction ``type`` of each sample in ``sample_ids`` such a
        correction names, in ``sample_ids`` order.

    Raises:
        RuntimeError: When a type's count differs from the build report's.
    """
    wanted = {
        c.sampleid_sha256: c.type
        for c in trial_config(study_of(platform)).corrections
        if c.type in CORRECTED_TYPES
    }
    hits = (
        pl.DataFrame({"SampleID": sample_ids})
        .with_columns(
            pl.col("SampleID")
            .map_elements(digest, return_dtype=pl.String)
            .replace_strict(wanted, default=None, return_dtype=pl.String)
            .alias("type")
        )
        .drop_nulls("type")
    )
    applied = json.loads(manifest_build_report_path(platform).read_text())["corrections"]["applied"]
    for kind in CORRECTED_TYPES:
        found = hits.filter(pl.col("type") == kind).height
        # The build report lists only the correction types it applied at least once.
        if found != applied.get(kind, 0):
            raise RuntimeError(
                f"{platform}: {found} input samples carry a {kind} correction, "
                f"the manifest build applied {applied.get(kind, 0)}"
            )
    return hits


def _identity_rows(platform: str, nonbridge: pl.DataFrame) -> list[dict]:
    """Breakdown rows of the genetic identity check.

    The samples checked, then each outcome, then the corrected samples of each type that
    QC kept (Status != FAIL), then the samples whose participant has no genotype whatever
    their outcome.
    """
    sid = PLATFORM_SPECS[platform].sample_id
    corrected = corrected_samples(platform, nonbridge[sid])
    genotyped = genotyped_subjects(platform)
    outcome = nonbridge.select(
        identity_outcome(nonbridge, sid, corrected["SampleID"], genotyped).alias("o")
    )["o"]
    counts = dict(outcome.value_counts().iter_rows())
    kept = corrected.join(
        nonbridge.filter(COHORT).select(pl.col(sid).alias("SampleID")), on="SampleID", how="semi"
    )
    by_type = dict(kept.group_by("type").len().iter_rows())

    def row(stage: str, order: int, reason: str, label: str, n: int) -> dict:
        return _row(platform, stage, order, reason=reason, label=label, n_samples=int(n))

    rows = [row(IDENTITY_CHECK, 0, IDENTITY_CHECKED, "Samples checked", nonbridge.height)]
    rows += [
        row(IDENTITY_CHECK, i, slug, label, counts.get(slug, 0))
        for i, (slug, label) in enumerate(IDENTITY_OUTCOMES, start=1)
    ]
    rows += [
        row(CORRECTION, i, kind, CORRECTION_LABELS[kind], by_type.get(kind, 0))
        for i, kind in enumerate(CORRECTED_TYPES, start=1)
    ]
    missing = nonbridge.filter(~pl.col("USUBJID").is_in(genotyped.implode())).height
    rows.append(row(GENOTYPE_COVERAGE, 0, NOT_GENOTYPED, "Participant not genotyped", missing))
    return rows


def _visit_rows(platform: str, adsl: pl.DataFrame, final_samples: int) -> list[dict]:
    """Analysis-ready samples per arm and week, checked against the final sample count."""
    per_cell = (
        cohort_samples(platform)
        .collect()
        .join(adsl, on="USUBJID", how="left")
        .drop_nulls("arm")
        .group_by("arm", WEEK_COL)
        .len()
    )
    # The arms of the analysis are every arm in the cohort: the counts must cover it.
    if per_cell["len"].sum() != final_samples:
        raise RuntimeError(
            f"{platform}: samples per arm and week sum to {per_cell['len'].sum()}, "
            f"not the {final_samples} analysis-ready samples"
        )
    counts = {(arm, week): n for arm, week, n in per_cell.iter_rows()}
    weeks = sorted(per_cell[WEEK_COL].unique().to_list())
    return [
        _row(
            platform,
            FINAL_VISIT,
            i,
            reason=f"week_{week}",
            label=f"Week {week}",
            arm=arm,
            n_samples=int(counts.get((arm, week), 0)),
        )
        for arm in ARM_ORDER
        for i, week in enumerate(weeks)
    ]


def _adsl_arm() -> pl.DataFrame:
    """ADSL analysis arm and randomized assigned arm (TRT01P)."""
    return (
        pl.scan_parquet(PHENO_DIR / "adsl.parquet")
        .select(
            "USUBJID",
            pl.col("TRT01A")
            .replace_strict(ARM_MAP, default=None, return_dtype=pl.String)
            .alias("arm"),
            pl.when(pl.col("RANDFL") == "Y")
            .then(pl.col("TRT01P").replace_strict(ARM_MAP, default=None, return_dtype=pl.String))
            .otherwise(None)
            .alias("randomized_arm"),
        )
        .collect()
    )


def _arm_rows(platform: str, stage: str, subjects: pl.DataFrame, label: str | None) -> list[dict]:
    """The overall and per-arm subject rows of one stage."""
    by_arm = dict(subjects.group_by("arm").len().iter_rows())
    rows = [_row(platform, stage, 0, label=label, n_subjects=subjects.height)]
    rows += [
        _row(platform, stage, i, arm=arm, n_subjects=int(by_arm.get(arm, 0)))
        for i, arm in enumerate(ARM_ORDER, start=1)
    ]
    return rows


def platform_breakdown(platform: str, adsl: pl.DataFrame) -> list[dict]:
    """The breakdown rows of one platform.

    Raises:
        RuntimeError: When the staging identities fail (final != input minus exclusions,
            final != non-bridge Status != FAIL) or the category totals differ from the QC
            exclusion table.
    """
    df = read_sample_summary(platform)
    randomized = adsl.select(pl.col("randomized_arm").alias("arm")).drop_nulls("arm")
    rows = _arm_rows(platform, "randomized", randomized, "Randomized (analyzed arms)")
    adsl = adsl.drop("randomized_arm")

    # Rows are made unique by sample ID, as in the QC exclusion table.
    sid = PLATFORM_SPECS[platform].sample_id
    nonbridge = df.filter(~pl.col(BRIDGE_FLAG)).unique(
        subset=sid, keep="first", maintain_order=True
    )
    reasons = active_reasons(nonbridge)
    charged = nonbridge.with_columns(charged_reason(reasons).alias("_reason"))
    final_set = charged.filter(pl.col("_reason").is_null())
    totals = category_totals(charged, reasons)

    # Staging identities; they stop the build if they drift.
    expected_final = nonbridge.height - sum(totals.values())
    if final_set.height != expected_final:
        raise RuntimeError(
            f"{platform}: staging identity broken ({final_set.height} != {expected_final})"
        )
    not_fail = nonbridge.filter(COHORT)
    if final_set.height != not_fail.height:
        raise RuntimeError(
            f"{platform}: final ({final_set.height}) != non-bridge Status!=FAIL "
            f"({not_fail.height}); the analysis cohort rule is Status != FAIL"
        )
    table_totals = table_category_totals(read_exclusion_table(platform))
    if totals != table_totals:
        raise RuntimeError(
            f"{platform}: category totals {totals} != the QC exclusion table's {table_totals}"
        )

    cohort = (
        nonbridge.select("USUBJID").unique().join(adsl, on="USUBJID", how="left").drop_nulls("arm")
    )
    rows += _arm_rows(platform, "cohort", cohort, "Proteomics cohort")
    rows.append(_row(platform, "input", label="Non-bridge samples", n_samples=nonbridge.height))

    counts = dict(charged.group_by("_reason").len().drop_nulls("_reason").iter_rows())
    for stage in FLAG_CATEGORIES:
        for i, (reason, label, _) in enumerate(reasons[stage], start=1):
            rows.append(
                _row(
                    platform,
                    stage,
                    i,
                    reason=reason,
                    label=label,
                    n_samples=int(counts.get(reason, 0)),
                )
            )

    rows.append(
        _row(
            platform,
            REASSIGNED,
            label="Reassigned to genotype-matched participant",
            n_samples=reassigned_count(platform, nonbridge),
        )
    )
    rows.extend(_identity_rows(platform, nonbridge))
    rows.extend(_visit_rows(platform, adsl, final_set.height))

    final_ppts = final_set.select("USUBJID").unique().join(adsl, on="USUBJID", how="left")
    final_rows = _arm_rows(platform, "final", final_ppts, None)
    final_rows[0]["n_samples"] = final_set.height
    rows.extend(final_rows)
    return rows


def build_breakdown() -> pl.DataFrame:
    """Compute the QC-flow breakdown of both platforms.

    Returns:
        Tidy long frame in the artifact schema of the module docstring, one row per fact.
    """
    adsl = _adsl_arm()
    rows = [row for platform in PLATFORMS for row in platform_breakdown(platform, adsl)]
    return pl.DataFrame(rows, schema=SCHEMA)
