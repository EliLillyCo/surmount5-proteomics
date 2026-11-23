"""The contract between the QC stage and everything downstream of it.

SURMOUNT-5 is one trial measured on two platforms, so a QC run is a platform run:
``surmount5_olink`` (Olink Explore HT) or ``surmount5_soma`` (SomaScan 11K). The QC stage
(``analysis.qc.study.run_study``) writes, per run and named by the run's stem, a sample
summary, an assay summary, an exclusion table, the identity events, a long value table
and the sample manifest it built into ``<qc_root>/<study>/``. This module owns what the
rest of the repository may rely on in those files: their names, the required columns,
each platform's identifier and value columns, the rule that selects the analysis cohort,
and the registry of exclusion flags. Every reader outside ``analysis/qc/`` goes through
it.

Readers take a platform (``"olink"`` or ``"soma"``); ``study_of`` gives its run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from analysis.design import (
    PLATFORMS,
    RAW_CONTROL_VALUE,
    RAW_TREAT_VALUE,
    STUDY_STEM,
    VISIT_TO_WEEK,
)
from analysis.qc_paths import study_qc_root

TRIAL = STUDY_STEM
# Platform -> QC run (``PLATFORMS`` is ``analysis.design``'s, re-exported here).
PLATFORM_STUDIES: dict[str, str] = {platform: f"{TRIAL}_{platform}" for platform in PLATFORMS}

STATUS_LEVELS: tuple[str, ...] = ("PASS", "WARN", "FAIL")
# Sample-flag columns are Boolean and non-null.
FLAG_PREFIXES: tuple[str, ...] = ("FAIL_", "WARN_")

# The visit a sample is analyzed at. It is null for a FAIL sample and can differ from the
# tube's own VISITNUM.
ANALYSIS_VISIT = "VISITNUM_ANALYSIS"
# The analysis visits and their weeks (baseline is week 0): ``analysis.design.VISIT_TO_WEEK``,
# imported above.
WEEK_COL = "WEEK"


@dataclass(frozen=True)
class PlatformSpec:
    """The identifier and value columns of one platform's QC outputs.

    Attributes:
        sample_id: Sample identifier column (``SampleID`` / ``SampleId``).
        protein_id: Assay identifier column (``OlinkID`` / ``SeqId``).
        protein_name: Assay name column of the assay summary.
        long_suffix: File suffix of the long value table.
        value_cols: Value columns of the long table.
        sample_status: Per-sample QC verdict column of the long table.
    """

    sample_id: str
    protein_id: str
    protein_name: str
    long_suffix: str
    value_cols: tuple[str, ...]
    sample_status: str = "SampleQC_Status"


PLATFORM_SPECS: dict[str, PlatformSpec] = {
    # QC reads PCNormalizedNPX; the long table carries NPX too (the analysis layer).
    "olink": PlatformSpec(
        "SampleID", "OlinkID", "Assay", "npx_long.parquet", ("NPX", "PCNormalizedNPX")
    ),
    # Non-ANML RFU (linear scale).
    "soma": PlatformSpec("SampleId", "SeqId", "Target", "rfu_long.parquet", ("RFU",)),
}

# Sample summary columns, with the dtype every reader parses them as (the visits as
# Float64 whether the writer formats them as integers or floats). The sample ID column
# is the platform's.
SAMPLE_REQUIRED: dict[str, type[pl.DataType]] = {
    "USUBJID": pl.String,
    "VISITNUM": pl.Float64,
    ANALYSIS_VISIT: pl.Float64,
    "Status": pl.String,
}


def _group_name(arm: str) -> str:
    """The QC's column-safe name of a treatment arm (``ProteomicsQC._standardize_group_name``)."""
    value = arm.replace("/", "_")
    value = re.sub(r"\s+(\d)", r"-\1", value)
    value = re.sub(r"(mg|mcg|ug|g)\s+", r"\1_", value)
    return value.replace(" ", "")


# Per-arm detection-rate columns of the assay summary, and the names the tables use.
ARM_DETECTION_COLUMNS: dict[str, str] = {
    f"Detection_Rate_{_group_name(RAW_TREAT_VALUE)}": "Detection_Rate_TZP",
    f"Detection_Rate_{_group_name(RAW_CONTROL_VALUE)}": "Detection_Rate_SEMA",
}
ASSAY_REQUIRED_COMMON: tuple[str, ...] = (
    "UniProt",
    "Intra_CV",
    "Detection_Rate_All",
    *ARM_DETECTION_COLUMNS,
)

# Exclusion table columns: one row per arm, status, reason, root reason and visit.
EXCLUSION_REQUIRED: tuple[str, ...] = (
    "STUDY",
    "arm",
    "status",
    "reason",
    "root_reason",
    "visit",
    "n_samples",
    "n_subjects",
)
EXCLUDED = "Excluded"
# Exclusion-table reasons for tubes with no sample-summary row (tubes in the manifest with
# no assay data).
TABLE_ONLY_REASONS: tuple[str, ...] = ("FAIL_Not_Assayed",)

# The analysis cohort: every sample that did not FAIL (WARN samples stay in).
COHORT: pl.Expr = pl.col("Status") != "FAIL"


TECHNICAL = "technical"
IDENTITY = "identity"
LONGITUDINAL = "longitudinal"
# The order the exclusion flow draws its stages in.
FLAG_CATEGORIES: tuple[str, ...] = (TECHNICAL, IDENTITY, LONGITUDINAL)


@dataclass(frozen=True)
class FlagSpec:
    """One FAIL flag of the sample summary and how the exclusion flow reports it.

    Attributes:
        flag: Column name in the sample summary.
        category: Exclusion stage the flag belongs to, one of ``FLAG_CATEGORIES``.
        reason: Machine slug of the reported reason; flags sharing a reason are
            counted together.
        label: Display label of the reason.
        study_specific: True for a flag the flow reports only when it fires on one of
            the run's samples: a criterion a run may not apply, or a data-linkage check.
            A flag missing from a platform's summary is not reported.
    """

    flag: str
    category: str
    reason: str
    label: str
    study_specific: bool = False


# Every FAIL flag a sample summary may carry, in the QC's precedence order: an excluded
# sample is charged to the first flag it fails, as in the QC's own exclusion table, and
# so to that flag's category. A correction exclusion outranks every other flag. The
# vendor sample flag is FAIL_Sample_QC on Olink and FAIL_RowCheck on SomaScan.
FLAG_REGISTRY: tuple[FlagSpec, ...] = (
    FlagSpec("FAIL_Correction_Exclude", IDENTITY, "correction", "Adjudicated exclusion", True),
    FlagSpec("FAIL_No_Manifest_Match", TECHNICAL, "no_manifest", "Not in sample manifest", True),
    FlagSpec("FAIL_Sample_QC", TECHNICAL, "sample_qc", "Sample-QC fail", True),
    FlagSpec("FAIL_PCA_Outlier", TECHNICAL, "pca", "PCA outlier"),
    FlagSpec("FAIL_Median_IQR_Outlier", TECHNICAL, "median_iqr", "Median / IQR outlier"),
    FlagSpec("FAIL_RowCheck", TECHNICAL, "rowcheck", "Row-check fail", True),
    FlagSpec("FAIL_Identity_Foreign", IDENTITY, "identity_foreign", "Identity mismatch"),
    FlagSpec(
        "FAIL_Identity_Unresolved", IDENTITY, "identity_unresolved", "Unresolved identity", True
    ),
    FlagSpec("FAIL_No_Treatment_Arm", LONGITUDINAL, "no_arm", "No treatment arm", True),
    FlagSpec("FAIL_No_ADSL_Match", LONGITUDINAL, "no_adsl", "Not in ADSL", True),
    FlagSpec("FAIL_No_Treatment_Window", LONGITUDINAL, "no_window", "No treatment dates", True),
    FlagSpec("FAIL_No_ADT", LONGITUDINAL, "no_adt", "No sample date", True),
    FlagSpec("FAIL_After_Treatment", LONGITUDINAL, "post_treatment", "Post-treatment sample"),
    FlagSpec(
        "FAIL_Not_Analysis_Visit", LONGITUDINAL, "not_analysis_visit", "Unscheduled visit", True
    ),
    FlagSpec("FAIL_Duplicate_Visit", LONGITUDINAL, "duplicate_visit", "Duplicate visit", True),
    FlagSpec(
        "FAIL_Baseline_After_Dose", LONGITUDINAL, "baseline_after_dose", "Post-dose baseline", True
    ),
    FlagSpec("FAIL_No_Baseline", LONGITUDINAL, "no_baseline", "No baseline sample"),
    FlagSpec("FAIL_Only_Baseline", LONGITUDINAL, "only_baseline", "Only baseline sample"),
)

# Marks bridge tubes, which are not trial samples. The QC keeps them out of the sample
# summary (the column is all False) and marks them in its manifest instead.
BRIDGE_FLAG = "FAIL_Bridge_Sample"
BRIDGE_COL = "BridgeSample"
# Identity-event action of a tube moved to the subject its genotype matches.
REASSIGNED = "reassigned"
# Sample-summary flag columns the genetic identity check sets: FAIL flags exclude the
# tube, WARN flags keep it.
IDENTITY_FAIL_PREFIX = "FAIL_Identity_"
IDENTITY_WARN_PREFIX = "WARN_Identity_"


def flag_category(flag: str) -> str:
    """The exclusion category of a registered FAIL flag.

    Args:
        flag: A ``FLAG_REGISTRY`` flag.

    Returns:
        Its category.

    Raises:
        KeyError: When the flag is not registered.
    """
    return {spec.flag: spec.category for spec in FLAG_REGISTRY}[flag]


def study_of(platform: str) -> str:
    """The QC run of a platform.

    Args:
        platform: ``olink`` or ``soma``.

    Returns:
        ``surmount5_olink`` or ``surmount5_soma``.
    """
    return PLATFORM_STUDIES[platform]


def platform_of(study: str) -> str:
    """The platform of a QC run.

    Args:
        study: ``surmount5_olink`` or ``surmount5_soma``.

    Returns:
        ``olink`` or ``soma``.
    """
    return {s: p for p, s in PLATFORM_STUDIES.items()}[study]


def qc_stem(study: str) -> str:
    """File stem of a run's QC outputs: the run itself (``surmount5_olink``).

    Args:
        study: QC run.

    Returns:
        The stem the QC stage names its per-run files with.
    """
    platform_of(study)
    return study


def qc_study_name(study: str) -> str:
    """The name the QC run gives itself (its ``STUDY`` value): the stem in capitals.

    Args:
        study: QC run.

    Returns:
        The QC study name.
    """
    return qc_stem(study).upper()


def qc_dir(platform: str) -> Path:
    """QC output directory of a platform: ``<qc_root>/surmount5_<platform>``.

    Args:
        platform: ``olink`` or ``soma``.

    Returns:
        The directory.
    """
    return study_qc_root(study_of(platform))


def _file(platform: str, suffix: str) -> Path:
    return qc_dir(platform) / f"{qc_stem(study_of(platform))}.{suffix}"


def sample_summary_path(platform: str) -> Path:
    """Per-sample QC summary TSV of a platform."""
    return _file(platform, "sample_qc_summary.tsv")


def assay_summary_path(platform: str) -> Path:
    """Per-assay QC summary TSV of a platform."""
    return _file(platform, "assay_qc_summary.tsv")


def exclusion_table_path(platform: str) -> Path:
    """Exclusion table TSV of a platform."""
    return _file(platform, "exclusion_table.tsv")


def identity_events_path(platform: str) -> Path:
    """Identity events of a platform: one row per flagged or reassigned tube."""
    return _file(platform, "identity_events.parquet")


def dna_link_path(platform: str) -> Path:
    """Genotype link of a platform run: one row per participant with a genotype."""
    return _file(platform, "dna_link.parquet")


def long_table_path(platform: str) -> Path:
    """Long value table of a platform (Olink NPX, SomaScan non-ANML RFU)."""
    return _file(platform, PLATFORM_SPECS[platform].long_suffix)


def manifest_build_report_path(platform: str) -> Path:
    """The report the manifest build wrote for a platform run."""
    return qc_dir(platform) / "manifest_build_report.json"


def manifest_path(platform: str) -> Path:
    """The sample manifest the QC built for a platform run, bridge tubes included."""
    return qc_dir(platform) / "manifest.parquet"


def genotyped_subjects(platform: str) -> pl.Series:
    """Participants whose genotype the identity check could compare against.

    Args:
        platform: ``olink`` or ``soma``.

    Returns:
        The ``USUBJID`` of every row of the run's genotype link.
    """
    return pl.read_parquet(dna_link_path(platform), columns=["USUBJID"])["USUBJID"].unique()


def sample_schema(platform: str) -> dict[str, type[pl.DataType]]:
    """The sample-summary columns every reader may rely on, with their dtypes.

    Args:
        platform: ``olink`` or ``soma``.

    Returns:
        The platform's sample ID column first, then ``SAMPLE_REQUIRED``.
    """
    return {PLATFORM_SPECS[platform].sample_id: pl.String, **SAMPLE_REQUIRED}


def sample_summary_problems(frame: pl.DataFrame, platform: str) -> list[str]:
    """Ways a sample summary breaks the contract.

    Args:
        frame: The summary as ``read_sample_summary`` parses it.
        platform: ``olink`` or ``soma``.

    Returns:
        One message per violation; empty when the summary conforms.
    """
    required = sample_schema(platform)
    problems = [f"missing column {c}" for c in required if c not in frame.columns]
    if BRIDGE_FLAG not in frame.columns:
        problems.append(f"missing column {BRIDGE_FLAG}")
    if problems:
        return problems
    for column, dtype in required.items():
        if frame.schema[column] != dtype:
            problems.append(f"{column} is {frame.schema[column]}, expected {dtype}")
    bad_status = set(frame["Status"].unique().to_list()) - set(STATUS_LEVELS)
    if bad_status:
        problems.append(f"Status values outside {STATUS_LEVELS}: {sorted(map(str, bad_status))}")
    for column in frame.columns:
        if not column.startswith(FLAG_PREFIXES):
            continue
        if frame.schema[column] != pl.Boolean:
            problems.append(f"{column} is {frame.schema[column]}, expected Boolean")
        elif frame[column].null_count():
            problems.append(f"{column} has nulls")
    registered = {spec.flag for spec in FLAG_REGISTRY} | {BRIDGE_FLAG}
    unregistered = [c for c in frame.columns if c.startswith("FAIL_") and c not in registered]
    if unregistered:
        problems.append(f"FAIL flags missing from FLAG_REGISTRY: {unregistered}")
    scheduled = [float(v) for v in VISIT_TO_WEEK]
    on_schedule = pl.col(ANALYSIS_VISIT).is_in(scheduled).fill_null(False)
    off_schedule = frame.filter(COHORT & ~on_schedule).height
    if off_schedule:
        problems.append(
            f"{off_schedule} cohort samples have no {ANALYSIS_VISIT} among {list(VISIT_TO_WEEK)}"
        )
    return problems


def read_sample_summary(platform: str) -> pl.DataFrame:
    """A platform's sample summary, checked against the contract.

    Args:
        platform: ``olink`` or ``soma``.

    Returns:
        One row per tube, with the contract's dtypes.

    Raises:
        ValueError: When the summary breaks the contract.
    """
    path = sample_summary_path(platform)
    header = pl.read_csv(path, separator="\t", n_rows=0).columns
    frame = pl.read_csv(
        path,
        separator="\t",
        schema_overrides={c: t for c, t in sample_schema(platform).items() if c in header},
        infer_schema_length=None,
    )
    problems = sample_summary_problems(frame, platform)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return frame


def read_assay_summary(platform: str) -> pl.DataFrame:
    """A platform's assay summary, its per-arm detection rates under their table names.

    Args:
        platform: ``olink`` or ``soma``.

    Returns:
        One row per assay, with ``Detection_Rate_TZP`` and ``Detection_Rate_SEMA``.

    Raises:
        ValueError: When a required column is missing.
    """
    spec = PLATFORM_SPECS[platform]
    path = assay_summary_path(platform)
    frame = pl.read_csv(path, separator="\t", infer_schema_length=None)
    required = (spec.protein_id, spec.protein_name, *ASSAY_REQUIRED_COMMON)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return frame.rename(ARM_DETECTION_COLUMNS)


def read_exclusion_table(platform: str) -> pl.DataFrame:
    """A platform's exclusion table, checked to name the platform's own run.

    Args:
        platform: ``olink`` or ``soma``.

    Returns:
        The table.

    Raises:
        ValueError: When a required column is missing or a row names another run.
    """
    path = exclusion_table_path(platform)
    frame = pl.read_csv(path, separator="\t", infer_schema_length=None)
    missing = [c for c in EXCLUSION_REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    names = frame["STUDY"].unique().sort().to_list()
    if names != [qc_study_name(study_of(platform))]:
        raise ValueError(f"{path}: STUDY values {names} are not {study_of(platform)}'s QC name")
    return frame


def bridge_samples(platform: str) -> pl.DataFrame:
    """The bridge tubes of a platform's QC manifest.

    Args:
        platform: ``olink`` or ``soma``.

    Returns:
        ``SampleID`` of every manifest row with ``BRIDGE_COL`` set.
    """
    return (
        pl.scan_parquet(manifest_path(platform))
        .filter(pl.col(BRIDGE_COL))
        .select("SampleID")
        .unique()
        .sort("SampleID")
        .collect()
    )


def reassigned_samples(platform: str) -> pl.DataFrame:
    """The tubes a platform's identity check moved to the subject their genotype matches.

    Args:
        platform: ``olink`` or ``soma``.

    Returns:
        ``SampleID``, ``USUBJID_LABELED`` and ``USUBJID_INFERRED`` of every identity
        event whose action is ``REASSIGNED``.
    """
    return (
        pl.scan_parquet(identity_events_path(platform))
        .filter(pl.col("action") == REASSIGNED)
        .select("SampleID", "USUBJID_LABELED", "USUBJID_INFERRED")
        .sort("SampleID")
        .collect()
    )


def cohort_samples(platform: str) -> pl.LazyFrame:
    """The platform's analysis-cohort samples at the proteomics visits.

    Args:
        platform: ``olink`` or ``soma``.

    Returns:
        The platform's sample ID column, ``USUBJID``, ``VISITNUM`` (the analysis visit,
        as an integer) and the analysis week (``WEEK``) of every sample that passes
        ``COHORT`` at a visit of ``VISIT_TO_WEEK``.
    """
    sid = PLATFORM_SPECS[platform].sample_id
    return (
        pl.scan_csv(
            sample_summary_path(platform),
            separator="\t",
            schema_overrides=sample_schema(platform),
            infer_schema_length=None,
        )
        # Select the cohort at the scheduled visits before the integer cast, so a
        # fractional unscheduled visit elsewhere in the summary never reaches it.
        .filter(COHORT, pl.col(ANALYSIS_VISIT).is_in([float(v) for v in VISIT_TO_WEEK]))
        # Rows are made unique by sample ID.
        .unique(subset=sid, keep="first", maintain_order=True)
        .select(
            sid,
            "USUBJID",
            pl.col(ANALYSIS_VISIT).cast(pl.Int32).alias("VISITNUM"),
            pl.col(ANALYSIS_VISIT)
            .cast(pl.Int32)
            .replace_strict(VISIT_TO_WEEK, return_dtype=pl.Int32)
            .alias(WEEK_COL),
        )
    )
