"""Runs the QC pipeline on one trial's Olink NPX or SomaScan ADAT data.

Runs sample and assay QC, the sex and age concordance models, the pQTL and identity
checks and the clinical-data filters, then writes the QC outputs into ``--output-dir``
(published in one rename, so a failed run leaves the previous outputs). A trial's
config (``--config``) sets the thresholds and the manifest source.

    python -m analysis.qc.driver --platform olink|soma --proteomics-data <npx|adat> \
        --config <toml>

``analysis.qc.study`` builds these arguments for the two platform runs from
``analysis/qc/config_{olink,soma}.toml`` and the paths config.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import polars as pl

from analysis.qc.age_concordance import PROTAGE_TABLE_PATH
from analysis.qc.config import TrialConfig, load_config
from analysis.qc.core import OlinkQC, SomaQC, apply_reassignments
from analysis.qc.io import scan_file
from analysis.qc.manifest import (
    MANIFEST_SCHEMA,
    OPTIONAL_MANIFEST_COLUMNS,
    build_manifest,
    visit_records,
)
from analysis.qc.platform import get_platform
from analysis.qc.sex_concordance import SEX_TABLE_PATH
from analysis.qc.utils import publish_output_dir, stage_output_dir

logger = logging.getLogger(__name__)

DEFAULT_STUDY_NAME = "my_study"

# The subject column is an ADaM name that is never renamed; the whole pipeline assumes
# it.
USUBJID_COL = "USUBJID"

# Columns of a build_manifest frame, defined once in manifest.py.
BUILT_MANIFEST_COLUMNS = tuple(MANIFEST_SCHEMA)


def load_bridge_manifest(path, manifest_df: pl.DataFrame) -> pl.DataFrame | None:
    """Loads the bridge manifest's PatientID rows that have AGE and SEX.

    A path is read, and a missing file raises ``FileNotFoundError``. With no path the
    run proceeds only when every null-USUBJID row of ``manifest_df`` has ``BridgeSample``
    true; those rows fail as bridges (``FAIL_Bridge_Sample``).

    Args:
        path: The bridge manifest file, or ``None``.
        manifest_df: The manifest, checked for null-USUBJID rows when there is no path.

    Returns:
        The bridge rows, or ``None`` without a path.

    Raises:
        ValueError: If there is no path and a null-USUBJID row is not a bridge tube.
    """
    if not path:
        flag = pl.col("BridgeSample")
        if manifest_df.schema["BridgeSample"] == pl.String:
            flag = flag.str.strip_chars().str.to_lowercase().is_in(["y", "true"])
        unmapped = manifest_df.filter(pl.col(USUBJID_COL).is_null())
        not_bridge = unmapped.filter(~flag.fill_null(False))
        if not_bridge.height:
            raise ValueError(
                f"{not_bridge.height} manifest rows have a null {USUBJID_COL} and are not a bridge "
                "tube (BridgeSample is not true), and no --bridge-manifest was given"
            )
        logger.info(
            f"No bridge manifest given: {unmapped.height} bridge samples (null {USUBJID_COL}, "
            "BridgeSample true) stay without a subject, AGE or SEX"
        )
        return None
    logger.info(f"Loading bridge manifest from {path}...")
    return (
        scan_file(path)
        .select(["PatientID", "AGE", "SEX"])
        .collect()
        .filter(pl.col("AGE").is_not_null() & pl.col("SEX").is_not_null())
    )


def normalize_sample_id_column(df, platform: str = "olink"):
    """Renames the sample and plate ID columns to the names the platform's QC reads.

    ``OlinkQC`` reads ``SampleID`` and ``PlateID``; ``SomaQC`` reads ``SampleId`` and
    ``PlateId``.
    """
    columns = df.collect_schema().names() if hasattr(df, "collect_schema") else df.columns
    renames = {}
    for olink_col, soma_col in (("SampleID", "SampleId"), ("PlateID", "PlateId")):
        expected_col = olink_col if platform == "olink" else soma_col
        other_col = soma_col if platform == "olink" else olink_col
        if other_col in columns and expected_col not in columns:
            logger.info(f"Renaming '{other_col}' to '{expected_col}'")
            renames[other_col] = expected_col
    return df.rename(renames) if renames else df


def check_manifest_options(options, config: TrialConfig) -> None:
    """Checks the manifest and ADSL sources against the trial config.

    With ``[manifest]`` QC builds the manifest from the raw vendor key, so
    ``--mapping-manifest`` is rejected, ``--bridge-manifest`` too (a built manifest has
    no PatientID to join it on), and ``--adsl`` may be left out (it defaults to
    ``[manifest] adsl``) or must name the same file. ``--prebuilt-manifest`` (a parquet
    in the build's schema) replaces the build and is accepted only with
    ``[manifest]``: the ADaM visit records and the screening dates still come from
    ``[manifest]``. Without ``[manifest]``, ``--mapping-manifest`` and ``--adsl`` are
    required. The clinical step needs ``[visits]`` either way.

    Raises:
        SystemExit: If the options conflict with the config.
    """
    if config.visits_baseline is None:
        raise SystemExit("error: --config must give [visits] baseline and analysis")
    if config.manifest is None:
        if options.prebuilt_manifest_path:
            raise SystemExit(
                "error: --prebuilt-manifest needs a config with [manifest]: it replaces the build "
                "but the ADaM visit records still come from [manifest]"
            )
        if not options.manifest_path:
            raise SystemExit(
                "error: --mapping-manifest is required unless the config has [manifest]"
            )
        if not options.adsl_path:
            raise SystemExit("error: --adsl is required unless the config has [manifest]")
        return
    if options.manifest_path:
        raise SystemExit(
            "error: --mapping-manifest cannot be used when the config has [manifest]: "
            "QC builds the manifest from the raw vendor key"
        )
    if options.bridge_manifest_path:
        raise SystemExit("error: --bridge-manifest cannot be used when the config has [manifest]")
    if options.adsl_path and os.path.realpath(options.adsl_path) != os.path.realpath(
        config.manifest.adsl
    ):
        raise SystemExit(
            f"error: --adsl {options.adsl_path} differs from [manifest] adsl {config.manifest.adsl}"
        )
    if options.adlb_path and os.path.realpath(options.adlb_path) != os.path.realpath(
        config.manifest.adlb
    ):
        raise SystemExit(
            f"error: --adlb {options.adlb_path} differs from [manifest] adlb {config.manifest.adlb}"
        )


def resolve_adlb_path(options, config: TrialConfig) -> str | None:
    """Returns the trial's ADLB: ``[manifest] adlb`` when the config has it, else ``--adlb``."""
    return config.manifest.adlb if config.manifest is not None else options.adlb_path


def resolve_output_dir(options, config: TrialConfig) -> None:
    """Sets ``options.output_dir`` to ``{platform}_qc_results`` when it is absent."""
    if options.output_dir is None:
        options.output_dir = f"{get_platform(config.platform).name}_qc_results"


def resolve_study_name(options, config: TrialConfig) -> None:
    """Sets ``options.study_name`` to the config's study_name.

    ``load_config`` fills the config's study_name from ``--study-name`` (or the default)
    when the TOML leaves it unset, so one name reaches the STUDY column and the identity
    events.

    Raises:
        SystemExit: If ``--study-name`` contradicts the config's.
    """
    if options.study_name and options.study_name != config.study_name:
        raise SystemExit(
            f"error: --study-name {options.study_name} differs from the config's "
            f"study_name {config.study_name}"
        )
    options.study_name = config.study_name


def qc_manifest_from_build(
    manifest: pl.DataFrame, config: TrialConfig | None = None
) -> pl.LazyFrame:
    """Returns the QC manifest from a ``build_manifest`` frame.

    The optional columns (``VOLUME_NOTE``) are typed when present. With ``config``, a
    config that names a volume-note column needs the column in the manifest.

    Raises:
        ValueError: If a contract column is missing or mistyped.
    """
    missing = [c for c in BUILT_MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"Built manifest lacks {', '.join(missing)}")
    if (
        config is not None
        and config.manifest is not None
        and config.manifest.columns.volume_note
        and "VOLUME_NOTE" not in manifest.columns
    ):
        raise ValueError(
            "The config's [manifest.columns] volume_note needs a VOLUME_NOTE column in the built "
            "manifest, which this one lacks: rebuild it with the current config"
        )
    mistyped = [
        f"{c} is {manifest.schema[c]}, expected {t}"
        for c, t in {**MANIFEST_SCHEMA, **OPTIONAL_MANIFEST_COLUMNS}.items()
        if c in manifest.columns and manifest.schema[c] != t
    ]
    if mistyped:
        raise ValueError(f"Built manifest column types: {'; '.join(mistyped)}")
    return manifest.lazy()


def trial_screening_dates(options, config: TrialConfig):
    """Returns the last screening date per subject for the pre-dose baseline exception.

    The dates come from the trial's ADLB (``resolve_adlb_path``) plus
    ``config.advs_path`` when set; the latest date from
    either file wins. ``None`` when there is no ADLB.
    """
    from analysis.qc.clinical import screening_dates

    adlb = resolve_adlb_path(options, config)
    if not adlb:
        logger.warning(
            "No ADLB or ADVS: the pre-dose baseline exception cannot check screening dates"
        )
        return None
    paths = [adlb] + ([config.advs_path] if config.advs_path else [])
    return screening_dates([scan_file(p) for p in paths], config.visits_screening)


def identity_visit_records(options, config: TrialConfig):
    """Returns the dated ADaM visit records the identity step places moved tubes against.

    They come from the tube-record sources the manifest build reads: the ``[manifest]``
    LB and ADLB when ``lb`` is set, else ADLB and ADIG, plus ADVS for the (subject,
    visit) pairs they lack; with no ``[manifest]``, ``--adlb`` alone. ``None`` when
    identity is off.

    Raises:
        ValueError: If identity is on and there is no ADLB.
    """
    if not config.identity_enabled:
        return None
    if config.manifest is not None:
        m = config.manifest
        records = visit_records(
            scan_file(m.adlb),
            scan_file(m.advs),
            scan_file(m.adig) if m.adig and not m.lb else None,
            scan_file(m.lb) if m.lb else None,
        )
    elif options.adlb_path:
        records = visit_records(scan_file(options.adlb_path), None)
    else:
        raise ValueError(
            "identity needs dated visit records: give --adlb, or [manifest] adlb in the config"
        )
    return records.collect()


def _median_visits_per_subject(qc) -> float:
    """Returns the median number of distinct samples per USUBJID."""
    sid = qc.sample_id_col
    adsl_merged = qc.adsl_merged
    frame = adsl_merged.collect() if hasattr(adsl_merged, "collect") else adsl_merged
    return (
        frame.group_by("USUBJID")
        .agg(pl.col(sid).n_unique().alias("n"))
        .select(pl.col("n").median())
        .item()
    )


def _run_pqtl_concordance(qc, config: TrialConfig) -> None:
    """Scores genotype-proteome concordance and keeps it on ``qc``.

    ``qc.pqtl_scores`` (per sample) and ``qc.pqtl_evidence`` (score matrix and
    genotypes) feed ``analysis.qc.identity.identity_check``.
    """
    from analysis.qc.pqtl_reference import pqtl_concordance_for_qc

    qc.pqtl_scores, qc.pqtl_evidence = pqtl_concordance_for_qc(qc, config)
    logger.info(pqtl_score_summary(qc.pqtl_scores))


def _fmt_mean(values: pl.Series) -> str:
    """Formats the mean to two decimals, or "n/a" for an empty or all-null series."""
    mean = values.mean()
    return "n/a" if mean is None else f"{mean:.2f}"


def pqtl_score_summary(scores: pl.DataFrame) -> str:
    """Builds one log line of sample counts, calls and mean ``pqtl_self_z``.

    Calls (match, mismatch, none) are split by label kind. A sample whose labeled
    subject is genotyped has a ``pqtl_self_z`` (it is scored); one with an ungenotyped
    label has none and can still be called ``mismatch`` from its best-other match alone.
    """
    has_self = pl.col("pqtl_self_z").is_not_null()
    scored = scores.filter(has_self)
    unscored = scores.filter(~has_self)
    clean = scored.filter(pl.col("pqtl_call") != "mismatch")

    def calls(frame: pl.DataFrame) -> str:
        n = {
            c: frame.filter(pl.col("pqtl_call") == c).height for c in ("match", "mismatch", "none")
        }
        return f"{n['match']} match, {n['mismatch']} mismatch, {n['none']} none"

    return (
        f"pQTL concordance: {scores.height} samples; genotyped label {scored.height} "
        f"({calls(scored)}); ungenotyped label {unscored.height} ({calls(unscored)}); "
        f"mean pqtl_self_z {_fmt_mean(scored['pqtl_self_z'])} ({_fmt_mean(clean['pqtl_self_z'])} "
        f"over {clean.height} samples without a mismatch call)"
    )


def share_identity_matrix(qc, config: TrialConfig) -> None:
    """Builds the identity matrix once for the pQTL stage and the identity stage.

    Parks it on ``qc.shared_identity_matrix`` (``analysis.qc.identity.matrix_for``
    hands it to both stages) when both run, so the matrix and its residuals are computed
    once; ``None`` otherwise.
    """
    from analysis.qc.identity import identity_matrix

    qc.shared_identity_matrix = (
        identity_matrix(qc) if config.pqtl_enabled and config.identity_enabled else None
    )


def run_identity_step(qc, config: TrialConfig) -> None:
    """Runs the identity check, applies its reassignments and records its flags.

    The identity FAIL flags enter ``failed_samples``, so the clinical step that follows
    treats those tubes as absent and cascades their subjects; the WARN flags change
    Status only. Runs before ``remove_sample_unsuitable``.
    """
    from analysis.qc.identity import identity_check, identity_flags

    events = identity_check(qc, config)
    if events.height == 0:
        logger.info("Identity check: no events detected")
        return
    # Reassigned samples take the inferred subject's covariates, group and re-evaluated
    # sex/age and arm flags, and keep their own draw date.
    apply_reassignments(qc, events)
    flags = identity_flags(events)
    for flag, ids in flags.items():
        qc._flag_samples(ids, flag)

    type_counts = events.group_by("event_type").len().sort("event_type")
    action_counts = events.group_by("action").len().sort("action")
    logger.info(
        f"Identity check: {events.height} events by type ("
        + ", ".join(f"{n} {etype}" for etype, n in type_counts.iter_rows())
        + "); by action ("
        + ", ".join(f"{n} {action}" for action, n in action_counts.iter_rows())
        + "); by flag ("
        + ", ".join(f"{len(ids)} {flag}" for flag, ids in flags.items())
        + ")"
    )


def run_identity_and_concordance(
    qc,
    *,
    config: TrialConfig,
    sex_table_path,
    sex_col,
    sex_c,
    age_col,
    age_threshold,
    age_std_threshold,
    protage_table_path,
    visit_records=None,
):
    """Runs the sex and age models, pQTL concordance and the identity step.

    The order is sex model, age model, pQTL, identity check, reassignment. Sex and age
    produce predictions (WARN only); identity is the arbiter. Single-visit samples get
    no fallback: the clinical longitudinal filters remove them downstream.
    """
    sex_result = qc.remove_sample_sex_concordance(
        sex_table_path=sex_table_path, sex_col=sex_col, C=sex_c
    )
    if "error" in sex_result:
        raise ValueError(f"sex concordance model failed: {sex_result['error']}")

    # The age model needs repeat visits.
    median_visits = _median_visits_per_subject(qc)
    if median_visits >= 2:
        age_result = qc.remove_sample_age_concordance(
            age_col=age_col,
            age_threshold=age_threshold,
            std_threshold=age_std_threshold,
            protage_table_path=protage_table_path,
        )
        if "error" in age_result:
            raise ValueError(f"age concordance model failed: {age_result['error']}")
    else:
        logger.info("Single-visit study: skipping age concordance.")

    # pQTL concordance runs before the identity check. load_config guarantees the
    # catalog and genotype dir when it is enabled.
    share_identity_matrix(qc, config)
    if config.pqtl_enabled:
        _run_pqtl_concordance(qc, config)

    if config.identity_enabled:
        # The placement guard reads the visit records.
        qc.visit_records = visit_records
        run_identity_step(qc, config)
        # The reassignments that follow change what the matrix was built from.
        qc.shared_identity_matrix = None
    else:
        logger.info("Identity check disabled by config.")


def _option(options, name: str, config_value):
    """Returns the command-line value of ``name`` when given, else the config's."""
    value = getattr(options, name)
    return config_value if value is None else value


def build_qc_steps(qc, options):
    """Returns the QC pipeline as ``(step_name, callable)`` pairs in run order.

    The steps are those of the run's platform: Olink adds the LOD before detection and
    the duplicate-assay check; SomaScan adds saturation, signal-to-background and the
    assay-flag consolidation after detection.
    """
    config = options.trial_config
    olink = options.platform == "olink"
    steps = [
        (
            "remove_sample_qc",
            (lambda: qc.remove_sample_qc(missing_threshold=options.missing_threshold))
            if olink
            else (lambda: qc.remove_sample_qc()),
        ),
        (
            "filter_assay_qc",
            (lambda: qc.filter_assay_qc(warn_threshold=options.assay_warn_threshold))
            if olink
            else (lambda: qc.filter_assay_qc()),
        ),
        (
            "filter_assay_cv",
            lambda: qc.filter_assay_cv(
                intra_threshold=_option(options, "intra_cv_threshold", config.intra_cv),
                inter_threshold=options.inter_cv_threshold,
            ),
        ),
    ]
    if olink:
        # olink_lod runs after filter_assay_cv and before filter_assay_detection.
        steps.append(
            (
                "olink_lod",
                lambda: qc.olink_lod(
                    lod_method=options.lod_method, lod_file_path=options.lod_file_path
                ),
            )
        )
    steps.append(
        (
            "filter_assay_detection",
            lambda: qc.filter_assay_detection(
                sample_threshold=_option(options, "detection_threshold", config.detection_rate)
            ),
        )
    )
    if olink:
        steps.append(
            (
                "filter_assay_duplicates",
                lambda: qc.filter_assay_duplicates(corr_threshold=options.duplicate_corr_threshold),
            )
        )
    else:
        steps += [
            (
                "filter_assay_saturation",
                lambda: qc.filter_assay_saturation(
                    rfu_threshold=config.saturation_rfu, sample_frac=config.saturation_frac
                ),
            ),
            (
                "filter_assay_sb_ratio",
                lambda: qc.filter_assay_sb_ratio(sb_threshold=config.sb_ratio),
            ),
            ("finalize_assay_flags", lambda: qc.finalize_assay_flags()),
        ]
    steps += [
        (
            "remove_sample_outliers",
            lambda: qc.remove_sample_outliers(
                n_components=_option(options, "outlier_n_components", config.qc_outlier_n_pcs),
                n_std=_option(options, "outlier_n_std", config.qc_outlier_n_std),
            ),
        ),
        (
            "identity_and_concordance",
            lambda: run_identity_and_concordance(
                qc,
                config=config,
                sex_table_path=options.sex_table_path,
                sex_col=options.sex_col,
                sex_c=_option(options, "sex_concordance_c", config.sex_c),
                age_col=options.age_col,
                age_threshold=options.age_threshold,
                age_std_threshold=options.age_std_threshold,
                protage_table_path=options.protage_table_path,
                visit_records=identity_visit_records(options, config),
            ),
        ),
        (
            "remove_sample_unsuitable",
            lambda: qc.remove_sample_unsuitable(
                sample_date_col=options.sample_date_col,
                screening_dates=trial_screening_dates(options, config),
            ),
        ),
        (
            "exclusion_summary",
            lambda: qc.exclusion_summary(
                sample_date_col=options.sample_date_col,
                screening_dates=trial_screening_dates(options, config),
            ),
        ),
    ]
    return steps


def _comma_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",")]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses and validates the command line.

    The comma-separated options (``--clinical-vars``, ``--force-categorical``,
    ``--group-col``) come back as lists.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--platform", choices=["olink", "soma"], required=True)
    ap.add_argument(
        "--proteomics-data",
        dest="proteomics_data_path",
        required=True,
        help="Olink NPX file or SomaScan ADAT",
    )
    ap.add_argument(
        "--mapping-manifest",
        dest="manifest_path",
        help="Mapping manifest; required unless the --config TOML has [manifest], and "
        "rejected if it does",
    )
    ap.add_argument(
        "--prebuilt-manifest",
        dest="prebuilt_manifest_path",
        help="A built manifest (parquet in manifest.MANIFEST_SCHEMA) used instead of "
        "building one; only with a --config that has [manifest]",
    )
    ap.add_argument("--adsl", dest="adsl_path", help="ADSL; defaults to [manifest] adsl")
    ap.add_argument("--adlb", dest="adlb_path", help="ADLB; defaults to [manifest] adlb")
    ap.add_argument("--bridge-manifest", dest="bridge_manifest_path")
    ap.add_argument("--protage-table", dest="protage_table_path", default=str(PROTAGE_TABLE_PATH))
    ap.add_argument("--sex-concordance-table", dest="sex_table_path", default=str(SEX_TABLE_PATH))
    ap.add_argument(
        "--output-dir", dest="output_dir", help="Output directory [default: <platform>_qc_results]"
    )
    ap.add_argument(
        "--output-stem",
        dest="output_stem",
        help="Prefix of the output files [default: the study name]",
    )
    ap.add_argument(
        "--study-name",
        dest="study_name",
        help=f"Study name; defaults to the --config study_name, else {DEFAULT_STUDY_NAME}",
    )
    ap.add_argument(
        "--clinical-vars",
        dest="clinical_vars",
        type=_comma_list,
        default=_comma_list("AGE,SEX,TRT01A,COUNTRY,SITEID,RACE"),
    )
    ap.add_argument(
        "--force-categorical",
        dest="force_categorical",
        type=_comma_list,
        default=_comma_list("SITEID,VISITNUM"),
    )
    ap.add_argument("--group-col", dest="group_col", type=_comma_list, default=["TRT01A"])
    ap.add_argument("--missing-threshold", dest="missing_threshold", type=float, default=0.2)
    ap.add_argument("--assay-warn-threshold", dest="assay_warn_threshold", type=float, default=0.1)
    ap.add_argument("--lod-method", dest="lod_method", default="NCLOD")
    ap.add_argument("--lod-file", dest="lod_file_path")
    ap.add_argument("--intra-cv-threshold", dest="intra_cv_threshold", type=float)
    ap.add_argument("--inter-cv-threshold", dest="inter_cv_threshold", type=float, default=25.0)
    ap.add_argument("--detection-threshold", dest="detection_threshold", type=float)
    ap.add_argument(
        "--duplicate-corr-threshold", dest="duplicate_corr_threshold", type=float, default=0.8
    )
    ap.add_argument("--outlier-n-components", dest="outlier_n_components", type=int)
    ap.add_argument("--outlier-n-std", dest="outlier_n_std", type=float)
    ap.add_argument("--sex-col", dest="sex_col", default="SEX")
    ap.add_argument("--sex-concordance-c", dest="sex_concordance_c", type=float)
    ap.add_argument("--age-col", dest="age_col", default="VISITAGE")
    ap.add_argument("--age-threshold", dest="age_threshold", type=float, default=15.0)
    ap.add_argument("--age-std-threshold", dest="age_std_threshold", type=float, default=5.5)
    ap.add_argument("--sample-date-col", dest="sample_date_col", default="ADT")
    ap.add_argument(
        "--config",
        dest="config_path",
        help="Trial TOML; it must give [visits] baseline and analysis",
    )
    ap.add_argument(
        "--data-col",
        dest="data_col",
        help="QC data column; overrides the config's data_col",
    )
    ap.add_argument("--random-seed", dest="random_seed", type=int, default=0)
    options = ap.parse_args(argv)
    if options.lod_method == "FixedLOD" and not options.lod_file_path:
        ap.error("--lod-file is required with --lod-method FixedLOD")
    return options


def _load_manifest(options, config: TrialConfig, work_dir: str) -> pl.LazyFrame:
    """Returns the QC manifest: prebuilt, built from ``[manifest]``, or a mapping file."""
    if config.manifest is not None and options.prebuilt_manifest_path:
        logger.info(f"Loading the prebuilt manifest from {options.prebuilt_manifest_path}...")
        return qc_manifest_from_build(pl.read_parquet(options.prebuilt_manifest_path), config)
    if config.manifest is not None:
        logger.info("Building the manifest from the config's [manifest]...")
        return qc_manifest_from_build(build_manifest(config, Path(work_dir))[0], config)
    logger.info(f"Loading manifest from {options.manifest_path}...")
    manifest = scan_file(options.manifest_path)
    # Numeric-looking SampleID/PatientID values infer as integers when every row is
    # numeric, which then fails to join the proteomics data's string SampleID.
    manifest_id_cols = [
        c for c in manifest.collect_schema().names() if c in ("SampleID", "SampleId", "PatientID")
    ]
    if manifest_id_cols:
        manifest = manifest.with_columns([pl.col(c).cast(pl.Utf8) for c in manifest_id_cols])
    return manifest


def _attach_bridge_subjects(
    adsl: pl.LazyFrame, manifest_df: pl.DataFrame, bridge_manifest: pl.DataFrame
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Adds ``BRIDGE_<PatientID>`` ADSL rows and manifest subjects for the bridge tubes.

    Raises:
        ValueError: If the bridge manifest gives one donor two AGE or SEX values.
    """
    bridge_adsl = (
        manifest_df.filter(pl.col(USUBJID_COL).is_null())
        .join(bridge_manifest, on=pl.col("PatientID").cast(pl.Utf8), how="inner")
        .select(
            [
                pl.concat_str([pl.lit("BRIDGE_"), pl.col("PatientID").cast(pl.Utf8)]).alias(
                    USUBJID_COL
                ),
                pl.col("AGE").cast(pl.Float64),
                pl.col("SEX"),
            ]
        )
    )
    conflicting = (
        bridge_adsl.group_by(USUBJID_COL)
        .agg(pl.col("AGE").n_unique(), pl.col("SEX").n_unique())
        .filter((pl.col("AGE") > 1) | (pl.col("SEX") > 1))[USUBJID_COL]
        .sort()
        .to_list()
    )
    if conflicting:
        raise ValueError(
            "bridge manifest has conflicting AGE or SEX rows for " + ", ".join(conflicting)
        )
    bridge_adsl = bridge_adsl.unique(subset=[USUBJID_COL], maintain_order=True)
    manifest = manifest_df.with_columns(
        pl.when(pl.col(USUBJID_COL).is_null())
        .then(pl.concat_str([pl.lit("BRIDGE_"), pl.col("PatientID").cast(pl.Utf8)]))
        .otherwise(pl.col(USUBJID_COL))
        .alias(USUBJID_COL)
    ).lazy()
    logger.info(f"Added {bridge_adsl.height} bridge samples to ADSL")
    return pl.concat([adsl, bridge_adsl.lazy()], how="diagonal"), manifest


def run(options: argparse.Namespace, config: TrialConfig | None = None) -> None:
    """Runs the QC pipeline for parsed ``options`` and publishes its outputs.

    Loads the config and inputs, runs the QC steps and writes the results into a
    staging directory beside ``--output-dir`` that is renamed into place at the end, so
    a failed run leaves the previous outputs and a rerun leaves no stale file.

    Args:
        options: The parsed command line (``parse_args``).
        config: The trial config, built by the caller; None loads ``--config``.
    """
    if config is None:
        config = load_config(
            options.config_path,
            platform=options.platform,
            study_name=options.study_name or DEFAULT_STUDY_NAME,
        )
    options.trial_config = config
    if config.pqtl_enabled:
        # Fail before the QC run: enabled pQTL never skips.
        from analysis.qc.genotype_check import check_genotype_dir
        from analysis.qc.pqtl_reference import require_pqtl_inputs

        require_pqtl_inputs(config)
        check_genotype_dir(
            config.pqtl_genotype_dir, config.pqtl_expected_chromosomes, config.pqtl_pgen_pattern
        )
    resolve_study_name(options, config)
    resolve_output_dir(options, config)
    check_manifest_options(options, config)
    stem = options.output_stem or options.study_name
    adsl_path = options.adsl_path or config.manifest.adsl
    # The data column is the command-line flag, else the config's, else the platform's.
    data_col = options.data_col or config.data_col

    staging = stage_output_dir(options.output_dir, stem, get_platform(config.platform).name)
    try:
        work_dir = str(staging)
        manifest = _load_manifest(options, config, work_dir)
        logger.info(f"Loading ADSL from {adsl_path}...")
        adsl = scan_file(adsl_path).with_columns(pl.lit(options.study_name).alias("STUDY"))
        manifest_df = manifest.with_columns(pl.lit(options.study_name).alias("STUDY")).collect()
        manifest = manifest_df.lazy()

        # Bridge samples in a mapping manifest are the null-USUBJID rows; a built
        # manifest flags them itself.
        if config.manifest is None and manifest_df.filter(pl.col(USUBJID_COL).is_null()).height:
            logger.info("Processing bridge samples in manifest...")
            bridge_manifest = load_bridge_manifest(options.bridge_manifest_path, manifest_df)
            if bridge_manifest is not None:
                adsl, manifest = _attach_bridge_subjects(adsl, manifest_df, bridge_manifest)

        qc_class = OlinkQC if options.platform == "olink" else SomaQC
        qc = qc_class(
            options.proteomics_data_path,
            project_name=options.study_name,
            save_prefix=f"{work_dir}/{stem}",
            adsl=adsl,
            manifest=normalize_sample_id_column(manifest, options.platform),
            usubjid_col=USUBJID_COL,
            clinical_vars=options.clinical_vars,
            force_categorical=options.force_categorical,
            group_col=options.group_col,
            random_seed=options.random_seed,
            data_col=data_col,
            config=config,
        )
        for step_name, step_func in build_qc_steps(qc, options):
            logger.info(f"Running {step_name}...")
            step_func()

        logger.info("Saving results...")
        qc.save_results()
        qc.close()
        publish_output_dir(staging, options.output_dir, stem)
        staging = None
        logger.info(f"QC pipeline completed. Outputs in {options.output_dir}")
    finally:
        if staging is not None:
            # The run failed: its partial outputs never reach --output-dir.
            shutil.rmtree(staging)


def main(argv: list[str] | None = None) -> int:
    """Runs the QC pipeline from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
