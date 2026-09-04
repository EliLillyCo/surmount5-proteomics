"""SURMOUNT-5-specific QC workflow adapter for Olink and SomaScan platforms."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from paths import PHENO_DIR, QC_ROOT, RAW_PROTEOMICS_ROOT

from analysis.qc import OlinkQC, SomaQC

logger = logging.getLogger(__name__)

OLINK_CLINICAL_VARS = [
    "AGE",
    "SEX",
    "TRT01A",
    "WGTBL",
    "COUNTRY",
    "SITEID",
    "RACE",
]
SOMA_CLINICAL_VARS = [
    "AGE",
    "SEX",
    "TRT01A",
    "WEIGHTBL",
    "COUNTRY",
    "SITEID",
    "RACE",
]
FORCE_CATEGORICAL = ["SITEID", "VISITNUM"]


@dataclass(frozen=True)
class QCInputPaths:
    platform: str
    data_path: Path
    manifest_path: Path
    adsl_path: Path
    adlb_path: Path
    bridge_manifest_path: Path | None


def _project_root() -> Path:
    return QC_ROOT.parents[1]


def _default_bridge_manifest_path() -> Path:
    return (
        _project_root()
        / "reference_data"
        / "olink_bridge_samples"
        / "plasma.manifest.csv"
    )


def _resolve_single_path(directory: Path, patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if not matches:
            continue
        if len(matches) > 1:
            raise ValueError(
                f"Expected exactly one file under {directory} matching {pattern}, found: "
                + ", ".join(str(path.name) for path in matches)
            )
        return matches[0]
    raise FileNotFoundError(f"No files matched {patterns} under {directory}")


def resolve_input_paths(
    platform: str,
    *,
    bridge_manifest_path: str | Path | None = None,
) -> QCInputPaths:
    platform = platform.lower()
    if platform not in {"olink", "soma"}:
        raise ValueError("platform must be 'olink' or 'soma'")

    if platform == "olink":
        data_dir = RAW_PROTEOMICS_ROOT / "olink"
        data_path = _resolve_single_path(data_dir, ["*olink*.parquet"])
    else:
        data_dir = RAW_PROTEOMICS_ROOT / "somalogic"
        data_path = _resolve_single_path(data_dir, ["*qcCheck_*.adat", "*.adat"])

    manifest_path = _resolve_single_path(
        data_dir,
        [
            "surmount5.standardized_manifest.csv",
            "surmount5.standardized_manifest.csv",
            "*.standardized_manifest.csv",
        ],
    )
    bridge_path = (
        Path(bridge_manifest_path)
        if bridge_manifest_path is not None
        else _default_bridge_manifest_path()
    )
    return QCInputPaths(
        platform=platform,
        data_path=data_path,
        manifest_path=manifest_path,
        adsl_path=PHENO_DIR / "adsl.parquet",
        adlb_path=PHENO_DIR / "adlb.parquet",
        bridge_manifest_path=bridge_path if bridge_path.exists() else None,
    )


def output_prefix(
    platform: str, *, output_root: str | Path = QC_ROOT, study_id: str = "surmount5"
) -> Path:
    platform = platform.lower()
    if platform not in {"olink", "soma"}:
        raise ValueError("platform must be 'olink' or 'soma'")
    output_root = Path(output_root)
    return output_root / f"{study_id}_{platform}" / f"{study_id}_{platform}"


def _load_adsl() -> pl.LazyFrame:
    return pl.scan_parquet(PHENO_DIR / "adsl.parquet").with_columns(
        pl.lit(None).cast(pl.Date).alias("LTRESCDT"),
        pl.col("TRT01A").str.replace_all(" or MTD", ""),
    )


def _load_adlb() -> pl.LazyFrame:
    return pl.scan_parquet(PHENO_DIR / "adlb.parquet")


def _assert_manifest_columns(manifest: pl.LazyFrame, platform: str) -> None:
    schema = set(manifest.collect_schema().names())
    required = {"USUBJID", "VISITNUM", "ADT"}
    sample_id_col = "SampleID" if platform == "olink" else "SampleId"
    required.add(sample_id_col)
    missing = sorted(required - schema)
    if missing:
        raise ValueError(
            f"{platform} manifest is missing required columns for manuscript QC: {', '.join(missing)}"
        )


def _load_manifest(platform: str, manifest_path: Path) -> pl.LazyFrame:
    manifest = pl.scan_csv(manifest_path)
    if platform == "soma":
        manifest = manifest.with_columns(pl.col("SampleID").alias("SampleId"))
    _assert_manifest_columns(manifest, platform)
    return manifest


def _augment_with_bridge_samples(
    adsl: pl.LazyFrame,
    manifest: pl.LazyFrame,
    bridge_manifest_path: Path | None,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    if bridge_manifest_path is None:
        logger.warning("Bridge manifest not found; proceeding without bridge samples.")
        return adsl, manifest

    manifest_schema = set(manifest.collect_schema().names())
    if "PatientID" not in manifest_schema:
        logger.warning(
            "Manifest has no PatientID column; proceeding without bridge samples."
        )
        return adsl, manifest

    bridge_meta = (
        pl.read_csv(bridge_manifest_path)
        .select(["PatientID", "AGE", "SEX"])
        .with_columns(pl.col("PatientID").cast(pl.Utf8))
        .filter(pl.col("AGE").is_not_null() & pl.col("SEX").is_not_null())
    )
    manifest_df = manifest.collect()
    bridge_candidates = manifest_df.filter(
        pl.col("USUBJID").is_null() & pl.col("PatientID").is_not_null()
    ).with_columns(pl.col("PatientID").cast(pl.Utf8).alias("_bridge_patient_id"))

    if bridge_candidates.is_empty():
        return adsl, manifest

    bridge_adsl = (
        bridge_candidates.join(
            bridge_meta.rename({"PatientID": "_bridge_patient_id"}),
            on="_bridge_patient_id",
            how="inner",
        )
        .select(
            [
                pl.concat_str([pl.lit("BRIDGE_"), pl.col("_bridge_patient_id")]).alias(
                    "USUBJID"
                ),
                pl.col("AGE").cast(pl.Float64),
                pl.col("SEX"),
            ]
        )
        .unique(subset=["USUBJID"])
    )

    manifest_with_bridge = manifest_df.with_columns(
        pl.when(pl.col("USUBJID").is_null() & pl.col("PatientID").is_not_null())
        .then(pl.concat_str([pl.lit("BRIDGE_"), pl.col("PatientID").cast(pl.Utf8)]))
        .otherwise(pl.col("USUBJID"))
        .alias("USUBJID")
    ).lazy()

    if bridge_adsl.is_empty():
        logger.warning(
            "No bridge samples matched the bridge manifest; proceeding without bridge ADSL rows."
        )
        return adsl, manifest_with_bridge

    logger.info("Added %s bridge samples to ADSL", bridge_adsl.height)
    return pl.concat([adsl, bridge_adsl.lazy()], how="diagonal"), manifest_with_bridge


def build_olink_qc(
    *,
    save_prefix: str | Path,
    study_label: str = "SURMOUNT-5",
    random_seed: int = 0,
    bridge_manifest_path: str | Path | None = None,
) -> tuple[OlinkQC, pl.LazyFrame, QCInputPaths]:
    paths = resolve_input_paths("olink", bridge_manifest_path=bridge_manifest_path)
    adsl, manifest = _augment_with_bridge_samples(
        _load_adsl(),
        _load_manifest("olink", paths.manifest_path),
        paths.bridge_manifest_path,
    )
    adlb = _load_adlb()
    qc = OlinkQC(
        npx_path=paths.data_path,
        data_col="PCNormalizedNPX",
        project_name=study_label,
        save_prefix=str(save_prefix),
        adsl=adsl,
        manifest=manifest,
        usubjid_col="USUBJID",
        clinical_vars=OLINK_CLINICAL_VARS,
        force_categorical=FORCE_CATEGORICAL,
        group_col=["TRT01A"],
        random_seed=random_seed,
    )
    return qc, adlb, paths


def build_soma_qc(
    *,
    save_prefix: str | Path,
    study_label: str = "SURMOUNT-5",
    random_seed: int = 0,
    bridge_manifest_path: str | Path | None = None,
) -> tuple[SomaQC, pl.LazyFrame, QCInputPaths]:
    paths = resolve_input_paths("soma", bridge_manifest_path=bridge_manifest_path)
    adsl, manifest = _augment_with_bridge_samples(
        _load_adsl(),
        _load_manifest("soma", paths.manifest_path),
        paths.bridge_manifest_path,
    )
    adlb = _load_adlb()
    qc = SomaQC(
        adat_path=paths.data_path,
        project_name=study_label,
        save_prefix=str(save_prefix),
        adsl=adsl,
        manifest=manifest,
        usubjid_col="USUBJID",
        clinical_vars=SOMA_CLINICAL_VARS,
        force_categorical=FORCE_CATEGORICAL,
        group_col=["TRT01A"],
        random_seed=random_seed,
    )
    return qc, adlb, paths


def run_olink_qc(
    *,
    save_prefix: str | Path,
    study_label: str = "SURMOUNT-5",
    random_seed: int = 0,
    bridge_manifest_path: str | Path | None = None,
) -> tuple[Path, QCInputPaths]:
    qc, adlb, paths = build_olink_qc(
        save_prefix=save_prefix,
        study_label=study_label,
        random_seed=random_seed,
        bridge_manifest_path=bridge_manifest_path,
    )
    qc.remove_sample_qc()
    qc.filter_assay_qc()
    qc.filter_assay_cv(intra_threshold=25.0)
    qc.filter_assay_detection(sample_threshold=0.2)
    qc.remove_sample_outliers()
    qc.remove_sample_sex_concordance()
    qc.remove_sample_age_concordance(
        age_col="VISITAGE",
        age_threshold=15,
        std_threshold=5.5,
    )
    qc.remove_sample_unsuitable()
    qc.check_assay_biomarker_correlation(lab_data=adlb)
    qc.save_results()
    return Path(f"{save_prefix}.npx_long.parquet"), paths


def run_soma_qc(
    *,
    save_prefix: str | Path,
    study_label: str = "SURMOUNT-5",
    random_seed: int = 0,
    bridge_manifest_path: str | Path | None = None,
) -> tuple[Path, QCInputPaths]:
    qc, adlb, paths = build_soma_qc(
        save_prefix=save_prefix,
        study_label=study_label,
        random_seed=random_seed,
        bridge_manifest_path=bridge_manifest_path,
    )
    qc.remove_sample_qc()
    qc.filter_assay_qc()
    qc.filter_assay_cv(intra_threshold=10.0)
    qc.filter_assay_detection(sample_threshold=0.9)
    qc.filter_assay_saturation(rfu_threshold=80_000, sample_frac=0.50)
    qc.filter_assay_sb_ratio(sb_threshold=2.0)
    qc.finalize_assay_flags()
    qc.remove_sample_outliers()
    qc.remove_sample_sex_concordance()
    qc.remove_sample_age_concordance(
        age_col="VISITAGE",
        age_threshold=15,
        std_threshold=5.5,
    )
    qc.remove_sample_unsuitable()
    qc.check_assay_biomarker_correlation(lab_data=adlb)
    qc.save_results()
    return Path(f"{save_prefix}.rfu_long.parquet"), paths
