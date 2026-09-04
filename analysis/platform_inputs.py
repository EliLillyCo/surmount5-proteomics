"""SURMOUNT-5 platform-specific analysis frames built from mounted QC outputs."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from paths import PHENO_DIR, QC_OLINK_DIR, QC_SOMA_DIR, first_existing

BASELINE_VISITNUM = 2
VISIT_LABELS: dict[int, int] = {8: 24, 20: 72}
VISITNUMS: tuple[int, ...] = (BASELINE_VISITNUM, *VISIT_LABELS.keys())
POST_BASELINE_VISITS: tuple[int, ...] = tuple(
    visit for visit in VISITNUMS if visit != BASELINE_VISITNUM
)

RAW_CONTROL_VALUE = "SEMA 2.4mg or MTD"
RAW_TREAT_VALUE = "TZP 15mg or MTD"
CONTROL_VALUE = "SEMA2.4mgorMTD"
TREAT_VALUE = "TZP15mgorMTD"

MMRM_COVARIATE_CONFIG: dict[str, list[str] | str] = {
    "cat_covariates": ["SEX"],
    "cont_covariates": ["AGE"],
    "covar_label": "AGE_SEX",
}

MEDIATION_COVARIATE_CONFIG: dict[str, list[str] | str] = {
    "cat_covariates": ["SEX"],
    "cont_covariates": ["AGE", "WGTBL"],
    "covar_label": "AGE_SEX",
}


@dataclass(frozen=True)
class PreparedPlatformInputs:
    platform: str
    data: pl.LazyFrame
    annotation: pl.DataFrame
    response_var: str
    response_var_bl: str
    protein_id_col: str


def _load_adsl() -> pl.LazyFrame:
    return pl.scan_parquet(PHENO_DIR / "adsl.parquet").with_columns(
        pl.col("WEIGHTBL").alias("WGTBL")
    )


def _load_advs() -> pl.LazyFrame:
    return pl.scan_parquet(PHENO_DIR / "advs.parquet")


def _load_sample_map(path, sample_col: str) -> pl.LazyFrame:
    return (
        pl.scan_csv(path, separator="\t", schema_overrides={"VISITNUM": pl.Float64})
        .with_columns(pl.col("VISITNUM").cast(pl.Int32, strict=False))
        .filter(pl.col("VISITNUM").is_in(VISITNUMS))
        .rename({sample_col: "sample_id"})
    )


def _qc_file(root: str, name: str) -> Path:
    base = QC_OLINK_DIR if root == "olink" else QC_SOMA_DIR
    return base / name


def _build_weight_wide(advs: pl.LazyFrame) -> pl.LazyFrame:
    return (
        advs.with_columns(pl.col("VISITNUM").cast(pl.Int32, strict=False))
        .filter(
            pl.col("PARAMCD") == "WEIGHTKG",
            pl.col("VISITNUM").is_in(VISITNUMS),
        )
        .select("USUBJID", "VISITNUM", pl.col("AVAL").alias("weight_kg"))
        .collect()
        .pivot(
            on="VISITNUM",
            index="USUBJID",
            values="weight_kg",
            aggregate_function="first",
        )
        .rename({str(visit): f"WGTV{visit}" for visit in VISITNUMS})
        .with_columns(
            [
                (
                    (pl.col(f"WGTV{visit}") - pl.col(f"WGTV{BASELINE_VISITNUM}"))
                    / pl.col(f"WGTV{BASELINE_VISITNUM}")
                    * 100
                ).alias(f"PCTCHG_WGTV{visit}")
                for visit in POST_BASELINE_VISITS
            ]
        )
        .lazy()
    )


def _visit_specific_percent_change(prefix: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col("VISITNUM") == BASELINE_VISITNUM)
        .then(pl.lit(0.0))
        .otherwise(
            pl.coalesce(
                [
                    pl.when(pl.col("VISITNUM") == visit).then(pl.col(f"{prefix}{visit}"))
                    for visit in POST_BASELINE_VISITS
                ]
            )
        )
        .alias(alias)
    )


def _normalize_olink_expression() -> pl.LazyFrame:
    expr = (
        pl.scan_parquet(
            _qc_file("olink", "surmount5_olink.npx_long.parquet")
        )
        .filter(pl.col("SampleQC_Status") != "FAIL")
        .rename({"SampleID": "sample_id"})
    )
    columns = set(expr.collect_schema().names())
    if "NPX" in columns:
        return expr
    if "PCNormalizedNPX" in columns:
        return expr.rename({"PCNormalizedNPX": "NPX"})
    raise ValueError(
        "Expected either 'NPX' or 'PCNormalizedNPX' in the Olink QC long parquet."
    )


def _normalize_soma_expression() -> pl.LazyFrame:
    expr = (
        pl.scan_parquet(
            _qc_file("soma", "surmount5_soma.rfu_long.parquet")
        )
        .filter(pl.col("SampleQC_Status") != "FAIL")
        .rename({"SampleId": "sample_id"})
    )
    columns = set(expr.collect_schema().names())
    if "RFU" not in columns:
        raise ValueError("Expected 'RFU' in the SomaScan QC long parquet.")
    return expr.with_columns(pl.col("RFU").log(2).alias("log2_RFU"))


def _olink_annotation() -> pl.DataFrame:
    return pl.read_csv(
        _qc_file("olink", "surmount5_olink.assay_qc_summary.tsv"),
        separator="\t",
    ).select(["OlinkID", "Assay", pl.col("Status").alias("allQC")])


def _soma_annotation() -> pl.DataFrame:
    return pl.read_csv(
        _qc_file("soma", "surmount5_soma.assay_qc_summary.tsv"),
        separator="\t",
    ).select(
        ["SeqId", pl.col("Target").alias("Assay"), pl.col("Status").alias("allQC")]
    )


_REQUIRED_COLUMNS = {
    "USUBJID",
    "VISITNUM",
    "TRT01A",
    "AGE",
    "SEX",
}


def _select_available(frame: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    available = set(frame.collect_schema().names())
    missing_required = _REQUIRED_COLUMNS - available
    if missing_required:
        raise ValueError(
            f"Platform frame is missing required columns: {sorted(missing_required)}"
        )
    return frame.select([column for column in columns if column in available])


def _build_olink_frame(*, active_comparator_only: bool) -> PreparedPlatformInputs:
    adsl = _load_adsl()
    weight_wide = _build_weight_wide(_load_advs())
    sample_map = _load_sample_map(
        _qc_file("olink", "surmount5_olink.sample_qc_summary.tsv"),
        "SampleID",
    )
    expr = _normalize_olink_expression()

    baseline = (
        expr.join(sample_map, on="sample_id")
        .filter(pl.col("VISITNUM") == BASELINE_VISITNUM)
        .select("USUBJID", "OlinkID", pl.col("NPX").alias("NPXBL"))
        .unique(subset=["USUBJID", "OlinkID"])
    )

    data = (
        expr.join(sample_map, on="sample_id")
        .join(adsl, on="USUBJID")
        .join(weight_wide, on="USUBJID", how="left")
        .join(baseline, on=["USUBJID", "OlinkID"], how="left")
        .with_columns(
            _visit_specific_percent_change("PCTCHG_WGTV", "PCTCHG_WGT"),
        )
    )
    if active_comparator_only:
        data = data.filter(pl.col("TRT01A").is_in([RAW_CONTROL_VALUE, RAW_TREAT_VALUE]))

    return PreparedPlatformInputs(
        platform="olink",
        data=_select_available(
            data,
            [
                "sample_id",
                "OlinkID",
                "NPX",
                "NPXBL",
                "USUBJID",
                "VISITNUM",
                "AGE",
                "SEX",
                "TRT01A",
                "WGTBL",
                "PCTCHG_WGT",
                "HBA1CBL",
            ],
        ).rename({"sample_id": "SampleID"}),
        annotation=_olink_annotation(),
        response_var="NPX",
        response_var_bl="NPXBL",
        protein_id_col="OlinkID",
    )


def _build_soma_frame(*, active_comparator_only: bool) -> PreparedPlatformInputs:
    adsl = _load_adsl()
    weight_wide = _build_weight_wide(_load_advs())
    sample_map = _load_sample_map(
        _qc_file("soma", "surmount5_soma.sample_qc_summary.tsv"),
        "SampleId",
    )
    expr = _normalize_soma_expression()

    baseline = (
        expr.join(sample_map, on="sample_id")
        .filter(pl.col("VISITNUM") == BASELINE_VISITNUM)
        .select("USUBJID", "SeqId", pl.col("log2_RFU").alias("log2_RFUBL"))
        .unique(subset=["USUBJID", "SeqId"])
    )

    data = (
        expr.join(sample_map, on="sample_id")
        .join(adsl, on="USUBJID")
        .join(weight_wide, on="USUBJID", how="left")
        .join(baseline, on=["USUBJID", "SeqId"], how="left")
        .with_columns(
            _visit_specific_percent_change("PCTCHG_WGTV", "PCTCHG_WGT"),
        )
    )
    if active_comparator_only:
        data = data.filter(pl.col("TRT01A").is_in([RAW_CONTROL_VALUE, RAW_TREAT_VALUE]))

    return PreparedPlatformInputs(
        platform="soma",
        data=_select_available(
            data,
            [
                "sample_id",
                "SeqId",
                "log2_RFU",
                "log2_RFUBL",
                "USUBJID",
                "VISITNUM",
                "AGE",
                "SEX",
                "TRT01A",
                "WGTBL",
                "PCTCHG_WGT",
                "HBA1CBL",
            ],
        ).rename({"sample_id": "SampleId"}),
        annotation=_soma_annotation(),
        response_var="log2_RFU",
        response_var_bl="log2_RFUBL",
        protein_id_col="SeqId",
    )


def prepare_mmrm_inputs(platform: str) -> PreparedPlatformInputs:
    if platform == "olink":
        return _build_olink_frame(active_comparator_only=False)
    if platform == "soma":
        return _build_soma_frame(active_comparator_only=False)
    raise ValueError(f"Unsupported platform: {platform}")


def prepare_mediation_inputs(platform: str) -> PreparedPlatformInputs:
    if platform == "olink":
        return _build_olink_frame(active_comparator_only=True)
    if platform == "soma":
        return _build_soma_frame(active_comparator_only=True)
    raise ValueError(f"Unsupported platform: {platform}")
