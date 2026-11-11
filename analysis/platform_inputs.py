"""SURMOUNT-5 platform-specific analysis frames built from the QC outputs.

The QC outputs are read through ``analysis.qc_contract``: the analysis cohort is every
sample with ``Status != "FAIL"`` at its analysis visit (``VISITNUM_ANALYSIS``). Olink is modeled on NPX (QC ran
on PCNormalizedNPX) with the datapoints Olink's per-datapoint ``AssayQC`` flag marks
``"FAIL"`` set to missing (their rows are left out of the long frame); SomaScan on log2 of the non-ANML RFU.
Every assay is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from analysis.design import (
    BASELINE_VISITNUM,
    CAT_COVARIATES,
    COVAR_LABEL,
    POST_BASELINE_VISITS,
    RAW_CONTROL_VALUE,
    RAW_TREAT_VALUE,
    VISITNUMS,
)

# Re-exports of the study-design constants.
from analysis.design import CONTROL_VALUE as CONTROL_VALUE
from analysis.design import TREAT_VALUE as TREAT_VALUE
from analysis.design import VISIT_LABELS as VISIT_LABELS
from paths import PHENO_DIR

# Olink assays (by OlinkID) excluded from the ranking-based analyses (Supp Fig 5, pathway
# ranked statistic).
OLINK_EXCLUDED_ASSAYS: tuple[str, ...] = ("OID42503",)
# Platform -> assays set aside before any ranking, enrichment or signature.
EXCLUDED_ASSAYS: dict[str, tuple[str, ...]] = {"olink": OLINK_EXCLUDED_ASSAYS, "soma": ()}

MMRM_COVARIATE_CONFIG: dict[str, list[str] | str] = {
    "cat_covariates": list(CAT_COVARIATES),
    "cont_covariates": ["AGE"],
    "covar_label": COVAR_LABEL,
}


@dataclass(frozen=True)
class PreparedPlatformInputs:
    """One platform's long analysis frame and assay annotation, as the MMRM and mediation read them.

    Attributes:
        platform: ``"olink"`` or ``"soma"``.
        data: One row per sample and assay in the analysis cohort, joined to ADSL and the
            visit body weight: the sample ID (``SampleID`` / ``SampleId``), the assay ID, the
            response and its baseline value, ``USUBJID``, ``VISITNUM``, ``AGE``, ``SEX``,
            ``TRT01A``, ``WGTBL``, ``PCTCHG_WGT`` (0 at baseline) and ``HBA1CBL`` (each column
            that ADSL supplies; the ADaM key columns are required).
        annotation: Assay ID and assay name as ``Assay``.
        response_var: The modeled value column, ``NPX`` (Olink) or ``log2_RFU`` (SomaScan).
        response_var_bl: Its baseline column, ``NPXBL`` or ``log2_RFUBL``.
        protein_id_col: The assay key, ``OlinkID`` or ``SeqId``.
    """

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


def _build_weight_wide(advs: pl.LazyFrame) -> pl.LazyFrame:
    """Body weight per subject at each proteomics visit, wide, with the percent changes.

    Rows are selected on the recorded VISITNUM. Repeated weights of one subject and
    visit must agree (they collapse to one) or it raises.
    """
    long = (
        advs.filter(
            pl.col("PARAMCD") == "WEIGHTKG",
            pl.col("VISITNUM").is_in([float(v) for v in VISITNUMS]),
        )
        .select("USUBJID", pl.col("VISITNUM").cast(pl.Int32), pl.col("AVAL").alias("weight_kg"))
        .collect()
    )
    conflicts = long.group_by("USUBJID", "VISITNUM").agg(
        pl.col("weight_kg").unique().sort().alias("weights")
    ).filter(pl.col("weights").list.len() > 1)
    if conflicts.height:
        raise ValueError(
            f"{conflicts.height} (USUBJID, VISITNUM) weight rows disagree, "
            f"e.g. {conflicts.sort('USUBJID', 'VISITNUM').head(3).to_dicts()}"
        )
    return (
        long.unique(subset=["USUBJID", "VISITNUM"], keep="first", maintain_order=True)
        .pivot(
            on="VISITNUM",
            index="USUBJID",
            values="weight_kg",
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


def _cohort(platform: str) -> pl.LazyFrame:
    """The platform's analysis cohort: sample ID as ``sample_id``, USUBJID, VISITNUM."""
    from analysis.qc_contract import PLATFORM_SPECS, cohort_samples

    return cohort_samples(platform).select(
        pl.col(PLATFORM_SPECS[platform].sample_id).alias("sample_id"), "USUBJID", "VISITNUM"
    )


def _olink_expression() -> pl.LazyFrame:
    """Olink NPX per sample and assay; datapoints the vendor's AssayQC flag marks FAIL are missing."""
    from analysis.qc_contract import long_table_path

    return (
        pl.scan_parquet(long_table_path("olink"))
        .filter(pl.col("AssayQC") != "FAIL")
        .select(pl.col("SampleID").alias("sample_id"), "OlinkID", "NPX")
    )


def _soma_expression() -> pl.LazyFrame:
    """log2 of the SomaScan non-ANML RFU."""
    from analysis.qc_contract import long_table_path

    return pl.scan_parquet(long_table_path("soma")).select(
        pl.col("SampleId").alias("sample_id"), "SeqId", pl.col("RFU").log(2).alias("log2_RFU")
    )


def _annotation(platform: str) -> pl.DataFrame:
    """Assay ID and name from the platform's assay summary."""
    from analysis.qc_contract import PLATFORM_SPECS, read_assay_summary

    spec = PLATFORM_SPECS[platform]
    return read_assay_summary(platform).select(spec.protein_id, pl.col(spec.protein_name).alias("Assay"))


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
        raise ValueError(f"Platform frame is missing required columns: {sorted(missing_required)}")
    return frame.select([column for column in columns if column in available])


def _one_value_per_visit(data: pl.LazyFrame, protein_col: str) -> pl.LazyFrame:
    """Raise if any (USUBJID, VISITNUM, protein) has more than one value row."""
    dupes = (
        data.group_by("USUBJID", "VISITNUM", protein_col)
        .len()
        .filter(pl.col("len") > 1)
        .collect()
    )
    if dupes.height:
        raise ValueError(
            f"{dupes.height} (USUBJID, VISITNUM, {protein_col}) keys have more than one "
            f"value row, e.g. {dupes.head(3).to_dicts()}"
        )
    return data


def _build_olink_frame(*, active_comparator_only: bool) -> PreparedPlatformInputs:
    adsl = _load_adsl()
    weight_wide = _build_weight_wide(_load_advs())
    sample_map = _cohort("olink")
    expr = _olink_expression()

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
    data = _one_value_per_visit(data, "OlinkID")

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
        annotation=_annotation("olink"),
        response_var="NPX",
        response_var_bl="NPXBL",
        protein_id_col="OlinkID",
    )


def _build_soma_frame(*, active_comparator_only: bool) -> PreparedPlatformInputs:
    adsl = _load_adsl()
    weight_wide = _build_weight_wide(_load_advs())
    sample_map = _cohort("soma")
    expr = _soma_expression()

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
    data = _one_value_per_visit(data, "SeqId")

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
        annotation=_annotation("soma"),
        response_var="log2_RFU",
        response_var_bl="log2_RFUBL",
        protein_id_col="SeqId",
    )


def prepare_mmrm_inputs(platform: str) -> PreparedPlatformInputs:
    """The MMRM frame for one platform: every arm in the QC analysis cohort.

    Reads the QC long table, cohort and assay summary through ``analysis.qc_contract`` and
    ``adsl.parquet`` / ``advs.parquet`` under ``PHENO_DIR``. Olink keeps NPX with the
    datapoints the vendor's AssayQC flag marks FAIL set to missing; SomaScan takes log2 of the non-ANML RFU.

    Args:
        platform: ``"olink"`` or ``"soma"``.

    Returns:
        The prepared frame, annotation and column names.

    Raises:
        ValueError: If ``platform`` is neither, ADVS weights for one subject and visit
            disagree, a subject x visit x assay key has more than one value row, or a
            required ADaM column is missing.
    """
    if platform == "olink":
        return _build_olink_frame(active_comparator_only=False)
    if platform == "soma":
        return _build_soma_frame(active_comparator_only=False)
    raise ValueError(f"Unsupported platform: {platform}")


def prepare_mediation_inputs(platform: str) -> PreparedPlatformInputs:
    """The mediation frame for one platform: as :func:`prepare_mmrm_inputs`, restricted to the two compared arms.

    Rows are kept only where ``TRT01A`` is the raw tirzepatide or semaglutide label
    (``RAW_TREAT_VALUE`` / ``RAW_CONTROL_VALUE`` in ``analysis.design``).

    Args:
        platform: ``"olink"`` or ``"soma"``.

    Returns:
        The prepared frame, annotation and column names.

    Raises:
        ValueError: As :func:`prepare_mmrm_inputs`.
    """
    if platform == "olink":
        return _build_olink_frame(active_comparator_only=True)
    if platform == "soma":
        return _build_soma_frame(active_comparator_only=True)
    raise ValueError(f"Unsupported platform: {platform}")
