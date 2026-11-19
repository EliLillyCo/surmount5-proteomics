"""Proteomics assay vs. matched clinical-lab biomarker: pair mapping, loaders and Spearman fits.

For each protein whose assay has a directly matched clinical laboratory analyte, ``fit_pair``
correlates the QC cohort's protein values (Olink NPX; SomaScan log2 RFU) with the lab value
(Spearman). Extended Data Fig 3 draws the baseline fits and Tables S7/S8 list every timepoint;
both call the loaders and ``fit_pair`` here, so a correlation the figure prints is the
table's.

Proteomics and laboratory visits are matched using ``TIMEPOINTS``. A subject's repeated lab rows at a visit are averaged; nonpositive lab values and
nonpositive SomaScan RFU are dropped; a pair needs at least ``MIN_N`` subjects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from analysis.design import BASELINE_VISITNUM, VISIT_TO_WEEK
from analysis.qc_contract import cohort_samples

MIN_N = 10
# Lab visit and week matched to baseline proteomics.
LAB_SCREENING_VISITNUM = 1
LAB_SCREENING_WEEK = -2
# (proteomics VISITNUM, lab VISITNUM, proteomics week, lab week), in visit order.
TIMEPOINTS: list[tuple[int, int, int, int]] = [
    (visit, LAB_SCREENING_VISITNUM, week, LAB_SCREENING_WEEK)
    if visit == BASELINE_VISITNUM
    else (visit, visit, week, week)
    for visit, week in VISIT_TO_WEEK.items()
]


@dataclass(frozen=True)
class PairSpec:
    """One assay matched to one lab test.

    Attributes:
        gene: Label of the assay's protein (gene symbol or analyte name).
        assay_id: OlinkID or SeqId.
        lab_name: Lab test name.
        paramcd: ADLB ``PARAMCD`` of the lab test.
    """

    gene: str
    assay_id: str
    lab_name: str
    paramcd: str


OLINK_PAIRS = [
    PairSpec("INS-CPEPTIDE", "OID44722", "Insulin", "INSUP28S"),
    PairSpec("GGT1", "OID45132", "GGT", "GGTE17S"),
    PairSpec("GOT1", "OID45135", "AST", "ASTE01S"),
    PairSpec("CST3", "OID45345", "Cystatin C", "CYSTZB5S"),
    PairSpec("PNLIP", "OID45398", "Lipase", "LIPAE09T"),
    PairSpec("CALCA", "OID43443", "Calcitonin", "CLCTGB7S"),
    PairSpec("TSHB", "OID45001", "TSH", "TSHQ04S"),
    PairSpec("FSHB", "OID43729", "FSH", "FSHQ06S"),
]

SOMA_PAIRS = [
    PairSpec("INS", "4883-56", "Insulin", "INSUP28S"),
    PairSpec("GPT", "3709-4", "ALT", "ALTE03S"),
    PairSpec("GOT1", "4912-17", "AST", "ASTE01S"),
    PairSpec("ALPL", "16926-44", "ALP", "ALP47E"),
    PairSpec("CK-MM", "2670-67", "Creatine Kinase", "CKE12T"),
    PairSpec("CK-MB", "3714-49", "Creatine Kinase", "CKE12T"),
    PairSpec("ALB", "18380-78", "Albumin", "ALBF11S"),
    PairSpec("CST3", "2609-59", "Cystatin C", "CYSTZB5S"),
    PairSpec("PNLIP", "15613-16", "Lipase", "LIPAE09T"),
    PairSpec("CALCA", "29468-2", "Calcitonin", "CLCTGB7S"),
    PairSpec("TSH", "3521-16", "TSH", "TSHQ04S"),
    PairSpec("AMY2A", "18917-53", "Amylase", "AMYLE56S"),
    PairSpec("FSH", "3032-11", "FSH", "FSHQ06S"),
]


def load_subjects(platform: str, sample_col: str, visitnum: int) -> pl.DataFrame:
    """The QC analysis cohort's samples at one analysis visit, sorted by sample ID."""
    return (
        cohort_samples(platform)
        .filter(pl.col("VISITNUM") == visitnum)
        .select(sample_col, "USUBJID")
        .sort(sample_col)
        .collect()
    )


def load_proteomics(
    parquet_path: Path,
    sample_col: str,
    assay_col: str,
    value_col: str,
    valid_samples: pl.DataFrame,
    assay_ids: list[str],
) -> pl.DataFrame:
    """``USUBJID``, assay ID and value of the given assays in the given samples."""
    sample_ids = valid_samples[sample_col].to_list()
    df = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col(sample_col).is_in(sample_ids) & pl.col(assay_col).is_in(assay_ids))
        .select(sample_col, assay_col, value_col)
        .collect()
    )
    return (
        df.join(valid_samples, on=sample_col)
        .select("USUBJID", assay_col, value_col)
        .sort(["USUBJID", assay_col])
    )


def load_lab_values(adlb: Path, paramcds: list[str], lab_visitnum: int) -> pl.DataFrame:
    """Each subject's mean positive lab value per ``PARAMCD`` at one lab visit."""
    return (
        pl.scan_parquet(adlb)
        .filter(
            pl.col("PARAMCD").is_in(paramcds)
            & (pl.col("VISITNUM") == lab_visitnum)
            & pl.col("AVAL").is_not_null()
            & (pl.col("AVAL") > 0)
        )
        .group_by("USUBJID", "PARAMCD", maintain_order=True)
        .agg(pl.col("AVAL").mean())
        .collect()
        .sort(["USUBJID", "PARAMCD"])
    )


@dataclass(frozen=True)
class PairFit:
    """Spearman fit of one assay against its lab test.

    Attributes:
        spec: The pair.
        rho: Spearman correlation.
        pvalue: Its two-sided p-value.
        n: Subjects in the fit.
        protein_vals: Protein values in the fit (log2 when the platform is log-transformed).
        lab_vals: Lab values in the fit, same subjects and order.
    """

    spec: PairSpec
    rho: float
    pvalue: float
    n: int
    protein_vals: np.ndarray
    lab_vals: np.ndarray


def fit_pair(
    spec: PairSpec,
    prot_df: pl.DataFrame,
    lab_df: pl.DataFrame,
    assay_col: str,
    value_col: str,
    log2_transform: bool,
) -> PairFit | None:
    """Spearman correlation of one assay with its lab test over the subjects having both.

    Args:
        spec: The pair.
        prot_df: ``load_proteomics`` output.
        lab_df: ``load_lab_values`` output.
        assay_col: ``OlinkID`` or ``SeqId``.
        value_col: Protein value column (``NPX`` or ``RFU``).
        log2_transform: Take log2 of positive protein values (SomaScan RFU).

    Returns:
        The fit, or None when fewer than ``MIN_N`` subjects remain.
    """
    prot = prot_df.filter(pl.col(assay_col) == spec.assay_id).select("USUBJID", value_col)
    lab = lab_df.filter(pl.col("PARAMCD") == spec.paramcd).select("USUBJID", "AVAL")
    merged = prot.join(lab, on="USUBJID").drop_nulls().sort("USUBJID")
    if len(merged) < MIN_N:
        return None

    x = merged[value_col].to_numpy().astype(float)
    y = merged["AVAL"].to_numpy().astype(float)
    if log2_transform:
        valid = (x > 0) & np.isfinite(x)
        x, y = x[valid], y[valid]
        x = np.log2(x)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[mask], y[mask]
    if len(x) < MIN_N:
        return None

    rho, pvalue = stats.spearmanr(x, y)
    return PairFit(spec=spec, rho=rho, pvalue=pvalue, n=len(x), protein_vals=x, lab_vals=y)
