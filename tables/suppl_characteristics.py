"""Baseline characteristics of the SURMOUNT-5 proteomic substudy.

One-sheet Table-1-style summary across the substudy cohort, defined as
randomized TZP/SEMA participants with a non-FAIL baseline sample on Olink
OR SomaScan. Includes per-arm and total demographics, anthropometrics,
metabolic markers, per-platform sample counts, and the number of proteins
differential between arms at baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import COVAR, PHENO_DIR, QC_OLINK_DIR, QC_SOMA_DIR, mmrm_dir  # noqa: E402

ADSL_PATH = PHENO_DIR / "adsl.parquet"
OLINK_SAMPLE_QC = QC_OLINK_DIR / "surmount5_olink.sample_qc_summary.tsv"
SOMA_SAMPLE_QC = QC_SOMA_DIR / "surmount5_soma.sample_qc_summary.tsv"

TZP = "TZP 15mg or MTD"
SEMA = "SEMA 2.4mg or MTD"
BASELINE_VISIT = 2


def _mean_sd(series: pl.Series, dp: int = 1) -> str:
    s = series.drop_nulls()
    if s.is_empty():
        return ""
    return f"{s.mean():.{dp}f} ({s.std():.{dp}f})"


def _n_pct(n: int, total: int) -> str:
    return f"{n} ({n / total * 100:.1f})" if total else "0 (0.0)"


def _baseline_diff_counts(platform: str) -> tuple[int, int]:
    path = (
        mmrm_dir(platform)
        / "finalRes"
        / "baseline"
        / f"surmount5_{COVAR}_proteomics_olinkAnalysis_Baseline_DiffRes_py.csv"
    )
    df = pl.read_csv(path, columns=["pVal", "fdr"])
    return (
        df.filter(pl.col("fdr") < 0.05).height,
        df.filter(pl.col("pVal") < 0.05).height,
    )


def _baseline_pass_usubjids(qc_path: Path) -> list[str]:
    return (
        pl.read_csv(qc_path, separator="\t", columns=["USUBJID", "VISITNUM", "Status"])
        .filter((pl.col("VISITNUM") == BASELINE_VISIT) & (pl.col("Status") != "FAIL"))[
            "USUBJID"
        ]
        .unique()
        .to_list()
    )


def build() -> list[tuple[str, pl.DataFrame]]:
    adsl_full = pl.read_parquet(ADSL_PATH).filter(pl.col("TRT01A").is_in([TZP, SEMA]))

    oly_subjects = _baseline_pass_usubjids(OLINK_SAMPLE_QC)
    soma_subjects = _baseline_pass_usubjids(SOMA_SAMPLE_QC)
    cohort = adsl_full.filter(
        pl.col("USUBJID").is_in(oly_subjects) | pl.col("USUBJID").is_in(soma_subjects)
    )
    tzp = cohort.filter(pl.col("TRT01A") == TZP)
    sema = cohort.filter(pl.col("TRT01A") == SEMA)
    n_tzp, n_sema, n_total = tzp.height, sema.height, cohort.height

    def _by_arm_count(usubjids: list[str]) -> tuple[int, int, int]:
        subset = cohort.filter(pl.col("USUBJID").is_in(usubjids))
        return (
            subset.filter(pl.col("TRT01A") == TZP).height,
            subset.filter(pl.col("TRT01A") == SEMA).height,
            subset.height,
        )

    oly_t, oly_s, oly_all = _by_arm_count(oly_subjects)
    som_t, som_s, som_all = _by_arm_count(soma_subjects)

    n_oly_fdr, n_oly_p = _baseline_diff_counts("olink")
    n_soma_fdr, n_soma_p = _baseline_diff_counts("soma")

    col_tzp = f"TZP 15mg (n={n_tzp})"
    col_sema = f"SEMA 2.4mg (n={n_sema})"
    col_total = f"Total (n={n_total})"

    def row(label, t, s, tot):
        return {"Characteristic": label, col_tzp: t, col_sema: s, col_total: tot}

    def header(label):
        return row(label, "", "", "")

    def cont(label, col, dp=1):
        return row(
            label,
            _mean_sd(tzp[col], dp),
            _mean_sd(sema[col], dp),
            _mean_sd(cohort[col], dp),
        )

    def cat_yn(label, col, val):
        nt = tzp.filter(pl.col(col) == val).height
        ns = sema.filter(pl.col(col) == val).height
        na = cohort.filter(pl.col(col) == val).height
        return row(label, _n_pct(nt, n_tzp), _n_pct(ns, n_sema), _n_pct(na, n_total))

    rows = [
        header("Demographics"),
        cont("  Age, years", "AGE"),
        cat_yn("  Female, n (%)", "SEX", "F"),
        header("Anthropometrics"),
        cont("  Body weight, kg", "WEIGHTBL"),
        cont("  BMI, kg/m²", "BMIBL"),
        cont("  Waist circumference, cm", "WSTCIRBL"),
        header("Metabolic"),
        cont("  HbA1c, %", "HBA1CBL", dp=2),
        cat_yn("  Prediabetes, n (%)", "PRDIAFL1", "Y"),
        header("Proteomic substudy"),
        row(
            "  Olink samples passing QC at baseline, n",
            str(oly_t),
            str(oly_s),
            str(oly_all),
        ),
        row(
            "  SomaScan samples passing QC at baseline, n",
            str(som_t),
            str(som_s),
            str(som_all),
        ),
        header("Baseline differential proteins"),
        row("  Olink, FDR < 0.05, n", "", "", str(n_oly_fdr)),
        row("  Olink, P < 0.05, n", "", "", str(n_oly_p)),
        row("  SomaScan, FDR < 0.05, n", "", "", str(n_soma_fdr)),
        row("  SomaScan, P < 0.05, n", "", "", str(n_soma_p)),
    ]

    return [("Characteristics", pl.DataFrame(rows))]
