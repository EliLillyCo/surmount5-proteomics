"""Baseline characteristics of the SURMOUNT-5 proteomic substudy.

One-sheet Table-1-style summary across the substudy cohort, defined as
randomized TZP/SEMA participants with a baseline sample in the Olink or
SomaScan QC analysis cohort. Includes per-arm and total demographics, anthropometrics,
metabolic markers, per-platform sample counts, and the number of proteins
differential between arms at baseline.
"""

from __future__ import annotations


import polars as pl

from analysis.mmrm_outputs import baseline_diff_csv
from analysis.design import BASELINE_VISITNUM, RAW_CONTROL_VALUE, RAW_TREAT_VALUE
from analysis.qc_contract import cohort_samples
from analysis.statistics import FDR_LABEL, FDR_THRESHOLD
from paths import PHENO_DIR

ADSL_PATH = PHENO_DIR / "adsl.parquet"

TZP = RAW_TREAT_VALUE
SEMA = RAW_CONTROL_VALUE


def _mean_sd(series: pl.Series, dp: int = 1) -> str:
    s = series.drop_nulls()
    if s.is_empty():
        return ""
    return f"{s.mean():.{dp}f} ({s.std():.{dp}f})"


def _n_pct(n: int, total: int) -> str:
    return f"{n} ({n / total * 100:.1f})" if total else "0 (0.0)"


def _baseline_diff_counts(platform: str) -> tuple[int, int]:
    path = baseline_diff_csv(platform)
    df = pl.read_csv(path, columns=["pVal", "fdr"])
    return (
        df.filter(pl.col("fdr") < FDR_THRESHOLD).height,
        df.filter(pl.col("pVal") < 0.05).height,
    )


def _baseline_pass_usubjids(platform: str) -> list[str]:
    """Participants with a baseline sample in the platform's QC analysis cohort."""
    return (
        cohort_samples(platform)
        .filter(pl.col("VISITNUM") == BASELINE_VISITNUM)
        .select("USUBJID")
        .unique()
        .sort("USUBJID")
        .collect()["USUBJID"]
        .to_list()
    )


def build() -> list[tuple[str, pl.DataFrame]]:
    """Return the one-sheet baseline characteristics table of the proteomic substudy.

    The cohort is the TZP and SEMA participants in ADSL with a baseline sample in the
    Olink or the SomaScan QC analysis cohort. Rows: demographics, anthropometrics and
    metabolic markers per arm and in total, per-platform baseline sample counts, and the
    number of proteins that differ between arms at baseline (FDR < 0.05 and P < 0.05, from
    the MMRM baseline contrast CSVs).
    """
    adsl_full = pl.read_parquet(ADSL_PATH).filter(pl.col("TRT01A").is_in([TZP, SEMA]))

    oly_subjects = _baseline_pass_usubjids("olink")
    soma_subjects = _baseline_pass_usubjids("soma")
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

    col_tzp = f"TZP (MTD 10 or 15 mg; n={n_tzp})"
    col_sema = f"SEMA (MTD 1.7 or 2.4 mg; n={n_sema})"
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
        row(f"  Olink, FDR < {FDR_LABEL}, n", "", "", str(n_oly_fdr)),
        row("  Olink, P < 0.05, n", "", "", str(n_oly_p)),
        row(f"  SomaScan, FDR < {FDR_LABEL}, n", "", "", str(n_soma_fdr)),
        row("  SomaScan, P < 0.05, n", "", "", str(n_soma_p)),
    ]

    return [("Characteristics", pl.DataFrame(rows))]
