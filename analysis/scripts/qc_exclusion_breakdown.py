"""QC exclusion-flow breakdown for Supplementary Figure 1 (study design).

Materializes a single committed artifact
(``analysis/outputs/qc_exclusion_breakdown.parquet``) holding the per-reason
sample/participant counts of the SURMOUNT-5 proteomics QC flow, so that
``figures/supp_fig1_study_design.py`` reads them from data instead of carrying
them as inline constants (per the no-hardcoding convention).

Staging (reproduces the net step totals in the figure's CONSORT flow)
---------------------------------------------------------------------
Working from the committed sample-QC summaries
(``<qc_root>/surmount5_{olink,soma}/*.sample_qc_summary.tsv``)::

    nonbridge    = ~FAIL_Bridge_Sample                         # 1824 / 1806
    technical    = nonbridge & any(<technical FAIL flags>)     # −25 / −31
    longitudinal = (nonbridge & ~technical)
                   & any(FAIL_Discontinued, _No_Baseline, _Only_Baseline)
                                                               # −111 / −124
    final        = remainder                                   # 1688/598, 1651/591

Technical FAIL flags are platform-specific:
    Olink:    FAIL_Missing_Rate, FAIL_PCA_Outlier, FAIL_Median_IQR_Outlier,
              FAIL_Sex_Concordance, FAIL_Age_Concordance
    SomaScan: FAIL_RowCheck_and_RFU_Outlier, FAIL_PCA_Outlier,
              FAIL_Median_IQR_Outlier, FAIL_Sex_Concordance, FAIL_Age_Concordance

Per-reason counts use SINGLE attribution: each excluded sample is charged to the
first criterion it fails in a fixed priority order, so the per-reason counts sum
exactly to the net step total (no double counting).

    Olink technical:    missing → PCA → median/IQR → sex/age
    SomaScan technical: RFU/row-check → PCA → median/IQR → sex/age
    longitudinal:       discontinuation → no-baseline → only-baseline

Final analysis-ready participant counts are split by treatment arm by joining the
retained ``USUBJID`` set to ``TRT01A`` from ADSL.

Artifact schema (tidy long form, one fact per row)
--------------------------------------------------
- platform       — ``olink`` | ``soma``
- stage          — ``input`` | ``technical`` | ``longitudinal`` | ``final``
- order          — bullet order within (platform, stage); 0 for non-bullet rows
- reason         — machine slug (null for ``input`` / overall ``final`` rows)
- label          — display label used in the figure callouts (null where N/A)
- arm            — ``TZP`` | ``SEMA`` for per-arm ``final`` rows (else null)
- n_samples      — sample count (null where the row carries only participants)
- n_participants — participant count (final rows only; else null)

Requires the dxfuse mounts (``qc_root``, ``pheno_root``). The written parquet is
committed so the figure renders without the mounts present.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from paths import PHENO_DIR, QC_OLINK_DIR, QC_SOMA_DIR, first_existing

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "analysis" / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "qc_exclusion_breakdown.parquet"

BRIDGE_FLAG = "FAIL_Bridge_Sample"

# (reason slug, display label, [FAIL flag columns]) in single-attribution
# priority order — earlier entries win when a sample fails several gates.
OLINK_TECHNICAL = [
    ("missing", "Missing values", ["FAIL_Missing_Rate"]),
    ("pca", "PCA outlier", ["FAIL_PCA_Outlier"]),
    ("median_iqr", "Median / IQR outlier", ["FAIL_Median_IQR_Outlier"]),
    ("sex_age", "Sex / age mismatch", ["FAIL_Sex_Concordance", "FAIL_Age_Concordance"]),
]
SOMA_TECHNICAL = [
    ("rfu_rowcheck", "RFU / row-check outlier", ["FAIL_RowCheck_and_RFU_Outlier"]),
    ("pca", "PCA outlier", ["FAIL_PCA_Outlier"]),
    ("median_iqr", "Median / IQR outlier", ["FAIL_Median_IQR_Outlier"]),
    ("sex_age", "Sex / age mismatch", ["FAIL_Sex_Concordance", "FAIL_Age_Concordance"]),
]
LONGITUDINAL = [
    ("discontinuation", "Treatment discontinuation", ["FAIL_Discontinued"]),
    ("no_baseline", "No baseline sample", ["FAIL_No_Baseline"]),
    ("only_baseline", "Only baseline sample", ["FAIL_Only_Baseline"]),
]

# Raw TRT01A → display arm label (matches analysis.platform_inputs raw values).
ARM_LABELS = {"TZP 15mg or MTD": "TZP", "SEMA 2.4mg or MTD": "SEMA"}

PLATFORMS = [
    ("olink", QC_OLINK_DIR, "olink", OLINK_TECHNICAL),
    ("soma", QC_SOMA_DIR, "soma", SOMA_TECHNICAL),
]


def _qc_summary(qc_dir: Path, study: str) -> pl.DataFrame:
    """Load a platform's sample-QC summary (all FAIL flags as raw strings)."""
    path = first_existing(
        qc_dir / f"surmount5_{study}.sample_qc_summary.tsv",
        qc_dir / f"surmount5_{study}.sample_qc_summary.tsv",
    )
    return pl.read_csv(path, separator="\t", infer_schema_length=0)


def _is_true(col: str) -> pl.Expr:
    return pl.col(col) == "true"


def _any_true(flags: list[str]) -> pl.Expr:
    return pl.any_horizontal([_is_true(c) for c in flags])


def _attribute(frame: pl.DataFrame, specs: list[tuple[str, str, list[str]]]) -> list[int]:
    """Single-attribution per-reason counts over ``specs`` (priority order)."""
    counts: list[int] = []
    assigned = pl.lit(False)
    for _, _, flags in specs:
        gate = _any_true(flags)
        counts.append(frame.filter(gate & ~assigned).height)
        assigned = assigned | gate
    return counts


def _adsl_arm() -> pl.DataFrame:
    return (
        pl.scan_parquet(PHENO_DIR / "adsl.parquet")
        .select("USUBJID", "TRT01A")
        .collect()
    )


def build_breakdown() -> pl.DataFrame:
    """Compute the full QC-flow breakdown across both platforms."""
    adsl = _adsl_arm()
    rows: list[dict] = []

    for platform, qc_dir, study, technical in PLATFORMS:
        df = _qc_summary(qc_dir, study)

        nonbridge = df.filter(~_is_true(BRIDGE_FLAG))
        tech_flags = [flag for _, _, flags in technical for flag in flags]
        technical_set = nonbridge.filter(_any_true(tech_flags))
        remainder = nonbridge.filter(~_any_true(tech_flags))
        long_flags = [flag for _, _, flags in LONGITUDINAL for flag in flags]
        longitudinal_set = remainder.filter(_any_true(long_flags))
        final_set = remainder.filter(~_any_true(long_flags))

        rows.append({
            "platform": platform, "stage": "input", "order": 0,
            "reason": None, "label": "Non-bridge samples", "arm": None,
            "n_samples": nonbridge.height, "n_participants": None,
        })

        for stage, specs, frame in [
            ("technical", technical, technical_set),
            ("longitudinal", LONGITUDINAL, longitudinal_set),
        ]:
            for i, ((reason, label, _), count) in enumerate(
                zip(specs, _attribute(frame, specs)), start=1
            ):
                rows.append({
                    "platform": platform, "stage": stage, "order": i,
                    "reason": reason, "label": label, "arm": None,
                    "n_samples": count, "n_participants": None,
                })

        final_ppts = final_set.select("USUBJID").unique()
        rows.append({
            "platform": platform, "stage": "final", "order": 0,
            "reason": None, "label": None, "arm": None,
            "n_samples": final_set.height, "n_participants": final_ppts.height,
        })

        arm_counts = (
            final_ppts.join(adsl, on="USUBJID", how="left")
            .with_columns(pl.col("TRT01A").replace_strict(ARM_LABELS, default=None).alias("arm"))
            .drop_nulls("arm")
            .group_by("arm")
            .len()
        )
        arm_map = dict(zip(arm_counts["arm"], arm_counts["len"]))
        for i, arm in enumerate(("TZP", "SEMA"), start=1):
            rows.append({
                "platform": platform, "stage": "final", "order": i,
                "reason": None, "label": None, "arm": arm,
                "n_samples": None, "n_participants": int(arm_map[arm]),
            })

    return pl.DataFrame(
        rows,
        schema={
            "platform": pl.String, "stage": pl.String, "order": pl.Int32,
            "reason": pl.String, "label": pl.String, "arm": pl.String,
            "n_samples": pl.Int64, "n_participants": pl.Int64,
        },
    )


def write_breakdown() -> Path:
    breakdown = build_breakdown()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    breakdown.write_parquet(OUTPUT_PATH)
    print(f"Saved → {OUTPUT_PATH}")
    for platform in ("olink", "soma"):
        sub = breakdown.filter(pl.col("platform") == platform)
        tech = sub.filter(pl.col("stage") == "technical")["n_samples"].sum()
        long = sub.filter(pl.col("stage") == "longitudinal")["n_samples"].sum()
        final = sub.filter((pl.col("stage") == "final") & pl.col("arm").is_null())
        print(
            f"  {platform:5s}  input {sub.filter(pl.col('stage') == 'input')['n_samples'].item():,}"
            f"  −{tech} technical  −{long} longitudinal"
            f"  → {final['n_samples'].item():,} samples / {final['n_participants'].item()} ppts"
        )
    return OUTPUT_PATH


if __name__ == "__main__":
    write_breakdown()
