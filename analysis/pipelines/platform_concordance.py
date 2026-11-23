"""Cross-platform correlation + GMM concordance for SURMOUNT-5.

Assays are paired on the UniProt accessions of the assay annotation
(``analysis.external.vendor_assay_tables``). Computes
per-assay-pair (SeqId × OlinkID) Spearman and Pearson correlations using baseline samples
only (the QC analysis cohort at the baseline analysis visit, ``analysis.qc_contract``),
then fits a 2-component Gaussian Mixture Model to identify concordant pairs. Olink values are PCNormalizedNPX; SomaScan
values are log2 of the non-ANML RFU.

Output: analysis/outputs/platform_concordance.parquet, one row per pair, with the mixture's
weights, means, standard deviations and threshold as constant columns
(``analysis.concordance.FIT_COLUMNS``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr

from analysis.concordance import fit_columns, fit_concordance_gmm
from analysis.design import BASELINE_VISITNUM
from analysis.external.vendor_assay_tables import read_annotation
from analysis.output_layout import ANALYSIS_OUTPUTS, PLATFORM_CONCORDANCE
from analysis.qc_contract import (
    assay_summary_path,
    cohort_samples,
    long_table_path,
)


# ---------------------------------------------------------------------------
# Protein pairing via the assay annotation's accessions
# ---------------------------------------------------------------------------


def _assay_accessions(
    annotation: pl.DataFrame, platform: str, assays: pl.DataFrame, id_col: str
) -> pl.DataFrame:
    """``id_col`` and ``UP`` (one accession), one row per analyzed assay and annotated accession.

    Raises:
        ValueError: If an analyzed assay has no row in the annotation.
    """
    own = annotation.filter(pl.col("Platform") == platform)
    missing = assays.join(own.select(pl.col("AssayID").alias(id_col)).unique(), on=id_col, how="anti")
    if missing.height:
        raise ValueError(
            f"{platform}: {missing.height} analyzed assays are not in the assay annotation, "
            f"e.g. {missing[id_col].head(5).to_list()}"
        )
    accessions = (
        own.select(pl.col("AssayID").alias(id_col), pl.col("UniProt").alias("UP"))
        .filter(pl.col("UP").is_not_null() & (pl.col("UP") != ""))
        .unique()
    )
    return assays.select(id_col).join(accessions, on=id_col, how="inner")


def build_protein_pairs(
    annotation: pl.DataFrame,
    soma_assays: pl.DataFrame,
    olink_assays: pl.DataFrame,
) -> pl.DataFrame:
    """Match SeqId and OlinkID pairs sharing at least one UniProt accession of the annotation.

    The accessions are the assay annotation's (``analysis.external.vendor_assay_tables``), one
    row per (assay, accession) on both platforms. An assay is paired
    with every assay of the other platform sharing any of its accessions.

    Args:
        annotation: The long annotation (``read_annotation()``): ``Platform``, ``AssayID``,
            ``UniProt``.
        soma_assays: The analyzed aptamers: ``SeqId``, ``Target``.
        olink_assays: The analyzed Olink assays: ``OlinkID``, ``Assay``.

    Returns:
        One row per (SeqId, OlinkID): SeqId, OlinkID, SomaTarget, OlinkAssay, SharedUniProt
        (sorted, semicolon-joined) and NSharedUniProt.

    Raises:
        ValueError: If an analyzed assay is not in the annotation.
    """
    soma_acc = _assay_accessions(annotation, "soma", soma_assays, "SeqId")
    olink_acc = _assay_accessions(annotation, "olink", olink_assays, "OlinkID")
    return (
        soma_acc.join(olink_acc, on="UP", how="inner")
        .group_by("SeqId", "OlinkID")
        .agg(
            pl.col("UP").unique().sort().str.join(";").alias("SharedUniProt"),
            pl.col("UP").n_unique().alias("NSharedUniProt"),
        )
        .join(soma_assays.select("SeqId", pl.col("Target").alias("SomaTarget")), on="SeqId", how="left")
        .join(olink_assays.select("OlinkID", pl.col("Assay").alias("OlinkAssay")), on="OlinkID", how="left")
        .select("SeqId", "OlinkID", "SomaTarget", "OlinkAssay", "SharedUniProt", "NSharedUniProt")
        .sort("SeqId", "OlinkID")
    )


# ---------------------------------------------------------------------------
# Sample matching (baseline only)
# ---------------------------------------------------------------------------


def match_baseline_samples() -> pl.DataFrame:
    """Match baseline samples across platforms by USUBJID.

    Takes each platform's QC analysis cohort (``Status != FAIL``) at the baseline
    analysis visit, then inner-joins on USUBJID.
    Returns DataFrame with: USUBJID, SomaSampleId, OlinkSampleID.
    """

    def baseline(platform: str, sid: str) -> pl.DataFrame:
        return (
            cohort_samples(platform)
            .filter(pl.col("VISITNUM") == BASELINE_VISITNUM)
            .select(sid, "USUBJID")
            .collect()
        )

    soma_bl = baseline("soma", "SampleId")
    olink_bl = baseline("olink", "SampleID")

    matched = soma_bl.join(olink_bl, on="USUBJID", how="inner")
    return matched.select("USUBJID", "SampleId", "SampleID")


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------


def _corr(x: np.ndarray, y: np.ndarray):
    """Pearson and Spearman on finite pairs; returns (pr, pp, sr, sp, n)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, n
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return pr, pp, sr, sp, n


def compute_correlations(
    soma_data_path: Path,
    olink_data_path: Path,
    protein_pairs: pl.DataFrame,
    sample_matches: pl.DataFrame,
) -> pl.DataFrame:
    """Compute per-pair correlations using baseline-matched samples.

    Uses log2(RFU) for Soma and PCNormalizedNPX for Olink.
    """
    t0 = time.time()

    soma_sample_ids = set(sample_matches["SampleId"].to_list())
    olink_sample_ids = set(sample_matches["SampleID"].to_list())

    # Load and filter to matched baseline samples that pass sample QC
    print("Loading Soma data …")
    soma = (
        pl.scan_parquet(soma_data_path)
        .filter(
            pl.col("SampleId").is_in(soma_sample_ids)
            & (pl.col("SampleQC_Status") != "FAIL")
        )
        .select("SampleId", "SeqId", "RFU")
        .collect()
    )

    print("Loading Olink data …")
    olink = (
        pl.scan_parquet(olink_data_path)
        .filter(
            pl.col("SampleID").is_in(olink_sample_ids)
            & (pl.col("SampleQC_Status") != "FAIL")
        )
        .select("SampleID", "OlinkID", "PCNormalizedNPX")
        .collect()
    )

    # Join USUBJID for alignment
    soma_with_subj = soma.join(
        sample_matches.select("SampleId", "USUBJID"), on="SampleId", how="inner"
    )
    olink_with_subj = olink.join(
        sample_matches.select("SampleID", "USUBJID"), on="SampleID", how="inner"
    )

    # Pivot to wide: rows=USUBJID, columns=assay_id
    print("Pivoting Soma …")
    soma_wide = soma_with_subj.pivot(
        on="SeqId", index="USUBJID", values="RFU", aggregate_function="first"
    )

    print("Pivoting Olink …")
    olink_wide = olink_with_subj.pivot(
        on="OlinkID",
        index="USUBJID",
        values="PCNormalizedNPX",
        aggregate_function="first",
    )

    # Align subjects
    common_subj = set(soma_wide["USUBJID"].to_list()) & set(
        olink_wide["USUBJID"].to_list()
    )
    soma_wide = soma_wide.filter(pl.col("USUBJID").is_in(common_subj)).sort("USUBJID")
    olink_wide = olink_wide.filter(pl.col("USUBJID").is_in(common_subj)).sort("USUBJID")
    print(f"  Common baseline subjects: {len(common_subj)}")

    soma_cols = set(soma_wide.columns) - {"USUBJID"}
    olink_cols = set(olink_wide.columns) - {"USUBJID"}

    pair_ids = list(zip(protein_pairs["SeqId"], protein_pairs["OlinkID"], strict=True))
    absent = [(s, o) for s, o in pair_ids if s not in soma_cols or o not in olink_cols]
    if absent:
        raise ValueError(
            f"{len(absent)} of {len(pair_ids)} assay pairs are missing from a baseline "
            f"wide frame (SeqId, OlinkID), e.g. {absent[:5]}"
        )

    # Compute correlations per pair
    n_pairs = len(protein_pairs)
    print(f"Computing correlations for {n_pairs:,} pairs …")

    results = []
    for i, row in enumerate(protein_pairs.iter_rows(named=True)):
        seq_id = row["SeqId"]
        olink_id = row["OlinkID"]

        x = soma_wide[seq_id].to_numpy().astype(np.float64)
        if (x <= 0).any():
            raise ValueError(
                f"SomaScan {seq_id} has {int((x <= 0).sum())} non-positive RFU values"
            )
        x = np.log2(x)
        y = olink_wide[olink_id].to_numpy().astype(np.float64)

        pr, pp, sr, sp, n = _corr(x, y)

        results.append(
            {
                "SeqId": seq_id,
                "OlinkID": olink_id,
                "SomaTarget": row["SomaTarget"],
                "OlinkAssay": row["OlinkAssay"],
                "SharedUniProt": row["SharedUniProt"],
                "NSharedUniProt": row["NSharedUniProt"],
                "NSamples": n,
                "PearsonR": pr,
                "PearsonP": pp,
                "SpearmanRho": sr,
                "SpearmanP": sp,
            }
        )

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1:,}/{n_pairs:,} ({time.time() - t0:.1f}s)")

    print(f"Done: {len(results):,} pairs with data in {time.time() - t0:.1f}s")
    return pl.DataFrame(results)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(output_dir: Path = ANALYSIS_OUTPUTS) -> pl.DataFrame:
    """Run cross-platform correlation + GMM concordance pipeline."""

    print("=" * 70)
    print("CROSS-PLATFORM CONCORDANCE (Baseline, GMM)")
    print("=" * 70)

    # 1. Build protein pairs
    print("\n[1/4] Building protein pairs …")
    pairs = build_protein_pairs(
        read_annotation(),
        pl.read_csv(assay_summary_path("soma"), separator="\t").select("SeqId", "Target"),
        pl.read_csv(assay_summary_path("olink"), separator="\t").select("OlinkID", "Assay"),
    )
    print(f"  Matched pairs: {len(pairs):,}")

    # 2. Match baseline samples
    print("\n[2/4] Matching baseline samples …")
    samples = match_baseline_samples()
    print(f"  Matched baseline subjects: {len(samples):,}")

    # 3. Compute correlations
    print("\n[3/4] Computing correlations …")
    corr_df = compute_correlations(
        long_table_path("soma"),
        long_table_path("olink"),
        pairs,
        samples,
    )

    # 4. Fit GMM
    print("\n[4/4] Fitting GMM …")
    rho_vals = corr_df["SpearmanRho"].to_numpy()
    gmm_result = fit_concordance_gmm(rho_vals)

    print(
        f"  GMM component means: μ₀={gmm_result['means'][0]:.3f}, μ₁={gmm_result['means'][1]:.3f}"
    )
    print(
        f"  GMM component stds:  σ₀={gmm_result['stds'][0]:.3f}, σ₁={gmm_result['stds'][1]:.3f}"
    )
    print(
        f"  GMM weights:         w₀={gmm_result['weights'][0]:.3f}, w₁={gmm_result['weights'][1]:.3f}"
    )
    print(f"  Threshold (P=0.5):   τ={gmm_result['threshold']:.3f}")

    # Annotate with GMM results
    corr_df = corr_df.with_columns(
        pl.Series("GMM_P_Concordant", gmm_result["posteriors"]),
    ).with_columns(
        (pl.col("GMM_P_Concordant") >= 0.5).alias("Concordant"),
    )

    # UniProt: the first shared accession; GeneSymbol: the Olink vendor assay name, else the SomaScan target
    corr_df = corr_df.with_columns(
        pl.col("SharedUniProt").str.split(";").list.first().alias("UniProt"),
    ).with_columns(
        pl.coalesce(
            pl.col("OlinkAssay"),
            pl.col("SomaTarget"),
        ).alias("GeneSymbol"),
    )

    # Final column order
    corr_df = corr_df.select(
        "SeqId",
        "OlinkID",
        "UniProt",
        "GeneSymbol",
        "SomaTarget",
        "OlinkAssay",
        "SharedUniProt",
        "NSharedUniProt",
        "NSamples",
        "SpearmanRho",
        "SpearmanP",
        "PearsonR",
        "PearsonP",
        "GMM_P_Concordant",
        "Concordant",
        *fit_columns(gmm_result),
    ).sort("SpearmanRho", descending=True)

    # Summary
    n_conc = corr_df.filter(pl.col("Concordant")).height
    n_total = corr_df.height
    print(
        f"\n  Concordant pairs: {n_conc:,} / {n_total:,} ({100 * n_conc / n_total:.1f}%)"
    )
    print(f"  Median Spearman rho: {corr_df['SpearmanRho'].median():.3f}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / PLATFORM_CONCORDANCE.name
    corr_df.write_parquet(out_path)
    print(f"\n  Saved → {out_path}")
    print("=" * 70)

    return corr_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_OUTPUTS)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
