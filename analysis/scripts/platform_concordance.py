"""Cross-platform correlation + GMM concordance for SURMOUNT-5.

Computes per-assay-pair (SeqId × OlinkID) Spearman and Pearson
correlations using baseline samples only (VISITNUM=2), then fits a
2-component Gaussian Mixture Model to identify concordant pairs.

Output: analysis/outputs/platform_concordance.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr
from sklearn.mixture import GaussianMixture

_SCRIPT_DIR = Path(__file__).resolve().parent
_ANALYSIS_DIR = _SCRIPT_DIR.parent
_OUTPUT_DIR = _ANALYSIS_DIR / "outputs"

sys.path.insert(0, str(_ANALYSIS_DIR.parent))
from paths import QC_OLINK_DIR, QC_SOMA_DIR, first_existing  # noqa: E402

SOMA_DIR = QC_SOMA_DIR
OLINK_DIR = QC_OLINK_DIR


# ---------------------------------------------------------------------------
# Protein pairing via shared UniProt
# ---------------------------------------------------------------------------


def build_protein_pairs(
    soma_assay_path: Path,
    olink_assay_path: Path,
) -> pl.DataFrame:
    """Match SeqId ↔ OlinkID pairs sharing at least one UniProt accession.

    Returns DataFrame with: SeqId, OlinkID, SomaTarget, OlinkAssay,
    SharedUniProt (semicolon-joined), NSharedUniProt.
    """
    soma = pl.read_csv(soma_assay_path, separator="\t").select(
        "SeqId", "Target", "UniProt"
    )
    olink = pl.read_csv(olink_assay_path, separator="\t").select(
        "OlinkID", "Assay", "UniProt"
    )

    # Explode Soma pipe-separated UniProts
    soma_exp = (
        soma.with_columns(pl.col("UniProt").str.split("|"))
        .explode("UniProt")
        .with_columns(pl.col("UniProt").str.strip_chars())
        .filter(pl.col("UniProt").is_not_null() & (pl.col("UniProt") != ""))
        .rename({"UniProt": "UP"})
    )

    # Olink has single UniProt per row (verified: no multi-UniProt entries)
    olink_exp = olink.with_columns(
        pl.col("UniProt").str.strip_chars().alias("UP")
    ).filter(pl.col("UP").is_not_null() & (pl.col("UP") != ""))

    # Inner join on shared UniProt
    merged = soma_exp.join(olink_exp, on="UP", how="inner")

    # Aggregate: group by (SeqId, OlinkID) → collect shared UniProts
    pairs = (
        merged.group_by(["SeqId", "OlinkID"])
        .agg(
            pl.col("Target").first().alias("SomaTarget"),
            pl.col("Assay").first().alias("OlinkAssay"),
            pl.col("UP").unique().sort().str.join(";").alias("SharedUniProt"),
            pl.col("UP").n_unique().alias("NSharedUniProt"),
        )
        .sort("SeqId", "OlinkID")
    )
    return pairs


# ---------------------------------------------------------------------------
# Sample matching (baseline only)
# ---------------------------------------------------------------------------


def match_baseline_samples(
    soma_sample_qc_path: Path,
    olink_sample_qc_path: Path,
) -> pl.DataFrame:
    """Match baseline samples across platforms by USUBJID.

    Filters to VISITNUM=2, Status != FAIL, then inner-joins on USUBJID.
    Returns DataFrame with: USUBJID, SomaSampleId, OlinkSampleID.
    """
    soma_sq = pl.read_csv(soma_sample_qc_path, separator="\t")
    olink_sq = pl.read_csv(olink_sample_qc_path, separator="\t")

    soma_bl = soma_sq.filter(
        (pl.col("VISITNUM") == 2) & (pl.col("Status") != "FAIL")
    ).select(pl.col("SampleId"), pl.col("USUBJID"))
    olink_bl = olink_sq.filter(
        (pl.col("VISITNUM") == 2) & (pl.col("Status") != "FAIL")
    ).select(pl.col("SampleID"), pl.col("USUBJID"))

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

    # Compute correlations per pair
    n_pairs = len(protein_pairs)
    print(f"Computing correlations for {n_pairs:,} pairs …")

    results = []
    for i, row in enumerate(protein_pairs.iter_rows(named=True)):
        seq_id = row["SeqId"]
        olink_id = row["OlinkID"]

        if seq_id not in soma_cols or olink_id not in olink_cols:
            continue

        x = soma_wide[seq_id].to_numpy().astype(np.float64)
        x = np.log2(np.clip(x, 1e-10, None))
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
# GMM concordance
# ---------------------------------------------------------------------------


def fit_concordance_gmm(
    spearman_values: np.ndarray,
) -> dict:
    """Fit 2-component GMM and compute per-observation P(concordant).

    Returns dict with: means, stds, weights, threshold, posteriors, labels.
    """
    X = spearman_values.reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
        n_init=10,
        max_iter=500,
        reg_covar=5e-3,
    )
    gmm.fit(X)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_

    # Ensure component 0 = low, component 1 = high
    order = np.argsort(means)
    means = means[order]
    stds = stds[order]
    weights = weights[order]

    # Reorder posteriors
    raw_resp = gmm.predict_proba(X)
    posteriors = raw_resp[:, order]

    # Threshold: where P(high | rho) = 0.5
    from scipy import stats as sp_stats

    margin = 0.5 * (means[1] - means[0])
    x_grid = np.linspace(means[0] - margin, means[1] + margin, 10000)
    d0 = weights[0] * sp_stats.norm.pdf(x_grid, means[0], stds[0])
    d1 = weights[1] * sp_stats.norm.pdf(x_grid, means[1], stds[1])
    total = d0 + d1
    p_high = np.where(total > 0, d1 / total, 0)
    cross_idx = np.where(np.diff(np.sign(p_high - 0.5)))[0]
    if len(cross_idx) > 0:
        midpoint = np.mean(means)
        best = cross_idx[np.argmin(np.abs(x_grid[cross_idx] - midpoint))]
        threshold = float(x_grid[best])
    else:
        threshold = float(np.mean(means))

    return {
        "means": means,
        "stds": stds,
        "weights": weights,
        "threshold": threshold,
        "posteriors": posteriors[:, 1],  # P(high component)
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    soma_dir: Path = SOMA_DIR,
    olink_dir: Path = OLINK_DIR,
    output_dir: Path = _OUTPUT_DIR,
) -> pl.DataFrame:
    """Run cross-platform correlation + GMM concordance pipeline."""

    print("=" * 70)
    print("CROSS-PLATFORM CONCORDANCE (Baseline, GMM)")
    print("=" * 70)

    # 1. Build protein pairs
    print("\n[1/4] Building protein pairs …")
    soma_assay_qc = first_existing(
        soma_dir / "surmount5_soma.assay_qc_summary.tsv",
        soma_dir / "surmount5_soma.assay_qc_summary.tsv",
    )
    olink_assay_qc = first_existing(
        olink_dir / "surmount5_olink.assay_qc_summary.tsv",
        olink_dir / "surmount5_olink.assay_qc_summary.tsv",
    )
    pairs = build_protein_pairs(
        soma_assay_qc,
        olink_assay_qc,
    )
    print(f"  Matched pairs: {len(pairs):,}")

    # 2. Match baseline samples
    print("\n[2/4] Matching baseline samples …")
    soma_sample_qc = first_existing(
        soma_dir / "surmount5_soma.sample_qc_summary.tsv",
        soma_dir / "surmount5_soma.sample_qc_summary.tsv",
    )
    olink_sample_qc = first_existing(
        olink_dir / "surmount5_olink.sample_qc_summary.tsv",
        olink_dir / "surmount5_olink.sample_qc_summary.tsv",
    )
    samples = match_baseline_samples(
        soma_sample_qc,
        olink_sample_qc,
    )
    print(f"  Matched baseline subjects: {len(samples):,}")

    # 3. Compute correlations
    print("\n[3/4] Computing correlations …")
    soma_long = first_existing(
        soma_dir / "surmount5_soma.rfu_long.parquet",
        soma_dir / "surmount5_soma.rfu_long.parquet",
    )
    olink_long = first_existing(
        olink_dir / "surmount5_olink.npx_long.parquet",
        olink_dir / "surmount5_olink.npx_long.parquet",
    )
    corr_df = compute_correlations(
        soma_long,
        olink_long,
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

    # Add gene symbol from UniProt map
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
    out_path = output_dir / "platform_concordance.parquet"
    corr_df.write_parquet(out_path)
    print(f"\n  Saved → {out_path}")
    print("=" * 70)

    return corr_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soma-dir", type=Path, default=SOMA_DIR)
    parser.add_argument("--olink-dir", type=Path, default=OLINK_DIR)
    parser.add_argument("--output-dir", type=Path, default=_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.soma_dir, args.olink_dir, args.output_dir)


if __name__ == "__main__":
    main()
