"""Data preparation: load MMRM results, map markers to genes, compute ranking metrics."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ranking metric functions
# ---------------------------------------------------------------------------


def _ranking_metric_fc(fc: float, pval: float) -> float:
    """Calculate -log10(pVal) * sign(log2(FC)).  FC is centered at 1."""
    if pd.isna(fc) or pd.isna(pval):
        return np.nan
    if pval == 0:
        pval = 1e-300
    sign = 1 if fc > 1 else (-1 if fc < 1 else 0)
    return -np.log10(pval) * sign


def _ranking_metric_pcbl(pcbl: float, pval: float) -> float:
    """Calculate -log10(pVal) * sign(PCBL).  PCBL is centered at 0."""
    if pd.isna(pcbl) or pd.isna(pval):
        return np.nan
    if pval == 0:
        pval = 1e-300
    sign = 1 if pcbl > 0 else (-1 if pcbl < 0 else 0)
    return -np.log10(pval) * sign


# ---------------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------------


def prepare_data(
    results_file: str,
    olink_mapping: dict[str, list[str]],
    somascan_mapping: dict[str, list[str]],
    qc_filter: str = "ALL",
    weighting: str = "equal",
) -> tuple[dict[str, float], list[str]]:
    """
    Load MMRM results, map markers to genes, compute ranking metric.

    When multiple markers map to the same gene, their ranking metrics are
    combined using the selected *weighting* strategy:
      - ``"equal"`` : simple arithmetic mean (default)
      - ``"ivw"``   : inverse-variance weighted mean (w_i = 1/SE_i^2)

    Returns ``(ranked_genes dict, significant_genes list)``.
    """
    df = pd.read_csv(results_file)
    logger.info(f"Loaded {len(df)} rows from {Path(results_file).name}")
    required_columns = {"marker", "pVal", "fdr"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{results_file} is missing required columns: {missing}")

    # Auto-detect platform
    sample = df["marker"].head(100).astype(str)
    is_olink = any(m.startswith("OID") for m in sample)
    mapping = olink_mapping if is_olink else somascan_mapping
    logger.info(f"Platform: {'Olink' if is_olink else 'Somascan'}")

    # QC filter
    if "allQC" in df.columns:
        before = len(df)
        if qc_filter == "PASS":
            df = df[df["allQC"] == "PASS"]
        elif qc_filter == "PASS_WARN":
            df = df[df["allQC"].isin(["PASS", "WARN"])]
        logger.info(f"QC {qc_filter}: {before} -> {len(df)} rows")
    else:
        logger.warning("No 'allQC' column — skipping QC filter")

    # Auto-detect effect column
    df = df.copy()
    if "FC" in df.columns:
        effect_col = "FC"
        se_col = "SE-FC"
        metric_fn = _ranking_metric_fc
    elif "PCBL" in df.columns:
        effect_col = "PCBL"
        se_col = "SE_PCBL"
        metric_fn = _ranking_metric_pcbl
    else:
        logger.error("Neither 'FC' nor 'PCBL' column found in results")
        return {}, []
    logger.info(f"Effect column: {effect_col}")

    # Check SE availability for IVW
    has_se = se_col in df.columns
    if weighting == "ivw" and not has_se:
        logger.warning(
            f"SE column '{se_col}' not found — falling back to equal weighting"
        )
        weighting = "equal"

    # Ranking metric (vectorised)
    df["ranking_metric"] = np.vectorize(metric_fn)(df[effect_col], df["pVal"])

    # Expand marker -> gene(s)
    rows = []
    skipped_nan = 0
    for _, r in df.iterrows():
        genes = mapping.get(r["marker"], [])
        se_val = r.get(se_col, np.nan) if has_se else np.nan
        for g in genes:
            if pd.isna(g):
                skipped_nan += 1
                continue
            row_dict = {
                "gene_symbol": g,
                "ranking_metric": r["ranking_metric"],
                "fdr": r["fdr"],
            }
            if weighting == "ivw":
                row_dict["se"] = se_val
            rows.append(row_dict)
    if skipped_nan:
        logger.info(f"Skipped {skipped_nan} NaN gene symbols")
    if not rows:
        logger.error("No genes mapped!")
        return {}, []

    expanded = pd.DataFrame(rows)

    if weighting == "ivw":
        # Inverse-variance weighted mean: w_i = 1 / SE_i^2
        expanded = expanded.dropna(subset=["ranking_metric", "se"])
        expanded = expanded[expanded["se"] > 0]
        expanded["weight"] = 1.0 / (expanded["se"] ** 2)

        def _ivw_agg(grp):
            w = grp["weight"]
            r = grp["ranking_metric"]
            return pd.Series(
                {
                    "ranking_metric": (w * r).sum() / w.sum(),
                    "fdr": grp["fdr"].min(),
                }
            )

        genes_agg = expanded.groupby("gene_symbol").apply(_ivw_agg)
        logger.info("Weighting: IVW (1/SE^2)")
    else:
        genes_agg = (
            expanded.groupby("gene_symbol")
            .agg(ranking_metric=("ranking_metric", "mean"), fdr=("fdr", "min"))
            .dropna(subset=["ranking_metric"])
        )
        logger.info("Weighting: equal (simple mean)")

    logger.info(f"Mapped {len(expanded)} pairs -> {len(genes_agg)} unique genes")

    ranked = genes_agg["ranking_metric"].to_dict()
    sig = genes_agg.index[genes_agg["fdr"] < 0.05].tolist()
    logger.info(f"Significant genes (FDR<0.05): {len(sig)}")
    return ranked, sig
