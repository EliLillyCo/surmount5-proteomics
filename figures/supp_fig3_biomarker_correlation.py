"""Supplemental figure: proteomics assay vs. clinical lab biomarker correlations.

Scatter plots of baseline protein values (NPX or log₂RFU) vs. matched clinical
lab measurements, with Spearman ρ annotations.  Layout: 5 rows × 5 columns —
Olink on top (rows 0–1, 8 pairs; last two cells of row 1 hidden), SomaScan
below (rows 2–4, 13 pairs ranked by |ρ|; last two cells of row 4 hidden).

All lab values use VISITNUM 1 (screening); proteomics from VISITNUM 2 (randomization).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import polars as pl
import ultraplot as uplt
from scipy import stats

from _internal._common import (
    BANNER_HEIGHT_IN,
    FS_AUX,
    FS_EMPH,
    FS_PANEL,
    NAT_W2,
    PHENO_DIR,
    QC_OLINK_DIR,
    QC_SOMA_DIR,
    c,
    init_figure_theme,
    save_figure,
)

# ---------------------------------------------------------------------------
# Lab-assay color mapping (consistent across platforms, all Lilly palette)
# ---------------------------------------------------------------------------


LAB_COLORS = {
    "FSH": "#5A9BD5",
    "GGT": c["bold_brown"],
    "Cystatin C": c["bold_green"],
    "Calcitonin": c["bold_grey"],
    "TSH": c["red"],
    "Lipase": c["vibrant_coral"],
    "Insulin": c["bold_blue"],
    "AST": c["vibrant_gold"],
    "ALT": "#E8873D",
    "ALP": "#7A5195",
    "Albumin": "#3CB371",
    "Creatine Kinase": c["black"],
    "Amylase": "#E07B9A",
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OLINK_NPX = QC_OLINK_DIR / "surmount5_olink.npx_long.parquet"
SOMA_RFU = QC_SOMA_DIR / "surmount5_soma.rfu_long.parquet"
OLINK_SAMPLE_QC = QC_OLINK_DIR / "surmount5_olink.sample_qc_summary.tsv"
SOMA_SAMPLE_QC = QC_SOMA_DIR / "surmount5_soma.sample_qc_summary.tsv"
ADLB = PHENO_DIR / "adlb.parquet"

# ---------------------------------------------------------------------------
# Protein-to-lab mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairSpec:
    gene: str
    assay_id: str
    lab_name: str
    paramcd: str


OLINK_PAIRS = [
    PairSpec("FSHB", "OID43729", "FSH", "FSHQ06S"),
    PairSpec("GGT1", "OID45132", "GGT", "GGTE17S"),
    PairSpec("CST3", "OID45345", "Cystatin C", "CYSTZB5S"),
    PairSpec("CALCA", "OID43443", "Calcitonin", "CLCTGB7S"),
    PairSpec("TSHB", "OID45001", "TSH", "TSHQ04S"),
    PairSpec("PNLIP", "OID45398", "Lipase", "LIPAE09T"),
    PairSpec("INS-CPEPTIDE", "OID44722", "Insulin", "INSUP28S"),
    PairSpec("GOT1", "OID45135", "AST", "ASTE01S"),
]

SOMA_PAIRS = [
    PairSpec("FSH", "3032-11", "FSH", "FSHQ06S"),
    PairSpec("PNLIP", "15613-16", "Lipase", "LIPAE09T"),
    PairSpec("CST3", "2609-59", "Cystatin C", "CYSTZB5S"),
    PairSpec("CK-MM", "2670-67", "Creatine Kinase", "CKE12T"),
    PairSpec("CK-MB", "3714-49", "Creatine Kinase", "CKE12T"),
    PairSpec("AMY2A", "18917-53", "Amylase", "AMYLE56S"),
    PairSpec("TSH", "3521-16", "TSH", "TSHQ04S"),
    PairSpec("INS", "4883-56", "Insulin", "INSUP28S"),
    PairSpec("GPT", "3709-4", "ALT", "ALTE03S"),
    PairSpec("GOT1", "4912-17", "AST", "ASTE01S"),
    PairSpec("ALPL", "16926-44", "ALP", "ALP47E"),
    PairSpec("ALB", "18380-78", "Albumin", "ALBF11S"),
    PairSpec("CALCA", "29468-2", "Calcitonin", "CLCTGB7S"),
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_baseline_subjects(qc_path: Path, sample_col: str) -> pl.DataFrame:
    return (
        pl.read_csv(qc_path, separator="\t")
        .filter((pl.col("VISITNUM") == 2) & (pl.col("Status") != "FAIL"))
        .select(sample_col, "USUBJID")
        .sort(sample_col)
    )


def _load_proteomics(
    parquet_path: Path,
    sample_col: str,
    assay_col: str,
    value_col: str,
    valid_samples: pl.DataFrame,
    assay_ids: list[str],
) -> pl.DataFrame:
    sample_ids = valid_samples[sample_col].to_list()
    df = (
        pl.scan_parquet(parquet_path)
        .filter(
            pl.col(sample_col).is_in(sample_ids) & pl.col(assay_col).is_in(assay_ids)
        )
        .select(sample_col, assay_col, value_col)
        .collect()
    )
    return (
        df.join(valid_samples, on=sample_col)
        .select("USUBJID", assay_col, value_col)
        .sort(["USUBJID", assay_col])
    )


def _load_lab_values(paramcds: list[str]) -> pl.DataFrame:
    return (
        pl.scan_parquet(ADLB)
        .filter(
            pl.col("PARAMCD").is_in(paramcds)
            & (pl.col("VISITNUM") == 1)
            & pl.col("AVAL").is_not_null()
            & (pl.col("AVAL") > 0)
        )
        .group_by("USUBJID", "PARAMCD", maintain_order=True)
        .agg(pl.col("AVAL").mean())
        .collect()
        .sort(["USUBJID", "PARAMCD"])
    )


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------


@dataclass
class CorrResult:
    spec: PairSpec
    rho: float
    pvalue: float
    n: int
    protein_vals: np.ndarray
    lab_vals: np.ndarray
    slope: float
    intercept: float


def compute_correlations(
    pairs: list[PairSpec],
    prot_df: pl.DataFrame,
    lab_df: pl.DataFrame,
    assay_col: str,
    value_col: str,
    log2_transform: bool = False,
) -> list[CorrResult]:
    results = []
    for spec in pairs:
        prot = prot_df.filter(pl.col(assay_col) == spec.assay_id).select(
            "USUBJID", value_col
        )
        lab = lab_df.filter(pl.col("PARAMCD") == spec.paramcd).select("USUBJID", "AVAL")
        merged = prot.join(lab, on="USUBJID").drop_nulls().sort("USUBJID")

        if len(merged) < 10:
            continue

        x = merged[value_col].to_numpy().astype(float)
        y = merged["AVAL"].to_numpy().astype(float)

        if log2_transform:
            valid = (x > 0) & np.isfinite(x)
            x, y = x[valid], y[valid]
            x = np.log2(x)

        mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
        x, y = x[mask], y[mask]

        if len(x) < 10:
            continue

        rho, pvalue = stats.spearmanr(x, y)
        log_y = np.log10(y)
        slope, intercept, *_ = stats.linregress(x, log_y)

        results.append(
            CorrResult(
                spec=spec,
                rho=rho,
                pvalue=pvalue,
                n=len(x),
                protein_vals=x,
                lab_vals=y,
                slope=slope,
                intercept=intercept,
            )
        )

    results.sort(
        key=lambda r: (-abs(r.rho), r.spec.gene, r.spec.assay_id, r.spec.lab_name)
    )
    return results


# ---------------------------------------------------------------------------
# Banner helpers
# ---------------------------------------------------------------------------


def _add_platform_banner(fig, ax_top_left, ax_top_right, label: str):
    """Full-width grey banner above a column group."""
    bb_left = ax_top_left.get_position()
    bb_right = ax_top_right.get_position()
    banner_height = BANNER_HEIGHT_IN / fig.get_figheight()
    banner_y = bb_left.y1 + 0.022
    banner = mpatches.FancyBboxPatch(
        (bb_left.x0, banner_y),
        bb_right.x1 - bb_left.x0,
        banner_height,
        boxstyle="square,pad=0",
        facecolor="#E8E8E8",
        edgecolor="none",
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.patches.append(banner)
    mid_x = (bb_left.x0 + bb_right.x1) / 2
    mid_y = banner_y + banner_height / 2
    fig.text(
        mid_x,
        mid_y,
        label,
        fontsize=FS_EMPH,
        fontweight="bold",
        ha="center",
        va="center",
    )


def _add_subplot_banner(fig, ax, gene: str, assay_id: str):
    """Small grey banner above an individual subplot: bold gene · grey assay ID."""
    bb = ax.get_position()
    banner_height = 0.016
    banner_y = bb.y1 + 0.001
    banner = mpatches.FancyBboxPatch(
        (bb.x0, banner_y),
        bb.x1 - bb.x0,
        banner_height,
        boxstyle="square,pad=0",
        facecolor="#F2F2F2",
        edgecolor="none",
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.patches.append(banner)
    mid_x = (bb.x0 + bb.x1) / 2
    mid_y = banner_y + banner_height / 2
    fig.text(
        mid_x - 0.002,
        mid_y,
        gene,
        fontsize=FS_AUX,
        fontweight="bold",
        ha="right",
        va="center",
    )
    fig.text(
        mid_x + 0.002,
        mid_y,
        f"· {assay_id}",
        fontsize=FS_AUX,
        color="#666666",
        ha="left",
        va="center",
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _plot_panel(
    axs,
    results: list[CorrResult],
    x_label: str,
    bottom_row_indices: set[int],
) -> None:
    """Fill axes with scatter + regression for results, colored by lab assay."""
    for i, ax in enumerate(axs):
        if i >= len(results):
            ax.set_visible(False)
            continue

        r = results[i]
        color = LAB_COLORS[r.spec.lab_name]
        ax.scatter(
            r.protein_vals,
            r.lab_vals,
            s=8,
            alpha=0.5,
            color=color,
            edgecolors="white",
            linewidths=0.3,
        )

        x_fit = np.linspace(r.protein_vals.min(), r.protein_vals.max(), 100)
        y_fit = 10 ** (r.slope * x_fit + r.intercept)
        ax.plot(x_fit, y_fit, color="black", alpha=0.5, lw=1.0)

        ax.set_yscale("log")
        if i in bottom_row_indices:
            ax.set_xlabel(x_label, fontsize=FS_AUX)
        else:
            ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=5)

        # Lab name in bottom-right
        ax.text(
            0.96,
            0.04,
            r.spec.lab_name,
            transform=ax.transAxes,
            fontsize=FS_AUX,
            fontweight="bold",
            va="bottom",
            ha="right",
            color="#444444",
        )

        # Correlation annotation top-left
        ax.text(
            0.04,
            0.96,
            f"ρ = {r.rho:.2f}\nn = {r.n}",
            transform=ax.transAxes,
            fontsize=FS_AUX,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
        )


def make_figure():
    init_figure_theme()

    # --- Load data ---
    olink_subjects = _load_baseline_subjects(OLINK_SAMPLE_QC, "SampleID")
    soma_subjects = _load_baseline_subjects(SOMA_SAMPLE_QC, "SampleId")

    olink_assay_ids = [p.assay_id for p in OLINK_PAIRS]
    soma_assay_ids = [p.assay_id for p in SOMA_PAIRS]

    olink_prot = _load_proteomics(
        OLINK_NPX,
        "SampleID",
        "OlinkID",
        "PCNormalizedNPX",
        olink_subjects,
        olink_assay_ids,
    )
    soma_prot = _load_proteomics(
        SOMA_RFU, "SampleId", "SeqId", "RFU", soma_subjects, soma_assay_ids
    )

    all_paramcds = list(dict.fromkeys(p.paramcd for p in OLINK_PAIRS + SOMA_PAIRS))
    lab_df = _load_lab_values(all_paramcds)

    # --- Compute correlations ---
    olink_results = compute_correlations(
        OLINK_PAIRS,
        olink_prot,
        lab_df,
        "OlinkID",
        "PCNormalizedNPX",
        log2_transform=False,
    )
    soma_results = compute_correlations(
        SOMA_PAIRS, soma_prot, lab_df, "SeqId", "RFU", log2_transform=True
    )

    olink_top = olink_results[:8]
    soma_top = soma_results[:13]

    # --- Create figure: 5 rows × 5 cols, square panels (aspect=1) ---
    # Olink occupies rows 0-1 (2 rows × 5 cols = 10 cells, 8 used).
    # SomaScan occupies rows 2-4 (3 rows × 5 cols = 15 cells, 13 used).
    # Enlarged hspace between rows 1 and 2 keeps the SomaScan banner well clear
    # of the Olink bottom-row x-tick labels and "NPX" axis label.
    fig, axs = uplt.subplots(
        nrows=5,
        ncols=5,
        figwidth=NAT_W2,
        figheight=9.05,
        hspace=("0.45in", "0.95in", "0.45in", "0.45in"),
        wspace=("0.24in",) * 4,
        left="0.25in",
        right="0.1in",
        top="0.55in",
        bottom="0.3in",
        share=False,
        aspect=1,
    )
    axs.format(abc=False)

    # Olink: rows 0-1, all 5 cols (10 cells; first 8 used, last 2 hidden)
    olink_axs = [axs[r * 5 + col] for r in range(2) for col in range(5)]
    # SomaScan: rows 2-4, all 5 cols (15 cells; first 13 used, last 2 hidden)
    soma_axs = [axs[r * 5 + col] for r in range(2, 5) for col in range(5)]

    # Olink bottom row = indices 5..7 (cells 8 and 9 are hidden, so 5,6,7 are
    # the visible bottom-most). SomaScan bottom-most visible per column:
    # cols 0-2 have 3 rows used (indices 10, 11, 12); cols 3-4 have 2 rows used
    # (indices 8, 9).
    olink_bottom = {5, 6, 7}
    soma_bottom = {8, 9, 10, 11, 12}

    _plot_panel(olink_axs, olink_top, "NPX", olink_bottom)
    _plot_panel(soma_axs, soma_top, "log₂(RFU)", soma_bottom)

    # --- Platform banners ---
    _add_platform_banner(fig, axs[0], axs[4], "Olink")
    _add_platform_banner(fig, axs[10], axs[14], "SomaScan")

    # --- Panel labels above platform banners ---
    bb_olink_l = axs[0].get_position()
    bb_soma_l = axs[10].get_position()
    olink_label_y = (
        bb_olink_l.y1 + 0.022 + BANNER_HEIGHT_IN / fig.get_figheight() + 0.006
    )
    soma_label_y = (
        bb_soma_l.y1 + 0.022 + BANNER_HEIGHT_IN / fig.get_figheight() + 0.006
    )
    fig.text(
        bb_olink_l.x0 - 0.01,
        olink_label_y,
        "a",
        fontsize=FS_PANEL,
        fontweight="bold",
        va="bottom",
        ha="right",
    )
    fig.text(
        bb_soma_l.x0 - 0.01,
        soma_label_y,
        "b",
        fontsize=FS_PANEL,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    # --- Subplot banners ---
    for i, r in enumerate(olink_top):
        _add_subplot_banner(fig, olink_axs[i], r.spec.gene, r.spec.assay_id)

    for i, r in enumerate(soma_top):
        _add_subplot_banner(fig, soma_axs[i], r.spec.gene, r.spec.assay_id)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    out_path = Path(__file__).resolve().parent / "supp_fig3_biomarker_correlation"
    save_figure(fig, str(out_path), formats=("pdf",))
