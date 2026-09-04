"""GIPR vs GLP1R cell-type specificity in pancreas (supplementary figure).

Single-panel dot plot of incretin-receptor expression across ten broad
pancreatic cell types from the Human Hormone Cell Atlas (Fei, Huang-Doran et
al., Science 2026; doi:10.1126/science.aeb2672). The panel anchors the
manuscript's PPY-as-GIP-marker hypothesis by showing that:

  * GIPR is expressed broadly across pancreatic endocrine cells (α, β, γ, δ)
    but not appreciably in exocrine or stromal populations
  * γ (PP) cells have the highest GIPR mean expression of any pancreatic
    cell type and the highest fraction of GIPR+ cells
  * GLP1R is restricted to β and δ cells; γ-cell GLP1R is essentially absent
  * Acinar cells show a small but distinct GLP1R signal (no GIPR)

Encoding
--------
* dot SIZE  = % of cells with non-zero counts in the layer
* dot COLOR = mean log1p-normalized expression
* per-gene colormaps (Reds for GIPR, Blues for GLP1R) and independent
  normalizations because the two genes' dynamic ranges differ by ~8x; a
  shared scale would crush the GLP1R column to white.

Data source
-----------
``data/external/hormone_atlas_pancreas_gpcr.parquet`` — recomputed by
``analysis/external/recompute_hormone_atlas_pancreas.py`` from the published
``Pancreas_annotated.h5ad`` (122,020 normal pancreas cells), aggregated over
the atlas's broad ``celltype_level1`` annotations.

Normalization
-------------
The atlas h5ad's ``X`` is log1p-normalized counts: per-cell UMI counts scaled
to a constant cell-total (counts-per-N, the scanpy default), then natural-log
transformed via ``log(1 + x)``. Per-cell-type means are arithmetic means of
those log1p values across all cells (including zero cells). "% expressing" is
the fraction of cells with non-zero counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _internal._common import (  # noqa: E402
    FS_AUX,
    FS_AXLABEL,
    FS_BODY,
    FS_NARR,
    NAT_W15B,
    c,
    init_figure_theme,
    nature_figsize,
    save_figure,
)

DATA_PATH = THIS_DIR.parent / "data" / "external" / "hormone_atlas_pancreas_gpcr.parquet"

# ---------------------------------------------------------------------------
# Display order. Endocrine α/β/γ/δ first, then exocrine and stromal
# populations. Cell-type ids match ``celltype_level1`` in the source h5ad.
# ---------------------------------------------------------------------------
PANCREAS_ORDER = [
    ("alpha_cell", "α (alpha)"),
    ("gamma_cell", "γ (PP / gamma)"),
    ("beta_cell", "β (beta)"),
    ("delta_cell", "δ (delta)"),
    ("ductal_epithelial_cell", "Ductal"),
    ("acinar_cell_idling", "Acinar (idling)"),
    ("acinar_cell_secretory", "Acinar (secretory)"),
    ("stellate_cell", "Stellate"),
    ("endothelial_cell", "Endothelial"),
    ("macrophage", "Macrophage"),
]
GAMMA_LABEL = "γ (PP / gamma)"

# Color tokens. Per-gene colormaps: Reds for GIPR, Blues for GLP1R; the gamma
# row is highlighted in pale gold.
COLOR_HIGHLIGHT = c.get("highlight", "#F4B400")


def _load_pancreas_matrix() -> tuple[
    list[tuple[str, str]],
    dict[tuple[str, str], dict],
]:
    """Return (display_order, lookup) where lookup keys are (cell_type, gene)."""
    df = pl.read_parquet(DATA_PATH)
    by_key = {(r["cell_type"], r["gene"]): r for r in df.to_dicts()}
    rows = []
    for ct_id, _ in PANCREAS_ORDER:
        for gene in ("GIPR", "GLP1R"):
            r = by_key.get((ct_id, gene))
            rows.append(
                dict(
                    cell_type=ct_id,
                    gene=gene,
                    pct=r["pct_expressing"] if r else 0.0,
                    mean=r["mean_expr"] if r else 0.0,
                    n=r["n_cells"] if r else 0,
                )
            )
    return PANCREAS_ORDER, {(r["cell_type"], r["gene"]): r for r in rows}


def make_figure() -> plt.Figure:
    """Render the GIPR/GLP1R pancreas dot plot."""
    init_figure_theme()
    order, mat = _load_pancreas_matrix()
    cell_labels = [lbl for _, lbl in order]
    n = len(order)

    # NAT_W15B canvas — leaves room on the right for the size legend +
    # two stacked colorbars without crowding the dot grid.
    fig_w, fig_h = nature_figsize(NAT_W15B, 0.27 * n + 1.0)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.24, 0.13, 0.46, 0.78])

    x_by_gene = {"GIPR": 0.0, "GLP1R": 1.0}

    # Dot-size scaling: small floor so even ~0.1% rows are visible, with a
    # 30% reference cap so the gamma/beta dots still fit visually.
    SIZE_MIN, SIZE_MAX = 6.0, 220.0
    PCT_REF = 30.0

    def _size(pct: float) -> float:
        if not np.isfinite(pct) or pct <= 0:
            return SIZE_MIN
        return SIZE_MIN + (SIZE_MAX - SIZE_MIN) * min(pct / PCT_REF, 1.0)

    # Per-gene colormap normalization on mean log1p expression. GLP1R's
    # dynamic range across pancreas cells is ~8x smaller than GIPR's, so a
    # shared norm would crush GLP1R to near-white.
    def _vmax(gene: str) -> float:
        vals = [mat[(ct, gene)]["mean"] for ct, _ in order]
        return max(vals) if max(vals) > 0 else 1.0

    norm_g = mcolors.Normalize(vmin=0.0, vmax=_vmax("GIPR"))
    norm_l = mcolors.Normalize(vmin=0.0, vmax=_vmax("GLP1R"))
    cmap_gipr = plt.get_cmap("Reds")
    cmap_glp = plt.get_cmap("Blues")

    for i, (ct_id, _label) in enumerate(order):
        y = n - 1 - i
        for gene in ("GIPR", "GLP1R"):
            x = x_by_gene[gene]
            r = mat[(ct_id, gene)]
            if r["pct"] > 0:
                cmap = cmap_gipr if gene == "GIPR" else cmap_glp
                norm = norm_g if gene == "GIPR" else norm_l
                # Lift the colormap floor a touch so the lowest values don't
                # render white-on-white.
                color = cmap(0.20 + 0.80 * norm(r["mean"]))
                ax.scatter(
                    [x], [y],
                    s=_size(r["pct"]),
                    color=color,
                    edgecolor="#444",
                    linewidth=0.4,
                    zorder=3,
                )
            else:
                ax.scatter(
                    [x], [y], s=SIZE_MIN,
                    facecolor="white",
                    edgecolor="#BBB",
                    linewidth=0.5,
                    zorder=3,
                )
            # Per-dot percentage annotation. Sub-1% values are summarized as
            # "<1%" in light grey so reviewers can see they're real (not
            # missing) without competing with the appreciable-expression rows.
            if r["pct"] >= 1.0:
                pct_text = f"{r['pct']:.1f}%"
                pct_color = "#444"
            elif r["pct"] > 0:
                pct_text = "<1%"
                pct_color = "#AAAAAA"
            else:
                pct_text = "0%"
                pct_color = "#AAAAAA"
            ax.text(
                x + 0.18, y, pct_text,
                fontsize=FS_AUX, va="center", ha="left",
                color=pct_color, zorder=4,
            )

    # Highlight the gamma row.
    gamma_idx = next(i for i, (_, lbl) in enumerate(order) if lbl == GAMMA_LABEL)
    gamma_y = n - 1 - gamma_idx
    ax.axhspan(gamma_y - 0.45, gamma_y + 0.45,
               color=COLOR_HIGHLIGHT, alpha=0.10, zorder=1)

    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(cell_labels)), fontsize=FS_BODY)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["GIPR", "GLP1R"], fontsize=FS_AXLABEL, fontstyle="italic")
    # Extra room on the right of each gene column for the % annotations.
    ax.set_xlim(-0.5, 1.85)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="both", which="both", length=0, pad=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#888")
        ax.spines[spine].set_linewidth(0.6)
    ax.set_title("Pancreas (Hormone Cell Atlas)", fontsize=FS_NARR, pad=4)

    # ---- Right-edge legend stack: size key + two colorbars ----
    # Single column with all three blocks left-aligned. Vertical anchors are
    # pinned absolutely (a previous attempt computed them relative to the
    # size-legend bottom and the GIPR colorbar's max-tick label collided
    # with the lowest "%" row).
    leg_x = 0.78
    label_x = leg_x          # titles flush with the dot / colorbar column
    val_x = leg_x + 0.07     # numeric labels (%, tick value) just to the right

    # 1. Size legend (top of stack).
    leg_pcts = [25, 10, 2]
    size_top_y = 0.88
    size_dy = 0.055
    fig.text(label_x, size_top_y + 0.045, "% expressing",
             fontsize=FS_AUX, va="bottom", ha="left", fontweight="bold")
    for i, p in enumerate(leg_pcts):
        ypos = size_top_y - i * size_dy
        leg_ax = fig.add_axes([leg_x, ypos - 0.025, 0.05, 0.05], frameon=False)
        leg_ax.set_xlim(-1, 1)
        leg_ax.set_ylim(-1, 1)
        leg_ax.set_xticks([])
        leg_ax.set_yticks([])
        for sp in leg_ax.spines.values():
            sp.set_visible(False)
        leg_ax.scatter([0], [0], s=_size(p), color="#888",
                       edgecolor="#444", linewidth=0.4)
        fig.text(val_x, ypos, f"{p}%", fontsize=FS_AUX, va="center")

    # 2. GIPR colorbar (middle).
    cbar_w = 0.018
    cbar_h = 0.16
    gipr_cbar_top = 0.62
    gipr_cbar_y = gipr_cbar_top - cbar_h
    cax_g = fig.add_axes([leg_x, gipr_cbar_y, cbar_w, cbar_h])
    sm_g = mcm.ScalarMappable(norm=norm_g, cmap=cmap_gipr)
    cb_g = fig.colorbar(sm_g, cax=cax_g)
    cb_g.set_label("")
    cb_g.ax.tick_params(labelsize=FS_AUX, length=2, width=0.4, pad=1)
    cb_g.set_ticks([0, _vmax("GIPR")])
    cb_g.ax.set_yticklabels(["0", f"{_vmax('GIPR'):.2f}"])
    cb_g.outline.set_linewidth(0.4)
    fig.text(label_x, gipr_cbar_top + 0.025,
             "GIPR mean expr.", fontsize=FS_AUX, va="bottom", ha="left",
             fontweight="bold")

    # 3. GLP1R colorbar (bottom). Gap below GIPR colorbar must clear the
    # GIPR "0" tick label and leave room above the GLP1R title.
    glp_cbar_top = gipr_cbar_y - 0.08
    glp_cbar_y = glp_cbar_top - cbar_h
    cax_l = fig.add_axes([leg_x, glp_cbar_y, cbar_w, cbar_h])
    sm_l = mcm.ScalarMappable(norm=norm_l, cmap=cmap_glp)
    cb_l = fig.colorbar(sm_l, cax=cax_l)
    cb_l.set_label("")
    cb_l.ax.tick_params(labelsize=FS_AUX, length=2, width=0.4, pad=1)
    cb_l.set_ticks([0, _vmax("GLP1R")])
    cb_l.ax.set_yticklabels(["0", f"{_vmax('GLP1R'):.2f}"])
    cb_l.outline.set_linewidth(0.4)
    fig.text(label_x, glp_cbar_top + 0.025,
             "GLP1R mean expr.", fontsize=FS_AUX, va="bottom", ha="left",
             fontweight="bold")

    return fig


if __name__ == "__main__":
    fig = make_figure()
    out_dir = Path(__file__).resolve().parent
    save_figure(fig, str(out_dir / "supp_fig15_gipr_glp1r_celltype"), formats=("pdf",))
    print("Wrote supp_fig15_gipr_glp1r_celltype.pdf")
