"""Mediation across-treatment figure: TZP 15 mg vs SEMA 2.4 mg.

Row 1: Olink and SomaScan global total-vs-direct scatters at Wk72.
Rows 2-3: paired Olink + SomaScan forest vignettes for the four manuscript
themes retained in the main figure.

Notes
-----
- Olink GCG is retained but renders as a truncated/capped diamond at the
  panel edge — the Olink antibody cross-reacts with GLP-1 so the signal is
  not glucagon biology. Use the SomaScan GCG aptamer for any glucagon claim.
- Markers → HGNC via uniprot_map (build_marker_genes from _volcano_figure.py).
- Counts in panel annotation use across-treatment BH-FDR < 0.05.
- Mediation classification (BH-FDR < 0.05 for both ACME and ADE):
    fully_mediated: ACME FDR<0.05, ADE FDR>=0.05
    partial:        ACME FDR<0.05, ADE FDR<0.05, same sign
    direct_only:    ADE  FDR<0.05, ACME FDR>=0.05
    inconsistent:   ACME FDR<0.05, ADE FDR<0.05, opposite signs
  ACME FDR is pre-computed by the mediation pipeline; ADE FDR is computed
  here via BH on ADE_pvalue, within each platform × visit panel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import ultraplot as uplt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke


class _TextOnlyHandler(HandlerBase):
    """Legend handler that collapses the handle swatch to zero width."""

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        handlebox.width = 0
        handlebox.sep = 0
        handlebox.set_visible(False)
        return []


from ._common import (
    ANALYSIS_DIR,
    BANNER_GAP_IN,
    BANNER_HEIGHT_IN,
    COLOR_NS,
    COVAR,
    FS_AXLABEL,
    FS_BODY,
    FS_DIRECTION,
    FS_FOREST_YLABEL,
    FS_GENE_LABEL,
    FS_LEGEND,
    FS_PILL,
    FS_TICK,
    LILLY_COLORS,
    MMRM_OLINK_DIR,
    MMRM_SOMA_DIR,
    PALETTE_VARIANT,
    RESULTS_DIR,
    build_marker_genes,
    deterministic_adjust_text,
    draw_banner,
    init_figure_theme,
    nature_figsize,
    place_panel_letter,
    save_figure,
)

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[1]
RESULTS = RESULTS_DIR
ANALYSIS = ANALYSIS_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MED_FILES = {
    "olink": MMRM_OLINK_DIR / "mediationRes" / f"surmount5_{COVAR}_mediation_PCTCHG_WGT.csv",
    "soma": MMRM_SOMA_DIR / "mediationRes" / f"surmount5_{COVAR}_mediation_PCTCHG_WGT.csv",
}

AC_FILES = {
    (plat, wk): RESULTS
    / f"surmount5_{plat}"
    / COVAR
    / "finalRes"
    / "acTrt"
    / f"surmount5_{COVAR}_proteomics_olinkAnalysis_acrossTrts_resCmps_"
    f"TZP15mgorMTDVSSEMA2.4mgorMTD@{wk}_py.csv"
    for plat in ("olink", "soma")
    for wk in (24, 72)
}

VISIT_TO_WEEK = {8: 24, 20: 72}
WEEK_TO_VISIT = {v: k for k, v in VISIT_TO_WEEK.items()}

FDR_THRESHOLD = 0.05  # across-treatment significance (volcano-figure convention)
ACME_FDR_THRESHOLD = 0.05  # mediation indirect-effect BH-FDR
ADE_FDR_THRESHOLD = (
    0.05  # mediation direct-effect BH-FDR (computed in classify_mediation)
)

# Forest themes — 8 themes in a 4×2 grid, organized by logical biology.
# Row 1 covers circulating hormones and pancreatic compartments; row 2
# covers lipid-handling and tissue/cell signatures. Insertion order matters:
# it drives row placement.
MAIN_THEME_LABELS = {
    # Row 1 — adipose / endocrine / digestive
    "Leptin & adiponectin": "Leptin &\nadiponectin",
    "GH–IGF axis": "GH–IGF axis",
    "Pancreatic exocrine": "Pancreatic\nexocrine",
    "Islet hormones": "Islet hormones",
    # Row 2 — tissue signatures / lipid transport
    "Activin & myogenesis": "Activin &\nmyogenesis",
    "Intestinal epithelium": "Intestinal\nepithelium",
    "Vascular": "Vascular",
    "Apolipoproteins & lipid handling": "Apolipoproteins\n& lipid handling",
}

MAIN_THEME_ROWS = (
    ("Leptin & adiponectin", "Activin & myogenesis"),
    ("Islet hormones", "Intestinal epithelium"),
)
MAIN_THEME_ORDER = tuple(name for row in MAIN_THEME_ROWS for name in row)

MAIN_THEMES = {
    # Leptin & adiponectin — LEP/LEPR/LEPROT are the leptin receptor axis,
    # ADIPOQ is the partnered adipokine; all four sig on at least one
    # platform at Wk72.
    "Leptin & adiponectin": ["LEP", "LEPR", "LEPROT", "ADIPOQ"],
    # GH-IGF axis — core hormone/receptor (GH1, GHR, IGF1, IGF1R) and the
    # IGFBP family + IGFALS (acid-labile subunit) + SHBG (carrier).
    "GH–IGF axis": [
        "GH1",
        "GHR",
        "IGF1",
        "IGF1R",
        "IGFBP1",
        "IGFBP2",
        "IGFBP3",
        "IGFBP4",
        "IGFBP5",
        "IGFBP6",
        "IGFALS",
        "SHBG",
    ],
    # Pancreatic exocrine — exocrine-specific REG family (REG3A, REG1A) plus
    # acinar zymogens: carboxypeptidases (CPA1, CPB1), chymotrypsins (CTRC,
    # CTRB1, CTRB2) and chymotrypsin-like CTRL, elastases (CELA1, CELA2A,
    # CELA3A), trypsins (PRSS1, PRSS2), trypsin inhibitor (SPINK1), colipase
    # (CLPS), cholesterol esterase (CEL) and lipases (PNLIP, PNLIPRP1,
    # PLA2G1B). CTRL is just under the FDR<0.05 threshold at Wk72 (fdr≈0.075,
    # negative effect) but is biologically core to the panel.
    "Pancreatic exocrine": [
        "REG3A",
        "REG1A",
        "CPA1",
        "CPB1",
        "CTRC",
        "CTRB1",
        "CTRB2",
        "CTRL",
        "CELA1",
        "CELA2A",
        "CELA3A",
        "PRSS1",
        "PRSS2",
        "SPINK1",
        "CLPS",
        "CEL",
        "PNLIP",
        "PNLIPRP1",
        "PLA2G1B",
    ],
    # Islet hormones — islet hormones, secretogranins, islet TFs.
    "Islet hormones": [
        "PPY",
        "CHGA",
        "CHGB",
        "SCG2",
        "SCG3",
        "PTPRN",
        "PTPRN2",
        "INS",
        "CPE",
        "ISL1",
        "SOX4",
    ],
    # Apolipoproteins & lipid handling — apolipoproteins, lipases,
    # lipid-handling receptors. TZP-vs-SEMA differential lipid effects
    # (APOD, APOF, APOL1, APOM, LIPG, LRP1 all sig at Wk72) form a
    # clinically meaningful cardiometabolic signal beyond weight change.
    "Apolipoproteins & lipid handling": [
        "APOB",
        "APOE",
        "APOA1",
        "APOA2",
        "APOA4",
        "APOC1",
        "APOC2",
        "APOC3",
        "APOM",
        "APOD",
        "APOL1",
        "APOF",
        "LIPG",
        "LPL",
        "LCAT",
        "CETP",
        "LIPC",
        "LIPA",
        "LDLR",
        "LRP1",
        "SCARB1",
        "PCSK9",
        "ABCA1",
        "ANGPTL3",
        "ANGPTL4",
        "ANGPTL8",
        "GPIHBP1",
        "LDLRAP1",
    ],
    # Intestinal epithelium — brush-border / epithelial markers with signal
    # in the data: epithelial adhesion (EPCAM, GPA33, CDH17, CDHR5),
    # brush-border enzymes (TREH, ANPEP, ALPI, DPP4, TMPRSS15), the cytoskeletal
    # marker VIL1, the Fc receptor FCAMR, the apical-membrane enterocyte marker
    # MAMDC4, and REG4 (regenerating-islet-derived 4 — a GI-tract / intestinal
    # goblet-cell marker, despite the "REG" name).
    "Intestinal epithelium": [
        "EPCAM",
        "GPA33",
        "TREH",
        "FCAMR",
        "MAMDC4",
        "REG4",
        "ANPEP",
        "VIL1",
        "ALPI",
        "DPP4",
        "TMPRSS15",
        "CDH17",
        "CDHR5",
    ],
    # Activin & myogenesis — myogenic TF (MYF5), myogenic-cell receptor
    # (NEO1), the Wnt antagonist DKK3, and the activin / myostatin /
    # follistatin family (MSTN, GDF11, FST, FSTL1, FSTL3, WFIKKN2, INHBA,
    # INHBB, INHBC, INHBE, INHA, ACVR2A, ACVR2B) that drives skeletal-muscle
    # mass regulation.
    "Activin & myogenesis": [
        "MYF5",
        "NEO1",
        "MSTN",
        "GDF11",
        "DKK3",
        "FST",
        "FSTL1",
        "FSTL3",
        "WFIKKN2",
        "INHBA",
        "INHBB",
        "INHBC",
        "INHBE",
        "INHA",
        "ACVR2A",
        "ACVR2B",
    ],
    # Vascular biology — endothelial markers (CLEC14A, ESM1, MCAM),
    # VEGF axis (VEGFB, VEGFD, NRP1), vascular adhesion (SELE),
    # angiopoietin axis (ANGPT2, ANGPTL2), RAAS (ACE2), endothelin (EDN1),
    # and ephrin/chemokine/guidance signalling (EFNB2, CXCL12, NOTCH3, ROBO1).
    "Vascular": [
        "VEGFB",
        "VEGFD",
        "NRP1",
        "ANGPT2",
        "ANGPTL2",
        "CXCL12",
        "EFNB2",
        "CLEC14A",
        "ESM1",
        "MCAM",
        "SELE",
        "EDN1",
        "ACE2",
        "NOTCH3",
        "ROBO1",
    ],
}

N_PROTEINS_PER_THEME = 7
N_ANNOT_ROW1 = 20  # genes to label in each Row 1 panel (split half up / half down)

# ---- Scatter label budget (per direction; doubled across up/down) ----------
# `N_VOLCANO_PER_DIR` is the dominant knob — headline differential genes ranked
# by |total| × −log10(ac_fdr). The per-class supplements guarantee each
# mediation class keeps representation even when its members miss the volcano
# cutoffs. Lowering these reduces scatter clutter; the union is de-duplicated so
# the realised label count is below the raw sum.
N_VOLCANO_PER_DIR = 12
N_SUPPRESSION_PER_DIR = 4
N_DIRECT_PER_DIR = 3
N_MEDIATED_PER_DIR = 3


# Colors (resolved at theme time).
# Scatter dots: hue encodes direction (navy = TZP higher, grey = SEMA higher),
# saturation encodes mediation class:
#   direct_only  : vivid (full navy / full grey) - solid fill
#   partial      : 35% fade toward white - solid fill
#   fully_mediated: 60% fade toward white - solid fill
#   inconsistent : vibrant gold - filled (rare; ACME and ADE both
#                  significant but opposite signs — the mediator cancels
#                  part of the direct effect)
#   ns/unresolved: neutral grey (across-treatment sig but neither
#                  ACME nor ADE individually significant)
def _fade(hex_color: str, frac: float) -> str:
    """Fade a hex color toward white by `frac` (0=no fade, 1=pure white)."""
    import matplotlib.colors as mcolors

    rgb = mcolors.to_rgb(hex_color)
    faded = tuple(c + (1.0 - c) * frac for c in rgb)
    return mcolors.to_hex(faded)


def _palette():
    c = LILLY_COLORS[PALETTE_VARIANT]
    # Manuscript arm colors: TZP = dark navy, SEMA = medium grey.
    TZP = "#0C376D"
    SEMA = "#5A5A5A"
    # Three intensity tiers per direction via programmatic fade toward white:
    #   direct        = full saturation (TZP navy / SEMA grey)
    #   partial       = 35 % fade toward white
    #   fully_mediated = 60 % fade toward white
    # The fade fractions keep each tier visibly distinct from both the NS
    # background and from each other.
    return {
        "ns": COLOR_NS,
        "direct_pos": TZP,
        "direct_neg": SEMA,
        "partial_pos": _fade(TZP, 0.35),
        "partial_neg": _fade(SEMA, 0.35),
        "fm_pos_fill": _fade(TZP, 0.60),
        "fm_neg_fill": _fade(SEMA, 0.60),
        # Unresolved: across-treatment significant but neither ACME nor
        # ADE individually reaches FDR < 0.05.  Single neutral colour
        # (no direction encoding — the decomposition is inconclusive).
        "unresolved": "#B8B8B8",
        # Inconsistent: rare (1–5 points/panel) — ACME and ADE both
        # significant but opposite signs.  `bold_grey` blended into the
        # NS cloud, so use `vibrant_gold` to stand out.
        "inconsistent": c["vibrant_gold"],
        "diagonal": c["black"],
        "axline": c["black"],
        "olink": TZP,
        "soma": SEMA,
        "total": c["black"],
        "direct": TZP,
        "mediated_gap": _fade(TZP, 0.35),
        # legend-name aliases (kept for forest-plot legends elsewhere)
        "partial": _fade(TZP, 0.35),
        "fully_mediated": _fade(TZP, 0.60),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_mediation(platform: str) -> pl.DataFrame:
    """Load mediation CSV, annotate with HGNC gene_label.

    Olink GCG is retained but is a known GLP-1 cross-reactivity artifact
    (the PEA antibody binds GLP-1, not glucagon). The scatter cap-detection
    machinery will render GCG as a truncated diamond at the panel edge so
    the reader can see the off-scale extreme, but no glucagon biological
    narrative should be built around the Olink GCG hit — defer to the
    SomaScan GCG aptamer for any glucagon claim.
    """
    mg = build_marker_genes()
    df = pl.read_csv(MED_FILES[platform])
    df = df.with_columns(
        pl.col("marker")
        .map_elements(lambda m: mg.get(m, []), return_dtype=pl.List(pl.Utf8))
        .alias("genes"),
    ).with_columns(
        pl.col("genes").list.join("|").alias("gene_label"),
        pl.lit(platform).alias("platform"),
    )
    return df


def load_across_treatment(platform: str, week: int) -> pl.DataFrame:
    mg = build_marker_genes()
    df = pl.read_csv(AC_FILES[(platform, week)])
    df = df.with_columns(
        pl.col("marker")
        .map_elements(lambda m: mg.get(m, []), return_dtype=pl.List(pl.Utf8))
        .alias("genes"),
    ).with_columns(
        pl.col("genes").list.join("|").alias("gene_label"),
        pl.lit(platform).alias("platform"),
        pl.lit(week).alias("week"),
    )
    return df


def detect_cap(
    vals: np.ndarray, ratio: float = 2.0
) -> tuple[float | None, float | None]:
    """Detect outliers warranting truncation (volcano-figure convention).

    Returns ``(cap_hi, cap_lo)``. Walks the sorted values inward from each
    extreme: on the positive side, while the current extreme exceeds the
    next value by ≥ ``ratio``, extends the cap inward (so multiple stacked
    outliers — e.g. Olink Wk24 with GCG at ADE ≈ −3.79 AND GH1 at ADE ≈
    −0.85 — all end up capped against the same reference value just outside
    the bulk of the non-outlier cloud). The cap sits at 1.2× the first
    "well-behaved" value so truncated markers land clearly outside the
    main cloud but inside the panel.

    Volcano uses ratio=3 for log2FC / −log10FDR space; here we use 2 because
    mediation effect-size magnitudes are smaller and a single point dragging
    the axis 60%+ is visually disruptive.
    """
    if len(vals) < 3:
        return (None, None)
    sv = np.sort(vals)

    # Positive side: walk from sv[-1] inward.
    cap_hi: float | None = None
    for k in range(len(sv) - 1, 0, -1):
        current = sv[k]
        nxt = sv[k - 1]
        if current > 0 and nxt > 0 and current > ratio * nxt:
            cap_hi = nxt * 1.2
        else:
            break

    # Negative side: walk from sv[0] outward.
    cap_lo: float | None = None
    for k in range(len(sv) - 1):
        current = sv[k]
        nxt = sv[k + 1]
        if current < 0 and nxt < 0 and current < ratio * nxt:
            cap_lo = nxt * 1.2
        else:
            break

    return (cap_hi, cap_lo)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR adjustment. Returns adjusted q-values, same
    shape as input. NaNs in pvals propagate as NaN q-values (excluded from
    the multiple-testing pool)."""
    p = np.asarray(pvals, dtype=float)
    mask = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    if mask.sum() == 0:
        return out
    p_valid = p[mask]
    n = p_valid.size
    order = np.argsort(p_valid)
    ranks = np.arange(1, n + 1, dtype=float)
    raw = p_valid[order] * n / ranks
    # Monotone-from-right cumulative minimum
    cummin = np.minimum.accumulate(raw[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(cummin, 1.0)
    out[mask] = q
    return out


def classify_mediation(
    df: pl.DataFrame,
    ac_df: pl.DataFrame | None = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Add a 'class' column to a mediation-results DataFrame.

    Classification uses **FDR < 0.05** for both ACME and ADE — matches the
    across-treatment volcano-figure standard.

    Both ACME and ADE FDR are computed via Benjamini–Hochberg **restricted
    to proteins with a significant across-treatment effect** (FDR < 0.05),
    because mediation decomposition is only meaningful when there is a
    total treatment effect to partition.  Proteins outside that gate
    receive ``NaN`` FDRs and classify as ``"ns"`` (unresolved).

    When *ac_df* is ``None``, BH is applied across all proteins in *df* and
    the pipeline's pre-computed ``ACME_fdr`` column is left unchanged.

    Parameters
    ----------
    df : pl.DataFrame
        Mediation results for one platform × visit.
    ac_df : pl.DataFrame, optional
        Across-treatment results containing ``marker`` and ``fdr`` columns.
        If given, only markers with ``fdr < 0.05`` at *week* enter the BH
        pool.
    week : int, optional
        Week filter applied to *ac_df* (required when *ac_df* is given).
    """
    if ac_df is not None:
        if week is None:
            raise ValueError("week is required when ac_df is provided")
        sig_markers = set(
            ac_df.filter((pl.col("week") == week) & (pl.col("fdr") < FDR_THRESHOLD))[
                "marker"
            ].to_list()
        )
        is_sig = df["marker"].is_in(sig_markers)
        sig_idx = np.where(is_sig.to_numpy())[0]
        sig_sub = df.filter(is_sig)
        # BH restricted to across-treatment significant proteins
        ade_q_sig = _bh_fdr(sig_sub["ADE_pvalue"].to_numpy())
        acme_q_sig = _bh_fdr(sig_sub["ACME_pvalue"].to_numpy())
        # Full-length columns: NaN for non-significant proteins
        ade_fdr_full = np.full(df.height, np.nan)
        acme_fdr_full = np.full(df.height, np.nan)
        ade_fdr_full[sig_idx] = ade_q_sig
        acme_fdr_full[sig_idx] = acme_q_sig
        df = df.with_columns(
            pl.Series("ADE_fdr", ade_fdr_full),
            pl.Series("ACME_fdr", acme_fdr_full),
        )
    else:
        ade_q = _bh_fdr(df["ADE_pvalue"].to_numpy())
        df = df.with_columns(pl.Series("ADE_fdr", ade_q))
    cls = (
        pl.when(
            (pl.col("ACME_fdr") < ACME_FDR_THRESHOLD)
            & (pl.col("ADE_fdr") < ADE_FDR_THRESHOLD)
            & (pl.col("ACME_estimate") * pl.col("ADE_estimate") < 0)
        )
        .then(pl.lit("inconsistent"))
        .when(
            (pl.col("ACME_fdr") < ACME_FDR_THRESHOLD)
            & (pl.col("ADE_fdr") < ADE_FDR_THRESHOLD)
        )
        .then(pl.lit("partial"))
        .when(
            (pl.col("ACME_fdr") < ACME_FDR_THRESHOLD)
            & (pl.col("ADE_fdr") >= ADE_FDR_THRESHOLD)
        )
        .then(pl.lit("fully_mediated"))
        .when(
            (pl.col("ADE_fdr") < ADE_FDR_THRESHOLD)
            & (pl.col("ACME_fdr") >= ACME_FDR_THRESHOLD)
        )
        .then(pl.lit("direct_only"))
        .otherwise(pl.lit("ns"))
    )
    return df.with_columns(cls.alias("class"))


# ---------------------------------------------------------------------------
# Row 1: global total-vs-direct scatters
# ---------------------------------------------------------------------------


def plot_global_scatter(
    ax,
    med_df: pl.DataFrame,
    ac_df: pl.DataFrame,
    week: int,
    platform: str,
    show_ylabel: bool = True,
) -> dict[str, int]:
    """Total-vs-direct effect scatter for one (platform, week) combo.

    Returns a counts dict for caption annotation.
    """
    pal = _palette()
    visit = WEEK_TO_VISIT[week]
    med = classify_mediation(
        med_df.filter(pl.col("visit") == visit),
        ac_df=ac_df,
        week=week,
    )
    # Drop rows with NaN total or ADE — these would break adjustText's KDTree
    # when passed via the `objects` parameter.
    med = med.filter(
        pl.col("total_estimate").is_finite() & pl.col("ADE_estimate").is_finite()
    )

    # Join with across-treatment for FDR<0.05 flag (same week)
    ac_sig_df = (
        ac_df.filter(pl.col("week") == week)
        .select(["marker", "fdr"])
        .rename({"fdr": "ac_fdr"})
    )
    plot_df = med.join(ac_sig_df, on="marker", how="left").with_columns(
        pl.col("ac_fdr").fill_null(1.0)
    )
    plot_df = plot_df.with_columns((pl.col("ac_fdr") < FDR_THRESHOLD).alias("ac_sig"))
    plot_df = plot_df.sort(["marker", "total_estimate", "ADE_estimate"])

    # ----- Outlier capping (volcano-figure convention) -----
    # Detect caps from significant points only so a single NS extreme can't
    # trigger capping. Then mark every point exceeding either cap, store
    # original values, and compute truncated display positions.
    sig_only_df = plot_df.filter(pl.col("ac_sig"))
    if sig_only_df.height >= 3:
        x_hi, x_lo = detect_cap(sig_only_df["total_estimate"].to_numpy())
        y_hi, y_lo = detect_cap(sig_only_df["ADE_estimate"].to_numpy())
    else:
        x_hi = x_lo = y_hi = y_lo = None

    cap_x_expr: pl.Expr = pl.lit(False)
    if x_hi is not None:
        cap_x_expr = cap_x_expr | (pl.col("total_estimate") > x_hi)
    if x_lo is not None:
        cap_x_expr = cap_x_expr | (pl.col("total_estimate") < x_lo)
    cap_y_expr: pl.Expr = pl.lit(False)
    if y_hi is not None:
        cap_y_expr = cap_y_expr | (pl.col("ADE_estimate") > y_hi)
    if y_lo is not None:
        cap_y_expr = cap_y_expr | (pl.col("ADE_estimate") < y_lo)

    plot_df = plot_df.with_columns(
        [
            pl.col("total_estimate").alias("_orig_total"),
            pl.col("ADE_estimate").alias("_orig_ADE"),
            cap_x_expr.alias("_cap_x"),
            cap_y_expr.alias("_cap_y"),
        ]
    ).with_columns((pl.col("_cap_x") | pl.col("_cap_y")).alias("_capped"))

    # Axis span — 8 % padding around the most extreme SIGNIFICANT non-capped
    # point. NS points are excluded: they're shown as a grey background and
    # their counts appear in the legend, but they shouldn't compress the
    # main visualization. The tight padding minimises empty space (Wk72
    # panels were ~30-60 % empty in y when |max sig y| << |max sig x|).
    # Label-vs-direction-cue collisions at the top edge (e.g. LDLRAP1 on
    # Olink Wk24) are handled by flipping `va="top"` for anchors in the top
    # 15 % of the panel — see the label-placement loop below.
    sig_non_capped = plot_df.filter(pl.col("ac_sig") & ~pl.col("_capped"))
    if sig_non_capped.height > 0:
        nc_x = float(sig_non_capped["total_estimate"].abs().max() or 0.0)
        nc_y = float(sig_non_capped["ADE_estimate"].abs().max() or 0.0)
        span = max(nc_x, nc_y, 0.05) * 1.08
    else:
        # Fallback: no sig non-capped points (rare). Use all non-capped.
        non_capped = plot_df.filter(~pl.col("_capped"))
        if non_capped.height > 0:
            nc_x = float(non_capped["total_estimate"].abs().max() or 0.0)
            nc_y = float(non_capped["ADE_estimate"].abs().max() or 0.0)
            span = max(nc_x, nc_y, 0.05) * 1.08
        else:
            span = (
                max(
                    abs(plot_df["total_estimate"].max() or 0.0),
                    abs(plot_df["total_estimate"].min() or 0.0),
                    0.1,
                )
                * 1.08
            )

    # Truncate capped points to 85 % of the axis range, preserving sign.
    # 85 % (vs the volcano's tighter 97 %) keeps the marker visibly displaced
    # from the corner annotations (↗ TZP > SEMA, ↙ SEMA > TZP, counts) but still close
    # to its true position — i.e. just enough to clear the corner labels.
    EDGE = span * 0.85
    disp_total = (
        pl.when(pl.col("_cap_x") & (pl.col("_orig_total") > 0))
        .then(pl.lit(EDGE))
        .when(pl.col("_cap_x") & (pl.col("_orig_total") < 0))
        .then(pl.lit(-EDGE))
        .otherwise(pl.col("total_estimate"))
    )
    disp_ADE = (
        pl.when(pl.col("_cap_y") & (pl.col("_orig_ADE") > 0))
        .then(pl.lit(EDGE))
        .when(pl.col("_cap_y") & (pl.col("_orig_ADE") < 0))
        .then(pl.lit(-EDGE))
        .otherwise(pl.col("ADE_estimate"))
    )
    plot_df = plot_df.with_columns(
        [
            disp_total.alias("_disp_total"),
            disp_ADE.alias("_disp_ADE"),
        ]
    )

    # Background NS points removed for visual clarity — only significant
    # (across-treatment FDR < 0.05) points are rendered.

    # Significant points — coloured by (class × direction).
    # Filled circles for direct & partial, open circles for fully-mediated.
    sig_nc = plot_df.filter(pl.col("ac_sig") & ~pl.col("_capped"))

    DOT_SIZE = 18
    DOT_EDGE = 0.4

    # Collect all scatter PathCollections so adjust_text can avoid them.
    scatter_objects: list = []

    def _draw_group(sub_df, *, facecolor, edgecolor, edgewidth, zorder=3):
        if sub_df.height == 0:
            return
        sc = ax.scatter(
            sub_df["_disp_total"].to_numpy(),
            sub_df["_disp_ADE"].to_numpy(),
            s=DOT_SIZE,
            facecolors=facecolor,
            edgecolors=edgecolor,
            lw=edgewidth,
            zorder=zorder,
        )
        scatter_objects.append(sc)

    # Plot z-order: direct > partial > mediated > unresolved > inconsistent.
    # Draw lowest-priority first so highest-priority lands on top in overlaps.

    # unresolved — neutral grey (weakest tier, zorder=2)
    _draw_group(
        sig_nc.filter(pl.col("class") == "ns"),
        facecolor=pal["unresolved"],
        edgecolor="white",
        edgewidth=DOT_EDGE,
        zorder=2,
    )
    # mediated — pale tinted fill, white edge (de-emphasized vs direct/partial)
    _draw_group(
        sig_nc.filter(
            (pl.col("class") == "fully_mediated") & (pl.col("total_estimate") > 0)
        ),
        facecolor=pal["fm_pos_fill"],
        edgecolor="white",
        edgewidth=DOT_EDGE,
        zorder=3,
    )
    _draw_group(
        sig_nc.filter(
            (pl.col("class") == "fully_mediated") & (pl.col("total_estimate") <= 0)
        ),
        facecolor=pal["fm_neg_fill"],
        edgecolor="white",
        edgewidth=DOT_EDGE,
        zorder=3,
    )
    # inconsistent — gold filled (rare; ACME and ADE opposite signs)
    _draw_group(
        sig_nc.filter(pl.col("class") == "inconsistent"),
        facecolor=pal["inconsistent"],
        edgecolor="white",
        edgewidth=DOT_EDGE,
        zorder=3,
    )
    # partial — light hue, white edge
    _draw_group(
        sig_nc.filter((pl.col("class") == "partial") & (pl.col("total_estimate") > 0)),
        facecolor=pal["partial_pos"],
        edgecolor="white",
        edgewidth=DOT_EDGE,
        zorder=4,
    )
    _draw_group(
        sig_nc.filter((pl.col("class") == "partial") & (pl.col("total_estimate") <= 0)),
        facecolor=pal["partial_neg"],
        edgecolor="white",
        edgewidth=DOT_EDGE,
        zorder=4,
    )
    # direct — vivid hue, white edge (drawn LAST so it lands on top of overlaps)
    _draw_group(
        sig_nc.filter(
            (pl.col("class") == "direct_only") & (pl.col("total_estimate") > 0)
        ),
        facecolor=pal["direct_pos"],
        edgecolor="white",
        edgewidth=DOT_EDGE,
        zorder=5,
    )
    _draw_group(
        sig_nc.filter(
            (pl.col("class") == "direct_only") & (pl.col("total_estimate") <= 0)
        ),
        facecolor=pal["direct_neg"],
        edgecolor="white",
        edgewidth=DOT_EDGE,
        zorder=5,
    )

    # Count summary — uses all sig (including capped).
    all_sig = plot_df.filter(pl.col("ac_sig"))
    counts: dict[str, int] = {
        cls: all_sig.filter(pl.col("class") == cls).height
        for cls in ("fully_mediated", "partial", "direct_only", "inconsistent", "ns")
    }

    # Capped points — directional triangle / diamond markers sized larger
    # so they read as "truncated, beyond axis" vs. regular dots. Class
    # encoding mirrors the non-capped scheme above. NS capped points are
    # dropped: their counts are reported in the legend's `NS (n)` entry but
    # we don't draw grey diamonds at the panel edge, since those compete
    # with the significant capped markers for attention.
    capped = plot_df.filter(pl.col("_capped") & pl.col("ac_sig"))
    for r in capped.iter_rows(named=True):
        if r["_cap_x"] and r["_cap_y"]:
            mk = "D"
        elif r["_cap_x"]:
            mk = "<" if r["_orig_total"] < 0 else ">"
        else:
            mk = "v" if r["_orig_ADE"] < 0 else "^"
        pos = r["total_estimate"] > 0
        if r["class"] == "inconsistent":
            face, edge, ew = pal["inconsistent"], "white", DOT_EDGE
        elif r["class"] == "ns":
            face, edge, ew = pal["unresolved"], "white", DOT_EDGE
        elif r["class"] == "fully_mediated":
            face = pal["fm_pos_fill"] if pos else pal["fm_neg_fill"]
            edge, ew = "white", DOT_EDGE
        elif r["class"] == "partial":
            face = pal["partial_pos"] if pos else pal["partial_neg"]
            edge, ew = "white", DOT_EDGE
        else:  # direct_only
            face = pal["direct_pos"] if pos else pal["direct_neg"]
            edge, ew = "white", DOT_EDGE
        ax.scatter(
            [r["_disp_total"]],
            [r["_disp_ADE"]],
            marker=mk,
            s=DOT_SIZE * 2.2,
            facecolors=face,
            edgecolors=edge,
            lw=ew,
            zorder=5,
        )

    # Reference lines: solid diagonal (y=x), solid x and y axes.
    # Use the panel-level `span` (40 % padded) so the diagonal stretches
    # corner-to-corner regardless of where extreme points land.
    ax.plot(
        [-span, span], [-span, span], color=pal["diagonal"], lw=0.6, alpha=0.7, zorder=2
    )
    ax.axhline(0, color=pal["axline"], lw=0.4, alpha=0.5, zorder=2)
    ax.axvline(0, color=pal["axline"], lw=0.4, alpha=0.5, zorder=2)

    # Create direction-cue texts NOW (before adjust_text) so they can be
    # passed as obstacles to avoid. Without this, near-edge gene labels
    # (e.g. LDLRAP1 on Olink Wk24, ADE≈0.71 ≈ 92 % of span) drift into the
    # `↗ TZP > SEMA` corner cue because adjust_text doesn't know it
    # exists yet. We also set xlim/ylim here so transAxes coordinates land
    # at the correct positions.
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal", adjustable="box")
    EDGE_MARGIN = 0.008  # 0.8 % from top/bottom border
    DIAG_INSET = 0.04  # offset from right/left edge to clear the diagonal
    halo = [withStroke(linewidth=2.5, foreground="white")]
    cue_tzp = ax.text(
        1 - DIAG_INSET,
        1 - EDGE_MARGIN,
        "↗ TZP > SEMA",
        transform=ax.transAxes,
        fontsize=FS_DIRECTION,
        color=pal["direct_pos"],
        ha="right",
        va="top",
        zorder=6,
        path_effects=halo,
    )
    cue_sema = ax.text(
        DIAG_INSET,
        EDGE_MARGIN,
        "↙ SEMA > TZP",
        transform=ax.transAxes,
        fontsize=FS_DIRECTION,
        color=pal["direct_neg"],
        ha="left",
        va="bottom",
        zorder=6,
        path_effects=halo,
    )

    # ----- Label selection: one label per HGNC gene -----
    # ONLY genes significant in across-treatment (FDR<0.05) are considered —
    # the `sig` filter above guarantees this. Primary ranker is the
    # across-treatment FDR (smallest first) split by direction, mirroring the
    # volcano figure. ADE/ACME extremes layered in afterward as supplements.
    # Capped points are ALWAYS labelled so the reader sees what's truncated.
    # Multi-gene markers contribute one entry per HGNC symbol.
    sig_all = plot_df.filter(pl.col("ac_sig"))  # includes capped + non-capped sig
    gene_points: dict[str, list[dict]] = {}
    for r in sig_all.iter_rows(named=True):
        for gene in r.get("genes") or []:
            gene_points.setdefault(gene, []).append(r)

    def _row_tiebreak(r: dict) -> tuple[str, str, float, float]:
        return (
            str(r.get("marker") or ""),
            "|".join(r.get("genes") or []),
            float(r.get("_orig_total") or 0.0),
            float(r.get("_orig_ADE") or 0.0),
        )

    def _best_by(rows, col, mode="abs"):
        if mode == "min":
            return sorted(rows, key=lambda r: (float(r[col]),) + _row_tiebreak(r))[0]
        return sorted(
            rows,
            key=lambda r: (-abs(float(r[col])),) + _row_tiebreak(r),
        )[0]

    # ----- Class-aware label selection -----
    # Goal: each mediation class with significant signal gets representation,
    # split by direction so TZP > SEMA and SEMA > TZP both get coverage. Without this,
    # rare classes (esp. suppression, <=5 pts/panel) can be skipped entirely
    # because they happen to have moderate FDR — see the bottom-left of
    # Olink Wk72 where CPA1/PNLIPRP1 (suppression) and EPCAM (direct-SEMA)
    # had plenty of free panel space but missed every magnitude/FDR cutoff.
    def _best_in_class(rows, cls, key):
        cand = [r for r in rows if r["class"] == cls]
        return _best_by(cand, key) if cand else None

    def _direction_up(rows):
        return _best_by(rows, "total_estimate")["total_estimate"] > 0

    def _vol_score(rows):
        best = _best_by(rows, "total_estimate")
        return abs(best["total_estimate"]) * -np.log10(max(best["ac_fdr"], 1e-300))

    def _top_per_dir(pool, score_fn, n=3):
        """Top n per direction from pool=[(g, rows, ...)] ranked by score_fn."""
        scored_up = sorted(
            ((g, rs) for g, rs in pool if _direction_up(rs)),
            key=lambda kv: (-score_fn(kv[1]), kv[0]),
        )[:n]
        scored_dn = sorted(
            ((g, rs) for g, rs in pool if not _direction_up(rs)),
            key=lambda kv: (-score_fn(kv[1]), kv[0]),
        )[:n]
        return scored_up + scored_dn

    # Volcano pool: headline differential genes, ranked by combined
    # |total| × −log10(ac_fdr) so effect size and significance trade off.
    by_volcano = _top_per_dir(
        list(gene_points.items()),
        _vol_score,
        n=N_VOLCANO_PER_DIR,
    )

    # Top suppression per direction by |total|: rare class but the count
    # varies (1 on Soma Wk24, 10 on Soma Wk72). Labelling all of them
    # overloads dense panels; cap at top-5 per direction so the class is
    # always represented without crowding the upper-right cluster.
    sup_pool = [
        (g, rs)
        for g, rs in gene_points.items()
        if _best_in_class(rs, "inconsistent", "total_estimate") is not None
    ]
    suppression_pool = _top_per_dir(
        sup_pool,
        lambda rs: abs(
            _best_in_class(rs, "inconsistent", "total_estimate")["total_estimate"]
        ),
        n=N_SUPPRESSION_PER_DIR,
    )

    # Top direct-only per direction by |ADE|: weight-independent extremes.
    do_pool = [
        (g, rs)
        for g, rs in gene_points.items()
        if _best_in_class(rs, "direct_only", "ADE_estimate") is not None
    ]
    by_direct = _top_per_dir(
        do_pool,
        lambda rs: abs(
            _best_in_class(rs, "direct_only", "ADE_estimate")["ADE_estimate"]
        ),
        n=N_DIRECT_PER_DIR,
    )

    # Top fully-mediated per direction by |ACME|: weight-mediated extremes.
    fm_pool = [
        (g, rs)
        for g, rs in gene_points.items()
        if _best_in_class(rs, "fully_mediated", "ACME_estimate") is not None
    ]
    by_mediated = _top_per_dir(
        fm_pool,
        lambda rs: abs(
            _best_in_class(rs, "fully_mediated", "ACME_estimate")["ACME_estimate"]
        ),
        n=N_MEDIATED_PER_DIR,
    )

    # Capped points are always labelled (they're the most extreme) — but
    # ONLY when significant in across-treatment. NS capped points still get
    # a grey diamond drawn (to show they were measured beyond the displayed
    # range), but labelling them piles up coordinate annotations at the
    # corner (e.g. ZWILCH / PIWIL4 / PHF24 on Olink Wk24 all share
    # _disp_total≈-EDGE, _disp_ADE≈-EDGE and their stacked labels become an
    # illegible cloud).
    capped_pool: list[tuple[str, list[dict]]] = []
    capped_genes: set[str] = set()
    for r in capped.filter(pl.col("ac_sig")).iter_rows(named=True):
        for gene in r.get("genes") or []:
            if gene in capped_genes:
                continue
            capped_genes.add(gene)
            capped_pool.append((gene, gene_points.get(gene, [r])))

    # Anchor each selected gene at the matching analyte. Pool order matters
    # only for which row anchors the label (the gene set is a union); we put
    # capped first so they aren't bumped by the supplementary sets, then
    # suppression (rare-class guarantee), then volcano headliners, then the
    # per-class direction-split pools.
    selected: dict[str, dict] = {}
    for source, col, mode in (
        (capped_pool, "ac_fdr", "min"),
        (suppression_pool, "ac_fdr", "min"),
        (by_volcano, "ac_fdr", "min"),
        (by_direct, "ADE_estimate", "abs"),
        (by_mediated, "ACME_estimate", "abs"),
    ):
        for gene, rows in source:
            if gene not in selected:
                selected[gene] = _best_by(rows, col, mode=mode)

    # Split selection: capped genes get manual placement (volcano-style
    # gene name + grey coordinate annotation right next to the truncated
    # marker). Non-capped genes go through adjustText as usual.
    capped_selected = {g: r for g, r in selected.items() if r["_capped"]}
    selected = {g: r for g, r in selected.items() if not r["_capped"]}

    # One Text per gene. Same pattern as `_volcano_figure.py`:
    # start with `va="bottom"` so the text bbox sits naturally *above* the
    # marker (avoiding the zero-gradient degeneracy of `va="center"`), and
    # give every label a white stroke halo so any transit overlap stays
    # legible.
    #
    # Anchors in the top 15 % of the panel flip to `va="top"` so the label
    # sits BELOW the marker — keeps near-edge labels (e.g. LDLRAP1 at
    # ADE≈0.71 on Olink Wk24, span≈0.77) out of the `↗ TZP > SEMA`
    # corner cue. The mirror also applies for the bottom 15 %.
    #
    # For multi-analyte genes (e.g. CHGA on SomaScan = 2 SeqIds), anchor the
    # label at the *centroid* of all matching analytes instead of one
    # arbitrary "primary" point. Lead lines then fan out to every analyte
    # from a position between them, rather than clustering at one extreme
    # which could push the label into the corner annotations.
    halo_text = [withStroke(linewidth=2.0, foreground="white")]
    texts = []
    label_anchors: list[tuple[str, float, float, bool]] = []
    for gene, anchor in sorted(
        selected.items(),
        key=lambda item: (item[1]["_disp_total"], item[1]["_disp_ADE"], item[0]),
    ):
        rows = gene_points[gene]
        multi = len(rows) > 1
        if multi:
            bx = float(np.mean([r["_disp_total"] for r in rows]))
            by = float(np.mean([r["_disp_ADE"] for r in rows]))
        else:
            bx = anchor["_disp_total"]
            by = anchor["_disp_ADE"]
        # Anchors in the top 15 % of the panel use va="top" so the label
        # sits BELOW the marker, away from the `↗ TZP > SEMA` corner cue.
        # Bottom-15 % anchors keep the default va="bottom" (label above
        # marker, away from the `↙ SEMA > TZP` cue).
        va_init = "top" if by > 0.85 * span else "bottom"
        texts.append(
            ax.text(
                bx,
                by,
                gene,
                fontsize=FS_GENE_LABEL,
                color="#222",
                ha="center",
                va=va_init,
                path_effects=halo_text,
                zorder=5,
            )
        )
        label_anchors.append((gene, bx, by, multi))

    if texts:
        # `min_arrow_len=12` (px) suppresses the primary lead line whenever
        # adjust_text leaves a label close to its anchor — i.e. clean cases
        # like CPA1 / PNLIPRP1 don't get a vestigial nub. Labels that had
        # to move (to dodge other labels) still keep their connector.
        #
        # `objects=[cue_tzp, cue_sema]` tells adjust_text to treat the
        # direction-cue texts as static obstacles, so near-edge labels are
        # pushed away from them rather than overlapping (LDLRAP1 used to
        # collide with `↗ TZP > SEMA` on Olink Wk24). The `force_static`
        # weight is raised from the default (0.1, 0.2) so the cues exert a
        # stronger repulsion than ordinary text-text avoidance.
        deterministic_adjust_text(
            texts,
            ax=ax,
            objects=[cue_tzp, cue_sema],
            arrowprops=dict(arrowstyle="-", color="#777", lw=0.4, alpha=0.7),
            expand=(1.2, 1.4),
            force_static=(0.6, 1.2),
            min_arrow_len=4,
        )

        # Post-pass safety net: adjust_text occasionally leaves a label
        # straddling the cue bbox when the panel is dense. Iterate over
        # labels, check display-space overlap with each cue, and shift any
        # offender down (top cue) or up (bottom cue) just enough to clear.
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        for cue, direction in [(cue_tzp, "down"), (cue_sema, "up")]:
            cue_bb = cue.get_window_extent(renderer=renderer)
            for txt in texts:
                tb = txt.get_window_extent(renderer=renderer)
                if tb.overlaps(cue_bb):
                    px, py = txt.get_position()
                    inv = ax.transData.inverted()
                    if direction == "down":
                        shift_px = (cue_bb.y0 - tb.y1) - 2  # 2 px clearance
                    else:
                        shift_px = (cue_bb.y1 - tb.y0) + 2
                    _, dy = inv.transform((0, shift_px)) - inv.transform((0, 0))
                    txt.set_position((px, py + dy))

    # Secondary lines: for any selected gene measured by >1 analyte, draw a
    # thin connector from the post-adjustText label position to *every*
    # analyte point of that gene. Since multi-analyte labels are anchored
    # at the centroid (not at one specific analyte), every analyte gets a
    # secondary line — there is no "primary" analyte to skip.
    for txt, (gene, bx, by, multi) in zip(texts, label_anchors):
        if not multi:
            continue
        rows = gene_points[gene]
        lx, ly = txt.get_position()
        for r in rows:
            ax.annotate(
                "",
                xy=(r["_disp_total"], r["_disp_ADE"]),
                xytext=(lx, ly),
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.3, alpha=0.7),
                zorder=4,
            )

    # Capped-point labels: each capped gene gets a 2-line label block
    # (gene name in dark, coordinates in grey below) placed near its
    # marker. Direction depends on the marker geometry:
    #
    #   • Triangle on the bottom/top edge (only y-capped): label goes to
    #     the SIDE (toward the panel interior in x).
    #   • Diamond at a corner (both x AND y capped):
    #       - alone at that y-edge → SIDE (matches the natural reading
    #         direction from corner toward the data cloud, e.g. GCG on
    #         Olink Wk72)
    #       - sharing the y-edge with another capped marker → VERTICAL
    #         (away from edge), so the diamond's label sits ABOVE/BELOW
    #         the marker and the companion's label sits to the SIDE; the
    #         two don't collide (e.g. GCG above + GH1 right on Olink Wk24)
    halo_label = [withStroke(linewidth=2.0, foreground="white")]
    SIDE_OFFSET_PT = 4.0  # horizontal gap from marker to side label
    SIDE_HALF_BLOCK_PT = 3.5  # side-placed block centred on marker y
    SIDE_DROP_PT = 2.0  # tiny downward shift of side blocks at
    # the bottom edge (or upward at top) so
    # a companion vertical-above label can
    # sit closer to its own marker without
    # the two labels touching
    VERTICAL_BASE_OFFSET_PT = 9.0  # vertical: above-placed label's coord
    # above marker — set to just clear the
    # companion side-label's top edge
    LINE_HEIGHT_PT = 7.0  # spacing between the two lines of a
    # vertical-above block

    from matplotlib import transforms

    # Group capped markers by y-edge so we can detect "companions".
    y_groups: dict[float, list[tuple[str, dict]]] = {}
    for gene, anchor in capped_selected.items():
        key = round(anchor["_disp_ADE"], 3)
        y_groups.setdefault(key, []).append((gene, anchor))

    for by, members in y_groups.items():
        for gene, anchor in members:
            ox = anchor["_orig_total"]
            oy = anchor["_orig_ADE"]
            bx = anchor["_disp_total"]
            cap_x = anchor["_cap_x"]
            cap_y = anchor["_cap_y"]
            is_corner = bool(cap_x and cap_y)
            on_left = anchor["_orig_total"] < 0
            on_bottom = anchor["_orig_ADE"] < 0
            has_companions = len(members) > 1

            # Direction decision: vertical (above/below marker) when the
            # diamond at a corner has a companion sharing the same y-edge;
            # horizontal (to the side) otherwise. In BOTH cases the two
            # text lines stack as a single grouped block with the gene
            # name on top and the coordinates directly below (closer to
            # the marker) — never split across the marker.
            # Bottom-left and top-right corners are the same corners used by
            # the directional cues, so keep those capped labels vertical even
            # when they have no companion marker at that edge.
            place_above = is_corner and (has_companions or (on_left == on_bottom))
            if place_above:
                x_off = 0
                ha = "center"
            else:
                x_off = (1 if on_left else -1) * SIDE_OFFSET_PT
                ha = "left" if on_left else "right"

            # Block layout:
            #   place_above=True  → block sits VERTICALLY above (or below)
            #     the marker. Both lines on the same side; gene name at
            #     the top, coord directly underneath. The base offset is
            #     large enough to clear any companion side-label that
            #     shares the same y-edge.
            #   place_above=False → block is CENTRED on the marker y,
            #     positioned to the side. Gene sits half-block above the
            #     marker line, coord sits half-block below. The marker
            #     glyph occupies the same y range but is offset to the
            #     left in x, so the gene+coord still read as a single
            #     2-line label "right next to" the marker.
            if place_above:
                block_dir = 1 if on_bottom else -1
                base_off = VERTICAL_BASE_OFFSET_PT
                coord_y_off = block_dir * base_off
                gene_y_off = block_dir * (base_off + LINE_HEIGHT_PT)
                gene_va = "bottom" if block_dir > 0 else "top"
                coord_va = gene_va
            else:
                # Tiny downward (for bottom-edge marker) or upward (for
                # top-edge) shift so a companion vertical-above label can
                # tuck in closer to its own marker.
                drop = -SIDE_DROP_PT if on_bottom else SIDE_DROP_PT
                gene_y_off = SIDE_HALF_BLOCK_PT + drop
                coord_y_off = -SIDE_HALF_BLOCK_PT + drop
                gene_va = "center"
                coord_va = "center"

            # Coordinates (grey).
            ax.text(
                bx,
                by,
                f"({ox:+.2f}, {oy:+.2f})",
                fontsize=FS_GENE_LABEL - 1,
                color="#888",
                ha=ha,
                va=coord_va,
                transform=transforms.offset_copy(
                    ax.transData,
                    fig=ax.figure,
                    x=x_off,
                    y=coord_y_off,
                    units="points",
                ),
                zorder=6,
                path_effects=halo_label,
            )
            # Gene name (dark).
            ax.text(
                bx,
                by,
                gene,
                fontsize=FS_GENE_LABEL,
                color="#222",
                ha=ha,
                va=gene_va,
                transform=transforms.offset_copy(
                    ax.transData,
                    fig=ax.figure,
                    x=x_off,
                    y=gene_y_off,
                    units="points",
                ),
                zorder=6,
                path_effects=halo_label,
            )

    ax.format(
        xlim=(-span, span),
        ylim=(-span, span),
        xlabel="Total effect",
        ylabel=("Direct effect" if show_ylabel else ""),
        xlabelsize=FS_AXLABEL,
        ylabelsize=FS_AXLABEL,
        xticklabelsize=FS_TICK,
        yticklabelsize=FS_TICK,
        title="",
    )

    # Island-style "Week 24" / "Week 72" pill at top-center (volcano convention).
    ax.text(
        0.5,
        0.97,
        f"Week {week}",
        transform=ax.transAxes,
        fontsize=FS_PILL,
        fontweight="bold",
        ha="center",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#F0F0F0",
            edgecolor="#CCCCCC",
            linewidth=0.4,
        ),
        zorder=6,
    )

    # Direction cues are created above (before adjust_text) so they can be
    # passed as obstacles to avoid; no need to draw them again here.

    # In-panel legend: single column, ordered by colour (red shades → blue
    # shades → suppression). All swatches use a white edge so mediated
    # (paler fill) reads as de-emphasized relative to direct/partial.
    def _swatch(face, edge, ew):
        return Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=face,
            markeredgecolor=edge,
            markersize=4.0,
            markeredgewidth=ew,
        )

    def _count_dir(cls_name, sign):
        cls_df = all_sig.filter(pl.col("class") == cls_name)
        if sign == "pos":
            return cls_df.filter(pl.col("total_estimate") > 0).height
        return cls_df.filter(pl.col("total_estimate") <= 0).height

    legend_handles: list = []
    legend_labels: list[str] = []

    # Order: TZP shades (TZP > SEMA) — vivid → light → palest, then SEMA
    # shades (SEMA > TZP) in the same order, then inconsistent last.
    legend_specs = [
        ("Direct, TZP > SEMA", "direct_only", "pos", pal["direct_pos"]),
        ("Partial, TZP > SEMA", "partial", "pos", pal["partial_pos"]),
        ("Mediated, TZP > SEMA", "fully_mediated", "pos", pal["fm_pos_fill"]),
        ("Direct, SEMA > TZP", "direct_only", "neg", pal["direct_neg"]),
        ("Partial, SEMA > TZP", "partial", "neg", pal["partial_neg"]),
        ("Mediated, SEMA > TZP", "fully_mediated", "neg", pal["fm_neg_fill"]),
    ]
    for label, cls_name, sign, color in legend_specs:
        n = _count_dir(cls_name, sign)
        if n:
            legend_handles.append(_swatch(color, "white", 0.4))
            legend_labels.append(f"{label} ({n})")

    if counts["ns"]:
        legend_handles.append(_swatch(pal["unresolved"], "white", 0.4))
        legend_labels.append(f"Unresolved ({counts['ns']})")

    if counts["inconsistent"]:
        legend_handles.append(_swatch(pal["inconsistent"], "white", 0.4))
        legend_labels.append(f"Inconsistent ({counts['inconsistent']})")

    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=FS_BODY,
        frameon=False,
        ncol=1,
        handlelength=0.65,
        handletextpad=0.22,
        labelspacing=0.14,
        borderpad=0.15,
        borderaxespad=0.15,
    )
    return counts


# ---------------------------------------------------------------------------
# Row 2: theme forest plots
# ---------------------------------------------------------------------------


def select_theme_genes_per_platform(
    theme_genes: list[str],
    med: pl.DataFrame,
    ac: pl.DataFrame,
    n_max: int = N_PROTEINS_PER_THEME,
    fdr_threshold: float = FDR_THRESHOLD,
) -> pl.DataFrame:
    """Pick top N proteins for a theme on one platform.

    Filter: FDR<`fdr_threshold` (default 0.05) in across-treatment Wk72 AND
    present in mediation results. The threshold defaults to the global
    significance cutoff but can be relaxed per-theme (e.g. Pancreatic
    exocrine uses 0.10 to surface near-significant acinar enzymes like
    CTRL alongside the formally significant ones — the formal FDR<0.05
    cutoff is unchanged for the rest of the analysis).
    Sort by |total_estimate| (mediation). Dedupe by gene_label.
    """
    m72 = classify_mediation(
        med.filter(pl.col("visit") == 20),
        ac_df=ac,
        week=72,
    )
    m72 = m72.filter(pl.col("genes").list.set_intersection(theme_genes).list.len() > 0)
    sig_markers = ac.filter((pl.col("week") == 72) & (pl.col("fdr") < fdr_threshold))[
        "marker"
    ].to_list()
    m72 = m72.filter(pl.col("marker").is_in(sig_markers))
    if m72.height == 0:
        return m72
    m72 = (
        m72.sort(pl.col("total_estimate").abs(), descending=True)
        .unique(subset=["gene_label"], keep="first", maintain_order=True)
        .head(n_max)
    )
    return m72


def build_aligned_rows(
    sel_ol: pl.DataFrame,
    sel_so: pl.DataFrame,
    full_ol: pl.DataFrame | None = None,
    full_so: pl.DataFrame | None = None,
) -> list[dict]:
    """Group Olink + Soma selections into one shared y-axis order.

    Multi-gene aware: two markers cluster onto the same row whenever any of
    their constituent HGNC genes match. This catches SomaScan heterodimer
    aptamers (e.g. SeqId for `ITGAV|ITGB3`) so they share a row with an
    Olink single-gene marker for either ITGAV or ITGB3.

    If `full_ol` / `full_so` are provided (classified mediation data at the
    relevant visit, including non-significant rows), each cluster that has
    no selection on a platform but DOES have measurable data on that
    platform gets a 'shadow' row attached — so a reader can see the
    cross-platform estimate even when it didn't reach FDR<0.05. Without
    this, INS-on-Soma at Wk72 (measured, total=-0.03, ns) would show as
    "missing" when in fact the aptamer recorded a value.

    Each returned cluster has:
        display       - "|".join(sorted(union_of_genes))  (y-tick label)
        olink_label   - gene_label from sel_ol that matched, or None
        soma_label    - gene_label from sel_so that matched, or None
        olink_shadow  - mediation row (dict) from full_ol that matches but
                        wasn't selected, or None (and likewise for soma)
        sort_val      - signed total_estimate from whichever platform has
                        the larger |total| (drives top-to-bottom row order)

    Sort: ascending signed total → SEMA > TZP at the bottom of the y-axis,
    TZP > SEMA at the top, matching the per-platform `df.sort('total_estimate')`
    convention.
    """

    def genes_of(label: str) -> set[str]:
        return set((label or "").split("|"))

    clusters: list[dict] = []

    def find_or_create(label: str, total: float, platform: str) -> None:
        g = genes_of(label)
        key = f"{platform}_label"
        for c in clusters:
            if c["genes"] & g:
                # Overlap — merge into existing cluster.
                c["genes"] |= g
                # Multiple markers from the same platform mapping to one
                # cluster (rare; only happens when separate single-gene
                # Olink markers both overlap a Soma heterodimer): keep the
                # larger-magnitude one as the cluster's representative for
                # that platform.
                if c[key] is None or abs(total) > abs(c.get(f"{key}_total", 0.0)):
                    c[key] = label
                    c[f"{key}_total"] = total
                if abs(total) > abs(c["sort_val"]):
                    c["sort_val"] = total
                return
        clusters.append(
            {
                "genes": g.copy(),
                "olink_label": None,
                "soma_label": None,
                "olink_label_total": 0.0,
                "soma_label_total": 0.0,
                "sort_val": total,
                key: label,
                f"{key}_total": total,
            }
        )

    if sel_ol.height:
        for r in sel_ol.iter_rows(named=True):
            find_or_create(r["gene_label"], r["total_estimate"], "olink")
    if sel_so.height:
        for r in sel_so.iter_rows(named=True):
            find_or_create(r["gene_label"], r["total_estimate"], "soma")

    for c in clusters:
        c["display"] = "|".join(sorted(c["genes"]))
        c["olink_shadow"] = None
        c["soma_shadow"] = None

    # Fill shadow rows from full-mediation data for clusters where one
    # platform has no selection. Prefer the row with the largest |total|
    # among matching genes (picks the strongest non-significant trace).
    def _find_shadow(full: pl.DataFrame, cluster_genes: set[str]) -> dict | None:
        if full is None or full.height == 0:
            return None
        best = None
        best_mag = -1.0
        for r in full.iter_rows(named=True):
            row_genes = set(r.get("genes") or [])
            if not (row_genes & cluster_genes):
                continue
            mag = abs(r.get("total_estimate") or 0.0)
            if mag > best_mag:
                best = r
                best_mag = mag
        return best

    for c in clusters:
        if c["olink_label"] is None:
            c["olink_shadow"] = _find_shadow(full_ol, c["genes"])
        if c["soma_label"] is None:
            c["soma_shadow"] = _find_shadow(full_so, c["genes"])

    clusters.sort(key=lambda c: c["sort_val"])
    return clusters


def _forest_xlim_for_aligned(
    aligned: list[dict],
    sel_ol: pl.DataFrame,
    sel_so: pl.DataFrame,
) -> tuple[float, float] | None:
    """Symmetric/padded xlim covering the CI bars of every row actually
    rendered in the aligned forest pair — selected (significant on this
    platform) AND shadow (measured but not significant). Excludes
    non-aligned rows from the full theme data so the xlim is exactly the
    data that's drawn.
    """
    by_ol = (
        {r["gene_label"]: r for r in sel_ol.iter_rows(named=True)}
        if sel_ol.height
        else {}
    )
    by_so = (
        {r["gene_label"]: r for r in sel_so.iter_rows(named=True)}
        if sel_so.height
        else {}
    )
    lo, hi = float("inf"), float("-inf")
    for c in aligned:
        for plat, by_label in (("olink", by_ol), ("soma", by_so)):
            label = c.get(f"{plat}_label")
            r = by_label.get(label) if label else c.get(f"{plat}_shadow")
            if r is None:
                continue
            for col in ("total_ci_lower", "ADE_ci_lower"):
                v = r.get(col)
                if v is not None:
                    lo = min(lo, float(v))
            for col in ("total_ci_upper", "ADE_ci_upper"):
                v = r.get(col)
                if v is not None:
                    hi = max(hi, float(v))
    if not (lo < hi):
        return None
    pad = (hi - lo) * 0.06
    return (lo - pad, hi + pad)


def plot_theme_forest(
    ax,
    theme_name: str,
    df: pl.DataFrame,
    show_xlabel: bool = True,
    show_ylabel: bool = False,
    mono_platform: str | None = None,
    aligned_rows: list[dict] | None = None,
    xlim: tuple[float, float] | None = None,
    show_prop_mediated: bool = False,
    pm_header_y: float = 0.99,
    pm_header_va: str = "top",
    pm_header_offset_in: float = 0.0,
    label_line_offset_pt: float = 3.5,
) -> None:
    """Forest plot: Total + Direct (ADE) per protein, with CI bars.

    Mediated portion is the visual gap. Color by platform (Olink/Soma).
    If mono_platform is set, use that platform's color uniformly.

    If aligned_rows is provided, the panel renders the union y-axis from
    `build_aligned_rows` so that paired Olink/Soma panels line up row-for-
    row. Rows where this platform has no matching marker render as a blank
    line with a greyed-italic y-tick label so the absence itself is
    informative.
    """
    pal = _palette()
    from matplotlib import transforms

    # Legacy path: no alignment — original per-platform behaviour.
    if aligned_rows is None:
        if df.height == 0:
            ax.text(
                0.5,
                0.5,
                "(no genes\npass criteria)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=5.5,
                color="#888",
            )
            ax.format(xlabel="", ylabel="", title=theme_name, xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
            return
        rows = []
        for r in df.sort("total_estimate").iter_rows(named=True):
            gene = (r["gene_label"] or "").split("|")[0]
            if mono_platform is None:
                tag = "O" if r["platform"] == "olink" else "S"
                rows.append((f"{gene} ({tag})", r, "selected"))
            else:
                rows.append((gene, r, "selected"))
    else:
        # Aligned path: rows come from the union cluster list. Each row's
        # `status` controls colour:
        #   "selected" → navy (total>0, TZP > SEMA) or grey (total<=0, SEMA > TZP)
        #   "shadow"   → grey (measured but ns in across-treatment)
        #   "missing"  → no marker, greyed-italic gene label
        if mono_platform not in ("olink", "soma"):
            raise ValueError("aligned_rows requires mono_platform='olink' or 'soma'")
        label_key = f"{mono_platform}_label"
        shadow_key = f"{mono_platform}_shadow"
        by_label: dict[str, dict] = (
            {r["gene_label"]: r for r in df.iter_rows(named=True)} if df.height else {}
        )
        rows = []
        for cluster in aligned_rows:
            label = cluster[label_key]
            r = by_label.get(label) if label else None
            if r is not None:
                rows.append((cluster["display"], r, "selected"))
            elif cluster.get(shadow_key) is not None:
                rows.append((cluster["display"], cluster[shadow_key], "shadow"))
            else:
                rows.append((cluster["display"], None, "missing"))

    n = len(rows)
    if n == 0:
        ax.text(
            0.5,
            0.5,
            "(no genes\npass criteria)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=5.5,
            color="#888",
        )
        ax.format(xlabel="", ylabel="", title=theme_name, xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return

    yvals = np.arange(n)
    y_total_offset = 0.18
    y_direct_offset = -0.18

    ax.axvline(0, color=pal["axline"], lw=0.5, alpha=0.7, zorder=1)

    SHADOW_GREY = COLOR_NS
    MEDIATION_CONNECTOR = "#666666"

    for i, (_, r, status) in enumerate(rows):
        if status == "missing" or r is None:
            continue
        y = yvals[i]
        # Direction-based colours matching the scatter convention:
        # navy = TZP higher (positive total), grey = SEMA higher (negative
        # total), light grey = measured but not significant in across-treatment.
        if status == "shadow":
            color = SHADOW_GREY
        else:
            color = pal["direct_pos"] if r["total_estimate"] > 0 else pal["direct_neg"]
        ax.plot(
            [r["total_ci_lower"], r["total_ci_upper"]],
            [y + y_total_offset, y + y_total_offset],
            color=color,
            lw=0.9,
            zorder=2,
        )
        ax.scatter(
            [r["total_estimate"]],
            [y + y_total_offset],
            s=16,
            c=color,
            edgecolors=pal["axline"],
            lw=0.3,
            zorder=3,
        )
        # Direct (ADE) effect colour encodes ADE significance:
        # direction colour if ADE FDR<0.05, grey otherwise. A selected row
        # with a coloured Total but grey Direct visually flags
        # "fully-mediated" rows where the weight pathway accounts for the
        # effect. Shadow rows (already grey) stay grey throughout.
        ade_sig = r.get("ADE_fdr") is not None and r["ADE_fdr"] < ADE_FDR_THRESHOLD
        ade_color = color if (status == "selected" and ade_sig) else SHADOW_GREY
        ax.plot(
            [r["ADE_ci_lower"], r["ADE_ci_upper"]],
            [y + y_direct_offset, y + y_direct_offset],
            color=ade_color,
            lw=0.9,
            zorder=2,
        )
        ax.scatter(
            [r["ADE_estimate"]],
            [y + y_direct_offset],
            s=16,
            c="white",
            edgecolors=ade_color,
            lw=0.9,
            zorder=3,
        )
        # Mediation connector only for significant rows where mediation is
        # meaningful. A neutral grey keeps it from competing with the
        # red/blue direction signal in the markers.
        if status == "selected" and r["class"] in ("partial", "fully_mediated"):
            ax.plot(
                [r["total_estimate"], r["ADE_estimate"]],
                [y + y_total_offset, y + y_direct_offset],
                color=MEDIATION_CONNECTOR,
                lw=0.6,
                alpha=0.7,
                zorder=1,
            )

    # Proportion-mediated annotation: small percentage text to the right
    # of each row. Shown for all selected (across-treatment significant)
    # rows. Direction-colored when ACME is significant, grey otherwise.
    # A "PM" column header sits above the top row.
    if show_prop_mediated:
        base_y_trans = ax.get_yaxis_transform()
        pm_header_trans = ax.transAxes
        if pm_header_offset_in:
            pm_header_trans = transforms.offset_copy(
                ax.transAxes,
                fig=ax.figure,
                x=0,
                y=pm_header_offset_in * 72.0,
                units="points",
            )
        ax.text(
            PM_COLUMN_X,
            pm_header_y,
            "PM",
            transform=pm_header_trans,
            ha="left",
            va=pm_header_va,
            fontsize=FS_FOREST_YLABEL - 0.5,
            fontweight="bold",
            color="#666666",
            clip_on=False,
        )
        for i, (_, r, status) in enumerate(rows):
            if status != "selected" or r is None:
                continue
            pm = r.get("prop_mediated")
            if pm is None or not np.isfinite(pm):
                continue
            y = yvals[i]
            acme_sig = (
                r.get("ACME_fdr") is not None and r["ACME_fdr"] < ACME_FDR_THRESHOLD
            )
            if acme_sig:
                pm_color = (
                    pal["direct_pos"] if r["total_estimate"] > 0 else pal["direct_neg"]
                )
            else:
                pm_color = COLOR_NS
            pm_text = f"{pm:.2f}"
            ax.text(
                PM_COLUMN_X,
                y,
                pm_text,
                transform=base_y_trans,
                ha="left",
                va="center",
                fontsize=FS_FOREST_YLABEL - 1.2,
                color=pm_color,
                clip_on=False,
            )

    # Hide the default y-tick labels — we render custom two-line labels
    # (gene name + smaller grey marker ID) via ax.text below so the styling
    # can vary line-by-line, which yticklabels alone can't do.
    # ylim extends 0.7 units below the first row (y=0) and 0.7 above the
    # last row (y=n-1) so two-line gene labels at the top row have enough
    # clearance to sit below the banner without overlap. With less padding
    # (e.g. n-0.3 top), tall panels in the supplement render the topmost
    # label INTO the banner area because matplotlib's `clip_on=False` text
    # extends beyond the data area.
    fmt_kwargs = dict(
        yticks=list(yvals),
        yticklabels=[""] * n,
        ylim=(-0.7, n - 1 + 0.7),
        xlabel="Effect (Week 72)" if show_xlabel else "",
        ylabel="",
        xlabelsize=FS_AXLABEL,
        xticklabelsize=FS_TICK,
        title=theme_name,
    )
    if xlim is not None:
        fmt_kwargs["xlim"] = xlim
    ax.format(**fmt_kwargs)
    # Belt-and-suspenders: ultraplot's autoscale flips x-axis direction
    # based on point-add order, which produced mixed (positive, negative)
    # and (negative, positive) limits across panels. Re-assert the
    # negative-left / positive-right convention here so the figure always
    # matches the scatter row's `↙ SEMA > TZP / ↗ TZP > SEMA` direction cues.
    cur_lo, cur_hi = ax.get_xlim()
    if cur_lo > cur_hi:
        ax.set_xlim(cur_hi, cur_lo)
    ax.tick_params(axis="y", length=0)

    # Custom two-line y-axis labels. Top line = gene/cluster name; bottom
    # line = the platform's marker ID (OlinkID or SeqId) in smaller grey.
    # Empty (cross-platform-missing) rows show only the greyed-italic gene
    # name.
    #
    # The vertical offsets between gene name and marker ID use a point offset
    # (via transforms.offset_copy). Most panels use the house default, but
    # dense supplement themes can pass a tighter panel-specific value when a
    # two-line label stack would otherwise crowd adjacent rows.
    base_trans = ax.get_yaxis_transform()  # x = axes-fraction, y = data
    trans_up = transforms.offset_copy(
        base_trans,
        fig=ax.figure,
        x=0,
        y=label_line_offset_pt,
        units="points",
    )
    trans_dn = transforms.offset_copy(
        base_trans,
        fig=ax.figure,
        x=0,
        y=-label_line_offset_pt,
        units="points",
    )

    for i, (cluster_label, r, status) in enumerate(rows):
        y = yvals[i]
        if status == "missing" or r is None:
            # Empty row on this platform — fall back to the cluster union so
            # the reader can see what gene cluster the row represents.
            ax.text(
                FOREST_LABEL_X,
                y,
                cluster_label,
                transform=base_trans,
                ha="right",
                va="center",
                fontsize=FS_FOREST_YLABEL,
                color="#BBBBBB",
                style="italic",
                clip_on=False,
            )
        else:
            # Use the PLATFORM-SPECIFIC gene label (what this marker actually
            # measures) rather than the cluster union. e.g., the Olink
            # ITGAV-only assay shows "ITGAV", while the paired SomaScan
            # heterodimer aptamer shows "ITGAV|ITGB3" — same row, different
            # labels reflecting what each platform measured.
            label_text = r["gene_label"]
            ax.text(
                FOREST_LABEL_X,
                y,
                label_text,
                transform=trans_up,
                ha="right",
                va="center",
                fontsize=FS_FOREST_YLABEL,
                color="#222",
                clip_on=False,
            )
            ax.text(
                FOREST_LABEL_X,
                y,
                r["marker"],
                transform=trans_dn,
                ha="right",
                va="center",
                fontsize=FS_FOREST_YLABEL - 1.5,
                color="#999",
                clip_on=False,
            )


# ---------------------------------------------------------------------------
# Figure assembly helpers
# ---------------------------------------------------------------------------

MAIN_THEME_N_MAX: dict[str, int] = {
    "Leptin & adiponectin": 4,
}
MAIN_THEME_FDR: dict[str, float] = {}
SCATTER_BANNER_TAIL = " · TZP vs SEMA · mediator: % Δweight"
PAIR_LETTER_DY_IN = BANNER_HEIGHT_IN + BANNER_GAP_IN
PAIR_WSPACE = "0.62in"
OUTER_LABEL_MARGIN = 0.016
FOREST_LABEL_X = -OUTER_LABEL_MARGIN
PM_COLUMN_X = 1.0 + OUTER_LABEL_MARGIN


def plot_theme_pair(
    ax_ol,
    ax_so,
    *,
    theme_name: str,
    genes: list[str],
    med_ol: pl.DataFrame,
    med_so: pl.DataFrame,
    ac_olink_all: pl.DataFrame,
    ac_soma_all: pl.DataFrame,
    full_ol_72: pl.DataFrame,
    full_so_72: pl.DataFrame,
    n_max: int = N_PROTEINS_PER_THEME,
    fdr_threshold: float = FDR_THRESHOLD,
    pm_header_y: float = 0.99,
    pm_header_va: str = "top",
    pm_header_offset_in: float = 0.0,
    label_line_offset_pt: float = 3.5,
) -> None:
    sel_ol = select_theme_genes_per_platform(
        genes,
        med_ol,
        ac_olink_all,
        n_max=n_max,
        fdr_threshold=fdr_threshold,
    )
    sel_so = select_theme_genes_per_platform(
        genes,
        med_so,
        ac_soma_all,
        n_max=n_max,
        fdr_threshold=fdr_threshold,
    )
    theme_genes = set(genes)
    theme_ol = full_ol_72.filter(
        pl.col("genes").list.set_intersection(list(theme_genes)).list.len() > 0
    )
    theme_so = full_so_72.filter(
        pl.col("genes").list.set_intersection(list(theme_genes)).list.len() > 0
    )
    aligned = build_aligned_rows(sel_ol, sel_so, theme_ol, theme_so)
    shared_xlim = _forest_xlim_for_aligned(aligned, sel_ol, sel_so)
    plot_theme_forest(
        ax_ol,
        "",
        sel_ol,
        show_xlabel=True,
        show_ylabel=True,
        mono_platform="olink",
        aligned_rows=aligned,
        xlim=shared_xlim,
        show_prop_mediated=True,
        pm_header_y=pm_header_y,
        pm_header_va=pm_header_va,
        pm_header_offset_in=pm_header_offset_in,
        label_line_offset_pt=label_line_offset_pt,
    )
    plot_theme_forest(
        ax_so,
        "",
        sel_so,
        show_xlabel=True,
        show_ylabel=True,
        mono_platform="soma",
        aligned_rows=aligned,
        xlim=shared_xlim,
        show_prop_mediated=True,
        pm_header_y=pm_header_y,
        pm_header_va=pm_header_va,
        pm_header_offset_in=pm_header_offset_in,
        label_line_offset_pt=label_line_offset_pt,
    )


def add_forest_legend(fig) -> None:
    pal = _palette()

    def _glyph(face, edge, ew, marker="o"):
        return Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            markerfacecolor=face,
            markeredgecolor=edge,
            markersize=5,
            markeredgewidth=ew,
        )

    pm_handle = Line2D([], [], linestyle="none", marker="none")
    forest_handles = [
        (_glyph(pal["direct_pos"], pal["axline"], 0.3), "TZP > SEMA"),
        (_glyph(pal["direct_neg"], pal["axline"], 0.3), "SEMA > TZP"),
        (_glyph(COLOR_NS, pal["axline"], 0.3), "Not significant (FDR ≥ 0.05)"),
        (_glyph("#444", pal["axline"], 0.3), "Total effect"),
        (_glyph("white", "#444", 0.9), "Direct effect"),
        (Line2D([0], [0], color="#666666", lw=1.2), "Sig. mediation"),
        (pm_handle, "PM = proportion mediated"),
    ]
    fig.legend(
        handles=[h for h, _ in forest_handles],
        labels=[l for _, l in forest_handles],
        loc="b",
        ncols=7,
        frame=False,
        fontsize=FS_LEGEND,
        title=False,
        columnspacing=1.0,
        handlelength=1.2,
        handletextpad=0.5,
        handler_map={pm_handle: _TextOnlyHandler()},
    )


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------


def make_figure():
    init_figure_theme()

    # Load all data once
    med_ol = load_mediation("olink")
    med_so = load_mediation("soma")
    ac_olink_all = pl.concat(
        [
            load_across_treatment("olink", 24),
            load_across_treatment("olink", 72),
        ]
    )
    ac_soma_all = pl.concat(
        [
            load_across_treatment("soma", 24),
            load_across_treatment("soma", 72),
        ]
    )

    def _theme_row(
        theme_names: tuple[str, ...], start_pid: int
    ) -> tuple[list[int], dict[str, tuple[int, int]], list[int], int]:
        row: list[int] = []
        panel_ids: dict[str, tuple[int, int]] = {}
        spacer_ids: list[int] = []
        if len(theme_names) == 2:
            for theme_name in theme_names:
                panel_ids[theme_name] = (start_pid, start_pid + 1)
                row += [start_pid] * 3 + [start_pid + 1] * 3
                start_pid += 2
            return row, panel_ids, spacer_ids, start_pid
        if len(theme_names) == 1:
            spacer_ids.append(start_pid)
            row += [start_pid] * 3
            start_pid += 1
            theme_name = theme_names[0]
            panel_ids[theme_name] = (start_pid, start_pid + 1)
            row += [start_pid] * 3 + [start_pid + 1] * 3
            start_pid += 2
            spacer_ids.append(start_pid)
            row += [start_pid] * 3
            return row, panel_ids, spacer_ids, start_pid + 1
        raise ValueError(f"Unsupported theme row: {theme_names!r}")

    scatter_row = [1] * 6 + [2] * 6
    forest_row_1, forest_ids_1, _, next_pid = _theme_row(MAIN_THEME_ROWS[0], 3)
    forest_row_2, forest_ids_2, _, _ = _theme_row(MAIN_THEME_ROWS[1], next_pid)
    theme_panel_ids = forest_ids_1 | forest_ids_2

    figwidth, figheight = nature_figsize("2col", 9.45)
    fig, axs = uplt.subplots(
        array=[scatter_row, forest_row_1, forest_row_2],
        figwidth=figwidth,
        figheight=figheight,
        share=False,
        hratios=(1.65, 1.0, 1.0),
        hspace=("0.84in", "0.74in"),
        wspace=(PAIR_WSPACE,),
        left="0.58in",
        right="0.32in",
        top="0.44in",
        bottom="0.82in",
    )
    axs.format(abc=False)
    flat_axs = list(np.ravel(np.atleast_1d(axs)))
    pid_to_ax = {pid: ax for pid, ax in enumerate(flat_axs, start=1)}

    plot_global_scatter(
        pid_to_ax[1], med_ol, ac_olink_all, 72, "olink", show_ylabel=True
    )
    plot_global_scatter(
        pid_to_ax[2], med_so, ac_soma_all, 72, "soma", show_ylabel=False
    )

    # Pre-classify full mediation data at Wk72 so build_aligned_rows can
    # fill in non-significant Soma/Olink data for genes that were only
    # selected on the other platform (e.g. INS-on-Soma at Wk72:
    # measured, total = −0.03, but ac_fdr = 0.72 so it wouldn't make the
    # FDR-driven selection — we still want to *show* the SomaScan estimate
    # next to the strong Olink signal).
    full_ol_72 = classify_mediation(
        med_ol.filter(pl.col("visit") == 20),
        ac_df=ac_olink_all,
        week=72,
    )
    full_so_72 = classify_mediation(
        med_so.filter(pl.col("visit") == 20),
        ac_df=ac_soma_all,
        week=72,
    )

    for theme_name in MAIN_THEME_ORDER:
        panel_ol, panel_so = theme_panel_ids[theme_name]
        plot_theme_pair(
            pid_to_ax[panel_ol],
            pid_to_ax[panel_so],
            theme_name=theme_name,
            genes=MAIN_THEMES[theme_name],
            med_ol=med_ol,
            med_so=med_so,
            ac_olink_all=ac_olink_all,
            ac_soma_all=ac_soma_all,
            full_ol_72=full_ol_72,
            full_so_72=full_so_72,
            n_max=MAIN_THEME_N_MAX.get(theme_name, N_PROTEINS_PER_THEME),
            fdr_threshold=MAIN_THEME_FDR.get(theme_name, FDR_THRESHOLD),
        )

    add_forest_legend(fig)
    fig.canvas.draw()
    draw_banner(
        fig,
        pid_to_ax[1],
        pid_to_ax[1],
        bold_text="Olink",
        tail_text=SCATTER_BANNER_TAIL,
    )
    draw_banner(
        fig,
        pid_to_ax[2],
        pid_to_ax[2],
        bold_text="SomaScan",
        tail_text=SCATTER_BANNER_TAIL,
    )
    for theme_name in MAIN_THEME_ORDER:
        panel_ol, panel_so = theme_panel_ids[theme_name]
        draw_banner(
            fig,
            pid_to_ax[panel_ol],
            pid_to_ax[panel_so],
            bold_text=MAIN_THEME_LABELS[theme_name].replace("\n", " "),
            left_edge_text="Olink",
            right_edge_text="SomaScan",
        )

    fig.canvas.draw()
    place_panel_letter(fig, pid_to_ax[1], "a", extra_dy_in=PAIR_LETTER_DY_IN)
    place_panel_letter(fig, pid_to_ax[2], "b", extra_dy_in=PAIR_LETTER_DY_IN)
    for label, theme_name in zip("cdef", MAIN_THEME_ORDER, strict=False):
        panel_ol, _ = theme_panel_ids[theme_name]
        place_panel_letter(
            fig, pid_to_ax[panel_ol], label, extra_dy_in=PAIR_LETTER_DY_IN
        )

    return fig


if __name__ == "__main__":
    fig = make_figure()
    out_dir = _HERE
    save_figure(fig, str(out_dir / "fig4_mediation"), formats=("pdf",))
    print("Wrote fig4_mediation.pdf")
