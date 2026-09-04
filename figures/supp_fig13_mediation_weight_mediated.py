"""Supplementary mediation figure 1: mostly weight-mediated themes.

Supplementary mediation themes. The lower section uses an asymmetric layout so
the larger coagulation/complement panel can sit beside two smaller mostly
weight-mediated panels without wasting vertical space.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import ultraplot as uplt

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from _internal._common import (  # noqa: E402
    draw_banner,
    init_figure_theme,
    nature_figsize,
    place_panel_letter,
    save_figure,
)

from _internal._mediation_figure import (  # noqa: E402
    MAIN_THEMES,
    MAIN_THEME_LABELS,
    PAIR_LETTER_DY_IN,
    PAIR_WSPACE,
    add_forest_legend,
    classify_mediation,
    load_across_treatment,
    load_mediation,
    plot_theme_pair,
)

# Supplement theme collections reused across the split mediation supplements.
NEURAL_THEME_NAME = "Neural"
SUPPL_THEMES: dict[str, list[str]] = {
    "Coagulation\n& complement": [
        "C3",
        "C4A",
        "C4B",
        "C5",
        "C6",
        "C7",
        "C8A",
        "C8B",
        "C8G",
        "C9",
        "C1QA",
        "C1QB",
        "C1QC",
        "C1R",
        "C1S",
        "C2",
        "CFB",
        "CFH",
        "CFI",
        "CFP",
        "CFD",
        "CFHR1",
        "CFHR2",
        "CFHR3",
        "CFHR4",
        "F2",
        "F3",
        "F7",
        "F9",
        "F10",
        "F11",
        "F12",
        "F13A1",
        "F13B",
        "KLKB1",
        "TF",
        "SERPIND1",
        "SERPINC1",
        "SERPINF1",
        "SERPINA1",
        "SERPINA3",
        "APCS",
        "VNN1",
        "CPB2",
        "PROC",
        "PROS1",
        "KNG1",
        "FGA",
        "FGB",
        "FGG",
        "PLG",
        "THBD",
    ],
    "Hepatic\nacute-phase": [
        "CRP",
        "SAA1",
        "SAA2",
        "SAA4",
        "AHSG",
        "APCS",
        "HP",
        "LBP",
        "CP",
        "SERPINA1",
        "SERPINA3",
        "ITIH3",
        "ITIH4",
        "AMBP",
        "LRG1",
        "ORM1",
        "ORM2",
        "A1BG",
        "A2M",
        "FN1",
    ],
    "Steroid hormone\nmetabolism": [
        "HSD11B1",
        "HSD11B2",
        "HSD17B1",
        "HSD17B2",
        "HSD17B14",
        "CYP3A4",
        "CYP3A5",
        "CYP11B1",
        "CYP11B2",
        "CYP17A1",
        "CYP19A1",
        "AKR1C3",
        "AKR1C4",
        "SHBG",
        "STAR",
        "SRD5A1",
        "SRD5A2",
    ],
    NEURAL_THEME_NAME: [
        # Synaptic adhesion: trans-synaptic organisers, cell adhesion
        # molecules, and synaptic vesicle machinery.
        "SLITRK1",
        "SLITRK2",
        "SLITRK3",
        "SLITRK4",
        "NLGN1",
        "NRXN3",
        "NCAM2",
        "CADM2",
        "CLSTN2",
        "CBLN2",
        "CNTN4",
        "MDGA1",
        "PCDH10",
        "PCDHGA1",
        "GRID2",
        "PTPRD",
        "PTPRR",
        "KIRREL1",
        "CPLX2",
        "STX1A",
        "SYT1",
        "NXPH3",
        "SEZ6",
        "ISLR2",
        "B3GAT1",
        "ADGRB3",
        "NCS1",
        # Axon guidance / myelination.
        "NTRK3",
        "PLXNA1",
        "RTN4R",
        "UNC5D",
        "ANOS1",
        "PRTG",
        "ROR1",
        "CNTFR",
        "CHL1",
        "MPZ",
        "MOG",
        "NCAN",
        "FLRT2",
    ],
    "ECM & TGFβ\nremodeling": [
        "TGFBR3",
        "LTBP1",
        "LTBP2",
        "LTBP4",
        "ACVRL1",
        "VASN",
        "ITGAV",
        "ITGB3",
        "BCAN",
        "UST",
        "VCAN",
        "ADAM15",
        "KLK4",
        "SH3PXD2B",
        "COL1A1",
        "COL3A1",
        "COL4A1",
        "COL4A2",
        "COL6A1",
        "COL9A1",
        "COL9A3",
        "FBN1",
        "FBN2",
        "FBN3",
        "MMP2",
        "MMP9",
        "TIMP1",
        "TIMP2",
        "DCN",
        "BGN",
        "LUM",
        "PRELP",
    ],
    "Iron\nhomeostasis": [
        "SCARA5",
        "SLC39A14",
        "SLC40A1",
        "TF",
        "TFRC",
        "TFR2",
        "BMP6",
        "HJV",
        "HAMP",
        "FTL",
        "FTH1",
        "LCN2",
        "HFE",
        "FXN",
        "HEPH",
    ],
    "Mast cell\nsignaling": [
        "RET",
        "CRTAM",
        "NEDD9",
        "FUT4",
        "FUT7",
        "DPEP1",
        "GGT5",
        "KIT",
        "NR4A3",
        "S100A13",
        "S100A11",
        "S100A12",
        "KITLG",
        "TPSAB1",
        "TPSB2",
        "CMA1",
        "MS4A2",
        "FCER1A",
    ],
}

# Per-theme row caps. Neural gets a much higher cap because its panel
# sits on its own tall row in Supplementary Figure 2.
N_MAX_DEFAULT = 7
THEME_N_MAX: dict[str, int] = {
    NEURAL_THEME_NAME: 18,
}

SUPP1_THEME_ORDER = (
    "Coagulation\n& complement",
    "Hepatic\nacute-phase",
    "Steroid hormone\nmetabolism",
    "GH–IGF axis",
    "Apolipoproteins & lipid handling",
)


def theme_genes(theme_name: str) -> list[str]:
    if theme_name in MAIN_THEMES:
        return MAIN_THEMES[theme_name]
    return SUPPL_THEMES[theme_name]


def theme_label(theme_name: str) -> str:
    if theme_name in MAIN_THEME_LABELS:
        return MAIN_THEME_LABELS[theme_name].replace("\n", " ")
    return theme_name.replace("\n", " ")


def make_figure():
    init_figure_theme()

    med_ol = load_mediation("olink")
    med_so = load_mediation("soma")
    ac_ol = pl.concat(
        [
            load_across_treatment("olink", 24),
            load_across_treatment("olink", 72),
        ]
    )
    ac_so = pl.concat(
        [
            load_across_treatment("soma", 24),
            load_across_treatment("soma", 72),
        ]
    )

    # Pre-classify Wk72 mediation results for shadow-row lookups.
    full_ol_72 = classify_mediation(
        med_ol.filter(pl.col("visit") == 20),
        ac_df=ac_ol,
        week=72,
    )
    full_so_72 = classify_mediation(
        med_so.filter(pl.col("visit") == 20),
        ac_df=ac_so,
        week=72,
    )

    row1 = [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3
    row2 = [1] * 3 + [2] * 3 + [5] * 3 + [6] * 3
    row3 = [7] * 3 + [8] * 3 + [9] * 3 + [10] * 3
    theme_panel_ids: dict[str, tuple[int, int]] = {
        "Coagulation\n& complement": (1, 2),
        "Hepatic\nacute-phase": (3, 4),
        "Steroid hormone\nmetabolism": (5, 6),
        "GH–IGF axis": (7, 8),
        "Apolipoproteins & lipid handling": (9, 10),
    }

    figwidth, figheight = nature_figsize("2col", 7.3)
    fig, axs = uplt.subplots(
        array=[row1, row2, row3],
        figwidth=figwidth,
        figheight=figheight,
        share=False,
        hratios=(0.95, 0.95, 1.35),
        hspace=("0.56in", "0.72in"),
        wspace=(PAIR_WSPACE,),
        left="0.58in",
        right="0.32in",
        top="0.44in",
        bottom="0.82in",
    )
    axs.format(abc=False)
    flat_axs = list(np.ravel(np.atleast_1d(axs)))
    pid_to_ax = {pid: ax for pid, ax in enumerate(flat_axs, start=1)}

    for theme_name in SUPP1_THEME_ORDER:
        panel_ol, panel_so = theme_panel_ids[theme_name]
        plot_theme_pair(
            pid_to_ax[panel_ol],
            pid_to_ax[panel_so],
            theme_name=theme_name,
            genes=theme_genes(theme_name),
            med_ol=med_ol,
            med_so=med_so,
            ac_olink_all=ac_ol,
            ac_soma_all=ac_so,
            full_ol_72=full_ol_72,
            full_so_72=full_so_72,
            n_max=THEME_N_MAX.get(theme_name, N_MAX_DEFAULT),
        )

    add_forest_legend(fig)
    fig.canvas.draw()
    for theme_name in SUPP1_THEME_ORDER:
        panel_ol, panel_so = theme_panel_ids[theme_name]
        draw_banner(
            fig,
            pid_to_ax[panel_ol],
            pid_to_ax[panel_so],
            bold_text=theme_label(theme_name),
            left_edge_text="Olink",
            right_edge_text="SomaScan",
        )

    fig.canvas.draw()
    for letter, theme_name in zip("abcde", SUPP1_THEME_ORDER, strict=False):
        panel_ol, _ = theme_panel_ids[theme_name]
        place_panel_letter(fig, pid_to_ax[panel_ol], letter, extra_dy_in=PAIR_LETTER_DY_IN)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    out_dir = _HERE
    save_figure(fig, str(out_dir / "supp_fig13_mediation_weight_mediated"), formats=("pdf",))
    print("Wrote supp_fig13_mediation_weight_mediated.pdf")
