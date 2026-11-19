"""Curated mediation theme gene lists, shared by the manuscript figures.

The ``*_CANDIDATES`` dictionaries hold the full curated HGNC lists, unfiltered. The figures draw ``significant_theme_genes`` of a
list instead: the genes whose mediation total effect clears FDR < 0.05 on at least one
platform at the figure's visit (BH on the Sobel/Wald p-values, ``total_fdr`` from
``analysis.mediation.results.load_platform``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import polars as pl

from analysis.mediation.results import load_platform
from analysis.qc_contract import PLATFORMS
from analysis.statistics import FDR_THRESHOLD

NEURAL_THEME_NAME = "Neural"

# Fig. 4 themes. Insertion order is the row placement order.
MAIN_THEME_CANDIDATES: dict[str, list[str]] = {
    # Leptin & adiponectin — LEP/LEPR/LEPROT are the leptin receptor axis,
    # ADIPOQ is the partnered adipokine.
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
    # PLA2G1B). The figures draw those that clear the total-effect FDR gate
    # (``significant_theme_genes``).
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
    # lipid-handling receptors.
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
    # Intestinal epithelium — epithelial adhesion (EPCAM, GPA33, CDH17, CDHR5),
    # brush-border enzymes (TREH, ANPEP, ALPI, DPP4, TMPRSS15), the cytoskeletal
    # marker VIL1, the Fc receptor FCAMR, the apical-membrane enterocyte marker
    # MAMDC4, and REG4 (an intestinal goblet-cell marker).
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
    # INHBB, INHBC, INHBE, INHA, ACVR2A, ACVR2B).
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
    # and ephrin/chemokine/guidance signaling (EFNB2, CXCL12, NOTCH3, ROBO1).
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

# Supplementary Fig. 9 and 10 themes. Supplementary Fig. 9 also draws "GH–IGF axis" and
# "Apolipoproteins & lipid handling" from MAIN_THEME_CANDIDATES; Supplementary Fig. 10 draws
# "Vascular" and "Pancreatic exocrine".
SUPPLEMENT_THEME_CANDIDATES: dict[str, list[str]] = {
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
        # Synaptic adhesion: trans-synaptic organizers, cell adhesion
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

THEME_CANDIDATES: dict[str, list[str]] = {**MAIN_THEME_CANDIDATES, **SUPPLEMENT_THEME_CANDIDATES}


def significant_theme_genes(
    candidates: list[str],
    visit: int,
    marker_genes: Mapping[str, list[str]],
    *,
    platforms: tuple[str, ...] = PLATFORMS,
    loader: Callable[[str, int], pl.DataFrame] = load_platform,
) -> list[str]:
    """The candidate genes with a significant mediation total effect on at least one platform.

    Args:
        candidates: Curated HGNC symbols of a theme.
        visit: VISITNUM of the figure.
        marker_genes: Marker (OlinkID or SeqId) -> all HGNC symbols it measures, so a
            heterodimer aptamer counts for each subunit.
        platforms: Platforms to pool.
        loader: ``(platform, visit) -> results`` with ``marker`` and ``total_fdr``.

    Returns:
        The candidates (order and duplicates as given, de-duplicated) whose marker has
        ``total_fdr`` < ``FDR_THRESHOLD`` on any platform.
    """
    significant: set[str] = set()
    for platform in platforms:
        df = loader(platform, visit).filter(pl.col("total_fdr") < FDR_THRESHOLD)
        for marker in df["marker"]:
            significant.update(marker_genes.get(marker, []))
    return [g for g in dict.fromkeys(candidates) if g in significant]


def significant_themes(
    candidates: Mapping[str, list[str]],
    visit: int,
    marker_genes: Mapping[str, list[str]],
    *,
    platforms: tuple[str, ...] = PLATFORMS,
    loader: Callable[[str, int], pl.DataFrame] = load_platform,
) -> dict[str, list[str]]:
    """Filter every theme with ``significant_theme_genes``; themes left with no gene are dropped.

    Loads each platform once. The names of the dropped themes are the difference of the
    keys, so a caller can report them.
    """
    cache: dict[str, pl.DataFrame] = {p: loader(p, visit) for p in platforms}
    kept = {
        name: significant_theme_genes(
            genes, visit, marker_genes, platforms=platforms, loader=lambda p, _v: cache[p]
        )
        for name, genes in candidates.items()
    }
    return {name: genes for name, genes in kept.items() if genes}
