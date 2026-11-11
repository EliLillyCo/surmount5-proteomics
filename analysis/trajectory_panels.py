"""Proteins shown in the trajectory supplementary figures (Supplementary Figs. 2, 3 and 5), chosen by a week-72 rule.

Criterion at the late visit (``LATE_WEEK``): the between-treatment (TZP - SEMA) FDR is below
``FDR_THRESHOLD``, the within-arm percent change from baseline has the same sign in both arms,
and the arm with the larger |change| has a within-arm FDR below ``FDR_THRESHOLD``. The rule
splits into three figures:

* ``TZP_INDUCED``: larger arm TZP, change > 0.
* ``TZP_SUPPRESSED``: larger arm TZP, change < 0.
* ``SEMA_GREATER``: larger arm SEMA, either direction.

Set aside before ranking: ``EXCLUDED_MARKERS`` (assays), ``EXOCRINE_GENES`` (drawn in the
exocrine pancreas figures) and ``SHOWN_IN_MAIN`` (drawn in Fig. 1 or Fig. 5). One assay is
kept per gene, the one with the lowest late between-treatment FDR; panels are ranked by that
FDR, ties by marker, and each platform shows at most ``TOP_K``. ``display_order`` then arranges the chosen panels by
trajectory class for drawing.
"""

from __future__ import annotations

import polars as pl

from analysis.mmrm_outputs import across_treatment_csv
from analysis.design import ARMS, LATE_WEEK
from analysis.output_layout import trajectory_path
from analysis.platform_inputs import EXCLUDED_ASSAYS as EXCLUDED_MARKERS
from analysis.statistics import FDR_THRESHOLD
from analysis.marker_genes import load_marker_to_gene
from analysis.protein_labels import marker_labels

TOP_K = 8
TZP_INDUCED, TZP_SUPPRESSED, SEMA_GREATER = "tzp_induced", "tzp_suppressed", "sema_greater"
RULES = (TZP_INDUCED, TZP_SUPPRESSED, SEMA_GREATER)


# Pancreatic acinar genes: shown in the exocrine pancreas figures (Extended Data Fig. 4 and
# Supplementary Fig. 4), imported by figures/supp_fig_exocrine_pancreas.py.
EXOCRINE_GENES: frozenset[str] = frozenset(
    {
        # Pancreatic acinar enzymes / zymogens / cofactors
        "CPA1",
        "CPB1",
        "CTRB1",
        "CTRB2",
        "CTRC",
        "CTRL",
        "SPINK1",
        "PNLIP",
        "PNLIPRP1",
        "PNLIPRP2",
        "PRSS1",
        "PRSS2",
        "PRSS3",
        "CELA1",
        "CELA2A",
        "CELA3A",
        "CELA3B",
        "AMY1A",
        "AMY1B",
        "AMY1C",
        "AMY2A",
        "CEL",
        "CLPS",
        "GP2",
        "PLA2G1B",
        "CUZD1",  # zymogen granule protein
        # Acinar regulators / regenerating family
        "SERPINI2",  # pancpin: acinar-specific serpin
        "REG1A",
        "REG1B",
        "REG3A",
        "REG3G",
        "KLK1",
    }
)

# Genes drawn in Fig. 1 (trajectory panels) and Fig. 5 (PPY).
SHOWN_IN_MAIN: frozenset[str] = frozenset(
    {
        # Fig. 1 trajectory panel.
        "LEP",
        "IGFBP2",
        "PNLIPRP1",
        "FABP4",
        "PAPPA",
        "GAST",
        "COL26A1",
        "AMDHD1",
        "ADIPOQ",
        "AZGP1",
        "NPW",
        "CRP",
        # Fig. 5 (PPY).
        "PPY",
    }
)


def qualifying(frame: pl.DataFrame, rule: str) -> pl.DataFrame:
    """Rows of ``frame`` meeting the week-72 rule for one figure, before exclusions and ranking.

    ``frame`` needs ``fdr_between_late``, ``pcbl_{TZP,SEMA}_late`` (within-arm percent change
    from baseline at the late visit) and ``fdr_{TZP,SEMA}_late`` (within-arm FDR).
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    tzp, sema = pl.col("pcbl_TZP_late"), pl.col("pcbl_SEMA_late")
    shared = (pl.col("fdr_between_late") < FDR_THRESHOLD) & (tzp * sema > 0)
    tzp_larger = shared & (tzp.abs() > sema.abs()) & (pl.col("fdr_TZP_late") < FDR_THRESHOLD)
    if rule == TZP_INDUCED:
        return frame.filter(tzp_larger & (tzp > 0))
    if rule == TZP_SUPPRESSED:
        return frame.filter(tzp_larger & (tzp < 0))
    return frame.filter(shared & (sema.abs() > tzp.abs()) & (pl.col("fdr_SEMA_late") < FDR_THRESHOLD))


def _eligible(frame: pl.DataFrame, rule: str, excluded_markers: tuple[str, ...]) -> pl.DataFrame:
    """Qualifying rows after the exclusions, one assay per gene, ranked."""
    return (
        qualifying(frame, rule)
        .filter(
            ~pl.col("marker").is_in(excluded_markers)
            & ~pl.col("gene").is_in(EXOCRINE_GENES | SHOWN_IN_MAIN)
        )
        .sort("fdr_between_late", "marker")
        .unique("gene", keep="first", maintain_order=True)
    )


def select_panels(
    frame: pl.DataFrame, rule: str, *, top_k: int = TOP_K, excluded_markers: tuple[str, ...] = ()
) -> pl.DataFrame:
    """The ``top_k`` panels of one figure: rule, exclusions, best assay per gene, ranked by late between-treatment FDR."""
    return _eligible(frame, rule, excluded_markers).head(top_k)


def display_order(panels: list[tuple[str, str, str, str]]) -> list[tuple[str, str, str, str]]:
    """Drawing order of chosen (marker, gene, TZP class, SEMA class) panels, given in rank order.

    Panels with the same class in both arms come first (increases before decreases), then
    those where the arms differ and TZP has a class, then those with no TZP change ("Null").
    Rank order is kept within each group (the sort is stable).
    """

    def group(panel: tuple[str, str, str, str]) -> tuple[int, int]:
        _, _, tzp, sema = panel
        if tzp == sema:
            return 0, 0 if tzp.endswith("↑") else 1
        return (2 if tzp == "Null" else 1), 0

    return sorted(panels, key=group)


def count_qualifying(frame: pl.DataFrame, rule: str, *, excluded_markers: tuple[str, ...] = ()) -> int:
    """Genes meeting the rule once exclusions are applied (what the figure legend reports)."""
    return _eligible(frame, rule, excluded_markers).height


def load_panel_frame(platform: str) -> pl.DataFrame:
    """Per-marker gene, late within-arm changes and FDRs (trajectory parquet) joined to the late between-treatment FDR."""
    trajectory = pl.read_parquet(trajectory_path(platform))
    between = pl.read_csv(across_treatment_csv(platform, LATE_WEEK)).select(
        "marker", pl.col("fdr").alias("fdr_between_late")
    )
    genes = load_marker_to_gene(platform)
    labels = marker_labels(platform)
    return trajectory.select(
        "marker",
        pl.col("marker").replace_strict(genes, default=pl.col("marker")).alias("gene"),
        pl.col("marker").replace_strict(labels, default=pl.col("marker")).alias("label"),
        *(
            expr
            for arm in tuple(ARMS)
            for expr in (
                pl.col(f"PCBL_{arm}_{LATE_WEEK}").alias(f"pcbl_{arm}_late"),
                pl.col(f"fdr_{arm}_{LATE_WEEK}").alias(f"fdr_{arm}_late"),
            )
        ),
    ).join(between, on="marker")


def panel_summary(platform: str, rule: str) -> dict:
    """Qualifying count and chosen markers of one figure on one platform."""
    frame = load_panel_frame(platform)
    excluded = EXCLUDED_MARKERS[platform]
    chosen = select_panels(frame, rule, excluded_markers=excluded)
    return {
        "n_qualifying": count_qualifying(frame, rule, excluded_markers=excluded),
        "chosen": chosen["marker"].to_list(),
        "genes": chosen["gene"].to_list(),
    }
