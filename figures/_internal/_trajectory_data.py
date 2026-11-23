"""Stage outputs the trajectory line panels read: within-arm %CFB, between-treatment FDR, classes.

Every trajectory figure (Fig 1, Fig 5 and the supplementary trajectory figures) reads its
panels through these loaders. Each reader is cached per platform (and arm and week), so a
figure that draws many panels reads each CSV or parquet once.
"""

from __future__ import annotations

from functools import lru_cache

import polars as pl

from analysis.design import ARMS, POST_BASELINE_WEEKS
from analysis.external.vendor_assay_tables import assay_genes
from analysis.mmrm_outputs import across_treatment_csv, pcbl_csv
from analysis.output_layout import UNIPROT_MAP, trajectory_path

# {(arm, week): (PCBL, SE_PCBL, lower CI, upper CI)} of one assay, post-baseline weeks only.
PanelData = dict[tuple[str, int], tuple[float, float, float, float]]


@lru_cache(maxsize=8)
def load_pcbl(platform: str, arm_full: str, week: int) -> pl.DataFrame:
    """Within-arm percent change from baseline of every assay at one week.

    Args:
        platform: ``"olink"`` or ``"soma"``.
        arm_full: Analysis ``TRT01A`` value (a value of ``analysis.design.ARMS``).
        week: Post-baseline week.

    Returns:
        ``marker``, ``PCBL``, ``SE_PCBL``, ``PCBL_lower_CI``, ``PCBL_upper_CI``.
    """
    return pl.read_csv(pcbl_csv(platform, arm_full, week)).select(
        "marker", "PCBL", "SE_PCBL", "PCBL_lower_CI", "PCBL_upper_CI"
    )


@lru_cache(maxsize=4)
def load_across_treatment(platform: str, week: int) -> pl.DataFrame:
    """TZP-vs-SEMA between-treatment contrast of every assay at one week.

    Args:
        platform: ``"olink"`` or ``"soma"``.
        week: Post-baseline week.

    Returns:
        ``marker``, ``FC``, ``fdr``.
    """
    return pl.read_csv(across_treatment_csv(platform, week)).select("marker", "FC", "fdr")


@lru_cache(maxsize=4)
def load_trajectory_table(platform: str) -> pl.DataFrame:
    """The platform's trajectory-class table (``analysis/outputs/trajectory_<platform>.parquet``)."""
    return pl.read_parquet(trajectory_path(platform))


@lru_cache(maxsize=1)
def genes_per_platform() -> dict[str, frozenset[str]]:
    """Gene symbols measured on each platform, from the UniProt map."""
    m = pl.read_parquet(UNIPROT_MAP)
    return {
        "olink": frozenset(m.filter(pl.col("present_olink"))["gene_symbol"].to_list()),
        "soma": frozenset(m.filter(pl.col("present_soma"))["gene_symbol"].to_list()),
    }


def best_marker_for_gene(platform: str, gene: str) -> str | None:
    """The gene's assay with the largest absolute post-baseline %CFB in either arm.

    Args:
        platform: ``"olink"`` or ``"soma"``.
        gene: Gene symbol.

    Returns:
        The assay ID, or None when no assay of the gene is in the trajectory table.
    """
    markers = [m for m, genes in assay_genes(platform).items() if gene in genes]
    if not markers:
        return None
    candidates = (
        load_trajectory_table(platform)
        .filter(pl.col("marker").is_in(markers))
        .with_columns(
            pl.max_horizontal(
                *(
                    pl.col(f"PCBL_{arm}_{week}").abs()
                    for arm in ARMS
                    for week in POST_BASELINE_WEEKS
                )
            ).alias("max_abs")
        )
        .sort("max_abs", descending=True)
    )
    return None if candidates.is_empty() else candidates["marker"][0]


def marker_pcbl(platform: str, marker: str) -> PanelData:
    """Within-arm %CFB with SE and 95% CI of one assay at each post-baseline week.

    Args:
        platform: ``"olink"`` or ``"soma"``.
        marker: Assay ID.

    Returns:
        ``{(arm, week): (PCBL, SE, lower CI, upper CI)}``, arms in ``ARMS`` order; a week the
        assay has no row for is absent.
    """
    data: PanelData = {}
    for arm_short, arm_full in ARMS.items():
        for week in POST_BASELINE_WEEKS:
            row = load_pcbl(platform, arm_full, week).filter(pl.col("marker") == marker)
            if row.is_empty():
                continue
            data[(arm_short, week)] = (
                row["PCBL"][0],
                row["SE_PCBL"][0],
                row["PCBL_lower_CI"][0],
                row["PCBL_upper_CI"][0],
            )
    return data


def across_treatment_fdr(platform: str, marker: str) -> dict[int, float | None]:
    """Between-treatment FDR of one assay at each post-baseline week.

    Args:
        platform: ``"olink"`` or ``"soma"``.
        marker: Assay ID.

    Returns:
        ``{week: fdr}``, None for a week the assay has no row for.
    """
    out: dict[int, float | None] = {}
    for week in POST_BASELINE_WEEKS:
        row = load_across_treatment(platform, week).filter(pl.col("marker") == marker)
        out[week] = row["fdr"][0] if not row.is_empty() else None
    return out


def trajectory_classes(platform: str, marker: str) -> tuple[str, str] | None:
    """(TZP, SEMA) trajectory class of one assay, or None when it is not in the table."""
    row = load_trajectory_table(platform).filter(pl.col("marker") == marker)
    if row.is_empty():
        return None
    return row["trajectory_TZP"][0], row["trajectory_SEMA"][0]
