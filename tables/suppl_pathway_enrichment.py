"""Pathway-enrichment supplementary tables.

Two sheets, one per method:

  cameraPR — directional set test on the moderated-t ranking.
    One row per (term × database × platform × week × direction) at FDR<0.05.
    `lead_genes_full` is camera's full leading-edge gene list (unfiltered).

  ORA — Fisher's exact overrepresentation on the per-cell sig-gene list.
    One row per (term × database × platform × week) at FDR<0.05.
    No `direction` column (ORA is non-directional). `lead_genes_full` is
    restricted to genes also FDR<0.05 in the across-treatment contrast on
    the same (platform, week).

Both sheets carry a trailing `Display_Label` column that is non-null only
for terms that are curated into a published figure (fig3 or supp_fig11). The
displayed label is resolved through ``figures._internal._pathway_labels``
so the table text matches the figure text exactly — the cleaned database
term, with hand-shortened forms applied where appropriate and curated
overrides for two terms whose source DB phrasing was misleading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from paths import COVAR, MMRM_OLINK_DIR, MMRM_SOMA_DIR, mmrm_dir  # noqa: E402

from _annotations import load_marker_gene_pairs  # noqa: E402
from figures._internal._pathway_labels import pathway_label  # noqa: E402

FIGURE_MAPPING_CSV = REPO / "figures" / "configs" / "pathway_figure_terms.csv"
ORA_FIGURE_MAPPING_CSV = REPO / "figures" / "configs" / "pathway_ora_figure_terms.csv"
FDR_THRESHOLD = 0.05


def _files(method: str) -> dict[tuple[str, int], Path]:
    """Locate all 4 (platform × week) result files for a given method."""
    sub = "ora_combined.csv" if method == "ora" else "camera_combined.csv"
    contrast = "TZP15mgorMTDVSSEMA2.4mgorMTD"
    return {
        ("olink", 24): MMRM_OLINK_DIR / "pathwayRes" / "acTrt" / f"{contrast}@24" / sub,
        ("olink", 72): MMRM_OLINK_DIR / "pathwayRes" / "acTrt" / f"{contrast}@72" / sub,
        ("soma", 24): MMRM_SOMA_DIR / "pathwayRes" / "acTrt" / f"{contrast}@24" / sub,
        ("soma", 72): MMRM_SOMA_DIR / "pathwayRes" / "acTrt" / f"{contrast}@72" / sub,
    }


def _protein_files() -> dict[tuple[str, int], Path]:
    contrast = "TZP15mgorMTDVSSEMA2.4mgorMTD"
    return {
        (platform, week): (
            mmrm_dir(platform)
            / "finalRes"
            / "acTrt"
            / f"surmount5_{COVAR}_proteomics_olinkAnalysis_acrossTrts_resCmps_{contrast}@{week}_py.csv"
        )
        for platform in ("olink", "soma")
        for week in (24, 72)
    }


def _load_camera() -> pl.DataFrame:
    """Long-format cameraPR results across all 4 panels and 12 databases."""
    frames = []
    for (platform, week), path in _files("camera").items():
        df = (
            pl.read_csv(path)
            .select(
                [
                    pl.col("Gene_Set_Database").alias("database"),
                    pl.col("Term").alias("original_term"),
                    pl.col("Direction").alias("direction"),
                    pl.col("FDR q-val").alias("fdr_q"),
                    pl.col("N_Genes").alias("n_lead"),
                    pl.col("Total_Genes").alias("total_genes"),
                    pl.col("Lead_genes").alias("lead_genes_full"),
                ]
            )
            .with_columns(
                pl.lit(platform).alias("platform"),
                pl.lit(week).alias("week"),
            )
        )
        frames.append(df)
    return pl.concat(frames)


def _load_ora() -> pl.DataFrame:
    """Long-format ORA results across all 4 panels and 12 databases.

    ORA "Overlap" is a "k/K" string; split for n_lead / total_genes.
    """
    frames = []
    for (platform, week), path in _files("ora").items():
        df = (
            pl.read_csv(path)
            .with_columns(pl.col("Overlap").str.split("/").alias("_ov"))
            .with_columns(
                pl.col("_ov").list.get(0).cast(pl.Int64).alias("n_lead"),
                pl.col("_ov").list.get(1).cast(pl.Int64).alias("total_genes"),
            )
            .select(
                [
                    pl.col("Gene_Set_Database").alias("database"),
                    pl.col("Term").alias("original_term"),
                    pl.col("Adjusted P-value").alias("fdr_q"),
                    pl.col("n_lead"),
                    pl.col("total_genes"),
                    pl.col("Genes").alias("lead_genes_full"),
                ]
            )
            .with_columns(
                pl.lit(platform).alias("platform"),
                pl.lit(week).alias("week"),
            )
        )
        frames.append(df)
    return pl.concat(frames)


def _sig_genes_for_cell(
    proteins: pl.DataFrame,
    marker_gene: pl.DataFrame,
    sig_threshold: float,
) -> set[str]:
    matched = proteins.join(marker_gene, on="marker", how="inner").select(
        "gene_symbol",
        "fdr",
    )
    # Fallback: markers absent from the canonical marker→gene map use the vendor
    # Assay label so significant proteins still contribute to the lead-gene filter.
    fallback = proteins.join(marker_gene, on="marker", how="anti").select(
        pl.col("Assay").alias("gene_symbol"),
        "fdr",
    )
    merged = pl.concat([matched, fallback], how="vertical_relaxed")
    return set(
        merged.filter(pl.col("fdr") < sig_threshold)
        .get_column("gene_symbol")
        .drop_nulls()
        .unique()
        .to_list()
    )


def _per_cell_sig_genes(
    sig_threshold: float = FDR_THRESHOLD,
) -> dict[tuple[str, int], set[str]]:
    """(platform, week) → set of contrast-sig gene_symbols at the cell.

    Backed by the canonical many-to-many marker→gene lookup, so heterodimer
    SomaScan SeqIds (e.g. 20187-10 "Integrin aVb3" → both ITGB3 and ITGAV)
    correctly contribute significance to every gene they measure.
    """
    marker_gene = load_marker_gene_pairs()
    return {
        cell: _sig_genes_for_cell(
            pl.read_csv(path).select(["marker", "Assay", "fdr"]),
            marker_gene,
            sig_threshold,
        )
        for cell, path in _protein_files().items()
    }


def _filter_ora_to_sig_lead_genes(
    ora: pl.DataFrame,
    sig_threshold: float = FDR_THRESHOLD,
) -> pl.DataFrame:
    """For each ORA row, restrict `lead_genes_full` to genes that are
    FDR<sig_threshold in the across-treatment contrast on the same
    (platform, week). The upstream ORA pipeline occasionally includes
    stragglers using a slightly different threshold/mapping — this filter
    keeps the reported lead-gene strings coherent with the contrast data.
    Also updates `n_lead` to the filtered count so the column stays
    interpretable as "lead genes in this row".
    """
    sig_by_cell = _per_cell_sig_genes(sig_threshold)
    new_genes: list[str | None] = []
    new_n: list[int] = []
    for r in ora.iter_rows(named=True):
        raw = r["lead_genes_full"]
        if raw is None or raw == "":
            new_genes.append(None)
            new_n.append(0)
            continue
        sig_set = sig_by_cell.get((r["platform"], r["week"]), set())
        kept = [g for g in raw.split(";") if g in sig_set]
        new_genes.append(";".join(kept) if kept else None)
        new_n.append(len(kept))
    return ora.with_columns(
        pl.Series("lead_genes_full", new_genes),
        pl.Series("n_lead", new_n, dtype=pl.Int64),
    )


def _attach_display_label(df: pl.DataFrame, mapping: pl.DataFrame) -> pl.DataFrame:
    """Mark each row that appears in the figure mapping with its rendered label.

    The rendered ``display_label`` is the same string the figure paints in its
    left gutter (cleaned database term, with hand-shortened forms / curated
    overrides applied) — see ``_pathway_labels.pathway_label``. Rows whose
    (database, original_term) is not in any figure mapping get a null label.
    """
    rendered = mapping.select(
        [
            "database",
            "original_term",
            pl.col("original_term")
            .map_elements(pathway_label, return_dtype=pl.Utf8)
            .alias("display_label"),
        ]
    )
    return df.join(rendered, on=["database", "original_term"], how="left")


def _figure_mapping() -> pl.DataFrame:
    """Union of fig3 + supp_fig11 mappings, deduped on (database, original_term).

    fig3 takes precedence on collisions so its theme/curated context wins; the
    label resolution is identical for both since pathway_label() is a pure
    function of original_term.
    """
    fig3 = pl.read_csv(FIGURE_MAPPING_CSV).select(
        ["database", "original_term", "display_label", "theme_id", "theme_name"]
    )
    ora = pl.read_csv(ORA_FIGURE_MAPPING_CSV).select(
        ["database", "original_term", "display_label", "theme_id", "theme_name"]
    )
    combined = pl.concat([fig3, ora]).unique(
        subset=["database", "original_term"], keep="first"
    )
    return combined


def build() -> list[tuple[str, pl.DataFrame]]:
    """Return list of (sheet_name, DataFrame) for the pathway tables."""
    mapping = _figure_mapping()

    camera = _attach_display_label(_load_camera(), mapping).filter(
        pl.col("fdr_q") < FDR_THRESHOLD
    )
    ora = _attach_display_label(
        _filter_ora_to_sig_lead_genes(_load_ora()),
        mapping,
    ).filter(pl.col("fdr_q") < FDR_THRESHOLD)

    sort_keys = [
        pl.col("display_label").is_null().cast(pl.Int64),  # figure rows first
        "display_label",
        "platform",
        "week",
        "fdr_q",
    ]

    camera_sheet = camera.sort(sort_keys, nulls_last=True).select(
        [
            pl.col("database").alias("Database"),
            pl.col("original_term").alias("Term"),
            pl.col("direction").alias("Direction"),
            pl.col("platform").alias("Platform"),
            pl.col("week").alias("Week"),
            pl.col("fdr_q").alias("FDR"),
            pl.col("n_lead").alias("N_Lead"),
            pl.col("total_genes").alias("Total_Genes"),
            pl.col("lead_genes_full").alias("Lead_Genes"),
            pl.col("display_label").alias("Display_Label"),
        ]
    )

    ora_sheet = ora.sort(sort_keys, nulls_last=True).select(
        [
            pl.col("database").alias("Database"),
            pl.col("original_term").alias("Term"),
            pl.col("platform").alias("Platform"),
            pl.col("week").alias("Week"),
            pl.col("fdr_q").alias("FDR"),
            pl.col("n_lead").alias("N_Lead"),
            pl.col("total_genes").alias("Total_Genes"),
            pl.col("lead_genes_full").alias("Lead_Genes"),
            pl.col("display_label").alias("Display_Label"),
        ]
    )

    return [("cameraPR", camera_sheet), ("ORA", ora_sheet)]
