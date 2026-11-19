"""Pathway-enrichment supplementary tables.

Two sheets, one per method:

  cameraPR — directional set test on the t-like ranking z = log2 estimate / SE
    (inter-gene correlation 0.01, sets of 5-500 measured genes).
    One row per (term × database × platform × week × direction) at FDR<0.05.
    `lead_genes_full` is camera's full leading-edge gene list (unfiltered).

  ORA — Fisher's exact overrepresentation on the per-cell gene-level query (a gene's P is the
    Simes combination of its assays' P, BH over genes, FDR<0.05).
    One row per (term × database × platform × week) at FDR<0.05.
    No `direction` column (ORA is non-directional). The ORA overlap is gene-level;
    `Lead_Genes` lists overlap genes that also have an assay at FDR<0.05
    in the same platform and week, and `N_Lead` counts that screened list.
    `N_Lead` can be smaller than the overlap used for the ORA P value.

Both sheets carry a trailing `Display_Label` column that is non-null only
for terms that are curated into a published figure (fig3 or supp_fig_pathway_ora). The
displayed label is resolved through ``analysis.pathway_labels``
so the table text matches the figure text exactly — the cleaned database
term with the label overrides of ``analysis.pathway_labels``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from analysis.design import PLATFORMS, POST_BASELINE_WEEKS
from analysis.marker_genes import load_marker_gene_pairs
from analysis.mmrm_outputs import PATHWAY_NUMERIC_SCHEMAS, across_treatment_csv, pathway_csv
from analysis.pathway_labels import pathway_label
from analysis.statistics import FDR_THRESHOLD

REPO = Path(__file__).resolve().parent.parent

FIGURE_MAPPING_CSV = REPO / "figures" / "configs" / "pathway_figure_terms.csv"
ORA_FIGURE_MAPPING_CSV = REPO / "figures" / "configs" / "pathway_ora_figure_terms.csv"


def _files(method: str) -> dict[tuple[str, int], Path]:
    """Locate all 4 (platform × week) result files for a given method."""
    return {
        (platform, week): pathway_csv(platform, week, method)
        for platform in PLATFORMS
        for week in POST_BASELINE_WEEKS
    }


def _protein_files() -> dict[tuple[str, int], Path]:
    return {
        (platform, week): across_treatment_csv(platform, week)
        for platform in PLATFORMS
        for week in POST_BASELINE_WEEKS
    }


def _load_camera() -> pl.DataFrame:
    """Long-format cameraPR results across all 4 panels and 7 databases."""
    frames = []
    for (platform, week), path in _files("camera").items():
        df = (
            pl.read_csv(path, schema_overrides=PATHWAY_NUMERIC_SCHEMAS["camera"])
            .select(
                [
                    pl.col("Gene_Set_Database").alias("database"),
                    pl.col("Term").alias("original_term"),
                    pl.col("Direction").alias("direction"),
                    pl.col("FDR q-val").alias("fdr_q"),
                    pl.col("N_Genes").alias("n_lead"),
                    pl.col("Total_Genes").alias("total_genes"),
                    pl.col("Lead_genes").alias("lead_genes_full"),
                    pl.col("inter_gene_cor"),
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
    """Long-format ORA results across all 4 panels and 7 databases.

    ORA "Overlap" is a "k/K" string; split for n_lead / total_genes.
    """
    frames = []
    for (platform, week), path in _files("ora").items():
        df = (
            pl.read_csv(path, schema_overrides=PATHWAY_NUMERIC_SCHEMAS["ora"])
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
    # A marker with no marker→gene entry uses its vendor Assay label.
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

    Backed by the canonical many-to-many marker→gene lookup, so assay-level
    significance propagates to every gene a marker maps to.
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
    FDR<sig_threshold in the between-treatment contrast on the same
    (platform, week), and sets `n_lead` to the filtered count.
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
    overrides applied) — see ``analysis.pathway_labels.pathway_label``. Rows whose
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
    """Union of fig3 + supp_fig_pathway_ora mappings, deduped on (database, original_term).

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
        "database",
        "original_term",
    ]

    camera_sheet = camera.sort([*sort_keys, "direction"], nulls_last=True).select(
        [
            pl.col("database").alias("Database"),
            pl.col("original_term").alias("Term"),
            pl.col("direction").alias("Direction"),
            pl.col("platform").alias("Platform"),
            pl.col("week").alias("Week"),
            pl.col("fdr_q").alias("FDR"),
            pl.col("n_lead").alias("N_Measured"),
            pl.col("total_genes").alias("Total_Genes"),
            pl.col("lead_genes_full").alias("Lead_Genes"),
            pl.col("inter_gene_cor").alias("Inter_Gene_Cor"),
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
