# Supplementary Tables Developer Guide

Reference for creating or modifying supplementary tables in this directory.

## Directory Structure

```
tables/
├── build_all.py                Master builder: assembles all sheets into one .xlsx
├── style.py                    Shared Excel styling (Nature Medicine spec)
├── _annotations.py             Shared marker annotation helper (UniProt/gene)
│
├── suppl_assay_qc.py           Tables S1–S2: Assay-level QC (Olink, SomaScan)
├── suppl_characteristics.py    Table S3: Baseline characteristics (Table-1 style)
├── suppl_baseline.py           Tables S4–S5: Baseline arm comparison
├── suppl_concordance.py        Table S6: Olink × SomaScan concordance
├── suppl_biomarker_correlation.py  Tables S7–S8: Proteomics vs clinical labs
├── suppl_trajectory.py         Tables S9–S10: Temporal proteomic response
├── suppl_across_treatment.py   Tables S11–S12: TZP vs SEMA treatment comparison
├── suppl_pathway_enrichment.py Tables S13–S14: Pathway enrichment (cameraPR, ORA)
├── suppl_mediation.py          Tables S15–S16: Weight-mediation decomposition
│
└── supplementary_tables.xlsx   Output workbook (16 sheets + Contents)
```

**Naming conventions:**
- `suppl_*.py` — table builder modules (each exposes `build()`)
- `_*.py` — internal shared modules (not standalone)
- `build_all.py` — orchestrator (run this to regenerate the workbook)

## Quick Start (Running the Build)

```bash
uv run python tables/build_all.py
```

Produces `tables/supplementary_tables.xlsx` with a styled Contents sheet,
hyperlinked TOC, and all 16 data sheets.

This workbook build expects mounted QC/MMRM/pathway/mediation inputs configured
via `paths.yaml`. For a no-data helper check, run:

```bash
uv run pytest analysis/tests/test_supplementary_tables.py -v
```

## Module Contract

Every `suppl_*.py` module must expose:

```python
def build() -> list[tuple[str, pl.DataFrame]]:
    """Return (sheet_name, polars_dataframe) pairs."""
    ...
```

The return list is ordered; `build_all.py` indexes into it via `MANIFEST`.

## Adding a New Table

1. Create `suppl_newanalysis.py` implementing the `build()` contract.
2. Import it in `build_all.py`.
3. Add entries to `MANIFEST`. Each tuple is `(tab_name, description, module, sheet_index)`.
4. If the sheet needs a footnote, add it to `FOOTNOTES`.
5. If new abbreviations are introduced, add them to `ABBREVIATIONS`.
6. Run `uv run python tables/build_all.py` to verify.

## MANIFEST (Sheet Order)

| Tab | Description | Module | Index |
|-----|-------------|--------|-------|
| Table S1 | Olink Explore HT assay QC | suppl_assay_qc | 0 |
| Table S2 | SomaScan 11K assay QC | suppl_assay_qc | 1 |
| Table S3 | Baseline characteristics of the proteomic substudy | suppl_characteristics | 0 |
| Table S4 | Baseline comparison, TZP vs SEMA (Olink) | suppl_baseline | 0 |
| Table S5 | Baseline comparison, TZP vs SEMA (SomaScan) | suppl_baseline | 1 |
| Table S6 | Cross-platform concordance (Olink × SomaScan) | suppl_concordance | 0 |
| Table S7 | Biomarker correlation (Olink vs clinical labs) | suppl_biomarker_correlation | 0 |
| Table S8 | Biomarker correlation (SomaScan vs clinical labs) | suppl_biomarker_correlation | 1 |
| Table S9 | Temporal proteomic response (Olink) | suppl_trajectory | 0 |
| Table S10 | Temporal proteomic response (SomaScan) | suppl_trajectory | 1 |
| Table S11 | Treatment comparison, TZP vs SEMA (Olink) | suppl_across_treatment | 0 |
| Table S12 | Treatment comparison, TZP vs SEMA (SomaScan) | suppl_across_treatment | 1 |
| Table S13 | Pathway enrichment — cameraPR (TZP vs SEMA) | suppl_pathway_enrichment | 0 |
| Table S14 | Pathway enrichment — ORA (TZP vs SEMA) | suppl_pathway_enrichment | 1 |
| Table S15 | Weight-mediation results, TZP vs SEMA (Olink) | suppl_mediation | 0 |
| Table S16 | Weight-mediation results, TZP vs SEMA (SomaScan) | suppl_mediation | 1 |

### Sheet-specific footnotes

Footnotes (italicized below the data block) are registered in the `FOOTNOTES` dict
in `build_all.py`. Currently configured:

| Tab | Footnote |
|-----|----------|
| Table S3 | Data are mean (SD) for continuous variables and n (%) for categorical variables unless otherwise indicated. |
| Table S7 | Lab values from VISITNUM 1 (screening); proteomics from VISITNUM 2 (randomization) for Wk0. |
| Table S8 | Lab values from VISITNUM 1 (screening); proteomics from VISITNUM 2 (randomization) for Wk0. |

## Shared Module: `_annotations.py`

Provides `build_marker_annotations() -> tuple[pl.DataFrame, pl.DataFrame]` returning
`(olink_annot, soma_annot)` DataFrames keyed by `marker` with columns:

| Column | Description |
|--------|-------------|
| `marker` | Platform-native ID (OlinkID or SeqId) |
| `UniProt` | UniProt accession(s), `"; "` joined if multi-mapped |
| `Gene_Symbol` | HGNC gene symbol(s), `"; "` joined if multi-mapped |
| `Assay` (Olink) / `Target` (Soma) | Vendor assay name |

Source: `analysis/outputs/uniprot_map.parquet` (authoritative mapping). Marker
IDs and vendor labels are paired positionally via simultaneous explode, so
multi-marker rows keep the correct Assay/Target names. `_annotations.py` also
provides `load_marker_gene_pairs()` for pathway tables, preserving many-to-many
marker→gene mappings without importing figure code.

## Excel Styling (`style.py`)

### Sheet Layout

```
Row 1: Table title (bold) — e.g. "Table S1: Olink Explore HT assay QC"
Row 2: Blank separator
Row 3: Column headers (bold, light gray fill #F2F2F2, top+bottom border)
Row 4+: Data rows
Last row: Bottom border
+2 below last: Footnote (italic, if configured)
```

Freeze panes at row 4 (headers always visible).

### Number Formats

`_get_number_format(col_name)` in `style.py` assigns an Excel number format by
matching column-name patterns (in priority order):

| Format | Matched when column name... | Examples |
|--------|-----------------------------|----------|
| `0.00E+00` | contains `P-value` or `FDR`, or equals `fdr_q` / `Spearman_P` | P-value, FDR, Spearman_P |
| `0.000` | starts with `log2FC` or `Detection_Rate_`, contains `_CI_`, or is in the `DECIMAL3_EXACT` set | log2FC, ACME_CI_lower, Spearman_Rho, Prop_Mediated |
| `0.00` | starts with `SE_` | SE_Wk24vsWk72 (overridden to `0.000` via DECIMAL3_EXACT) |
| `0.0` | starts with `PCBL` or equals `Intra_CV` | PCBL_TZP, PCBL_SEMA, Intra_CV |

Columns matching no pattern are left unformatted. The `DECIMAL3_EXACT` set holds
exact column names that take three decimals but do not fit the prefix/substring
patterns (e.g. `Spearman_Rho`, `Prob_Concordant`, `ACME`, `ADE`, `Total`).

To add formatting for new columns, extend the pattern checks in
`_get_number_format()` (or add the column to `DECIMAL3_EXACT`).

### Section Headers

Rows where column A is populated but all other columns are blank/None are
auto-bolded (used by the characteristics Table-1 format).

### Contents Sheet

`create_toc()` renders:
- Hyperlinked table names (click to jump)
- Table descriptions
- Abbreviations block (styled mini-table below)

## Data Sources

| Source | Path pattern | Used by |
|--------|-------------|---------|
| MMRM results | `paths.mmrm_dir(platform) / "finalRes"` | baseline, trajectory, across_treatment |
| Pathway results | `paths.mmrm_dir(platform) / "pathwayRes/acTrt"` | pathway_enrichment |
| Mediation results | `paths.mmrm_dir(platform) / "mediationRes"` | mediation |
| QC summaries | `paths.QC_OLINK_DIR` / `paths.QC_SOMA_DIR`; table readers use `surmount5_olink.assay_qc_summary.tsv` and `surmount5_soma.rfu_long.parquet` under those roots | assay_qc, biomarker_correlation, characteristics |
| Analysis outputs | `analysis/outputs/` | assay_qc annotations, concordance, trajectory classes, genotype frame |
| ADaM data | `paths.PHENO_DIR` | characteristics, biomarker_correlation |

## Conventions

- **Polars only** in build logic — conversion to pandas happens once in `build_all.py` for openpyxl.
- **No QC filtering** — all markers included regardless of allQC WARN status.
- **ADaM column names** preserved as-is (USUBJID, VISITNUM, AVAL, etc.).
- **Covariate key** — all MMRM results use `covar_AGE_SEX` subdirectory.
- **Rounding** — done in each module before returning (`.round(4)` for stats, varies for PCBL).
- **Sort order** — each module defines its own sort; generally Gene_Symbol > ID > Week.
- **Manifest** — `analysis/manifest.yaml` records the public script → input → output flow.
- **Mounted-input expectation** — `tables/build_all.py` is not a no-data smoke test; it expects mounted QC/MMRM/pathway/mediation inputs configured in `paths.yaml`.
- **Naming convention** — root directories come from `paths.py`; all file references use the `surmount5_*` prefix.
- **Cross-module imports** — shared marker metadata stays within `tables/_annotations.py`; table builders do not depend on figure scripts.

## Abbreviations (Contents Sheet)

The abbreviations block on the Contents sheet is defined by the `ABBREVIATIONS`
list in `build_all.py` (the single source of truth). Edit that list to add or
revise entries rather than duplicating it here.
