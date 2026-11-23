# Supplementary Tables Developer Guide

Reference for creating or modifying supplementary tables in this directory.

## Directory Structure

```
tables/
├── build_all.py                Master builder: assembles all sheets into one .xlsx
├── style.py                    Shared Excel styling
├── _annotations.py             Shared marker annotation helper (UniProt/gene)
├── __init__.py                 Makes `tables` a package (imports resolve from the repo root via PYTHONPATH)
│
├── suppl_assay_qc.py           Tables S1–S2: Assay-level QC and assay-quality annotation
├── suppl_characteristics.py    Table S3: Baseline characteristics (Table-1 style)
├── suppl_baseline.py           Tables S4–S5: Baseline arm comparison
├── suppl_concordance.py        Table S6: Olink × SomaScan concordance
├── suppl_biomarker_correlation.py  Tables S7–S8: Proteomics vs clinical labs
├── suppl_trajectory.py         Tables S9–S10: Temporal proteomic response
├── suppl_across_treatment.py   Tables S11–S12: TZP vs SEMA treatment comparison
├── suppl_sex_interaction.py    Tables S13–S14: Sex × treatment interaction on TZP vs SEMA
├── suppl_pathway_enrichment.py Tables S15–S16: Pathway enrichment (cameraPR, ORA)
├── suppl_mediation.py          Tables S17–S18: Weight-mediation decomposition (analytic P, unconditional FDR, six-way class, proportion mediated with its 95% bootstrap CI)
│
└── supplementary_tables.xlsx   Output workbook (18 sheets + Contents)
```

**Naming conventions:**
- `suppl_*.py` — table builder modules (each exposes `build()`)
- `_*.py` — internal shared modules (not standalone)
- `build_all.py` — orchestrator (run this to regenerate the workbook)

## Quick Start (Running the Build)

```bash
pixi run python tables/build_all.py
```

Produces `tables/supplementary_tables.xlsx` with a styled Contents sheet,
hyperlinked TOC, and all 18 data sheets.

This workbook build expects mounted QC/MMRM/pathway/mediation inputs configured
via `paths.yaml`. For a no-data helper check, run `pixi run validate`.

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
2. Add entries to `MANIFEST` in `build_all.py`. Each tuple is `(tab_name, description, module, sheet_index)`;
   `build_all.py` imports the module by name (`importlib.import_module("tables.<module>")`).
3. If the sheet needs a footnote, add it to `FOOTNOTES`.
4. If new abbreviations are introduced, add them to `ABBREVIATIONS`.
5. If the module reads a new `analysis/outputs/` file, list the module in that stage's
   `downstream` in `analysis/manifest.yaml`, so the build's pre-flight checks the file.
6. Run `pixi run python tables/build_all.py --list-tables`, then `pixi run python tables/build_all.py` to verify.

## MANIFEST (Sheet Order)

| Tab | Description | Module | Index |
|-----|-------------|--------|-------|
| Table S1 | Olink Explore HT assay annotations | suppl_assay_qc | 0 |
| Table S2 | SomaScan 11K assay annotations | suppl_assay_qc | 1 |
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
| Table S13 | Sex × treatment interaction, TZP vs SEMA (Olink) | suppl_sex_interaction | 0 |
| Table S14 | Sex × treatment interaction, TZP vs SEMA (SomaScan) | suppl_sex_interaction | 1 |
| Table S15 | Pathway enrichment — cameraPR (TZP vs SEMA) | suppl_pathway_enrichment | 0 |
| Table S16 | Pathway enrichment — ORA (TZP vs SEMA) | suppl_pathway_enrichment | 1 |
| Table S17 | Weight-mediation results, TZP vs SEMA (Olink) | suppl_mediation | 0 |
| Table S18 | Weight-mediation results, TZP vs SEMA (SomaScan) | suppl_mediation | 1 |

### Sheet-specific footnotes

Footnotes (italicized below the data block) are registered in the `FOOTNOTES` dict
in `build_all.py`. Currently configured:

| Tab | Footnote |
|-----|----------|
| Table S1, Table S2 | Assay/HGNC annotation, external catalog citations and matching rules. |
| Table S3 | Descriptive statistics and maximum tolerated dose. |
| Table S4, Table S5 | Baseline contrast scale, covariates, direction and FDR family. |
| Table S6 | Baseline correlation, mixture posterior and shared-accession gene ordering. |
| Table S7, Table S8 | Screening/visit pairing, minimum paired participants and how a P value below double precision is shown. |
| Table S9, Table S10 | Within-arm estimates, trajectory classes and FDR families. |
| Table S11, Table S12 | Between-arm contrast scale, covariates and FDR families. |
| Table S13, Table S14 | Sex × treatment model, contrast scale, direction categories and FDR families. |
| Table S15, Table S16 | Pathway analysis and figure term mapping via `Display_Label`. |
| Table S17, Table S18 | Analytic ACME/ADE/total-effect tests and bootstrap proportion-mediated intervals. |

## Assay-quality annotation (Tables S1–S2)

`suppl_assay_qc.py` appends three assay-quality columns to the per-assay QC of Tables S1 (Olink) and
S2 (SomaScan). They are built by `analysis/assay_annotation.py` (pure functions over
frames, importable without mounted data; thin loaders `load_catalogs` and
`load_concordance`). Every assay row stays; the build raises if the annotation does not
cover every QC assay. The per-assay QC columns before them are the assay ID, `Gene_Symbol`,
`HGNC_ID`, `UniProt`, `Intra_CV` and `Detection_Rate_All` / `_SEMA` / `_TZP`.

| Column | Definition |
|--------|------------|
| `Cis_pQTL_Catalogs` | Catalogs (5 Olink, 8 SomaScan) with at least one cis-pQTL for the assay at -log10 p >= 7.30103 (p < 5e-8), in catalog order; blank when none. UKB-PPP, deCODE and Fenland publish only associations past stricter thresholds (p < 1.7e-11, p < 1.8e-9 and p < 1e-11), so for them the effective threshold is the publisher's; the Table S1/S2 footnote says so. The number of catalogs is the number listed. |
| `Other_Platform_Cis_pQTL` | The assay's UniProt has a cis-pQTL in at least one catalog of the other platform. |
| `Eldjarn2023_Tier` | The three tiers of Eldjarn et al. 2023. 1: cis-pQTL on both platforms and a concordant partner assay. 2: cis-pQTL on either platform, not Tier 1. 3: no cis-pQTL on either platform. Concordant: any partner pair in `analysis/outputs/platform_concordance.parquet` with `Concordant` (Table S6 lists the pairs). An assay with no partner pair is never Tier 1. |

`build_annotation` also returns the catalog count, the strongest -log10 p and its catalog, the per-catalog match method
(`native` or `UniProt`) and the concordance posterior and call; the tables do not print them.

Inputs and handling:

- **Catalogs**: the 13 published catalogs of `analysis/assay_annotation.py` (five Olink
  and eight SomaScan: deCODE, ARIC, Fenland, CKB Atlas, MESA, Genes & Health, AGES and
  INTERVAL), the roster of the two QC configs
  (`reference_catalog` in `analysis/qc/config_{olink,soma}.toml`), read from
  `data/external/pqtl/parsed/` (`paths.yaml` `pqtl_catalog_dir`; untracked staging,
  checksum-pinned). `ckb_soma` is the 2026 CKB Atlas. One threshold for all catalogs, ARIC included. The
  `SIGN_EXCLUSIONS` proteins (derived from the catalogs) are dropped and -log10 p is
  recomputed from beta/se where it is missing or >= 300 (`fill_underflowed_nlp`).
- **Matching**: SomaScan catalogs and Genes & Health Olink (Explore HT) by assay ID
  (SeqId with `_` read as `-`); UKB-PPP, JHS (Tahir et al. 2024), MESA and CKB Olink (Explore 3072) by
  UniProt accession token (split on `; _ , |`).

## External HPA and Genes & Health Seer annotations (Tables S1–S2)

The tables add three external annotation columns:

| Column | Source |
|---|---|
| `HPA_Secretome_Location` | HPA v25.1 bulk `Secretome location` |
| `HPA_Enriched_Tissues` | HPA v25.1 bulk `RNA tissue specific nTPM`, tissue names only (values dropped) |
| `GH_Seer_MS_Rho` | Pietzner et al. 2026 (Nat Genet) Genes & Health supplement, sheet ST5 (Olink Explore HT vs Seer Proteograph) or ST6 (SomaScan 11K vs Seer), `Spearman  corr.`, numeric |

HPA joins by `HGNC_ID` → HGNC `ensembl_gene_id` → HPA `Ensembl`; multi-gene assays keep the tables' `|` (genes of one
accession) and `; ` (accessions) delimiters. `GH_Seer_MS_Rho` matches by assay ID only, the reagent (Explore HT OlinkID, read
from the sheet's `<gene>_<digits>` ID as `OID<digits>`; SomaScan SeqId, `seq.10010.10` read as `10010-10`); the UniProt
column is not used. An assay matched to several Seer protein groups keeps the group with the largest N (ties: a group whose UniProt is one of the assay's accessions, then the lowest numeric Seer ID), and the value is blank when
that N is below 100 or the sheet has no correlation. The
workbook is the SHA-256-pinned `genes_health` file the pQTL catalogs read (`gh_seer_workbook`). Pietzner et al., *Nature
Genetics* (2026), doi:10.1038/s41588-026-02697-6; HPA v25.1 (CC BY 4.0; data available from v25.proteinatlas.org).

HPA_Secretome_Location gives HPA's predicted secretion destination; HPA_Enriched_Tissues names the tissues in which HPA reports the gene's RNA as enriched or enhanced (blank when none). GH_Seer_MS_Rho is the Spearman correlation of the same assay with nanoparticle-enrichment mass spectrometry (Seer Proteograph) in Genes & Health (Pietzner et al., Nat Genet 2026; same Olink Explore HT and SomaScan 11K platforms), matched by assay ID; blank when no matched assay or N<100. When an assay matched several Seer protein groups with N≥100, the group quantified in the most participants is shown.

## Table S6 columns (cross-platform concordance)

One row per Olink assay × SomaScan aptamer pair that shares a UniProt accession of the assay annotation
(`analysis/outputs/platform_concordance.parquet`).

| Column | Description |
|--------|-------------|
| `OlinkID`, `SeqId` | The pair |
| `Shared_UniProt` | Every shared accession, `;`-joined, sorted |
| `Shared_Gene_Symbol` | The annotated genes of each shared accession, `;`-joined in the order of `Shared_UniProt`; the genes of one accession (identical proteins from paralogous loci) `|`-joined, alphabetical (`CGB3|CGB5|CGB8`), as in Tables S1–S2 |
| `Spearman_Rho`, `Spearman_P`, `N` | Baseline Spearman correlation of the pair (Olink PCNormalizedNPX, SomaScan log2 RFU), its P and the participants |
| `Prob_Concordant`, `Is_Concordant` | Posterior probability of the high-correlation component of the two-component Gaussian mixture, and the call (> 0.5) |

## Shared Module: `_annotations.py`

Provides `build_marker_annotations(hgnc_ids=False) -> tuple[pl.DataFrame, pl.DataFrame]` returning
`(olink_annot, soma_annot)` DataFrames keyed by `marker` with columns:

| Column | Description |
|--------|-------------|
| `marker` | Platform-native ID (OlinkID or SeqId) |
| `UniProt` | UniProt accession(s), `"; "` joined if multi-mapped |
| `Gene_Symbol` | HGNC gene symbol(s) as UniProt 2026_03 records them, `"; "` joined across accessions, `|` within one accession (paralogs) |
| `HGNC_ID` (Tables S1–S2 only) | HGNC ID of each `Gene_Symbol` entry, same order and separators; an empty element when a gene has no HGNC identifier |
| `Assay` (Olink) / `Target` (Soma) | Vendor assay name |

Source: `analysis/outputs/uniprot_map.parquet` (authoritative mapping). Marker
IDs and vendor labels are paired positionally via simultaneous explode, so
multi-marker rows keep the correct Assay/Target names. The pathway tables take their
many-to-many marker→gene pairs from `analysis.marker_genes.load_marker_gene_pairs()`,
the same function the figures use.

## Excel Styling (`style.py`)

### Sheet Layout

```
Row 1: Table title (bold) — e.g. "Table S1: Olink Explore HT assay annotations"
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
| `0.00` | starts with `SE_` | SE_Wk72vsWk24 (overridden to `0.000` via DECIMAL3_EXACT) |
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

Every MMRM, pathway and mediation file name is built in `analysis/mmrm_outputs.py` or
`analysis/mediation/results.py`, under `paths.mmrm_dir(platform)`; every QC file is read
through `analysis/qc_contract.py`. The table modules never spell out a file name of either.

| Source | Reader | Used by |
|--------|--------|---------|
| MMRM `finalRes` (`acTrt`, `pcblRes`, `withinTrt`, `baseline`) | `mmrm_outputs.across_treatment_csv`, `pcbl_csv`, `within_treatment_csv`, `baseline_diff_csv` | across_treatment, trajectory, baseline, characteristics |
| MMRM `rawOutput/*_chg_diff.csv` (contrast `StdErr`, `df`) | `mmrm_outputs.chg_diff_csv`, `with_contrast_stderr` | across_treatment, trajectory |
| Sex interaction results | `mmrm_outputs.sexint_csv` | sex_interaction |
| Pathway results (`pathwayRes/acTrt/<contrast>@<week>/`) | `mmrm_outputs.pathway_csv` | pathway_enrichment |
| Mediation results (`mediationRes/`) | `mediation.results.load_platform` | mediation |
| QC assay summary, long value table, analysis cohort | `qc_contract.read_assay_summary`, `long_table_path`, `cohort_samples` | assay_qc, biomarker_correlation, characteristics |
| Parsed pQTL catalogs | `analysis.qc_paths.PQTL_CATALOG_DIR` (`paths.yaml` `pqtl_catalog_dir`, default `data/external/pqtl/parsed/`) | assay_qc annotation |
| HPA v25.1 bulk, Genes & Health supplement (Pietzner et al. 2026; ST5, ST6), HGNC complete set | `analysis.external.assay_table_annotations.read_hpa`, `read_gh_seer`, `read_hgnc` (SHA-256 verified) | assay_qc S1/S2 external annotations |
| Analysis outputs | `analysis/outputs/` | assay_qc annotation and concordance (`platform_concordance.parquet`), trajectory (`trajectory_<platform>.parquet`), every module through `uniprot_map.parquet` |
| ADaM data | `paths.PHENO_DIR` (`paths.yaml` `pheno_root`) | characteristics, biomarker_correlation |

`build_all.py` first checks that the `analysis/outputs/` files these modules read are present
(`manifest_stages.require_stage_outputs("tables")`) and names the manifest stage that
writes any missing one.

## Conventions

- **Polars only** in build logic — conversion to pandas happens once in `build_all.py` for openpyxl.
- **No assay filtering** — every modeled assay is included.
- **ADaM column names** preserved as-is (USUBJID, VISITNUM, AVAL, etc.).
- **Rounding** — done in each module before returning (`.round(4)` for stats, varies for PCBL).
- **Sort order** — each module defines its own sort; generally Gene_Symbol > ID > Week.
- **Manifest** — `analysis/manifest.yaml` records the public script → input → output flow.
- **Mounted-input expectation** — `tables/build_all.py` is not a no-data smoke test; it expects mounted QC/MMRM/pathway/mediation inputs configured in `paths.yaml`.
- **Paths** — root directories come from `paths.py` (`paths.yaml`); a missing file raises at the first read.
- **Cross-module imports** — shared marker metadata stays within `tables/_annotations.py`; table builders do not depend on figure scripts, except `suppl_pathway_enrichment.py`, which reads the two figure term lists (`figures/configs/pathway_figure_terms.csv`, `pathway_ora_figure_terms.csv`).

## Abbreviations (Contents Sheet)

The abbreviations block on the Contents sheet is defined by the `ABBREVIATIONS`
list in `build_all.py` (the single source of truth). Edit that list to add or
revise entries rather than duplicating it here.
