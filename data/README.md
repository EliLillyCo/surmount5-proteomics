# Data

This repository does not include clinical trial data. Protected raw and
participant-level data are governed by the SURMOUNT-5 trial data-sharing
agreement and cannot be redistributed.

## Trial registration

SURMOUNT-5 is registered at
[ClinicalTrials.gov (NCT05822830)](https://clinicaltrials.gov/study/NCT05822830).

## Proteomics platforms

The analysis uses two proteomics platforms run on matched plasma samples at
Weeks 0, 24, and 72:

- **Olink Explore HT** (5,416 assays, log2 NPX scale)
- **SomaScan 11K** (10,771 aptamers, linear RFU scale)

## Clinical phenotype data

Clinical phenotype data follows CDISC ADaM standards. The MMRM models adjust
for age and sex as covariates. Treatment arms are tirzepatide 15 mg or MTD
versus semaglutide 2.4 mg or MTD.

## Data access

To reproduce the analyses, copy `paths.example.yaml` to `paths.yaml` and fill
in the local paths to the mounted data roots. See the top-level
[README](../README.md) for details.

## External reference data

### Vendor panel manifests

The cross-platform protein ID mapping (`analysis/outputs/uniprot_map.parquet`)
is built from two vendor panel reference files stored at
`analysis/reference/`. These are publicly available from the vendor websites.

**Olink Explore HT** (`analysis/reference/olink_explore_ht.tsv`):
download from https://olink.com/products/olink-explore-ht.

Required columns:

| Column | Description |
|---|---|
| `OlinkID` | Olink assay identifier |
| `Gene_Symbol` | HGNC gene symbol (pipe-delimited for multi-mapped assays) |
| `UniProt_ID` | UniProt accession (pipe-delimited, positionally aligned with Gene_Symbol) |

**SomaScan 11K** (`analysis/reference/somascan_11k.tsv`):
download from https://menu.somalogic.com.

Required columns:

| Column | Description |
|---|---|
| `SeqId` | SomaScan aptamer identifier (e.g. "12345-6") |
| `Gene_Symbol` | HGNC gene symbol (pipe-delimited for multi-mapped aptamers) |
| `UniProt_ID` | UniProt accession (pipe-delimited, positionally aligned with Gene_Symbol) |

Regenerate the cross-platform mapping with
`uv run python analysis/scripts/build_uniprot_map.py`.

### Published supplementary data

Three external datasets are used by the supplementary figures. All are derived
from published sources and can be regenerated locally.

**Hormone Cell Atlas** (Fei, Huang-Doran et al., *Science* 2026,
[doi:10.1126/science.aeb2672](https://doi.org/10.1126/science.aeb2672)):

- *Table S6A* (hormone-receiving cell type atlas): download the supplementary
  zip from
  https://www.science.org/doi/suppl/10.1126/science.aeb2672/suppl_file/science.aeb2672_tables_s1_to_s11.zip
  and extract Table S6A to `data/external/science_aeb2672_tableS6A.parquet`.
- *Pancreas GIPR/GLP1R per-cell-type expression* (recomputed from the pancreas
  h5ad for supp_fig15): regenerate with
  `uv run python analysis/external/recompute_hormone_atlas_pancreas.py`
  (downloads ~911 MB h5ad from
  https://cellgeni.cog.sanger.ac.uk/hormonecellatlas/download/)

### Published QC reference tables

The sex- and age-concordance QC checks (`analysis/qc/sex_concordance.py`,
`analysis/qc/age_concordance.py`) require two publicly available supplementary
tables.  Both are cited in the manuscript (refs 52, 53) and are **not
distributed with the repository** (`*.tsv` is gitignored).

Fetch and extract them with:

```bash
uv run python analysis/scripts/qc_reference_tables.py
```

Expected output locations (hardcoded in the QC modules):

```
analysis/qc/data/41467_2025_59034_MOESM3_ESM.Table_S2.tsv   # sex associations
analysis/qc/data/41591_2024_3164_MOESM3_ESM.Table_S1.tsv    # ProtAge proteins
```

| Table | Publication | DOI |
|---|---|---|
| Sex associations (Table S2) | Koprulu M, et al. *Nat Commun* 16, 4001 (2025) | [10.1038/s41467-025-59034-4](https://doi.org/10.1038/s41467-025-59034-4) |
| ProtAge 204 proteins (Table S1) | Argentieri MA, et al. *Nat Med* 30, 2450–2460 (2024) | [10.1038/s41591-024-03164-7](https://doi.org/10.1038/s41591-024-03164-7) |

**Maretty et al. STEP 1/2 semaglutide proteomics** (Maretty L et al., *Nat Med*
2025, [doi:10.1038/s41591-024-03355-2](https://doi.org/10.1038/s41591-024-03355-2)):
used for the cross-study comparison figures (supp_fig9, supp_fig10, fig5).
Regenerate with `uv run python analysis/scripts/maretty_step_data.py`, which
downloads the supplementary Excel from the publisher and writes
`analysis/outputs/maretty_step_data.parquet`.

## Derived outputs

All derived parquets under `analysis/outputs/` are regenerable and not
committed to the repository. After downloading the vendor panel manifests
(see above) and configuring `paths.yaml`, regenerate the cross-platform
protein mapping with:

```bash
uv run python analysis/scripts/build_uniprot_map.py
```

This writes `analysis/outputs/uniprot_map.parquet`, which maps protein IDs
across Olink and SomaScan panels and is used by the figure and table code.
