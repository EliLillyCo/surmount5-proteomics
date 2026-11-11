# SURMOUNT-5 proteomics paper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Analysis code for the manuscript on the plasma proteomic response to tirzepatide versus
semaglutide in adults with obesity (SURMOUNT-5), measured on Olink Explore HT and SomaScan 11K.

**Trial data are not included.** Summary-level proteomics results are available for
visualization through a public dashboard at <https://surmount5-proteomics.lilly.com>, and the
data are fully available in the Supplementary Tables. Requests for non-commercial use of
de-identified individual-level participant data from SURMOUNT-5 may be submitted to
[vivli.org](https://vivli.org). Every analysis stage, figure and table needs the trial data;
[Without the trial data](#without-the-trial-data) lists what runs without.

## Study

- **Trial**: SURMOUNT-5 ([NCT05822830](https://clinicaltrials.gov/study/NCT05822830)), phase 3b,
  open-label: tirzepatide (15 mg or maximum tolerated dose) vs semaglutide (2.4 mg or maximum
  tolerated dose) in adults with obesity or overweight without type 2 diabetes.
- **Proteomics timepoints**: weeks 0, 24 and 72.
- **Platforms**: Olink Explore HT (NPX, log2) and SomaScan 11K (RFU, linear).

## Repository structure

```
paths.py / paths.example.yaml   mount roots of the trial data and writable output roots
analysis/
  manifest.yaml     ordered stage list, inputs and outputs (see below)
  pipelines/        the entry point of every stage
  design.py         visit schedule, arm labels and other study constants
  qc_contract.py    what downstream code may rely on in the QC outputs
  qc/               Olink and SomaScan sample and assay QC
  mmrm/             MMRM engine (rpy2 -> R mmrm / emmeans)
  pathway/          cameraPR and over-representation gene-set enrichment
  mediation/        weight mediation (NumPy bootstrap)
  pharmacogenomics/ GIPR E354Q genotype x treatment model
  external/         checksum-verified downloads of third-party inputs
  tools/            manifest checker, stage runner and other repository tooling
scripts/            repository setup (setup-fonts)
figures/            publication figure scripts (figures/_internal: theme and helpers)
tables/             supplementary table builders
data/               third-party input sources and checksums (data/external/README.md)
```

## Requirements

- Linux x86-64 (the environment is locked for `linux-64`).
- `curl` and `awk` from the operating system.
- About 3 GB of disk for the environment and 1.5 GB for the third-party inputs, besides the
  trial data.

## Setup

```bash
pixi install --locked
pixi run check-r       # verify pinned R package versions
pixi run setup-fonts   # Inter font for figures (rerun after every pixi install; figures raise without it)
cp paths.example.yaml paths.yaml
```

Fill in `paths.yaml` with the mount roots of the trial data, the writable output roots and the
three workbooks that only a browser can download (`data/external/README.md`), or set
`SURMOUNT5_PATHS_YAML` to point at such a file. Only `mmrm_root`, `qc_root` and `pheno_root`
are required for the code to load; the checks in
[Without the trial data](#without-the-trial-data) run with the placeholder values.

## Reproducing the paper

`analysis/manifest.yaml` lists every stage in order with its command, inputs and outputs.

```bash
pixi run reproduce                  # every stage in order, then figures and tables
pixi run stage run <id> [<id> ...]  # one or more stages, in the order given
pixi run paper                      # figures and tables from existing outputs
```

The stages, in order:

<!-- stages:begin -->
| Stage | What it does |
|---|---|
| `qc_reference` | fetch the QC's published reference tables and cis-pQTL catalogs, parse them and stage the pinned SomaScan catalogs, checksum-verified |
| `hpa_disease_atlas` | download the HPA pan-disease blood atlas differential-expression table, checksum-verified |
| `ensembl_uniprot` | download the Ensembl release 116 gene to UniProt cross-references, checksum-verified |
| `hormone_atlas_pancreas_gpcr` | per-cell-type GIPR and GLP1R expression from the Hormone Cell Atlas pancreas h5ad |
| `science_aeb2672_tableS6A` | GIP-receiving cell types, Science Table S6A (Fei et al. 2026), from a browser-downloaded workbook |
| `vendor_assay_tables` | assay annotation from the vendor assay lists (Olink's public list, the SomaLogic menu) with the tracked UniProt gene-update layer, multi-target assays resolved per accession |
| `qc` | Olink and SomaScan sample and assay QC from the raw vendor files, ADaM and genotypes |
| `proteome_pca` | pooled per-platform proteome PCA and REML mixed-model variance attribution |
| `qc_exclusion_breakdown` | per-gate QC exclusion counts for the CONSORT diagram |
| `uniprot_map` | Olink and SomaScan marker to gene and UniProt map, the analyzed assays of the QC assay summaries with the accessions and genes of the assay annotation |
| `mmrm` | primary MMRM of change from baseline (TZP vs SEMA, weeks 24 and 72, age and sex), both platforms |
| `mmrm_sex_interaction` | treatment x visit x sex MMRM with per-sex covariance (Tables S13-S14) |
| `mediation` | weight mediation of the TZP vs SEMA effect at weeks 24 and 72, 100,000 bootstrap draws, seed 0 |
| `platform_concordance` | baseline Olink-SomaScan Spearman correlations of assay pairs sharing an annotated UniProt accession and their two-component GMM |
| `hcop` | HGNC's HCOP human-mouse ortholog predictions, for CellMarker's mouse gene sets |
| `gene_sets` | download the seven Enrichr gene-set libraries the pathway stage reads, checksum-verified |
| `pathway` | cameraPR and ORA over seven pinned Enrichr libraries on the TZP vs SEMA contrasts, both platforms |
| `trajectory_classification` | temporal response class of every assay in each arm (rules in analysis/trajectory.py) |
| `disease_reversal` | Fig 3b disease-signature reversal of the week-72 Olink TZP vs SEMA z, baseline-correlation rotation null |
| `maretty_step_data` | STEP 1 and STEP 2 semaglutide SomaScan effects from the Maretty et al. 2025 supplement |
| `wgs_pcs` | LD-pruned genome and SNPRelate genetic principal components for the E354Q model |
| `pharmacogenomics` | GIPR E354Q genotype x treatment MMRM for PPY (Supplementary Fig. 11) |
| `figures` | render every registered figure and write the Main, Extended Data and Supplementary PDFs |
| `tables` | build the 18-sheet supplementary tables workbook |
<!-- stages:end -->

`REPRODUCIBILITY.md` covers the mechanics: the provenance record each stage writes, which
stages `reproduce` skips and what `--force` clears, and the inputs this repository cannot
produce.

## Figures and tables

Each display item and the script or module that builds it. Figure PDFs are written to
`figures/output/` and the three bundles to `figures/` (the bundles carry a label strip per page
and no legends), and the supplementary workbook to `tables/supplementary_tables.xlsx`.

<!-- display-items:begin -->
| Item | Built by |
|---|---|
| Fig. 1 | [`figures/fig1_trajectory.py`](figures/fig1_trajectory.py) |
| Fig. 2 | [`figures/fig2_volcano.py`](figures/fig2_volcano.py) |
| Fig. 3 | [`figures/fig3_pathway.py`](figures/fig3_pathway.py) |
| Fig. 4 | [`figures/fig4_mediation.py`](figures/fig4_mediation.py) |
| Fig. 5 | [`figures/fig5_ppy.py`](figures/fig5_ppy.py) |
| Extended Data Fig. 1 | [`figures/supp_fig_study_design.py`](figures/supp_fig_study_design.py) |
| Extended Data Fig. 2 | [`figures/supp_fig_concordance.py`](figures/supp_fig_concordance.py) |
| Extended Data Fig. 3 | [`figures/supp_fig_biomarker_correlation.py`](figures/supp_fig_biomarker_correlation.py) |
| Extended Data Fig. 4 | [`figures/supp_fig_exocrine_pancreas.py`](figures/supp_fig_exocrine_pancreas.py) |
| Extended Data Fig. 5 | [`figures/supp_fig_cross_study.py`](figures/supp_fig_cross_study.py) |
| Extended Data Fig. 6 | [`figures/supp_fig_mediation_week24.py`](figures/supp_fig_mediation_week24.py) |
| Extended Data Fig. 7 | [`figures/supp_fig_gipr_glp1r_celltype.py`](figures/supp_fig_gipr_glp1r_celltype.py) |
| Supplementary Fig. 1 | [`figures/supp_fig_proteome_pca.py`](figures/supp_fig_proteome_pca.py) |
| Supplementary Fig. 2 | [`figures/supp_fig_tzp_induced.py`](figures/supp_fig_tzp_induced.py) |
| Supplementary Fig. 3 | [`figures/supp_fig_tzp_suppressed.py`](figures/supp_fig_tzp_suppressed.py) |
| Supplementary Fig. 4 | [`figures/supp_fig_exocrine_pancreas.py`](figures/supp_fig_exocrine_pancreas.py) |
| Supplementary Fig. 5 | [`figures/supp_fig_sema_greater.py`](figures/supp_fig_sema_greater.py) |
| Supplementary Fig. 6 | [`figures/supp_fig_tzp_opposing.py`](figures/supp_fig_tzp_opposing.py) |
| Supplementary Fig. 7 | [`figures/supp_fig_cross_study_pcbl.py`](figures/supp_fig_cross_study_pcbl.py) |
| Supplementary Fig. 8 | [`figures/supp_fig_pathway_ora.py`](figures/supp_fig_pathway_ora.py) |
| Supplementary Fig. 9 | [`figures/supp_fig_mediation_weight_mediated.py`](figures/supp_fig_mediation_weight_mediated.py) |
| Supplementary Fig. 10 | [`figures/supp_fig_mediation_heterogeneous.py`](figures/supp_fig_mediation_heterogeneous.py) |
| Supplementary Fig. 11 | [`figures/supp_fig_e354q.py`](figures/supp_fig_e354q.py) |
| Table S1 (Olink Explore HT assay annotations) | [`tables/suppl_assay_qc.py`](tables/suppl_assay_qc.py) |
| Table S2 (SomaScan 11K assay annotations) | [`tables/suppl_assay_qc.py`](tables/suppl_assay_qc.py) |
| Table S3 (Baseline characteristics of the proteomic substudy) | [`tables/suppl_characteristics.py`](tables/suppl_characteristics.py) |
| Table S4 (Baseline comparison, TZP vs SEMA (Olink)) | [`tables/suppl_baseline.py`](tables/suppl_baseline.py) |
| Table S5 (Baseline comparison, TZP vs SEMA (SomaScan)) | [`tables/suppl_baseline.py`](tables/suppl_baseline.py) |
| Table S6 (Cross-platform concordance (Olink × SomaScan)) | [`tables/suppl_concordance.py`](tables/suppl_concordance.py) |
| Table S7 (Biomarker correlation (Olink vs clinical labs)) | [`tables/suppl_biomarker_correlation.py`](tables/suppl_biomarker_correlation.py) |
| Table S8 (Biomarker correlation (SomaScan vs clinical labs)) | [`tables/suppl_biomarker_correlation.py`](tables/suppl_biomarker_correlation.py) |
| Table S9 (Temporal proteomic response (Olink)) | [`tables/suppl_trajectory.py`](tables/suppl_trajectory.py) |
| Table S10 (Temporal proteomic response (SomaScan)) | [`tables/suppl_trajectory.py`](tables/suppl_trajectory.py) |
| Table S11 (Treatment comparison, TZP vs SEMA (Olink)) | [`tables/suppl_across_treatment.py`](tables/suppl_across_treatment.py) |
| Table S12 (Treatment comparison, TZP vs SEMA (SomaScan)) | [`tables/suppl_across_treatment.py`](tables/suppl_across_treatment.py) |
| Table S13 (Sex × treatment interaction, TZP vs SEMA (Olink)) | [`tables/suppl_sex_interaction.py`](tables/suppl_sex_interaction.py) |
| Table S14 (Sex × treatment interaction, TZP vs SEMA (SomaScan)) | [`tables/suppl_sex_interaction.py`](tables/suppl_sex_interaction.py) |
| Table S15 (Pathway enrichment — cameraPR (TZP vs SEMA)) | [`tables/suppl_pathway_enrichment.py`](tables/suppl_pathway_enrichment.py) |
| Table S16 (Pathway enrichment — ORA (TZP vs SEMA)) | [`tables/suppl_pathway_enrichment.py`](tables/suppl_pathway_enrichment.py) |
| Table S17 (Weight-mediation results, TZP vs SEMA (Olink)) | [`tables/suppl_mediation.py`](tables/suppl_mediation.py) |
| Table S18 (Weight-mediation results, TZP vs SEMA (SomaScan)) | [`tables/suppl_mediation.py`](tables/suppl_mediation.py) |
<!-- display-items:end -->

## Without the trial data

These checks confirm that the environment installs and the code is consistent; they
reproduce no number in the paper.

```bash
cp paths.example.yaml paths.yaml   # the placeholder roots are enough; no mounts are needed
pixi run validate
```

`validate` runs the manifest check (it prints `Manifest/config checks OK`), the static
figure-style audit, and a byte-compile of `analysis`, `figures`, `tables` and `paths.py`.

## Citation

If you use this code, please cite the associated publication (see
[CITATION.cff](CITATION.cff)):

> Chiou J, Dunn JP, Dimitriadis GK, Falcon B, Yan D, O'Dushlaine C, Coghlan M, Harris C,
> Titchenell P. Differential plasma proteomic responses between tirzepatide and semaglutide in
> adults with obesity: an exploratory analysis of SURMOUNT-5. *TODO: journal*. 2026. DOI: TODO

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Eli Lilly and Company.
