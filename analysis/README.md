# Analysis code organization

This directory contains the manuscript-specific analysis code required to regenerate the derived outputs used by `figures/` and `tables/`.

The code is organized around the analysis steps that feed the paper figures and supplementary tables.

## Manifest

`analysis/manifest.yaml` is the central command map: its `analysis_outputs` stages are in run
order, each with its command, inputs, outputs and downstream consumers, and the README's
"Reproducing the paper" stage table and figure/table map are generated from it. Run order is in
`../README.md`, `pixi run manifest-list` prints the stage commands, and `../REPRODUCIBILITY.md`
covers what a rerun depends on.

```bash
pixi run python analysis/tools/publication_manifest.py list
pixi run python analysis/tools/publication_manifest.py check
pixi run python analysis/tools/publication_manifest.py audit-figures
pixi run python analysis/tools/publication_manifest.py readme --write   # regenerate the README blocks
```

## Environment management

This repo uses:

- one pixi environment, `pixi.toml` locked in `pixi.lock`, for the Python/R runtimes and every
  Python and R package (conda-forge and bioconda; `somadata` is the single PyPI exception)
- `pyproject.toml` for project metadata and tool configuration

Bring up the manuscript environment with:

```bash
pixi install --locked
pixi run check-r
```

`pixi install --locked` installs exactly what `pixi.lock` records. Change dependencies in
`pixi.toml` and relock with `pixi lock`; do not `pip install` or `uv` into the environment.

The pixi environment installs the R analysis packages from
`conda-forge` and `bioconda`:

- `mmrm` 0.3.19
- `emmeans` 2.0.3
- `mediation` 4.5.1
- `limma` 3.66.0
- `gdsfmt` 1.46.0
- `SNPRelate` 1.44.0
- `lme4` 2.0-1
- `arrow` 24.0.0

## Static validation

These checks do not require the trial data:

```bash
pixi run validate   # manifest check + figure-style audit + byte-compile
pixi run python analysis/tools/publication_manifest.py check
pixi run python analysis/tools/publication_manifest.py audit-figures
pixi run python -m compileall analysis figures tables
```

## Manuscript scripts

| Script | Purpose | Primary outputs |
|---|---|---|
| `pipelines/run_qc.py` | Run the SURMOUNT-5 Olink/SomaScan QC from the raw vendor files, sample keys, ADaM and genotypes | `<qc_root>/surmount5_{olink,soma}/surmount5_{olink,soma}.*` |
| `pipelines/run_mmrm.py` | Build SURMOUNT-5 Olink/Soma analysis frames from mounted QC outputs and run the primary `covar_AGE_SEX` manuscript MMRM model | `<mmrm_root>/surmount5_{olink,soma}/covar_AGE_SEX/...` |
| `pipelines/run_mmrm_sex.py` | Treatment × visit × sex MMRM with per-sex covariance (Tables S13–S14) | `<mmrm_root>/surmount5_{olink,soma}/covar_AGE_SEX_sexint/surmount5_sexint.csv` |
| `pipelines/qc_exclusion_breakdown.py` | Per-gate QC exclusion counts for the ED Fig 1 CONSORT diagram | `analysis/outputs/qc_exclusion_breakdown.parquet` |
| `python -m analysis.pathway batch` | cameraPR and ORA on the TZP − SEMA contrasts of every MMRM tree under `--base-dir` (`pathway/README.md`) | `<mmrm_root>/surmount5_{olink,soma}/covar_AGE_SEX/pathwayRes/acTrt/...` |
| `pipelines/run_mediation.py` | Build SURMOUNT-5 Olink/Soma active-comparator frames from mounted QC outputs and run manuscript mediation | `<mmrm_root>/surmount5_{olink,soma}/covar_AGE_SEX/mediationRes/...` |
| `pipelines/run_wgs_pcs.py` | Build the LD-pruned genome and SNPRelate principal components used by the E354Q pharmacogenomic model | mounted QA `pruned_genome/*` and `3100_pca/*_pc.csv` |
| `pipelines/build_uniprot_map.py` | Cross-platform Olink/SomaScan marker-to-gene/UniProt mapping | `analysis/outputs/uniprot_map.parquet` |
| `pipelines/platform_concordance.py` | Baseline Olink–SomaScan Spearman correlation of assay pairs sharing a UniProt accession of the assay annotation and the two-component GMM concordance call | `analysis/outputs/platform_concordance.parquet` |
| `pipelines/trajectory_classification.py` | Temporal response class assignments, both arms (rules in `trajectory.py`) | `analysis/outputs/trajectory_olink.parquet`, `trajectory_soma.parquet` |
| `pipelines/maretty_step_data.py` | STEP 1/2 semaglutide source-data extraction for ED Fig 5, Supp Fig 7 and the Fig 5 STEP forest | `analysis/outputs/maretty_step_data.parquet` |
| `pipelines/run_disease_reversal.py` | Fig 3b disease-signature reversal: HPA pan-disease blood atlas signatures against the Olink TZP − SEMA z, baseline-correlation rotation null (method in `disease_reversal.py`; HPA table fetched by `external/hpa_disease_atlas.py`) | `analysis/outputs/disease_reversal_olink_wk72.parquet` |
| `pharmacogenomics/run.py` | E354Q pharmacogenomic analysis frame and contrasts (Supp Fig 11) | `analysis/outputs/genotype_E354Q_*.parquet` |

The `external/` scripts stage and rebuild third-party inputs (`external/README.md`).

## Analysis modules

| Location | Scope |
|---|---|
| `pipelines/` | Manuscript-facing entrypoints: the `run_*.py` stages, the SURMOUNT-5-specific QC workflow adapter and the CLIs that write derived outputs under `analysis/outputs/` |
| `tools/` | The manifest command line (`publication_manifest.py`) and its modules (`manifest_stages`, `manifest_check`, `static_reads`, `readme_blocks`, `figure_audit`, `figure_baseline`), the stage runner, provenance (records and the check, with the import-closure walker) and the UniProt map |
| `qc/` | Sample and assay QC for Olink and SomaScan (`qc/README.md`) |
| `external/` | Checksum-verified downloads and rebuilds of third-party inputs (`external/README.md`) |
| `output_layout.py` | The one definition of `analysis/outputs/` and the stage outputs its writers and readers share |
| `design.py` | Config-free design constants: arms (raw and space-free `TRT01A`, display labels), proteomics visits and weeks, output stem `surmount5`, covariate label |
| `platform_inputs.py` | SURMOUNT-5 Olink/Soma preparation from mounted QC outputs + pheno data |
| `marker_genes.py`, `pathway_labels.py` | Marker → gene lookups and pathway display labels shared by figures and tables |
| `qc_flow.py`, `concordance.py`, `biomarker_correlation.py` | Single computations behind numbers a figure and a table (or stage) both report: the CONSORT counts, the ED2 GMM, the ED3 / Table S7–S8 Spearman fits |
| `mmrm/` | R `mmrm` + `emmeans` wrapper for the manuscript contrasts |
| `pathway/` | cameraPR and ORA generation from MMRM result CSVs |
| `mediation/` | numpy nonparametric-bootstrap mediation (100,000 resamples, seed 0) with Sobel/Wald p-values and the six-way class rule |
| `../pixi.toml` | Runtime and R package environment |
| `pharmacogenomics/` | E354Q pharmacogenomic analysis package, helpers, and `run.py` entrypoint |

Regenerate the paper QC outputs with:

```bash
pixi run stage run qc_reference   # fills the genotype and Fenland workbook paths from paths.yaml
pixi run python -m analysis.pipelines.run_qc --study surmount5_olink --output-dir <qc_root>
pixi run python -m analysis.pipelines.run_qc --study surmount5_soma --output-dir <qc_root>
pixi run python -m analysis.pipelines.qc_exclusion_breakdown
```

`analysis/qc/` is the sample and assay QC for Olink and SomaScan, configured by
`analysis/qc/config_{olink,soma}.toml`. It writes, per platform,
`<qc_root>/surmount5_<platform>/`: sample and assay summaries, the exclusion
table, identity events, pQTL scores, the DNA link and the long value table
(Olink `npx_long.parquet` with `NPX` and `PCNormalizedNPX`; SomaScan
`rfu_long.parquet` with the non-ANML `RFU`). QC runs on
PCNormalizedNPX; the MMRM, mediation and pathway inputs use NPX (datapoints the vendor's AssayQC
flag marks FAIL set to missing) and log2 RFU. `analysis/qc_contract.py` is the only
reader of those files.

The `analysis.mmrm` and `analysis.mediation` packages expect a
prepared analysis frame, not the raw QC parquet directly. The study wrappers
above perform the manuscript preparation step:

- Olink: the QC long table's `NPX` with the datapoints the vendor's AssayQC flag marks FAIL set to missing, then baseline `NPXBL`
- SomaScan: `RFU -> log2_RFU`, then baseline `log2_RFUBL`
- both platforms: join `USUBJID`/`VISITNUM`, derive `% weight change`,
  and map the proteomics visits to weeks 0, 24 and 72 (`analysis.design`)

## Scope

This repository contains the analysis workflow and the figure and table code, which shares its
styling through `figures/_internal/`.

For Supplementary Fig. 11, the genetic PCs are rebuilt
in-repo with `pixi run python analysis/pipelines/run_wgs_pcs.py`. That wrapper calls
`analysis/pharmacogenomics/build_pruned_genome.py` to produce the
autosomal pruned genome from the merged QC PLINK files and then
`analysis/pharmacogenomics/compute_wgs_pcs.R` to run SNPRelate PCA and
write the mounted QA PCA CSV used by `analysis/pharmacogenomics/run.py`.
