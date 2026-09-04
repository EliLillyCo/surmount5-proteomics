# Analysis code organization

This directory contains the manuscript-specific analysis code required to regenerate the derived outputs used by `figures/` and `tables/`.

The public release is organized around the analysis steps that feed the paper figures and supplementary tables.

## Manifest

`analysis/manifest.yaml` is the central command map for manuscript commands,
protected input roots, generated outputs, and the analysis modules
maintained in this repository.

```bash
uv run python analysis/tools/publication_manifest.py list
uv run python analysis/tools/publication_manifest.py check
uv run python analysis/tools/publication_manifest.py audit-figures
```

## Environment management

This repo uses:

- `pixi.toml` for the Python/R runtimes and conda-managed manuscript R packages
- `pyproject.toml` and `uv.lock` for Python dependencies

Bring up the manuscript environment with:

```bash
pixi install --locked
pixi run sync
pixi run check-r
```

`pixi run sync` delegates to
`env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=.pixi/envs/default uv sync --locked --inexact`,
so the committed Python lockfile is enforced inside the pixi environment
without pruning pixi-managed packages such as `rpy2`. Use this task rather than
bare `uv sync --locked` when syncing the manuscript environment.

The pixi environment installs the R analysis packages from
`conda-forge` and `bioconda`:

- `mmrm` 0.3.17
- `emmeans` 2.0.3
- `mediation` 4.5.1
- `limma` 3.66.0
- `gdsfmt` 1.46.0
- `SNPRelate` 1.44.0

## Static validation

These checks do not require mounted protected inputs:

```bash
uv run pytest analysis/tests/
uv run python analysis/tools/publication_manifest.py check
uv run python analysis/tools/publication_manifest.py audit-figures
uv run python -m compileall analysis figures tables
```

## Active manuscript scripts

| Script | Purpose | Primary outputs |
|---|---|---|
| `pipelines/run_qc.py` | Regenerate the minimal SURMOUNT-5 Olink/Soma QC outputs from raw proteomics files plus pheno data | mounted `quality_control/surmount5_{olink,soma}/surmount5_{olink,soma}.*` |
| `pipelines/run_mmrm.py` | Build SURMOUNT-5 Olink/Soma analysis frames from mounted QC outputs and run the primary `covar_AGE_SEX` manuscript MMRM model | mounted `results/mmrm/*/covar_AGE_SEX/...` |
| `pipelines/run_mediation.py` | Build SURMOUNT-5 Olink/Soma active-comparator frames from mounted QC outputs and run manuscript mediation | mounted `results/mmrm/*/covar_AGE_SEX/mediationRes/...` |
| `pipelines/run_wgs_pcs.py` | Build the LD-pruned genome and SNPRelate principal components used by the E354Q pharmacogenomic model | mounted QA `pruned_genome/*` and `3100_pca/*_pc.csv` |
| `scripts/build_uniprot_map.py` | Cross-platform Olink/SomaScan marker-to-gene/UniProt mapping | `analysis/outputs/uniprot_map*.parquet` |
| `scripts/platform_concordance.py` | Cross-platform effect-size concordance | `analysis/outputs/platform_concordance.parquet` |
| `scripts/trajectory_classification.py` | Temporal response class assignments | `analysis/outputs/trajectory_olink.parquet`, `trajectory_soma.parquet` |
| `scripts/maretty_step_data.py` | STEP 1/2 semaglutide source-data extraction for the cross-study supplements and Figure 5 forest plots | `analysis/outputs/maretty_step_data.parquet` |
| `pharmacogenomics/run.py` | E354Q pharmacogenomic analysis frame and contrasts | `analysis/outputs/genotype_E354Q_*.parquet` |

## Analysis modules

| Location | Scope |
|---|---|
| `pipelines/` | Manuscript-facing `run_*.py` entrypoints plus the SURMOUNT-5-specific QC workflow adapter |
| `scripts/` | Analysis CLIs that write derived outputs under `analysis/outputs/` |
| `tools/` | Manifest, aliasing, and shared helper utilities used across analysis code |
| `qc/` | Minimal vendored reusable Olink/Soma QC engine |
| `platform_inputs.py` | SURMOUNT-5 Olink/Soma preparation from mounted QC outputs + pheno data |
| `mmrm/` | R `mmrm` + `emmeans` wrapper for the manuscript contrasts |
| `pathway/` | cameraPR and ORA generation from MMRM result CSVs |
| `mediation/` | R bootstrap mediation workflow with 100,000 replicates and Sobel indirect-effect p-values |
| `../pixi.toml` | Runtime and manuscript R package environment for the public repo |
| `pharmacogenomics/` | E354Q pharmacogenomic analysis package, helpers, and active `run.py` entrypoint |

Regenerate the paper QC outputs with:

```bash
uv run python analysis/pipelines/run_qc.py --platform olink
uv run python analysis/pipelines/run_qc.py --platform soma
```

The manuscript QC outputs are platform-specific:

- Olink: `surmount5_olink/surmount5_olink.npx_long.parquet` with `PCNormalizedNPX`
- SomaScan: `surmount5_soma/surmount5_soma.rfu_long.parquet` with `RFU`

The QC wrapper writes `surmount5_*` basenames under the configured QC root.

`pipelines/run_qc.py` vendors only the manuscript-used QC path: sample QC, assay QC,
CV/detection filtering, Soma saturation/sample-to-buffer filters, PCA and
summary-statistic outlier detection, sex/age concordance, clinical suitability,
and biomarker-correlation validation. Generates QC summary parquets/TSVs only.


The generic `analysis.mmrm` and `analysis.mediation` packages expect a
prepared analysis frame, not the raw QC parquet directly. The study wrappers
above perform the manuscript preparation step:

- Olink: `PCNormalizedNPX -> NPX`, then baseline `NPXBL`
- SomaScan: `RFU -> log2_RFU`, then baseline `log2_RFUBL`
- both platforms: join `USUBJID`/`VISITNUM`, derive `% weight change`,
  and set `baseline_visitnum=2` with visit labels `8->24`, `20->72`

For the active Figure 5 pharmacogenomics path, the genetic PCs are rebuilt
in-repo with `uv run python analysis/pipelines/run_wgs_pcs.py`. That wrapper calls
`analysis/pharmacogenomics/build_pruned_genome.py` to produce the
autosomal pruned genome from the merged QC PLINK files and then
`analysis/pharmacogenomics/compute_wgs_pcs.R` to run SNPRelate PCA and
write the mounted QA PCA CSV used by `analysis/pharmacogenomics/run.py`.
