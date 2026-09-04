# SURMOUNT-5 proteomics paper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Proteomics analysis code for the SURMOUNT-5 manuscript: dual-platform (Olink Explore HT + SomaScan 11K) characterization of tirzepatide vs semaglutide in patients with obesity.

## Study

- **Trial**: SURMOUNT-5, Phase 3b, open-label, active-comparator
- **Arms**: Tirzepatide 15 mg or maximum tolerated dose (MTD) vs Semaglutide 2.4 mg or MTD
- **Population**: Adults with obesity/overweight without type 2 diabetes (N=751 randomized)
- **Proteomics timepoints**: Weeks 0, 24, 72
- **Platforms**: Olink Explore HT (5,416 assays, NPX log2) and SomaScan 11K (10,771 aptamers, RFU linear)

## Repository structure

```
analysis/            Manuscript analysis scripts and pipeline modules
  manifest.yaml      Script/input/output manifest for paper reproduction
  pipelines/         Manuscript-facing pipeline entrypoints and study-specific glue
  scripts/           Analysis CLIs that write derived outputs
  tools/             Manifest, aliasing, and helper utilities
  qc/                Olink/Soma QC package
  mmrm/              R mmrm + emmeans wrapper
  pathway/           cameraPR + ORA wrapper
  mediation/         R-bootstrap mediation wrapper
data/                Data inventory and protected-data schema notes
figures/             Publication figure scripts
tables/              Supplementary table builders
results/             Derived MMRM/pathway/mediation outputs when locally available
```

## Setup

This repo uses `pixi.toml` for the runtime toolchain and conda-managed R packages, and `pyproject.toml` + `uv.lock` for Python package resolution.

```bash
pixi install --locked
pixi run sync
pixi run check-r
```

`pixi run sync` enforces the committed `uv.lock`, so Python dependencies are
not silently re-resolved during reproducibility checks.

Key Python dependencies: polars, pandas, somadata, ultraplot, pypdf, rpy2, scipy, scikit-learn, statsmodels.

`pixi.toml` pins Python 3.13.13 and R 4.5.3, then installs the
R analysis packages directly from `conda-forge` and `bioconda` for the
`analysis.mmrm`, `analysis.mediation`, and `analysis.pathway` entrypoints:

- `mmrm` 0.3.17
- `emmeans` 2.0.3
- `mediation` 4.5.1
- `limma` 3.66.0
- `gdsfmt` 1.46.0
- `SNPRelate` 1.44.0

## Usage

All `uv run` commands below assume you are inside `pixi shell`. Running
`uv run` outside of `pixi shell` will recreate a `.venv` instead of using
the pixi-managed environment (which provides R, rpy2, and the locked conda
packages). Start a session with:

```bash
pixi shell
```

Analysis scripts are run as Python modules. The public repo is
CLI-first; no notebooks are required for manuscript reproduction.

```bash
# List manuscript reproduction commands
uv run python analysis/tools/publication_manifest.py list

# Validate manifest paths and static figure style conventions
uv run python analysis/tools/publication_manifest.py check
uv run python analysis/tools/publication_manifest.py audit-figures
uv run python -m compileall analysis figures tables

# The static checks above do not require mounted protected inputs.
# The analysis, figure, and table build commands below do.

# Analysis CLIs
uv run python analysis/pipelines/run_qc.py --help
uv run python analysis/scripts/build_uniprot_map.py
uv run python -m analysis.mmrm --help
uv run python -m analysis.pathway --help
uv run python -m analysis.mediation --help

# Generate main figures
uv run python figures/fig1_trajectory.py
uv run python figures/fig2_volcano.py
uv run python figures/fig3_pathway.py
uv run python figures/fig4_mediation.py
uv run python figures/fig5_ppy.py

# Build a combined main-figure PDF (Figure 1-5 labels only)
uv run python figures/build_main_figures.py

# Build the combined Supplementary Figures PDF (A4 pages, standard 0.5 in page margins, 9/8 pt legends)
uv run python figures/build_supplementary_figures.py

# Build supplementary tables
uv run python tables/build_all.py
```

## Data Access

Clinical trial data is accessed via DNAnexus (dxfuse mount at `/mnt/project/`).
Copy `paths.example.yaml` to `paths.yaml` and fill in the mounted roots
(`mmrm_root`, `qc_root`, `pheno_root`, `geno_root`, `qa_root`,
`raw_proteomics_root`). `paths.py` resolves `surmount5_*` study directories
under those roots. See `analysis/manifest.yaml` for the manuscript
script-to-output manifest.

Protected raw and participant-level data are not included. The public codebase contains the code required to regenerate manuscript-derived outputs when the protected inputs are mounted.

## Citation

If you use this code, please cite the associated publication:

> Chiou J, Dunn JP, Dimitriadis GK, Falcon B, Yan D, O'Dushlaine C,
> Coghlan M, Harris C, Titchenell P. Differential Plasma Proteomic
> Responses between Tirzepatide and Semaglutide in Adults with Obesity:
> An Exploratory Analysis of SURMOUNT-5. *TODO: journal*. 2026.
> DOI: TODO

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## License

This project is licensed under the [MIT License](LICENSE).

Copyright © 2026 Eli Lilly and Company.
