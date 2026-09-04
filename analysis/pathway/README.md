# Pathway wrapper

`analysis/pathway/` regenerates the manuscript `pathwayRes` directories from
MMRM result files:

- marker-to-gene loading for Olink and SomaScan assay manifests
- gene-level ranking from the MMRM contrasts
- ORA with Enrichr-format gene-set libraries
- `limma::cameraPR` via rpy2
- single-file and batch CLIs that write `camera_combined.csv` and `ora_combined.csv`

Gene-set libraries are cached locally as JSON. A first run can populate the
cache with `--download-missing`; subsequent runs read the cached copies.

```bash
uv run python -m analysis.pathway single \
  path/to/surmount5_covar_AGE_SEX_proteomics_olinkAnalysis_acrossTrts_resCmps_TZP15mgorMTDVSSEMA2.4mgorMTD@24_py.csv \
  --output-dir path/to/pathway_output \
  --mapping /path/to/olink_explore_ht_assays_with_coords.tsv \
  --somascan-mapping /path/to/somascan_11k_assays.tsv \
  --download-missing
```
