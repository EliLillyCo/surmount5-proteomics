# Mediation wrapper

`analysis/mediation/` regenerates the manuscript `mediationRes` outputs with the paper method:

- `mediation::mediate(..., boot=TRUE)` in R
- bootstrap confidence intervals from 100,000 replicates by default
- Sobel p-values for the indirect effect
- direct-effect and total-effect p-values from the fitted linear models

The package writes the same column set used by `figures/fig4_mediation.py`,
`figures/supp_fig13_mediation_weight_mediated.py`, and `tables/suppl_mediation.py`.

```bash
uv run python -m analysis.mediation \
  --data path/to/analysis_frame.parquet \
  --annotation path/to/annotation.parquet \
  --output-dir path/to/results_root \
  --study-name surmount5 \
  --covar-label AGE_SEX \
  --protein-id-col OlinkID \
  --mediator PCTCHG_WGT \
  --visit 8 \
  --visit 20
```

The command writes
`{output_dir}/covar_{covar_label}/mediationRes/{study_name}_covar_{covar_label}_mediation_{mediator}.csv`.

The generic CLI expects a prepared analysis frame, not the raw mounted QC
parquet. For the manuscript SURMOUNT-5 runs from QC outputs, use:

```bash
uv run python analysis/pipelines/run_mediation.py --platform olink
uv run python analysis/pipelines/run_mediation.py --platform soma
```
