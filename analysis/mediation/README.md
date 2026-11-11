# Mediation engine

`analysis/mediation/` splits the TZP − SEMA effect on each protein into the part mediated
by percent weight change (ACME) and a direct part (ADE). It does this per platform and per
post-baseline visit. The engine is written in numpy and gives the same estimates as R's
`mediation::mediate`.

Modules:

- `bootstrap.py`: vectorized product-of-coefficients bootstrap for linear models. Each
  effect gets an empirical bootstrap p and an analytic p (Sobel for ACME; a Wald z of the
  coefficient, with a normal reference, for ADE and total).
- `core.py`: `run_mediation_single_numpy`, one protein at one visit. It uses OLS mediator,
  outcome and total-effect models, and `seed = SEED` (0).
- `pipeline.py`: `run_mediation_pipeline`, every protein in a fork-server pool. It writes
  per-visit checkpoints, then one CSV.
- `classify.py`: the six-way class rule (`ns`, `fully_mediated`, `partial`, `direct_only`,
  `inconsistent`, `unresolved`) on BH FDR of the analytic p-values.
- `results.py`: `load_platform(platform, visit)`, the reader every figure and table uses.
  It applies `classify_sobel`.
- `themes.py`: the curated theme gene lists of Fig 4 and Supp Figs 9-10.
- `constants.py`, `cli.py`: covariates (`SEX`; `AGE`, `WGTBL`; the baseline protein is
  always added), the mediator `PCTCHG_WGT`, and the shared `--protein-subset` / `--out` flags.

## Running

```bash
pixi run python analysis/pipelines/run_mediation.py --platform olink
pixi run python analysis/pipelines/run_mediation.py --platform soma
```

- **Defaults:** both platforms, the configured post-baseline visits, `--sims 100000`,
  and `--n-jobs -1` (all cores but one).
- **Inputs:** `prepare_mediation_inputs(platform)` (`analysis/platform_inputs.py`) builds
  the frame from `paths.yaml:qc_root` (QC long tables and cohort) and
  `paths.yaml:pheno_root` (`adsl.parquet`, `advs.parquet`). It keeps the two compared arms
  only.
- **Output:**
  `<mmrm_root>/surmount5_<platform>/covar_AGE_SEX/mediationRes/surmount5_covar_AGE_SEX_mediation_PCTCHG_WGT.csv`.
  Override the root with `--out`.

## Reuse guard

The run writes `<csv stem>.fingerprint.json` next to the CSV. It records the model
configuration, each visit's input content, the annotation and the marker x visit keys
written. A rerun reuses the CSV only when all of these match. Otherwise, or when the
fingerprint file is missing, it exits with `SystemExit` and says what differs. The
per-visit checkpoints (`_checkpoint_v<visit>.parquet`) carry the same guard, and the run
deletes them once the CSV is written.
