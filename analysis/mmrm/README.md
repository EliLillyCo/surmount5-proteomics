# MMRM engine

This package wraps R `mmrm` + `emmeans` (through rpy2) to fit the manuscript's
per-protein mixed models. It runs only the primary model `covar_AGE_SEX`: age, sex and
the baseline protein level. The model adjusts for age, sex and the baseline protein level.

Modules:

- `constants.py`: `MMRM_CONTROL_ARGS`, default covariates (`SEX`; `AGE`) and response columns
- `preparation.py`: change-from-baseline frame and the model formula (`build_formula`)
- `rinterface.py`: Polars to R conversion, factor releveling and emmeans extraction
- `fitting.py`: per-protein fits in a fork-server worker pool (`run_mmrm_parallel`)
- `reporting.py`: the `rawOutput/` and `finalRes/` files
- `pipeline.py`: `run_mmrm_pipeline`, the whole run for one platform
- `sex_interaction.py`: the treatment x visit x sex model behind Tables S13-S14

## Model

- Change model: `<response>_DIFF ~ <baseline> + AGE + TRT01A + VISITNUM + TRT01A:VISITNUM +
  SEX + us(VISITNUM | USUBJID)` (`preparation.build_formula`). Every `mmrm()` call passes
  `control = mmrm_control(method = "Kenward-Roger", vcov = "Kenward-Roger-Linear")` with
  r-mmrm 0.3.19 (matching SAS `DDFM=KR`).
- Baseline model: a linear model (`lm`) of the baseline level on the covariates and
  `TRT01A`, at baseline.
- `emmeans` uses `weights = "proportional"` for the LSmeans of both models.
- Reference levels are set at the R boundary (`analysis.rinterface.to_r_factors`, sorted
  levels): the comparator arm (`control_trt`) and the earliest post-baseline visit. So
  treatment contrasts are **TZP - SEMA** and within-arm time contrasts are **week 72 - week
  24** (`@72VS24`, FC = 72/24).
- Row order is fully determined (sort keys end in `USUBJID`), so reruns write the same rows.
- A failed fit raises `RuntimeError` naming the marker; nothing is skipped.
- Benjamini-Hochberg FDR (`analysis.statistics.bh_fdr`) is applied per written comparison:
  one treatment pair x visit, one arm x visit (PCBL), or one arm's 72-vs-24 contrast.

The engine takes a prepared frame (response, baseline response, `USUBJID`, `VISITNUM`,
`TRT01A`, covariates); it never reads the QC parquet itself. `analysis/platform_inputs.py`
builds that frame (`prepare_mmrm_inputs`).

## Running

```bash
pixi run python analysis/pipelines/run_mmrm.py --platform olink
pixi run python analysis/pipelines/run_mmrm.py --platform soma
```

Results go to `<mmrm_root>/surmount5_<platform>/covar_AGE_SEX/` (`mmrm_root` in
`paths.yaml`, the root every figure, table, pathway and mediation reader uses; override with
`--output-root`):

- `rawOutput/surmount5_covar_AGE_SEX_chg_lsmeans.csv`: columns
  `marker,time,trt,LSmean,StdErr,DF,tValue,Probt,Lower,Upper,pVal`
- `rawOutput/surmount5_covar_AGE_SEX_chg_diff.csv`: every contrast's raw log2 `diff` with
  its `StdErr` and `df`
- `rawOutput/*_baseline_{lsmeans,diff}.csv`
- `finalRes/acTrt/`, `finalRes/withinTrt/`, `finalRes/pcblRes/` and `finalRes/baseline/`: the
  formatted per-comparison files. Readers build their names through `analysis/mmrm_outputs.py`.

The across- and within-treatment files (`acTrt`, `withinTrt`) have one row per marker with
`marker,Assay,trt1,trt2,time` (or `trt,time1,time2`), then `FC,SE-FC,pVal,fdr,estimate,SE,df`:

- `FC` is the ratio of the two LS-means on the linear scale, `LSM1 / LSM2` with `LSM = 2^LSmean`.
- `SE-FC` is the independence approximation FC·√((SE₁/LSM₁)²+(SE₂/LSM₂)²) from the two per-arm
  (or per-visit) LS-means and their standard errors. It is not the standard error of the contrast.
- `estimate`, `SE` and `df` are the emmeans contrast itself on the model (log2) scale: `estimate`
  is log₂ `FC`, `SE` is the contrast's own standard error and `df` its Kenward-Roger degrees of
  freedom (the `diff`, `StdErr` and `df` of `rawOutput/*_chg_diff.csv`). `pVal` is this
  contrast's p-value.

From Python: `from analysis.mmrm import run_mmrm_pipeline`.
