# MMRM wrapper

This directory wraps R `mmrm` + `emmeans` to run the
proteomics mixed-model workflow for the manuscript.

Included modules:

- `constants.py`
- `preparation.py`
- `rinterface.py`
- `fitting.py`
- `reporting.py`
- `pipeline.py`
- `__main__.py`

Required behavior:

- model: change from baseline with baseline covariate and `TRTP * VISITNUM`
- unstructured covariance: `us(VISITNUM | USUBJID)`
- explicit treatment reference and ordered visit factors at the R boundary
- Satterthwaite degrees of freedom
- Benjamini-Hochberg FDR
- proportional arm weights for pooled contrasts
- per-protein parallel fitting keeps each R call bounded in size

Run the CLI with:

```bash
uv run python -m analysis.mmrm --help
```

The generic CLI expects a prepared analysis frame with the response column,
baseline response column, `USUBJID`, `VISITNUM`, `TRT01A`, and any covariates
already assembled. It does not read the raw mounted QC parquet directly.

For the manuscript SURMOUNT-5 runs from QC outputs, use:

```bash
uv run python analysis/pipelines/run_mmrm.py --platform olink
uv run python analysis/pipelines/run_mmrm.py --platform soma
```

This wrapper is intentionally restricted to the primary manuscript model
`covar_AGE_SEX` (age, sex, and baseline protein level through the MMRM change
model). It does not run the weight- or HbA1c-adjusted sensitivity variants.

Or call the API directly:

```python
from analysis.mmrm import run_mmrm_pipeline
```
