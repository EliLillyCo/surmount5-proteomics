"""Pooled per-platform PCA and order-independent mixed-model variance attribution.

Fixed-effect contributions use variance of each term's fitted contribution, normalized
by the sum of those variances plus participant and residual variance. This is an
order-independent variancePartition-style allocation, not sequential marginal R²;
correlated fixed contributions have covariance that the additive allocation does not
attribute to any one term.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

from analysis.design import BASELINE_VISITNUM, BASELINE_WEEK, POST_BASELINE_VISITS, RAW_CONTROL_VALUE, RAW_TREAT_VALUE, VISIT_TO_WEEK, WEEKS
from analysis.output_layout import PCA_SCORES, PCA_VARIANCE, PCA_PARTITION, pca_model_input
from analysis.platform_inputs import prepare_mmrm_inputs, _build_weight_wide, _load_advs
from analysis.qc_contract import PLATFORM_SPECS, PLATFORMS, cohort_samples, read_sample_summary, study_of
from analysis.qc_paths import study_data
from paths import PHENO_DIR

N_COMPONENTS = 10
PARAMS = {"BMI": "advs", "AVGSYSBP": "advs", "HBA1K48X": "adlb", "LDLCBDS": "adlb",
          "TRIGF52S": "adlb", "GFRBES4S": "adlb", "ALTE03S": "adlb"}
CONTINUOUS = ("QC_METRIC", "AGE", "BMI", "PCTCHG_WGT", "HBA1C", "SBP", "LDLC", "TG", "EGFR", "LOG_ALT")


def matrix_from_long(long: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray, list[str]]:
    """Order subjects and assays deterministically and keep every post-QC sample."""
    meta = long.select("sample_id", "USUBJID", "VISITNUM", "TRT01A").unique().sort("TRT01A", "USUBJID", "VISITNUM")
    if meta["sample_id"].n_unique() != meta.height:
        raise ValueError("sample has more than one participant/visit/arm")
    wide = long.pivot(on="assay_id", index="sample_id", values="value")
    assays = sorted(set(wide.columns) - {"sample_id"})
    matrix = meta.select("sample_id").join(wide, on="sample_id", how="left").select(assays).to_numpy().astype(np.float64)
    return meta.with_columns(pl.col("VISITNUM").replace_strict(VISIT_TO_WEEK, return_dtype=pl.Int32).alias("WEEK")), matrix, assays


def pca_scores(matrix: np.ndarray, weeks: np.ndarray, *, n_components: int = N_COMPONENTS) -> tuple[np.ndarray, np.ndarray, int]:
    """Median-impute, sample-unit-variance standardize, thin SVD, orient by week."""
    missing = int(np.isnan(matrix).sum())
    medians = np.nanmedian(matrix, axis=0)
    filled = np.where(np.isnan(matrix), medians, matrix)
    sd = filled.std(axis=0, ddof=1)
    if not np.isfinite(filled).all() or (sd <= 0).any():
        raise ValueError("all assays must have observed values and nonzero variance")
    standardized = (filled - filled.mean(axis=0)) / sd
    u, singular, _ = np.linalg.svd(standardized, full_matrices=False)
    scores = u[:, :n_components] * singular[:n_components]
    flip = np.where(scores[weeks == max(WEEKS)].mean(axis=0) < scores[weeks == BASELINE_WEEK].mean(axis=0), -1, 1)
    return scores * flip, singular[:n_components] ** 2 / np.sum(singular ** 2), missing


def _clinical() -> pl.DataFrame:
    """ADaM baseline records at proteomics baseline; scheduled records thereafter."""
    records = []
    for domain in sorted(set(PARAMS.values())):
        lf = pl.scan_parquet(PHENO_DIR / f"{domain}.parquet").filter(
            pl.col("PARAMCD").is_in([key for key, source in PARAMS.items() if source == domain]),
            pl.col("ANL01FL") == "Y",
            pl.col("DTYPE").is_null() | pl.col("DTYPE").is_in(("AVERAGE", "ORIGINAL")),
        )
        base = lf.filter(pl.col("ABLFL") == "Y").select("USUBJID", "PARAMCD", "AVAL", pl.lit(BASELINE_VISITNUM).alias("VISITNUM"))
        post = lf.filter(pl.col("VISITNUM").cast(pl.Int32).is_in(POST_BASELINE_VISITS)).select("USUBJID", "PARAMCD", "AVAL", pl.col("VISITNUM").cast(pl.Int32))
        records.append(pl.concat([base, post]).collect())
    rec = pl.concat(records).unique()
    if rec.group_by("USUBJID", "VISITNUM", "PARAMCD").len().filter(pl.col("len") > 1).height:
        raise ValueError("ADaM key has multiple values")
    clinical = rec.pivot(on="PARAMCD", index=["USUBJID", "VISITNUM"], values="AVAL")
    adsl = pl.read_parquet(PHENO_DIR / "adsl.parquet").select("USUBJID", "AGE", "SEX")
    weight = _build_weight_wide(_load_advs()).collect()
    return (clinical.join(adsl, on="USUBJID", how="left").join(weight, on="USUBJID", how="left")
            .with_columns(pl.when(pl.col("VISITNUM") == BASELINE_VISITNUM).then(pl.lit(0.))
                          .otherwise(pl.coalesce([pl.when(pl.col("VISITNUM") == visit).then(pl.col(f"PCTCHG_WGTV{visit}")) for visit in POST_BASELINE_VISITS])).alias("PCTCHG_WGT"))
            .select("USUBJID", "VISITNUM", "AGE", "SEX", "PCTCHG_WGT", "BMI",
                    pl.col("HBA1K48X").alias("HBA1C"), pl.col("AVGSYSBP").alias("SBP"),
                    pl.col("LDLCBDS").alias("LDLC"), pl.col("TRIGF52S").alias("TG"),
                    pl.col("GFRBES4S").alias("EGFR"), pl.col("ALTE03S").cast(pl.Float64).log().alias("LOG_ALT")))


def _technical(platform: str) -> pl.DataFrame:
    spec = PLATFORM_SPECS[platform]
    summary = read_sample_summary(platform).select(pl.col(spec.sample_id).alias("sample_id"), "Z_Median")
    if platform == "olink":
        plate = (pl.scan_parquet(study_data(study_of(platform)))
                 .filter(pl.col("SampleType") == "SAMPLE", pl.col("AssayType") == "assay")
                 .select(pl.col("SampleID").alias("sample_id"), pl.col("PlateID").str.extract(r"_(SP\d+)$").alias("PLATE"))
                 .unique().collect())
        if plate.group_by("sample_id").len().filter(pl.col("len") != 1).height:
            raise ValueError("Olink sample maps to multiple plates")
        return plate.join(summary, on="sample_id").select("sample_id", "PLATE", pl.col("Z_Median").alias("QC_METRIC"))
    import somadata
    adat = somadata.read_adat(str(study_data(study_of(platform))))
    sample = pl.from_pandas(adat.index.to_frame(index=False)).filter(pl.col("SampleType") == "Sample")
    return (sample.select(pl.col("SampleId").alias("sample_id"), pl.col("PlateId").alias("PLATE"), pl.col("HybControlNormScale").cast(pl.Float64).alias("QC_METRIC"))
            .join(summary.select("sample_id"), on="sample_id"))


def _r_partition(path: Path) -> pl.DataFrame:
    """Read parquet inside R; use lme4 REML and return only the small result boundary."""
    script = r'''
    suppressPackageStartupMessages({library(lme4); library(arrow)})
    d <- as.data.frame(read_parquet(input_path))
    for (v in c("WEEK", "ARM", "PLATE", "SEX", "USUBJID")) d[[v]] <- factor(d[[v]])
    cont <- c("QC_METRIC", "AGE", "BMI", "PCTCHG_WGT", "HBA1C", "SBP", "LDLC", "TG", "EGFR", "LOG_ALT")
    for (v in cont) d[[v]] <- as.numeric(scale(d[[v]]))
    terms <- c("WEEK", "ARM", "WEEK:ARM", "PLATE", "QC_METRIC", "SEX", "AGE", "BMI", "PCTCHG_WGT", "HBA1C", "SBP", "LDLC", "TG", "EGFR", "LOG_ALT")
    rows <- list()
    for (pc in paste0("PC", 1:10)) {
      fit <- lmer(as.formula(paste(pc, "~", paste(terms, collapse=" + "), "+ (1 | USUBJID)")), data=d, REML=TRUE, control=lmerControl(calc.derivs=FALSE))
      messages <- fit@optinfo$conv$lme4$messages
      non_singular_messages <- messages[!grepl("^boundary \\(singular\\) fit:", messages)]
      if (!identical(fit@optinfo$conv$opt, 0) || length(non_singular_messages)) stop(paste(pc, "mixed model did not converge"))
      mm <- model.matrix(fit); beta <- fixef(fit); assigned <- attr(mm, "assign")
      components <- attr(terms(fit), "term.labels")
      vc <- as.data.frame(VarCorr(fit))
      vp <- vc$vcov[vc$grp == "USUBJID"]; ve <- vc$vcov[vc$grp == "Residual"]
      vv <- vapply(seq_along(components), function(i) var(as.vector(mm[, assigned == i, drop=FALSE] %*% beta[assigned == i])), numeric(1))
      denominator <- sum(vv) + vp + ve
      if (!is.finite(denominator) || denominator <= 0) stop("invalid mixed-model variance")
      rows[[pc]] <- data.frame(PC=pc, component=c(components, "Participant", "Residual"), share=c(vv, vp, ve)/denominator, n=nrow(d))
    }
    do.call(rbind, rows)
    '''
    ro.globalenv["input_path"] = str(path)
    with pandas2ri.converter.context():
        result = ro.r(script)
    return pl.from_pandas(result)


def run() -> None:
    """Write scores, explained variance and REML mixed-model partitions for both platforms."""
    clinical = _clinical()
    all_scores, all_variance, all_partition = [], [], []
    PCA_SCORES.parent.mkdir(parents=True, exist_ok=True)
    for platform in PLATFORMS:
        prepared = prepare_mmrm_inputs(platform)
        sid = PLATFORM_SPECS[platform].sample_id
        long = (prepared.data.filter(pl.col("TRT01A").is_in((RAW_CONTROL_VALUE, RAW_TREAT_VALUE)))
                .select(pl.col(sid).alias("sample_id"), "USUBJID", "VISITNUM", "TRT01A",
                        pl.col(prepared.protein_id_col).alias("assay_id"), pl.col(prepared.response_var).alias("value")).collect())
        meta, matrix, assays = matrix_from_long(long)
        expected = cohort_samples(platform).select(pl.col(sid).alias("sample_id")).collect()
        if set(expected["sample_id"]) != set(meta["sample_id"]):
            raise ValueError(f"{platform}: post-QC cohort differs from PCA cohort")
        score, variance, missing = pca_scores(matrix, meta["WEEK"].to_numpy())
        scored = meta.with_columns([pl.Series(f"PC{i+1}", score[:, i]) for i in range(N_COMPONENTS)])
        all_scores.append(scored.with_columns(pl.lit(platform).alias("platform")))
        all_variance.append(pl.DataFrame({"platform": [platform] * N_COMPONENTS,
                           "PC": [f"PC{i+1}" for i in range(N_COMPONENTS)], "variance_explained": variance}))
        joined = scored.join(clinical, on=["USUBJID", "VISITNUM"], how="left").join(_technical(platform), on="sample_id", how="left")
        if joined.height != scored.height:
            raise ValueError("metadata joins changed the sample cohort")
        model = (joined.select("USUBJID", pl.col("WEEK").cast(pl.String), pl.col("TRT01A").alias("ARM"),
                               "PLATE", "SEX", *CONTINUOUS, *[f"PC{i}" for i in range(1, N_COMPONENTS + 1)])
                 .drop_nulls())
        if model.height == 0:
            raise ValueError("no complete clinical records for mixed model")
        model.write_parquet(pca_model_input(platform))
        result = _r_partition(pca_model_input(platform)).with_columns(pl.lit(platform).alias("platform"))
        all_partition.append(result)
        print(f"{platform}: samples={scored.height} assays={len(assays)} missing_assay_cells={missing} complete_model_cases={model.height}", flush=True)
    pl.concat(all_scores).write_parquet(PCA_SCORES)
    pl.concat(all_variance).write_parquet(PCA_VARIANCE)
    pl.concat(all_partition).write_parquet(PCA_PARTITION)
