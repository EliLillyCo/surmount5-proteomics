"""Checks age concordance with the ProtAge 204-protein clock.

The clock uses a fixed ElasticNet alpha (the published approach).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

# Third-party
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from analysis.qc.flags import WARN_AGE_CONCORDANCE, WARN_AGE_CONSISTENCY, WARN_AGE_PREDICTION
from analysis.qc.modeling import (
    CV_FOLDS,
    KNN_K,
    RETIRED_UNIPROT,
    build_feature_matrix,
    explode_assay_uniprot,
    grouped_splits,
)

# Local imports
from analysis.qc_paths import PROTAGE_TABLE

logger = logging.getLogger(__name__)


def load_protage_proteins(
    tsv_path: str,
    assay_proteins: pl.DataFrame,
    protage_uniprot_col: str = "UniProt ID",
    protein_id_col: str = "SeqId",
    uniprot_col: str = "UniProt",
) -> tuple[list[str], pl.DataFrame]:
    """Matches the ProtAge 204 proteins to the trial assays by UniProt ID.

    Args:
        tsv_path: ProtAge supplementary table (Table S1, TSV).
        assay_proteins: Assays, with ``protein_id_col`` and ``uniprot_col`` columns.
        protage_uniprot_col: UniProt column of the ProtAge table.
        protein_id_col: Assay ID column of ``assay_proteins``.
        uniprot_col: UniProt column of ``assay_proteins``.

    Returns:
        The matched assay IDs and a match-info frame with one row per matched pair.
    """
    logger.info(f"loading ProtAge proteins from {tsv_path}")

    # The ProtAge table marks missing values as "NA".
    protage_df = pl.read_csv(tsv_path, separator="\t", null_values=["NA"])
    logger.info(f"loaded {len(protage_df)} ProtAge proteins")

    # Match by UniProt ID; multi-UniProt assays are split so each UniProt can match.
    assay_uniprots = explode_assay_uniprot(assay_proteins, protein_id_col, uniprot_col)
    match_df = (
        protage_df.filter(pl.col(protage_uniprot_col).is_not_null())
        .with_columns(pl.col(protage_uniprot_col).replace(RETIRED_UNIPROT))  # as the assay side
        .join(
            assay_uniprots,
            left_on=protage_uniprot_col,
            right_on=uniprot_col,
            how="inner",
        )
        .select(
            [
                pl.col(protage_uniprot_col).alias("ProtAge_UniProt"),
                pl.col(protein_id_col),
                pl.lit("UniProt ID").alias("match_type"),
            ]
        )
        .sort("ProtAge_UniProt", protein_id_col)
        .unique(maintain_order=True)
    )

    if match_df.is_empty():
        logger.warning("no matches found!")
        return [], match_df

    n_uniprot_matched = match_df["ProtAge_UniProt"].n_unique()
    n_uniprot_table = protage_df[protage_uniprot_col].drop_nulls().n_unique()
    matched_protein_ids = match_df[protein_id_col].unique(maintain_order=True).to_list()
    logger.info(
        f"Matched {n_uniprot_matched}/{n_uniprot_table} unique ProtAge UniProts "
        f"to {len(matched_protein_ids)} assays by UniProt ID"
    )

    return matched_protein_ids, match_df


# Defaults of TrialConfig ``age_enet_alpha`` and ``age_enet_l1_ratio`` (ProtAge clock).
AGE_ENET_ALPHA: float = 0.1667
AGE_ENET_L1_RATIO: float = 0.5
# The ProtAge table (Argentieri 2024, Table S1), staged by analysis.external.qc_reference.
PROTAGE_TABLE_PATH = PROTAGE_TABLE


RIDGE_ALPHAS = np.logspace(-2, 4, 13)
AGE_MODEL_NAMES = ("elasticnet", "ridgecv", "ridgecv_grouped")


def _clip(z: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(z, -limit, limit)


class GroupedRidgeCV(RegressorMixin, BaseEstimator):
    """A ridge regressor whose alpha is chosen by grouped k-fold CV over subjects.

    A subject's tubes always fall in one fold
    (``modeling.grouped_splits``, seed 0, salt ``alpha``); the alpha with the smallest
    summed squared held-out error wins, the smallest on a tie, and the model is then
    refitted on all rows. Row order does not change the result.

    Args:
        alphas: Candidate ridge penalties.
        cv_folds: Number of grouped CV folds.
    """

    def __init__(self, alphas=RIDGE_ALPHAS, cv_folds: int = CV_FOLDS):
        """Stores the candidate penalties and the fold count; see the class docstring."""
        self.alphas = alphas
        self.cv_folds = cv_folds

    def fit(self, X, y, *, groups):
        """Selects the alpha by grouped CV and refits on all rows.

        Args:
            X: Feature matrix.
            y: Targets.
            groups: One subject label per row.

        Returns:
            The fitted estimator.
        """
        X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
        order, splits = grouped_splits(groups, self.cv_folds, seed=0, salt="alpha")
        Xs, ys = X[order], y[order]
        alphas = np.asarray(self.alphas, dtype=float)
        sse = np.zeros(len(alphas))
        for k, alpha in enumerate(alphas):
            for train, test in splits:
                pred = Ridge(alpha=alpha).fit(Xs[train], ys[train]).predict(Xs[test])
                sse[k] += float(np.sum((ys[test] - pred) ** 2))
        self.alpha_ = float(alphas[int(np.argmin(sse))])
        refit = Ridge(alpha=self.alpha_).fit(X, y)
        self.coef_, self.intercept_ = refit.coef_, refit.intercept_
        return self

    def predict(self, X):
        """Predicts targets from the refitted coefficients."""
        return np.asarray(X) @ self.coef_ + self.intercept_


def _fit_regressor(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Fits ``model``, passing subject labels on when it ends in a grouped ridge."""
    name, last = model.steps[-1]
    if isinstance(last, GroupedRidgeCV):
        return model.fit(X, y, **{f"{name}__groups": groups})
    return model.fit(X, y)


def _make_age_regressor(
    age_model: Literal["elasticnet", "ridgecv", "ridgecv_grouped"],
    alpha: float,
    l1_ratio: float,
    random_seed: int,
    feature_clip: float | None = None,
    cv_folds: int = CV_FOLDS,
):
    """Builds a scaler (optional clip to +-``feature_clip``) plus regressor pipeline.

    ``ridgecv`` fits a RidgeCV over alphas 1e-2..1e4; ``ridgecv_grouped``
    takes the same alphas but picks one by a grouped CV over subjects (``cv_folds``
    folds) and must be fitted through ``_fit_regressor``.
    """
    if age_model == "elasticnet":
        reg = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=random_seed,
            max_iter=10000,
            tol=1e-6,
            selection="cyclic",
        )
    elif age_model == "ridgecv":
        reg = RidgeCV(alphas=RIDGE_ALPHAS)
    elif age_model == "ridgecv_grouped":
        reg = GroupedRidgeCV(alphas=RIDGE_ALPHAS, cv_folds=cv_folds)
    else:
        raise ValueError(
            f"age_model must be one of {', '.join(AGE_MODEL_NAMES)}; got {age_model!r}"
        )
    if feature_clip is None:
        return make_pipeline(StandardScaler(), reg)
    return make_pipeline(
        StandardScaler(),
        FunctionTransformer(_clip, kw_args={"limit": feature_clip}),
        reg,
    )


def _bias_corrected_residuals(
    X_s: np.ndarray,
    y_s: np.ndarray,
    groups_s: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    oof_pred: np.ndarray,
    make_model,
    cv_folds: int = CV_FOLDS,
) -> np.ndarray:
    """Returns out-of-fold residuals minus a linear fit of residual on age.

    Residuals are predicted - actual. The residual-on-age line for each outer fold is
    fitted on that fold's training samples only, using inner grouped-CV predictions so
    training residuals are out-of-sample too, then applied to the held-out samples.
    Held-out labels never enter the correction.
    """
    corrected = np.empty(len(y_s))
    for train_idx, test_idx in splits:
        _, inner_splits = grouped_splits(groups_s[train_idx], cv_folds, salt="inner")
        inner_pred = np.empty(len(train_idx))
        for inner_train, inner_test in inner_splits:
            model = _fit_regressor(
                make_model(),
                X_s[train_idx][inner_train],
                y_s[train_idx][inner_train],
                groups_s[train_idx][inner_train],
            )
            inner_pred[inner_test] = model.predict(X_s[train_idx][inner_test])
        slope, intercept = np.polyfit(y_s[train_idx], inner_pred - y_s[train_idx], 1)
        corrected[test_idx] = (oof_pred[test_idx] - y_s[test_idx]) - (
            intercept + slope * y_s[test_idx]
        )
    return corrected


def prepare_age_data(
    qc_instance,
    age_map: pl.DataFrame,
    age_col: str,
    protein_subset: list[str] | None = None,
    max_missing: float = 0.1,
    fit_exclude=(),
    knn_k: int = KNN_K,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    pl.DataFrame,
    np.ndarray,
]:
    """Builds the wide-format feature matrix and age labels for the age model.

    Args:
        qc_instance: The QC instance holding the proteomics data.
        age_map: Sample-level ages with ``age_col`` and the ID columns.
        age_col: Name of the age column in ``age_map``.
        protein_subset: Assay IDs to use; empty means all.
        max_missing: An assay missing in more than this fraction of samples is left out.
        fit_exclude: Sample IDs kept out of training.
        knn_k: Neighbors of the KNN imputation.

    Returns:
        ``(X, y, sample_ids, groups, feature_cols, data_with_age, fit)``.
    """
    fm = build_feature_matrix(
        qc_instance,
        labels=age_map,
        label_cols=[age_col, "VISITNUM", "TRT01A"],
        protein_subset=protein_subset or [],
        max_missing=max_missing,
        fit_exclude=fit_exclude,
        knn_k=knn_k,
    )
    y = fm.frame[age_col].to_numpy().astype(float)
    return fm.X, y, fm.sample_ids, fm.groups, fm.feature_cols, fm.frame, fm.fit


def _participant_age_stats(
    results_df: pl.DataFrame,
    sample_id_col: str,
    representative_name: str = "Representative_SampleId",
) -> pl.DataFrame:
    """Summarizes age predictions per subject, independent of input row order.

    Rows are sorted by (USUBJID, sample ID) before aggregating, so the floating
    std/median and the ``first`` representative sample do not depend on row order.
    """
    return (
        results_df.sort(["USUBJID", sample_id_col])
        .group_by("USUBJID", maintain_order=True)
        .agg(
            [
                pl.col("Visit_Adj_Age").median().alias("Median_Actual_Age"),
                pl.col("Predicted_Age").median().alias("Median_Predicted_Age"),
                pl.col("Predicted_Age").std().alias("Std_Predicted_Age"),
                pl.col("Predicted_Age").count().alias("N_Timepoints"),
                pl.col(sample_id_col).first().alias(representative_name),
            ]
        )
    )


def detect_age_outliers(
    results_df: pl.DataFrame,
    sample_id_col: str,
    age_threshold: float | None = None,
    mad_threshold: float = 3.0,
    tukey_fence_multiplier: float = 3.0,
    std_threshold: float | None = None,
) -> tuple[pl.DataFrame, float, int, set]:
    """Flags age outliers by a MAD rule on residuals and a Tukey fence on subject std.

    Args:
        results_df: Age prediction results, one row per sample.
        sample_id_col: Name of the sample ID column.
        age_threshold: Fixed threshold for the absolute residual; replaces the
            MAD-based one when given.
        mad_threshold: MAD multiplier for the residual threshold.
        tukey_fence_multiplier: IQR multiplier of the Tukey fence on per-subject
            prediction std.
        std_threshold: Fixed threshold for the per-subject std; replaces the Tukey
            fence when given.

    Returns:
        ``results_df`` with ``Is_Discordant``, ``From_Inconsistent_Participant`` and
        ``Std_Predicted_Age`` added, the effective residual threshold, the number of
        subjects with more than two timepoints, and the set of inconsistent USUBJIDs.
    """
    residuals = results_df["Residual"].to_numpy()

    # 1.4826 scales the MAD to a normal-consistent sigma.
    median_residual = np.median(np.abs(residuals))
    mad = np.median(np.abs(residuals - np.median(residuals)))
    robust_threshold = median_residual + mad_threshold * mad * 1.4826

    if age_threshold is not None:
        effective_threshold = age_threshold
        logger.info(
            f"MAD-based threshold: {robust_threshold:.1f} years | Using user age_threshold: {age_threshold} years"
        )
    else:
        effective_threshold = robust_threshold
        logger.info(f"Using MAD-based threshold ({mad_threshold}σ): {robust_threshold:.1f} years")

    results_df = results_df.with_columns(
        [(pl.col("Abs_Residual") > effective_threshold).alias("Is_Discordant")]
    )

    # A std needs at least three timepoints to mean anything.
    participant_stats = _participant_age_stats(results_df, sample_id_col).filter(
        pl.col("N_Timepoints") > 2
    )

    n_multi_timepoint = len(participant_stats)
    logger.info(f"participants with multiple timepoints: {n_multi_timepoint}")

    inconsistent_usubjids = set()
    if len(participant_stats) > 0:
        q1 = participant_stats["Std_Predicted_Age"].quantile(0.25)
        q3 = participant_stats["Std_Predicted_Age"].quantile(0.75)
        iqr = q3 - q1
        tukey_threshold = q3 + tukey_fence_multiplier * iqr

        if std_threshold is not None:
            inconsistent_threshold = std_threshold
            logger.info(
                f"Tukey fence: {tukey_threshold:.1f} years std | Using user std_threshold: {std_threshold} years std"
            )
        else:
            inconsistent_threshold = tukey_threshold
            logger.info(
                f"Using Tukey fence (Q3 + {tukey_fence_multiplier}×IQR): {tukey_threshold:.1f} years std"
            )

        inconsistent = participant_stats.filter(
            pl.col("Std_Predicted_Age") > inconsistent_threshold
        )
        inconsistent_usubjids = set(inconsistent["USUBJID"].to_list())

        if len(inconsistent_usubjids) > 0:
            logger.info(
                f"Participants with inconsistent predictions (std > {inconsistent_threshold:.1f}): {len(inconsistent_usubjids)}"
            )

    results_df = results_df.with_columns(
        pl.col("USUBJID").is_in(list(inconsistent_usubjids)).alias("From_Inconsistent_Participant")
    )

    if len(participant_stats) > 0:
        results_df = results_df.join(
            participant_stats.select(["USUBJID", "Std_Predicted_Age"]),
            on="USUBJID",
            how="left",
            maintain_order="left",
            validate="m:1",
        )

    return results_df, effective_threshold, n_multi_timepoint, inconsistent_usubjids


def sample_age_concordance(
    qc_instance,
    protage_table_path: str | Path = PROTAGE_TABLE_PATH,
    age_col: str = "AGE",
    alpha: float = AGE_ENET_ALPHA,
    l1_ratio: float = AGE_ENET_L1_RATIO,
    age_threshold: float | None = None,
    mad_threshold: float = 3.0,
    tukey_fence_multiplier: float = 3.0,
    std_threshold: float | None = None,
    genes_to_plot: list[str] | None = None,
    random_seed: int = 0,
    protage_uniprot_col: str = "UniProt ID",
    age_model: Literal["elasticnet", "ridgecv", "ridgecv_grouped"] = "elasticnet",
    max_missing: float = 0.1,
    feature_clip: float | None = None,
    fit_exclude=(),
    cv_folds: int = CV_FOLDS,
    knn_k: int = KNN_K,
) -> dict:
    """Predicts age from the ProtAge protein subset and flags discordant samples.

    By default an ElasticNet with the published fixed alpha is fitted on the ProtAge
    proteins; predictions are out of fold over subject-grouped CV.

    Args:
        qc_instance: The OlinkQC instance; samples are flagged on it.
        protage_table_path: ProtAge supplementary table (TSV).
        age_col: Name of the age column in ``adsl_merged``.
        alpha: Fixed ElasticNet regularization strength.
        l1_ratio: ElasticNet mixing (0 is Ridge, 1 is Lasso).
        age_threshold: Fixed threshold for the absolute residual; replaces the
            MAD-based one when given.
        mad_threshold: MAD multiplier for the residual threshold.
        tukey_fence_multiplier: IQR multiplier of the Tukey fence on per-subject
            prediction std.
        std_threshold: Fixed threshold for the per-subject std; replaces the Tukey
            fence when given.
        genes_to_plot: Assay IDs the results list as the featured proteins; the top
            predictors when ``None`` or when none of them was selected.
        random_seed: Seed of the ElasticNet.
        protage_uniprot_col: UniProt column of the ProtAge table.
        age_model: ``"elasticnet"`` (fixed alpha and l1_ratio), ``"ridgecv"``
            (RidgeCV, alphas 1e-2..1e4) or ``"ridgecv_grouped"`` (same alphas, alpha
            chosen by a grouped CV over subjects).
        max_missing: An assay missing in more than this fraction of samples is left
            out.
        feature_clip: Clip standardized features to +-this inside each training fold;
            None is off.
        fit_exclude: Sample IDs kept out of training but still predicted out of
            fold.
        cv_folds: Grouped CV folds for the out-of-fold predictions and the bias
            correction's inner CV (at most; fewer subjects give fewer).
        knn_k: Neighbors of the KNN imputation of the feature matrix.

    Returns:
        A results dictionary. ``results_df`` carries the raw ``Residual`` (predicted -
        actual) and ``Age_Residual_Corrected`` (residual with its age-dependent bias
        removed). It is ``{"error": ...}`` when no ProtAge protein matches or no
        feature is selected.
    """
    _make_age_regressor(
        age_model, alpha, l1_ratio, random_seed, feature_clip, cv_folds
    )  # validates age_model

    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    uniprot_col = qc_instance.uniprot_col

    assay_proteins = (
        qc_instance.analyte_metadata.select([protein_id_col, uniprot_col]).unique().collect()
    )

    protage_protein_ids, match_df = load_protage_proteins(
        tsv_path=protage_table_path,
        assay_proteins=assay_proteins,
        protage_uniprot_col=protage_uniprot_col,
        protein_id_col=protein_id_col,
        uniprot_col=uniprot_col,
    )

    if len(protage_protein_ids) == 0:
        logger.error("no ProtAge proteins matched the assay data")
        return {"error": "no_protage_matches"}

    # VISITAGE is derived lazily, so compute it if it was asked for and is missing.
    if (
        age_col == "VISITAGE"
        and "VISITAGE" not in qc_instance.adsl_merged.collect_schema().names()
        and hasattr(qc_instance, "_compute_visitage")
    ):
        logger.info("VISITAGE requested but not yet computed - computing now")
        qc_instance.adsl_merged = qc_instance._compute_visitage()

    age_map = (
        qc_instance.adsl_merged.select([sample_id_col, age_col, qc_instance.usubjid_col])
        .filter(pl.col(age_col).is_not_null() & pl.col(qc_instance.usubjid_col).is_not_null())
        .collect(engine="streaming")
    )

    # VISITNUM comes from the manifest, when it has one.
    if (
        qc_instance.manifest is not None
        and "VISITNUM" in qc_instance.manifest.collect_schema().names()
    ):
        visitnum_map = (
            qc_instance.manifest.select([sample_id_col, "VISITNUM"])
            .collect()
            .sort([sample_id_col, "VISITNUM"], nulls_last=True)
            .unique(subset=[sample_id_col], keep="first", maintain_order=True)
        )
        age_map = age_map.join(visitnum_map, on=sample_id_col, how="left", validate="m:1")

    # TRT01A is carried into the results frame.
    if "TRT01A" in qc_instance.adsl_merged.collect_schema().names():
        trt_map = (
            qc_instance.adsl_merged.select([sample_id_col, "TRT01A"])
            .unique(subset=[sample_id_col])
            .collect()
        )
        age_map = age_map.join(trt_map, on=sample_id_col, how="left", validate="m:1")

    X, y, sample_ids, groups, feature_cols, data_with_age, fit = prepare_age_data(
        qc_instance=qc_instance,
        age_map=age_map,
        age_col=age_col,
        protein_subset=protage_protein_ids,
        max_missing=max_missing,
        fit_exclude=fit_exclude,
        knn_k=knn_k,
    )

    def make_model():
        return _make_age_regressor(age_model, alpha, l1_ratio, random_seed, feature_clip, cv_folds)

    # Out-of-fold predictions (scaler fitted inside each fold). Samples outside the
    # outlier fit are predicted but never trained on.
    order, cv_splits = grouped_splits(groups, cv_folds)
    X_s, y_s, fit_s = X[order], y[order], fit[order]
    n_splits = len(cv_splits)
    train_splits = [(train[fit_s[train]], test) for train, test in cv_splits]
    oof_pred = np.empty(len(y))
    groups_s = groups[order]
    for train_idx, test_idx in train_splits:
        model = _fit_regressor(make_model(), X_s[train_idx], y_s[train_idx], groups_s[train_idx])
        oof_pred[test_idx] = model.predict(X_s[test_idx])
    y_pred = np.empty(len(y))
    y_pred[order] = oof_pred
    corrected_s = _bias_corrected_residuals(
        X_s,
        y_s,
        groups_s,
        train_splits,
        oof_pred,
        make_model,
        cv_folds,
    )
    residuals_corrected = np.empty(len(y))
    residuals_corrected[order] = corrected_s

    # Fitted on all training rows for the coefficients only.
    regressor = _fit_regressor(make_model(), X_s[fit_s], y_s[fit_s], groups_s[fit_s])
    final_reg = regressor[-1]
    coefs = final_reg.coef_
    optimal_alpha = float(final_reg.alpha_) if age_model != "elasticnet" else alpha

    selected_indices = (
        np.where(coefs != 0)[0] if age_model == "elasticnet" else np.arange(len(coefs))
    )
    n_selected = len(selected_indices)

    if n_selected == 0:
        logger.warning(f"No features selected with alpha={optimal_alpha}. Try decreasing alpha.")
        return {"error": "no_features_selected", "alpha": optimal_alpha}

    logger.info(
        f"{age_model} uses {n_selected}/{len(feature_cols)} ProtAge proteins (alpha={optimal_alpha:.4f})"
    )

    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    residuals = y_pred - y

    logger.info(f"MAE: {mae:.2f} years | R²: {r2:.3f}")

    results_dict = {
        sample_id_col: sample_ids,
        "USUBJID": groups,
        "Visit_Adj_Age": y,
        "Predicted_Age": y_pred,
        "Residual": residuals,
        "Abs_Residual": np.abs(residuals),
        "Age_Residual_Corrected": residuals_corrected,
    }

    if "VISITNUM" in data_with_age.columns:
        visitnum_values = data_with_age["VISITNUM"].to_numpy()
        results_dict["VISITNUM"] = visitnum_values

    if "TRT01A" in data_with_age.columns:
        trt_values = data_with_age["TRT01A"].to_list()
        results_dict["TRT01A"] = trt_values

    results_df = pl.DataFrame(results_dict)

    results_df, effective_threshold, n_multi_timepoint, inconsistent_usubjids = detect_age_outliers(
        results_df=results_df,
        sample_id_col=sample_id_col,
        age_threshold=age_threshold,
        mad_threshold=mad_threshold,
        tukey_fence_multiplier=tukey_fence_multiplier,
        std_threshold=std_threshold,
    )

    discordant = results_df.filter(pl.col("Is_Discordant"))
    n_discordant = len(discordant)
    discordant_rate = n_discordant / len(results_df) * 100

    logger.info(
        f"Discordant samples (|residual| > {effective_threshold:.1f} years): {n_discordant} ({discordant_rate:.1f}%)"
    )

    # Only assay IDs are returned, not display names.
    all_top_features = sorted(
        [(feature_cols[i], coefs[i]) for i in selected_indices],
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    if genes_to_plot:
        top_features = []
        for protein_id in genes_to_plot:
            if protein_id in feature_cols:
                idx = feature_cols.index(protein_id)
                if idx in selected_indices:
                    top_features.append((protein_id, coefs[idx]))

        if not top_features:
            logger.warning(
                f"None of the requested protein IDs {genes_to_plot} were found. Using top predictors."
            )
            top_features = all_top_features[:5]
        else:
            logger.info(f"Using custom proteins for plotting: {[f[0] for f in top_features]}")
    else:
        top_features = all_top_features[:5]

    n_both = 0
    n_pred_only = 0
    n_cons_only = 0

    if n_discordant > 0 or len(inconsistent_usubjids) > 0:
        samples_both_outliers = results_df.filter(
            pl.col("Is_Discordant") & pl.col("From_Inconsistent_Participant")
        )
        samples_prediction_only = results_df.filter(
            pl.col("Is_Discordant") & ~pl.col("From_Inconsistent_Participant")
        )
        samples_consistency_only = results_df.filter(
            ~pl.col("Is_Discordant") & pl.col("From_Inconsistent_Participant")
        )

        n_both = len(samples_both_outliers)
        n_pred_only = len(samples_prediction_only)
        n_cons_only = len(samples_consistency_only)

        if n_both > 0:
            warn_both_ids = samples_both_outliers[sample_id_col].to_list()
            logger.warning(
                f"Marking {n_both} samples as WARN (outliers in both prediction and consistency)"
            )
            qc_instance._flag_samples(warn_both_ids, WARN_AGE_CONCORDANCE)

        warn_ids = []
        if n_pred_only > 0:
            pred_warn_ids = samples_prediction_only[sample_id_col].to_list()
            warn_ids.extend(pred_warn_ids)
            logger.warning(f"Marking {n_pred_only} samples as WARN (prediction outliers only)")
            qc_instance._track_sample_qc(pred_warn_ids, WARN_AGE_PREDICTION)

        if n_cons_only > 0:
            cons_warn_ids = samples_consistency_only[sample_id_col].to_list()
            warn_ids.extend(cons_warn_ids)
            logger.warning(
                f"Marking {n_cons_only} samples as WARN (from inconsistent participants only)"
            )
            qc_instance._track_sample_qc(cons_warn_ids, WARN_AGE_CONSISTENCY)

        if warn_ids:
            qc_instance.warned_samples.extend(warn_ids)

        qc_instance.qc_results["age_discordant_warn"] = n_both + n_pred_only + n_cons_only
        qc_instance.qc_results["age_mae"] = float(mae)
        qc_instance.qc_results["age_r2"] = float(r2)

    # Kept on the instance as identity evidence.
    qc_instance.age_oof_predictions = results_df.select(
        [sample_id_col, "Predicted_Age", "Age_Residual_Corrected"]
    )

    unique_groups = np.unique(groups)
    n_unique_groups = len(unique_groups)

    return {
        "mae": mae,
        "r2": r2,
        "n_discordant": n_discordant,
        "n_selected": n_selected,
        "feature_cols": feature_cols,
        "age_model": age_model,
        "optimal_alpha": optimal_alpha,
        "l1_ratio": l1_ratio,
        "n_splits": n_splits,
        "n_unique_groups": n_unique_groups,
        "n_both": n_both,
        "n_pred_only": n_pred_only,
        "n_cons_only": n_cons_only,
        "n_inconsistent_participants": len(inconsistent_usubjids),
        "auto_alpha": False,
        "alpha": alpha,
        "top_age_proteins": all_top_features[:5],
        "all_selected_proteins": all_top_features,
        "data_with_age": data_with_age,
        "results_df": results_df,
        "discordant": discordant,
        "top_features": top_features,
        "y": y,
        "y_pred": y_pred,
        "sample_ids": sample_ids,
        "residuals": residuals,
        "effective_threshold": effective_threshold,
        "inconsistent_usubjids": inconsistent_usubjids,
        "participant_stats": _participant_age_stats(
            results_df, sample_id_col, f"Representative_{sample_id_col}"
        ).filter(pl.col("N_Timepoints") > 1),
        "tukey_fence_multiplier": tukey_fence_multiplier,
        "protage_match_info": match_df,
        "n_protage_matched": len(protage_protein_ids),
        "sample_id_col": sample_id_col,
        "protein_id_col": protein_id_col,
    }
