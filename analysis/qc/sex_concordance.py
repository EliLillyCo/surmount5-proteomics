"""Sex concordance checks from out-of-fold sex predictions on published proteins."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from analysis.qc.corrections import (
    EXPECTED_SEX_COL,
    SEX_EXCEPTION_COL,
    with_expected_sex,
)
from analysis.qc.flags import WARN_SEX_CONCORDANCE
from analysis.qc.modeling import (
    CV_FOLDS,
    KNN_K,
    RETIRED_UNIPROT,
    build_feature_matrix,
    explode_assay_uniprot,
    grouped_splits,
)
from analysis.qc_paths import SEX_TABLE

logger = logging.getLogger(__name__)

# Single sex-discordance cutoff: P(male) > this for reported females, < 1 - this for
# reported males.
SEX_PROB_THRESHOLD = 0.8
# Default of TrialConfig ``sex_c``: the sex model's inverse regularization strength.
SEX_C = 0.1
# The published sex-association table (Koprulu 2025, Table S2) the ``qc_reference`` stage
# writes.
SEX_TABLE_PATH = SEX_TABLE


def load_sex_proteins(
    tsv_path: str,
    assay_proteins: pl.DataFrame,
    p_threshold: float = 1e-100,
    pval_col_sl: str = "pval.SL",
    pval_col_ol: str = "pval.OL",
    protein_id_col: str = "SeqId",
    uniprot_col: str = "UniProt",
    seqid_col: str | None = "SeqId",
) -> tuple[list[str], pl.DataFrame]:
    """Loads the sex-associated proteins and matches them to the assays.

    Keeps table rows significant on both of the table's platforms, then matches them
    to the assays by SeqId (SomaScan, when ``seqid_col`` is set and present) or by
    UniProt (Olink). Table row order is kept.

    Args:
        tsv_path: Path to the sex association table (Table S2).
        assay_proteins: Assay table with ``protein_id_col`` and ``uniprot_col``.
        p_threshold: Significance threshold on both platforms' p-values.
        pval_col_sl: The table's SomaLogic p-value column.
        pval_col_ol: The table's Olink p-value column.
        protein_id_col: Assay id column.
        uniprot_col: UniProt id column.
        seqid_col: SeqId column of ``assay_proteins``; None for Olink.

    Returns:
        The matched assay ids (one per matched table row) and the match table.
    """
    logger.info(f"loading sex-associated proteins from {tsv_path}")

    sex_assoc_df = pl.read_csv(
        tsv_path, separator="\t", null_values=["NA"], infer_schema_length=None
    )
    logger.info(f"loaded {len(sex_assoc_df)} proteins with sex association results")

    significant = sex_assoc_df.filter(
        (pl.col(pval_col_sl) < p_threshold) & (pl.col(pval_col_ol) < p_threshold)
    )

    logger.info(
        f"Found {len(significant)} proteins significant at p<{p_threshold} "
        f"on BOTH platforms ({pval_col_sl} & {pval_col_ol})"
    )

    if len(significant) == 0:
        logger.warning(f"no proteins passed p<{p_threshold} threshold!")
        return [], pl.DataFrame()

    def _stat(col: str) -> pl.Expr:
        return pl.col(col) if col in significant.columns else pl.lit(None, dtype=pl.Float64)

    stats = [
        _stat("beta.SL").alias("beta_SL"),
        pl.col(pval_col_sl).alias("pval_SL"),
        _stat("beta.OL").alias("beta_OL"),
        pl.col(pval_col_ol).alias("pval_OL"),
    ]

    if seqid_col is not None and seqid_col in assay_proteins.columns:
        logger.info(f"matching by {seqid_col} (SomaScan format)")
        # The reference table writes SeqIds as SeqId_<n>_<m>, and the assay data as <n>-<m>.
        seq_parts = pl.col("SeqId").str.strip_prefix("SeqId_").str.split("_")
        assay_by_seqid = assay_proteins.select(
            pl.col(seqid_col).alias("SeqId_Data"),
            pl.col(protein_id_col).alias("_assay"),
        ).unique(subset="SeqId_Data", keep="first", maintain_order=True)
        match_df = (
            significant.filter(
                pl.col("SeqId").str.starts_with("SeqId_") & (seq_parts.list.len() == 2)
            )
            .with_columns(seq_parts.list.join("-").alias("SeqId_Data"))
            .join(assay_by_seqid, on="SeqId_Data", how="inner", maintain_order="left")
            .select(
                pl.col("SeqId").alias("SeqId_Table"),
                "SeqId_Data",
                pl.col("UniProt"),
                pl.col("_assay").alias(protein_id_col),
                *stats,
                pl.lit("SeqId").alias("match_type"),
            )
        )
    else:
        logger.info(f"matching by {uniprot_col} (Olink format)")
        assay_uniprots = explode_assay_uniprot(assay_proteins, protein_id_col, uniprot_col)
        first_assay = assay_uniprots.unique(
            subset=uniprot_col, keep="first", maintain_order=True
        ).select(pl.col(uniprot_col).alias("UniProt"), pl.col(protein_id_col).alias("_assay"))
        match_df = (
            significant.filter(pl.col("UniProt").is_not_null())
            # Same remap as the assay side.
            .with_columns(pl.col("UniProt").replace(RETIRED_UNIPROT))
            .join(first_assay, on="UniProt", how="inner", maintain_order="left")
            .select(
                pl.col("UniProt"),
                pl.col("_assay").alias(protein_id_col),
                *stats,
                pl.lit("UniProt").alias("match_type"),
            )
        )

    if len(match_df) > 0:
        logger.info(
            f"Matched {match_df['UniProt'].n_unique()}/{significant['UniProt'].n_unique()} "
            f"sex-associated UniProts by {match_df['match_type'][0]} "
            f"({len(match_df)} table rows, {match_df[protein_id_col].n_unique()} assays)"
        )
    else:
        logger.warning("no sex-associated proteins matched!")

    return match_df[protein_id_col].to_list(), match_df


def _make_sex_clf(sex_model: Literal["l1", "l2"], C: float, random_seed: int) -> LogisticRegression:
    """Builds the sex-prediction classifier for the configured ``sex_model``.

    ``l2`` (ridge) keeps all sex-associated proteins and is the default; ``l1``
    (lasso) sparsifies to a subset of coefficients.

    Raises:
        ValueError: If ``sex_model`` is neither ``l1`` nor ``l2``.
    """
    if sex_model == "l2":
        return LogisticRegression(
            l1_ratio=0,
            C=C,
            solver="lbfgs",
            max_iter=10000,
            random_state=random_seed,
            class_weight="balanced",
            tol=1e-6,
        )
    if sex_model == "l1":
        return LogisticRegression(
            l1_ratio=1,
            C=C,
            solver="saga",
            max_iter=10000,
            random_state=random_seed,
            class_weight="balanced",
            tol=1e-6,
        )
    raise ValueError(f"sex_model must be 'l1' or 'l2', got {sex_model!r}")


def prepare_sex_data(
    qc_instance,
    sex_map: pl.DataFrame,
    sex_col: str = "SEX",
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
    """Builds the feature matrix and the male label (1 = M) for sex prediction.

    Args:
        qc_instance: An OlinkQC instance.
        sex_map: Sample ids with their sex label and subject id.
        sex_col: Sex column in ``sex_map`` (M/F).
        protein_subset: Assays to use as features.
        max_missing: An assay missing in more than this fraction of samples is left out.
        fit_exclude: Sample ids kept out of training.
        knn_k: Neighbors of the KNN imputation.

    Returns:
        ``(X, y, sample_ids, groups, feature_cols, data_with_sex, fit)``, where
        ``fit`` flags the rows the model may train on.
    """
    fm = build_feature_matrix(
        qc_instance,
        labels=sex_map,
        label_cols=[sex_col],
        protein_subset=protein_subset or [],
        max_missing=max_missing,
        fit_exclude=fit_exclude,
        knn_k=knn_k,
    )
    y = (fm.frame[sex_col].to_numpy() == "M").astype(int)
    return fm.X, y, fm.sample_ids, fm.groups, fm.feature_cols, fm.frame, fm.fit


def detect_sex_discordance(
    results_df: pl.DataFrame,
    probability_threshold: float = SEX_PROB_THRESHOLD,
    sex_col: str = "SEX",
) -> tuple[pl.DataFrame, int]:
    """Flags samples whose predicted sex contradicts the label with high confidence.

    A reported male is discordant when P(male) is below ``1 - probability_threshold``,
    a reported female when it is above ``probability_threshold``.

    Args:
        results_df: Predictions with ``sex_col`` and ``Predicted_Male_Probability``.
        probability_threshold: Confidence needed to call a sample discordant.
        sex_col: Sex column (M/F).

    Returns:
        ``results_df`` with a boolean ``Is_Discordant`` column, and the number of
        discordant samples.
    """
    discordant_expr = (
        (pl.col(sex_col) == "M")
        & (pl.col("Predicted_Male_Probability") < (1 - probability_threshold))
    ) | ((pl.col(sex_col) == "F") & (pl.col("Predicted_Male_Probability") > probability_threshold))

    results_df = results_df.with_columns([discordant_expr.alias("Is_Discordant")])

    n_discordant = results_df.filter(pl.col("Is_Discordant")).height

    logger.info(
        f"Discordant samples: {n_discordant} (probability threshold: {probability_threshold})"
    )

    return results_df, n_discordant


def sample_sex_concordance(
    qc_instance,
    sex_table_path: str | Path = SEX_TABLE_PATH,
    sex_col: str = "SEX",
    p_threshold: float = 1e-100,
    pval_col_sl: str = "pval.SL",
    pval_col_ol: str = "pval.OL",
    C: float = SEX_C,
    probability_threshold: float = SEX_PROB_THRESHOLD,
    random_seed: int = 0,
    sex_model: Literal["l1", "l2"] = "l2",
    max_missing: float = 0.1,
    feature_clip: float | None = None,
    fit_exclude=(),
    cv_folds: int = CV_FOLDS,
    knn_k: int = KNN_K,
) -> dict:
    """Predicts each sample's sex from published sex-associated proteins.

    Fits a logistic regression (ridge by default, lasso with ``sex_model="l1"``) on
    the proteins significantly associated with sex in the published table, with
    out-of-fold predictions grouped by subject. The label is the subject's expected
    sex (an ``expected_sex`` exception, else the recorded sex); the recorded value is
    reported as ``Reported_Sex``. Discordant samples get ``WARN_Sex_Concordance`` and
    stay in the age concordance analyses. ``Sex_Exception`` is always present, false
    for every tube when the config has no exception. The instance also keeps the
    out-of-fold probabilities as ``sex_oof_predictions`` for identity evidence.

    Args:
        qc_instance: An OlinkQC instance.
        sex_table_path: Path to the sex association table (Table S2).
        sex_col: Sex column (M/F).
        p_threshold: Significance threshold on both platforms' p-values.
        pval_col_sl: The table's SomaLogic p-value column.
        pval_col_ol: The table's Olink p-value column.
        C: Inverse regularization strength.
        probability_threshold: Confidence needed to flag discordance.
        random_seed: Seed of the classifier.
        sex_model: ``"l2"`` (ridge) or ``"l1"`` (lasso).
        max_missing: An assay missing in more than this fraction of samples is left out.
        feature_clip: Clips standardized features to +-this inside each training fold;
            None is off.
        fit_exclude: Sample ids kept out of training but still predicted out of fold.
        cv_folds: Grouped CV folds for the out-of-fold predictions (at most; fewer
            subjects give fewer).
        knn_k: Neighbors of the KNN imputation of the feature matrix.

    Returns:
        A dict of results (``results_df``, ``discordant``,
        model statistics and settings), or ``{"error": ...}`` when no protein matched
        or the lasso kept none, or ``{"skipped": True, ...}`` for a single-sex trial.
    """
    _make_sex_clf(sex_model, C, random_seed)  # Validates sex_model before any work.

    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    uniprot_col = qc_instance.uniprot_col

    protein_cols = [protein_id_col, uniprot_col]
    # SeqId matching is SomaScan only (protein_id_col is the SeqId there).
    seqid_col = protein_id_col if qc_instance.platform.name == "soma" else None

    assay_proteins = qc_instance.analyte_metadata.select(protein_cols).unique().collect()

    sex_protein_ids, match_df = load_sex_proteins(
        tsv_path=sex_table_path,
        assay_proteins=assay_proteins,
        p_threshold=p_threshold,
        pval_col_sl=pval_col_sl,
        pval_col_ol=pval_col_ol,
        protein_id_col=protein_id_col,
        uniprot_col=uniprot_col,
        seqid_col=seqid_col,
    )

    if len(sex_protein_ids) == 0:
        logger.error("no sex-associated proteins matched the assay data")
        return {"error": "no_sex_protein_matches"}

    # The label is the expected protein sex, not the recorded one.
    labeled = with_expected_sex(
        qc_instance.adsl_merged.select([sample_id_col, sex_col, qc_instance.usubjid_col]).collect(),
        qc_instance.expected_sex,
        qc_instance.usubjid_col,
        sex_col,
        EXPECTED_SEX_COL,
    ).filter(
        pl.col(EXPECTED_SEX_COL).is_in(["M", "F"]) & pl.col(qc_instance.usubjid_col).is_not_null()
    )
    recorded = labeled.select(sample_id_col, pl.col(sex_col).alias("Reported_Sex"))
    sex_map = labeled.select(
        sample_id_col, pl.col(EXPECTED_SEX_COL).alias(sex_col), qc_instance.usubjid_col
    )

    X, y, sample_ids, groups, feature_cols, _, fit = prepare_sex_data(
        qc_instance=qc_instance,
        sex_map=sex_map,
        sex_col=sex_col,
        protein_subset=sex_protein_ids,
        max_missing=max_missing,
        fit_exclude=fit_exclude,
        knn_k=knn_k,
    )

    if len(np.unique(y[fit])) < 2:
        logger.warning(
            f"Only one sex in trial ({np.unique(y)[0]}). "
            "Skipping sex concordance — cannot train classifier."
        )
        qc_instance.sex_oof_predictions = None
        return {"skipped": True, "reason": "single_class"}

    # Out-of-fold predictions; the scaler is fitted inside each fold.
    order, cv_splits = grouped_splits(groups, cv_folds)
    X_s, y_s, fit_s = X[order], y[order], fit[order]

    def standardize(scaler, rows):
        z = scaler.transform(rows)
        return z if feature_clip is None else np.clip(z, -feature_clip, feature_clip)

    oof_pred = np.empty(len(y))
    for train_idx, test_idx in cv_splits:
        # Samples outside the outlier fit are predicted but not trained on.
        train_idx = train_idx[fit_s[train_idx]]
        sc = StandardScaler().fit(X_s[train_idx])
        clf = _make_sex_clf(sex_model, C, random_seed)
        clf.fit(standardize(sc, X_s[train_idx]), y_s[train_idx])
        oof_pred[test_idx] = clf.predict_proba(standardize(sc, X_s[test_idx]))[:, 1]
    y_pred_proba = np.empty(len(y))
    y_pred_proba[order] = oof_pred

    # All-data model, for the coefficients only.
    scaler = StandardScaler().fit(X_s[fit_s])
    classifier = _make_sex_clf(sex_model, C, random_seed)
    classifier.fit(standardize(scaler, X_s[fit_s]), y_s[fit_s])
    coefs = classifier.coef_[0]

    if sex_model == "l2":
        selected_indices = np.arange(len(coefs))
        n_selected = len(selected_indices)
        logger.info(f"Ridge model uses all {n_selected} sex-associated proteins (C={C})")
    else:
        selected_indices = np.where(coefs != 0)[0]
        n_selected = len(selected_indices)
        if n_selected == 0:
            logger.warning(f"no features selected with C={C}. Try increasing C.")
            return {"error": "no_features_selected", "C": C}
        logger.info(
            f"Lasso selected {n_selected}/{len(feature_cols)} sex-associated proteins (C={C})"
        )

    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y, y_pred_binary)
    auc = roc_auc_score(y, y_pred_proba)

    tn, fp, fn, tp = confusion_matrix(y, y_pred_binary).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    logger.info(f"Accuracy: {accuracy:.3f} | AUC: {auc:.3f}")
    logger.info(f"Sensitivity (male): {sensitivity:.3f} | Specificity (female): {specificity:.3f}")

    y_str = np.array(["M" if val == 1 else "F" for val in y])
    y_pred_str = np.array(["M" if val == 1 else "F" for val in y_pred_binary])

    results_df = pl.DataFrame(
        {
            sample_id_col: sample_ids,
            "USUBJID": groups,
            sex_col: y_str,
            "Predicted_Sex": y_pred_str,
            "Predicted_Male_Probability": y_pred_proba,
        }
    )

    results_df, n_discordant = detect_sex_discordance(results_df, probability_threshold, sex_col)

    top_male = sorted(
        [(i, coefs[i]) for i in selected_indices if coefs[i] > 0],
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    top_female = sorted(
        [(i, coefs[i]) for i in selected_indices if coefs[i] < 0],
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    if top_male:
        top_male_idx = top_male[0][0]
        top_male_feature = feature_cols[top_male_idx]
    else:
        top_male_feature = "None"

    if top_female:
        top_female_idx = top_female[0][0]
        top_female_feature = feature_cols[top_female_idx]
    else:
        top_female_feature = "None"

    logger.info(f"top male predictor: {top_male_feature}")
    logger.info(f"top female predictor: {top_female_feature}")

    discordant = results_df.filter(pl.col("Is_Discordant"))

    if n_discordant > 0:
        discordant_ids = discordant[sample_id_col].to_list()
        logger.warning(
            f"Identified {n_discordant} sex-discordant samples (not excluded from age concordance analyses)"
        )
        qc_instance._flag_samples(discordant_ids, WARN_SEX_CONCORDANCE)
        qc_instance.qc_results["sex_discordant_warned"] = n_discordant

    qc_instance.sex_oof_predictions = results_df.select(
        [sample_id_col, "Predicted_Male_Probability"]
    )

    # Fixed schema: Sex_Exception is present even without exceptions.
    results_df = results_df.rename({sex_col: EXPECTED_SEX_COL}).join(
        recorded, on=sample_id_col, how="left", validate="1:1", maintain_order="left"
    )
    excepted = (
        []
        if qc_instance.expected_sex is None
        else qc_instance.expected_sex[qc_instance.usubjid_col].to_list()
    )
    results_df = results_df.with_columns(pl.col("USUBJID").is_in(excepted).alias(SEX_EXCEPTION_COL))

    return {
        "results_df": results_df,
        "n_sex_exceptions": 0
        if qc_instance.expected_sex is None
        else qc_instance.expected_sex.height,
        "discordant": discordant,
        "n_selected": n_selected,
        "n_features": len(feature_cols),
        "model_type": sex_model,
        "n_splits": len(cv_splits),
        "feature_cols": feature_cols,
        "top_male_assay": top_male_feature,
        "top_male_feature": top_male_feature,
        "top_female_assay": top_female_feature,
        "top_female_feature": top_female_feature,
        "C": C,
        "probability_threshold": probability_threshold,
        "sex_map": sex_map,
        "sex_col": sex_col,
        "random_seed": random_seed,
        "n_discordant": n_discordant,
        "n_samples": len(results_df),
        "accuracy": accuracy,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "p_threshold": p_threshold,
        "pval_columns": f"{pval_col_sl}, {pval_col_ol}",
        "method": "Published associations",
        "match_df": match_df,
        "sample_id_col": sample_id_col,
        "protein_id_col": protein_id_col,
    }
