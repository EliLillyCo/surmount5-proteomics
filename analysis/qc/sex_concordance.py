"""Sex concordance checks using machine learning predictions.

This module provides sex concordance using sex-associated proteins from
published studies, adapted for proteomics data.
"""

from __future__ import annotations

# Standard library
import gc
import logging
import warnings
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.preprocessing import StandardScaler

# Local imports
from .utils import impute_missing_values

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
# Koprulu et al. 2025 (Nat Commun 16:4001) — see data/README.md §Published QC
# reference tables.  Regenerate with: uv run python analysis/scripts/qc_reference_tables.py
DEFAULT_SEX_TABLE = DATA_DIR / "41467_2025_59034_MOESM3_ESM.Table_S2.tsv"


def load_sex_proteins(
    tsv_path: str,
    soma_proteins: pl.DataFrame,
    p_threshold: float = 1e-100,
    pval_col_sl: str = "pval.SL",
    pval_col_ol: str = "pval.OL",
    protein_id_col: str = "SeqId",
    uniprot_col: str = "UniProt",
    seqid_col: str | None = "SeqId",
) -> tuple[list[str], pd.DataFrame]:
    """
    Load and match sex-associated proteins to SomaScan data.

    Args:
        tsv_path: Path to sex association table (Table S2)
        soma_proteins: DataFrame with columns [protein_id_col, uniprot_col, seqid_col]
        p_threshold: P-value threshold for significance (default: 1e-100)
        pval_col_sl: SomaLogic p-value column name (default: 'pval.SL')
        pval_col_ol: Olink p-value column name (default: 'pval.OL')
        protein_id_col: Column name for protein ID (default: 'SeqId')
        uniprot_col: Column name for UniProt ID (default: 'UniProt')
        seqid_col: Column name for SeqId (default: 'SeqId', None for Olink)

    Returns:
        Tuple of (list of matched protein IDs, match info DataFrame)
    """
    logger.info(f"loading sex-associated proteins from {tsv_path}")

    # Load sex association table
    sex_assoc_df = pd.read_csv(Path(tsv_path), sep="\t")
    logger.info(f"loaded {len(sex_assoc_df)} proteins with sex association results")

    # Filter for proteins significant on both platforms
    significant = sex_assoc_df[
        (sex_assoc_df[pval_col_sl] < p_threshold)
        & (sex_assoc_df[pval_col_ol] < p_threshold)
    ].copy()

    logger.info(
        f"Found {len(significant)} proteins significant at p<{p_threshold} "
        f"on BOTH platforms ({pval_col_sl} & {pval_col_ol})"
    )

    if len(significant) == 0:
        logger.warning(f"no proteins passed p<{p_threshold} threshold!")
        return [], pd.DataFrame()

    # Match by SeqId (SomaScan) or UniProt (Olink).
    matches = []

    if seqid_col is not None and seqid_col in soma_proteins.columns:
        # SomaScan matching: Use SeqId with format conversion.
        logger.info(f"matching by {seqid_col} (SomaScan format)")

        for idx, row in significant.iterrows():
            # Get SeqId from table (format: SeqId_10001_7).
            seq_id_table = row.get("SeqId", None)

            if pd.notna(seq_id_table) and seq_id_table.startswith("SeqId_"):
                # Convert SeqId_10001_7 -> 10001-7.
                parts = seq_id_table.replace("SeqId_", "").split("_")
                if len(parts) == 2:
                    seq_id_data = f"{parts[0]}-{parts[1]}"

                    match = soma_proteins.filter(pl.col(seqid_col) == seq_id_data)
                    if len(match) > 0:
                        matches.append(
                            {
                                "SeqId_Table": seq_id_table,
                                "SeqId_Data": seq_id_data,
                                "UniProt": row.get("UniProt", np.nan),
                                protein_id_col: match[protein_id_col][0],
                                "beta_SL": row.get("beta.SL", np.nan),
                                "pval_SL": row.get(pval_col_sl, np.nan),
                                "beta_OL": row.get("beta.OL", np.nan),
                                "pval_OL": row.get(pval_col_ol, np.nan),
                                "match_type": "SeqId",
                            }
                        )
    else:
        # Olink matching: Use UniProt ID
        logger.info(f"matching by {uniprot_col} (Olink format)")

        for idx, row in significant.iterrows():
            uniprot_id = row.get("UniProt", None)

            if pd.notna(uniprot_id):
                match = soma_proteins.filter(pl.col(uniprot_col) == uniprot_id)
                if len(match) > 0:
                    matches.append(
                        {
                            "UniProt": uniprot_id,
                            protein_id_col: match[protein_id_col][0],
                            "beta_SL": row.get("beta.SL", np.nan),
                            "pval_SL": row.get(pval_col_sl, np.nan),
                            "beta_OL": row.get("beta.OL", np.nan),
                            "pval_OL": row.get(pval_col_ol, np.nan),
                            "match_type": "UniProt",
                        }
                    )

    match_df = pd.DataFrame(matches)

    if len(match_df) > 0:
        match_type = match_df.iloc[0]["match_type"] if len(match_df) > 0 else "Unknown"
        logger.info(
            f"Matched {len(match_df)}/{len(significant)} sex-associated proteins by {match_type}"
        )
    else:
        logger.warning("no sex-associated proteins matched!")

    return (
        match_df[protein_id_col].tolist() if len(match_df) > 0 else [],
        match_df,
    )


def prepare_sex_data(
    qc_instance,
    sex_map: pl.DataFrame,
    sex_col: str = "SEX",
    usubjid_col: str = "USUBJID",
    protein_subset: list[str] | None = None,
    random_seed: int = 0,
    sample_id_col: str = "SampleId",
    protein_id_col: str = "SeqId",
    data_col: str = "log10_scaled_RFU",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray | None,
    pl.DataFrame,
]:
    """
    Prepare data for sex prediction.

    Args:
        qc_instance: SomaQC instance
        sex_map: DataFrame with sample_id_col, sex column, and usubjid column
        sex_col: Column name for sex
        usubjid_col: Column name for unique subject ID
        protein_subset: Optional list of protein IDs to use (if None, uses all PASS proteins)
        random_seed: Random seed for imputation
        sample_id_col: Column name for sample ID (default: 'SampleId')
        protein_id_col: Column name for protein ID (default: 'SeqId')
        data_col: Column name for data values (default: 'log10_scaled_RFU')

    Returns:
        Tuple of (X, y, sample_ids, groups, feature_cols, study_values, data_with_sex)
    """
    # Get wide format data (samples × proteins), excluding failed proteins and samples
    # Use qc_instance.samples which is already filtered to biological samples only
    sample_data = qc_instance.samples

    # Exclude failed samples
    if qc_instance.failed_samples:
        sample_data = sample_data.filter(
            ~pl.col(sample_id_col).is_in(qc_instance.failed_samples)
        )
        logger.info(f"excluding {len(qc_instance.failed_samples)} failed samples")

    # Exclude both FAIL and WARN proteins (only use PASS assays)
    failed_set = set(qc_instance.failed_proteins)
    warned_set = set(qc_instance.warned_proteins)
    excluded_proteins = list(failed_set | warned_set)
    if excluded_proteins:
        sample_data = sample_data.filter(
            ~pl.col(protein_id_col).is_in(excluded_proteins)
        )
        logger.info(
            f"Excluding {len(failed_set)} FAIL and {len(warned_set)} WARN {protein_id_col}s (total unique: {len(excluded_proteins)})"
        )

    # Filter to protein subset if specified
    if protein_subset is not None:
        sample_data = sample_data.filter(pl.col(protein_id_col).is_in(protein_subset))
        logger.info(f"using {len(protein_subset)} sex-associated proteins")

    # Extract study info before pivot (if available)
    study_map = None
    if "STUDY" in sample_data.columns:
        study_map = sample_data.select([sample_id_col, "STUDY"]).unique().collect()

    # Pivot to wide format
    data_wide = (
        sample_data.filter(
            pl.col(sample_id_col).is_in(sex_map[sample_id_col])
            & pl.col(data_col).is_not_null()
            & ~pl.col(data_col).is_nan()
        )
        .select([sample_id_col, protein_id_col, data_col])
        .group_by([sample_id_col, protein_id_col])
        .agg(pl.col(data_col).mean())
        .collect()
        .pivot(index=sample_id_col, columns=protein_id_col, values=data_col)
    )

    # Join with sex labels
    data_with_sex = data_wide.join(sex_map, on=sample_id_col, how="inner")

    # Join with study info if available
    if study_map is not None:
        data_with_sex = data_with_sex.join(study_map, on=sample_id_col, how="left")

    if len(data_with_sex) < 10:
        raise ValueError(
            f"Insufficient samples with sex labels: {len(data_with_sex)} < 10"
        )

    # Sort for deterministic ordering
    data_with_sex = data_with_sex.sort([usubjid_col, sample_id_col])

    # Get feature columns (sorted for deterministic ordering)
    # Exclude non-protein columns
    exclude_cols = {sample_id_col, sex_col, usubjid_col, "STUDY"}

    feature_cols = sorted([c for c in data_with_sex.columns if c not in exclude_cols])

    # Extract arrays
    X = data_with_sex.select(feature_cols).to_numpy()
    y_str = data_with_sex[sex_col].to_numpy()
    y = (y_str == "M").astype(int)
    sample_ids = data_with_sex[sample_id_col].to_numpy()
    groups = data_with_sex[usubjid_col].to_numpy()

    # Store study values if present
    study_values = None
    if "STUDY" in data_with_sex.columns:
        study_values = data_with_sex["STUDY"].to_numpy()

    # Use cached imputed data if available
    if (
        qc_instance._imputed_data_cache is not None
        and qc_instance._imputed_sample_ids is not None
        and len(qc_instance._imputed_sample_ids) == len(sample_ids)
        and np.all(qc_instance._imputed_sample_ids == sample_ids)
    ):
        logger.info("using cached imputed data from PCA")
        X = qc_instance._imputed_data_cache
    else:
        X = impute_missing_values(X, random_seed=random_seed)

    return X, y, sample_ids, groups, feature_cols, study_values, data_with_sex


def train_lasso_sex_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_mask: np.ndarray,
    C: float = 0.1,
    random_seed: int = 0,
    deterministic: bool = True,
) -> tuple[LogisticRegression, int, list]:
    """
    Train Lasso logistic regression sex prediction model.

    Args:
        X: Feature matrix (already scaled)
        y: Sex labels (1=Male, 0=Female)
        groups: Group labels for cross-validation (USUBJID)
        train_mask: Boolean mask for training samples
        C: Regularization strength (inverse of alpha)
        random_seed: Random seed
        deterministic: Whether to use deterministic parallel execution

    Returns:
        Tuple of (fitted model, n_splits, cv_splits)
    """
    # Get cross-validation configuration (only on training samples)
    train_groups = groups[train_mask]
    unique_groups_sorted = np.sort(np.unique(train_groups))
    n_unique_groups = len(unique_groups_sorted)
    n_splits = min(5, n_unique_groups)

    if n_splits < 2:
        logger.warning(
            f"Only {n_unique_groups} unique participants in training set. Skipping cross-validation."
        )
        raise ValueError(f"Insufficient participants: {n_unique_groups} < 2")

    gkf = GroupKFold(n_splits=n_splits)
    logger.info(
        f"Using {n_splits}-fold grouped cross-validation ({n_unique_groups} unique participants in training set)"
    )

    # Get training data
    X_train = X[train_mask]
    y_train = y[train_mask]

    # Sort groups for deterministic splits
    group_sort_idx = np.argsort(train_groups)
    X_train_sorted = X_train[group_sort_idx]
    y_train_sorted = y_train[group_sort_idx]
    train_groups_sorted = train_groups[group_sort_idx]

    cv_splits = list(gkf.split(X_train_sorted, y_train_sorted, train_groups_sorted))

    # Create and fit classifier
    classifier = LogisticRegression(
        C=C,
        penalty="l1",
        solver="saga",
        class_weight="balanced",
        random_state=random_seed,
        max_iter=10000,
        tol=1e-6,
    )

    # Fit using sorted data for determinism
    classifier.fit(X_train_sorted, y_train_sorted)

    return classifier, n_splits, cv_splits


def detect_sex_discordance(
    results_df: pl.DataFrame,
    probability_threshold: float = 0.8,
    sex_col: str = "SEX",
) -> tuple[pl.DataFrame, int]:
    """
    Detect sex discordance based on prediction probability.

    Args:
        results_df: DataFrame with columns ['SampleId', 'USUBJID', sex_col, 'Predicted_Sex', 'Predicted_Male_Probability']
        probability_threshold: Minimum probability to flag discordance
        sex_col: Column name for sex

    Returns:
        Tuple of (updated results_df, n_discordant)
    """
    # Flag discordant samples (high-confidence only)
    discordant_expr = (
        (pl.col(sex_col) == "M")
        & (pl.col("Predicted_Male_Probability") < (1 - probability_threshold))
    ) | (
        (pl.col(sex_col) == "F")
        & (pl.col("Predicted_Male_Probability") > probability_threshold)
    )

    results_df = results_df.with_columns([discordant_expr.alias("Is_Discordant")])

    n_discordant = results_df.filter(pl.col("Is_Discordant")).height

    logger.info(
        f"Discordant samples: {n_discordant} (probability threshold: {probability_threshold})"
    )

    return results_df, n_discordant


def sample_sex_concordance(
    qc_instance,
    sex_table_path: str | Path = DEFAULT_SEX_TABLE,
    sex_col: str = "SEX",
    p_threshold: float = 1e-100,
    pval_col_sl: str = "pval.SL",
    pval_col_ol: str = "pval.OL",
    C: float = 0.1,
    probability_threshold: float = 0.8,
    genes_to_plot: list[str] | None = None,
    random_seed: int = 0,
    deterministic: bool = True,
    train_cohorts: list[str] | None = None,
) -> dict:
    """
    Sex concordance using sex-associated proteins from published studies.

    Uses Lasso (L1-regularized) logistic regression with proteins significantly
    associated with sex in published literature (p<1e-100 by default).

    Args:
        qc_instance: SomaQC instance
        sex_table_path: Path to sex association table (Table S2)
        sex_col: Column name for sex (M/F)
        p_threshold: P-value threshold for significance (default 1e-100)
        pval_col_sl: SomaLogic p-value column name (default 'pval.SL')
        pval_col_ol: Olink p-value column name (default 'pval.OL')
        C: Regularization strength (inverse of alpha, default 0.1)
        probability_threshold: Minimum probability to flag discordance (default 0.8)
        genes_to_plot: Optional list of gene names to plot
        random_seed: Random seed for reproducibility
        deterministic: If True, uses n_jobs=1 for deterministic results
        train_cohorts: Optional list of cohort names to use for training

    Returns:
        Dictionary with results
    """
    qc_instance._print_section("SEX CONCORDANCE CHECK (PUBLISHED ASSOCIATIONS)")

    # Get column names from instance
    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    protein_name_col = qc_instance.protein_name_col
    uniprot_col = qc_instance.uniprot_col
    data_col = qc_instance.data_col

    # For SomaScan, protein_id_col is SeqId; for Olink, it's OlinkID
    # SeqId matching is only used for SomaScan
    if protein_id_col == "SeqId":
        seqid_col = protein_id_col
    elif protein_id_col == "OlinkID":
        seqid_col = None
    else:
        raise ValueError(
            f"Unsupported platform: protein_id_col='{protein_id_col}'. Expected 'SeqId' or 'OlinkID'."
        )

    # Get unique proteins from analyte metadata for matching
    protein_cols = [protein_id_col, uniprot_col]

    soma_proteins = qc_instance.analyte_metadata.select(protein_cols).unique().collect()

    # Load and match sex-associated proteins
    sex_protein_ids, match_df = load_sex_proteins(
        tsv_path=sex_table_path,
        soma_proteins=soma_proteins,
        p_threshold=p_threshold,
        pval_col_sl=pval_col_sl,
        pval_col_ol=pval_col_ol,
        protein_id_col=protein_id_col,
        uniprot_col=uniprot_col,
        seqid_col=seqid_col,
    )

    if len(sex_protein_ids) == 0:
        logger.error("no sex-associated proteins matched to SomaScan data!")
        return {"error": "no_sex_protein_matches"}

    # Get sex map from adsl_merged
    sex_map = (
        qc_instance.adsl_merged.select(
            [sample_id_col, sex_col, qc_instance.usubjid_col]
        )
        .filter(
            pl.col(sex_col).is_in(["M", "F"])
            & pl.col(qc_instance.usubjid_col).is_not_null()
        )
        .collect()
    )

    # Prepare data with sex-associated protein subset
    X, y, sample_ids, groups, feature_cols, study_values, data_with_sex = (
        prepare_sex_data(
            qc_instance=qc_instance,
            sex_map=sex_map,
            sex_col=sex_col,
            usubjid_col=qc_instance.usubjid_col,
            protein_subset=sex_protein_ids,
            random_seed=random_seed,
            sample_id_col=sample_id_col,
            protein_id_col=protein_id_col,
            data_col=data_col,
        )
    )

    # Filter training samples by cohort if requested
    train_mask = np.ones(len(X), dtype=bool)
    if train_cohorts is not None and study_values is not None:
        train_mask = np.isin(study_values, train_cohorts)
        n_train = train_mask.sum()
        logger.info(
            f"Restricting model training to {len(train_cohorts)} cohort(s): {', '.join(train_cohorts)} ({n_train}/{len(X)} samples)"
        )
        if n_train < 20:
            logger.warning(
                f"Only {n_train} training samples in specified cohorts - may be insufficient"
            )
    elif train_cohorts is not None:
        logger.warning(
            "train_cohorts specified but 'STUDY' column not found - using all samples for training"
        )

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    classifier, n_splits, cv_splits = train_lasso_sex_model(
        X=X_scaled,
        y=y,
        groups=groups,
        train_mask=train_mask,
        C=C,
        random_seed=random_seed,
        deterministic=deterministic,
    )

    # Get predictions
    X_train = X_scaled[train_mask]
    y_train = y[train_mask]
    n_jobs = 1 if deterministic else -1
    y_pred_proba_train = cross_val_predict(
        classifier,
        X_train,
        y_train,
        cv=cv_splits,
        method="predict_proba",
        n_jobs=n_jobs,
    )[:, 1]

    # Apply to all samples
    y_pred_proba = classifier.predict_proba(X_scaled)[:, 1]
    coefs = classifier.coef_[0]

    selected_indices = np.where(coefs != 0)[0]
    n_selected = len(selected_indices)

    if n_selected == 0:
        logger.warning(f"no features selected with C={C}. Try increasing C.")
        return {"error": "no_features_selected", "C": C}

    logger.info(
        f"Lasso selected {n_selected}/{len(feature_cols)} sex-associated proteins (C={C})"
    )

    # Calculate metrics
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y, y_pred_binary)
    auc = roc_auc_score(y, y_pred_proba)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred_binary).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    logger.info(f"Accuracy: {accuracy:.3f} | AUC: {auc:.3f}")
    logger.info(
        f"Sensitivity (male): {sensitivity:.3f} | Specificity (female): {specificity:.3f}"
    )

    # Create results dataframe
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

    # Detect discordance
    results_df, n_discordant = detect_sex_discordance(
        results_df, probability_threshold, sex_col
    )

    # Get top male and female predictors for visualization
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

    # Get discordant samples
    discordant = results_df.filter(pl.col("Is_Discordant"))

    # Track discordant samples
    if n_discordant > 0:
        discordant_ids = discordant[sample_id_col].to_list()
        logger.warning(
            f"Identified {n_discordant} sex-discordant samples (not excluded from age concordance analyses)"
        )
        qc_instance._track_sample_qc(discordant_ids, "FAIL_Sex_Concordance")
        qc_instance.qc_results["sex_discordant_failed"] = n_discordant

    # Clean up large variables
    try:
        del X, X_scaled, y_pred_proba
    except Exception:
        pass
    gc.collect()

    # Return results in format expected by reporting module
    return {
        "results_df": results_df.rename({sex_col: "Reported_Sex"}),
        "discordant": discordant,
        "n_selected": n_selected,
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
