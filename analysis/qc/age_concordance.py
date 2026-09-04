"""Age concordance checks using machine learning predictions.

This module provides age concordance using the ProtAge 204 proteins with fixed alpha
(published clock approach), adapted for proteomics data.
"""

from __future__ import annotations

# Standard library
import gc
import logging
import warnings
from pathlib import Path

# Third-party
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.preprocessing import StandardScaler

# Local imports
from .utils import impute_missing_values

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
# Argentieri et al. 2024 (Nat Med 30:2450) — see data/README.md §Published QC
# reference tables.  Regenerate with: uv run python analysis/scripts/qc_reference_tables.py
DEFAULT_PROTAGE_TABLE = DATA_DIR / "41591_2024_3164_MOESM3_ESM.Table_S1.tsv"


def load_protage_proteins(
    tsv_path: str,
    soma_proteins: pl.DataFrame,
    protage_uniprot_col: str = "UniProt ID",
    protein_id_col: str = "SeqId",
    uniprot_col: str = "UniProt",
) -> tuple[list[str], pl.DataFrame]:
    """
    Load and match ProtAge 204 proteins to SomaScan data.

    Args:
        tsv_path: Path to ProtAge supplementary table (TableS1.tsv)
        soma_proteins: DataFrame with columns [protein_id_col, uniprot_col]
        protage_uniprot_col: Column name for UniProt in ProtAge table (default: 'UniProt ID')
        protein_id_col: Column name for protein ID (default: 'SeqId')
        uniprot_col: Column name for UniProt in proteomics data (default: 'UniProt')

    Returns:
        Tuple of (list of matched protein IDs, match info DataFrame)
    """
    logger.info(f"loading ProtAge proteins from {tsv_path}")

    # Load ProtAge table (treat "NA" as null values)
    protage_df = pl.read_csv(Path(tsv_path), separator="\t", null_values=["NA"])
    logger.info(f"loaded {len(protage_df)} ProtAge proteins")

    # Match by UniProt ID.
    matches_uniprot_df = (
        protage_df.filter(pl.col(protage_uniprot_col).is_not_null())
        .join(
            soma_proteins,
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
    )

    matches = matches_uniprot_df.to_dicts()

    if not matches:
        logger.warning("no matches found!")
        match_df = pl.DataFrame(
            {
                "ProtAge_UniProt": [],
                protein_id_col: [],
                "match_type": [],
            }
        )
        return [], match_df

    match_df = pl.DataFrame(matches)

    logger.info(
        f"Matched {len(match_df)}/{len(protage_df)} ProtAge proteins by UniProt ID"
    )

    matched_protein_ids = match_df[protein_id_col].to_list()

    return matched_protein_ids, match_df


def prepare_age_data(
    qc_instance,
    age_map: pl.DataFrame,
    age_col: str,
    usubjid_col: str,
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
    Prepare wide-format data for age prediction.

    Args:
        qc_instance: SomaQC instance
        age_map: DataFrame with columns [sample_id_col, age_col, usubjid_col]
        age_col: Column name for age values (e.g., 'AGE' or 'VISITAGE')
        usubjid_col: Subject identifier column name
        protein_subset: Optional list of protein IDs to filter to (e.g., ProtAge proteins)
        random_seed: Random seed for imputation
        sample_id_col: Column name for sample ID (default: 'SampleId')
        protein_id_col: Column name for protein ID (default: 'SeqId')
        data_col: Column name for data values (default: 'log10_scaled_RFU')

    Returns:
        Tuple of (X, y, sample_ids, groups, feature_cols, study_values, data_with_age)
    """
    # Get wide format data (samples × proteins), excluding failed proteins and samples
    # Use qc_instance.samples which is already filtered to biological samples only
    sample_data = qc_instance.samples

    # Exclude failed samples (e.g., PCA outliers)
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
            f"excluding {len(failed_set)} FAIL and {len(warned_set)} WARN {protein_id_col}s (total unique: {len(excluded_proteins)})"
        )

    # Filter to protein subset if provided (e.g., ProtAge proteins)
    if protein_subset:
        sample_data = sample_data.filter(pl.col(protein_id_col).is_in(protein_subset))
        logger.info(f"using {len(protein_subset)} selected proteins")

    # Extract study info before pivot (if available)
    study_map = None
    if "STUDY" in sample_data.columns:
        study_map = sample_data.select([sample_id_col, "STUDY"]).unique().collect()

    data_wide = (
        sample_data.filter(
            pl.col(sample_id_col).is_in(age_map[sample_id_col])
            & pl.col(data_col).is_not_null()
            & ~pl.col(data_col).is_nan()
        )
        .select([sample_id_col, protein_id_col, data_col])
        .group_by([sample_id_col, protein_id_col])
        .agg(pl.col(data_col).mean())
        .collect()
        .pivot(index=sample_id_col, columns=protein_id_col, values=data_col)
    )

    # Join with age labels
    data_with_age = data_wide.join(age_map, on=sample_id_col, how="inner")

    # Join with study info if available
    if study_map is not None:
        data_with_age = data_with_age.join(study_map, on=sample_id_col, how="left")

    if len(data_with_age) < 20:
        logger.warning("insufficient samples with age labels (need ≥20)")
        raise ValueError(f"Insufficient samples: {len(data_with_age)} < 20")

    # Sort by usubjid_col then sample_id_col for deterministic ordering in cross-validation
    data_with_age = data_with_age.sort([usubjid_col, sample_id_col])

    # Prepare features and labels
    # Sort feature columns for deterministic ordering
    # Exclude metadata columns
    feature_cols = sorted(
        [
            c
            for c in data_with_age.columns
            if c
            not in [
                sample_id_col,
                age_col,
                usubjid_col,
                "STUDY",
                "VISITNUM",
                "TRT01A",
            ]
        ]
    )
    X = data_with_age.select(feature_cols).to_numpy()
    y = data_with_age[age_col].to_numpy().astype(float)
    sample_ids = data_with_age[sample_id_col].to_numpy()
    groups = data_with_age[usubjid_col].to_numpy()

    # Store study values if present for train_cohorts filtering
    study_values = None
    if "STUDY" in data_with_age.columns:
        study_values = data_with_age["STUDY"].to_numpy()

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

    return X, y, sample_ids, groups, feature_cols, study_values, data_with_age


def train_elasticnet_age_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_mask: np.ndarray,
    alpha: float | None = None,
    l1_ratio: float = 0.5,
    auto_alpha: bool = True,
    random_seed: int = 0,
    deterministic: bool = True,
) -> tuple[ElasticNet, float, int, list]:
    """
    Train ElasticNet age prediction model.

    Args:
        X: Feature matrix (already scaled)
        y: Age labels
        groups: Group labels for cross-validation (USUBJID)
        train_mask: Boolean mask for training samples
        alpha: Regularization strength (if None and auto_alpha=True, will be optimized)
        l1_ratio: ElasticNet mixing parameter
        auto_alpha: Whether to auto-select alpha via grid search
        random_seed: Random seed
        deterministic: Whether to use deterministic parallel execution

    Returns:
        Tuple of (fitted model, optimal_alpha, n_splits, cv_splits)
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

    # Optimized grid search (only on training samples)
    if auto_alpha and alpha is None:
        logger.info(
            "automatically selecting optimal alpha using cross-validation on training samples"
        )
        alphas = np.logspace(-3, 1, 10)

        def evaluate_alpha(test_alpha, X_data, y_data, splits, l1_r, rseed):
            """Evaluate alpha with all data passed as arguments for deterministic parallel execution."""
            scores = np.zeros(len(splits))
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                reg = ElasticNet(
                    alpha=test_alpha,
                    l1_ratio=l1_r,
                    random_state=rseed,
                    max_iter=10000,
                    tol=1e-6,
                    warm_start=False,
                    selection="cyclic",
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=Warning)
                    reg.fit(X_data[train_idx], y_data[train_idx])
                scores[fold_idx] = reg.score(X_data[test_idx], y_data[test_idx])
            return test_alpha, scores.mean()

        n_jobs_search = 1 if deterministic else -1
        results = Parallel(n_jobs=n_jobs_search, verbose=0, batch_size=2)(
            delayed(evaluate_alpha)(
                test_alpha,
                X_train_sorted,
                y_train_sorted,
                cv_splits,
                l1_ratio,
                random_seed,
            )
            for test_alpha in alphas
        )

        # Use min with negative key for deterministic tie-breaking (always picks first on ties)
        best_alpha, best_score = max(
            results, key=lambda x: (x[1], -alphas.tolist().index(x[0]))
        )
        optimal_alpha = best_alpha
        logger.info(f"optimal alpha: {optimal_alpha:.4f} (CV R²={best_score:.3f})")
    else:
        optimal_alpha = alpha if alpha is not None else 0.1
        logger.info(f"using alpha={optimal_alpha}")

    # Create and fit regressor
    regressor = ElasticNet(
        alpha=optimal_alpha,
        l1_ratio=l1_ratio,
        random_state=random_seed,
        max_iter=10000,
        tol=1e-6,
        selection="cyclic",
    )

    # Fit using sorted data for determinism
    regressor.fit(X_train_sorted, y_train_sorted)

    return regressor, optimal_alpha, n_splits, cv_splits


def detect_age_outliers(
    results_df: pl.DataFrame,
    sample_id_col: str,
    age_threshold: float | None = None,
    mad_threshold: float = 3.0,
    tukey_fence_multiplier: float = 3.0,
    std_threshold: float | None = None,
) -> tuple[pl.DataFrame, float, int, set]:
    """
    Detect age outliers using MAD and Tukey fence methods.

    Args:
        results_df: DataFrame with age prediction results
        sample_id_col: Column name for sample ID
        age_threshold: Optional fixed threshold for absolute residual (overrides MAD-based threshold if provided)
        mad_threshold: MAD multiplier for outlier detection
        tukey_fence_multiplier: Tukey fence multiplier for participant consistency
        std_threshold: Optional fixed threshold for participant std (overrides Tukey fence if provided)

    Returns:
        Tuple of (updated results_df, effective_threshold, n_multi_timepoint, inconsistent_usubjids)
    """
    residuals = results_df["Residual"].to_numpy()

    # Data-driven outlier detection using MAD
    median_residual = np.median(np.abs(residuals))
    mad = np.median(np.abs(residuals - np.median(residuals)))
    robust_threshold = median_residual + mad_threshold * mad * 1.4826

    # Use user threshold if provided, otherwise use data-driven
    if age_threshold is not None:
        effective_threshold = age_threshold
        logger.info(
            f"MAD-based threshold: {robust_threshold:.1f} years | Using user age_threshold: {age_threshold} years"
        )
    else:
        effective_threshold = robust_threshold
        logger.info(
            f"Using MAD-based threshold ({mad_threshold}σ): {robust_threshold:.1f} years"
        )

    # Mark discordant samples based on threshold
    results_df = results_df.with_columns(
        [(pl.col("Abs_Residual") > effective_threshold).alias("Is_Discordant")]
    )

    # Calculate per-participant statistics for prediction consistency
    # Only calculate std for participants with >2 timepoints (need at least 3 for meaningful std)
    participant_stats = (
        results_df.group_by("USUBJID")
        .agg(
            [
                pl.col("Visit_Adj_Age").median().alias("Median_Actual_Age"),
                pl.col("Predicted_Age").median().alias("Median_Predicted_Age"),
                pl.col("Predicted_Age").std().alias("Std_Predicted_Age"),
                pl.col("Predicted_Age").count().alias("N_Timepoints"),
                pl.col(sample_id_col).first().alias("Representative_SampleId"),
            ]
        )
        .filter(pl.col("N_Timepoints") > 2)
    )

    n_multi_timepoint = len(participant_stats)
    logger.info(f"participants with multiple timepoints: {n_multi_timepoint}")

    # Pre-calculate which participants have inconsistent predictions
    inconsistent_usubjids = set()
    if len(participant_stats) > 0:
        # Calculate Tukey's fence for extreme outliers (Q3 + tukey_fence_multiplier*IQR)
        q1 = participant_stats["Std_Predicted_Age"].quantile(0.25)
        q3 = participant_stats["Std_Predicted_Age"].quantile(0.75)
        iqr = q3 - q1
        tukey_threshold = q3 + tukey_fence_multiplier * iqr

        # Use user threshold if provided, otherwise use data-driven
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

    # Mark samples from inconsistent participants
    results_df = results_df.with_columns(
        pl.col("USUBJID")
        .is_in(list(inconsistent_usubjids))
        .alias("From_Inconsistent_Participant")
    )

    # Join participant stats to add Std_Predicted_Age to results
    if len(participant_stats) > 0:
        results_df = results_df.join(
            participant_stats.select(["USUBJID", "Std_Predicted_Age"]),
            on="USUBJID",
            how="left",
        )

    return results_df, effective_threshold, n_multi_timepoint, inconsistent_usubjids


def sample_age_concordance(
    qc_instance,
    protage_table_path: str | Path = DEFAULT_PROTAGE_TABLE,
    age_col: str = "AGE",
    alpha: float = 0.1667,
    l1_ratio: float = 0.5,
    age_threshold: float | None = None,
    mad_threshold: float = 3.0,
    tukey_fence_multiplier: float = 3.0,
    std_threshold: float | None = None,
    genes_to_plot: list[str] | None = None,
    random_seed: int = 0,
    deterministic: bool = True,
    train_cohorts: list[str] | None = None,
    protage_uniprot_col: str = "UniProt ID",
) -> dict:
    """
    Age concordance using ProtAge 204 proteins with fixed alpha (published clock).

    Uses ElasticNet regression with the ProtAge protein subset and a fixed
    regularization parameter optimized on the ProtAge protein set.

    Args:
        qc_instance: SomaQC instance
        protage_table_path: Path to ProtAge supplementary table
        age_col: Column name for age
        alpha: Fixed regularization strength (from ProtAge optimization)
        l1_ratio: ElasticNet mixing (0=Ridge, 1=Lasso, 0.5=equal mix)
        age_threshold: Optional fixed threshold for absolute residual (overrides MAD-based threshold if provided)
        mad_threshold: MAD threshold multiplier for outlier detection
        tukey_fence_multiplier: Tukey fence multiplier for participant consistency
        std_threshold: Optional fixed threshold for participant std (overrides Tukey fence if provided)
        genes_to_plot: Optional list of gene names to plot
        random_seed: Random seed for reproducibility
        deterministic: If True, uses n_jobs=1 for deterministic results
        train_cohorts: Optional list of cohort names to use for training
        protage_uniprot_col: Column name for UniProt in ProtAge table (default: 'UniProt ID')

    Returns:
        Dictionary with results
    """
    qc_instance._print_section("AGE CONCORDANCE CHECK (PROTAGE CLOCK)")

    # Get column names from instance
    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    protein_name_col = qc_instance.protein_name_col
    uniprot_col = qc_instance.uniprot_col
    data_col = qc_instance.data_col

    # Infer platform from protein_id_col
    if protein_id_col == "SeqId":
        is_somascan = True
    elif protein_id_col == "OlinkID":
        is_somascan = False
    else:
        raise ValueError(
            f"Unsupported platform: protein_id_col='{protein_id_col}'. Expected 'SeqId' or 'OlinkID'."
        )

    # Get unique proteins from SomaScan data for matching
    soma_proteins = (
        qc_instance.analyte_metadata.select([protein_id_col, uniprot_col])
        .unique()
        .collect()
    )

    # Load and match ProtAge proteins
    protage_protein_ids, match_df = load_protage_proteins(
        tsv_path=protage_table_path,
        soma_proteins=soma_proteins,
        protage_uniprot_col=protage_uniprot_col,
        protein_id_col=protein_id_col,
        uniprot_col=uniprot_col,
    )

    if len(protage_protein_ids) == 0:
        logger.error("no ProtAge proteins matched to SomaScan data!")
        return {"error": "no_protage_matches"}

    # Get age map from adsl_merged
    # If user requests VISITAGE but it doesn't exist yet, compute it now
    if (
        age_col == "VISITAGE"
        and "VISITAGE" not in qc_instance.adsl_merged.collect_schema().names()
        and hasattr(qc_instance, "_compute_visitage")
    ):
        logger.info("VISITAGE requested but not yet computed - computing now")
        qc_instance.adsl_merged = qc_instance._compute_visitage()

    age_map = (
        qc_instance.adsl_merged.select(
            [sample_id_col, age_col, qc_instance.usubjid_col]
        )
        .filter(
            pl.col(age_col).is_not_null()
            & pl.col(qc_instance.usubjid_col).is_not_null()
        )
        .collect(engine="streaming")
    )

    # Join with manifest to get VISITNUM if available
    if (
        qc_instance.manifest is not None
        and "VISITNUM" in qc_instance.manifest.collect_schema().names()
    ):
        visitnum_map = qc_instance.manifest.select(
            [sample_id_col, "VISITNUM"]
        ).collect()
        age_map = age_map.join(visitnum_map, on=sample_id_col, how="left")

    # Join with ADSL to get TRT01A if available (for legend grouping in trajectory plot)
    if "TRT01A" in qc_instance.adsl_merged.collect_schema().names():
        trt_map = qc_instance.adsl_merged.select([sample_id_col, "TRT01A"]).collect()
        age_map = age_map.join(trt_map, on=sample_id_col, how="left")

    # Prepare data with ProtAge protein subset
    X, y, sample_ids, groups, feature_cols, study_values, data_with_age = (
        prepare_age_data(
            qc_instance=qc_instance,
            age_map=age_map,
            age_col=age_col,
            usubjid_col=qc_instance.usubjid_col,
            protein_subset=protage_protein_ids,
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

    # Train model with fixed alpha
    regressor, optimal_alpha, n_splits, cv_splits = train_elasticnet_age_model(
        X=X_scaled,
        y=y,
        groups=groups,
        train_mask=train_mask,
        alpha=alpha,
        l1_ratio=l1_ratio,
        auto_alpha=False,  # Use fixed alpha
        random_seed=random_seed,
        deterministic=deterministic,
    )

    # Get predictions
    X_train = X_scaled[train_mask]
    y_train = y[train_mask]
    n_jobs = 1 if deterministic else -1
    y_pred_train = cross_val_predict(
        regressor, X_train, y_train, cv=cv_splits, n_jobs=n_jobs
    )

    # Apply to all samples
    y_pred = regressor.predict(X_scaled)
    coefs = regressor.coef_

    selected_indices = np.where(coefs != 0)[0]
    n_selected = len(selected_indices)

    if n_selected == 0:
        logger.warning(
            f"No features selected with alpha={optimal_alpha}. Try decreasing alpha."
        )
        return {"error": "no_features_selected", "alpha": optimal_alpha}

    logger.info(
        f"ElasticNet selected {n_selected}/{len(feature_cols)} ProtAge proteins (alpha={optimal_alpha:.4f})"
    )

    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    residuals = y_pred - y

    logger.info(f"MAE: {mae:.2f} years | R²: {r2:.3f}")

    # Create initial results dataframe
    results_dict = {
        sample_id_col: sample_ids,
        "USUBJID": groups,
        "Visit_Adj_Age": y,
        "Predicted_Age": y_pred,
        "Residual": residuals,
        "Abs_Residual": np.abs(residuals),
    }

    # Add VISITNUM if available in data_with_age
    if "VISITNUM" in data_with_age.columns:
        visitnum_values = data_with_age["VISITNUM"].to_numpy()
        results_dict["VISITNUM"] = visitnum_values

    # Add TRT01A if available in data_with_age
    if "TRT01A" in data_with_age.columns:
        trt_values = data_with_age["TRT01A"].to_list()
        results_dict["TRT01A"] = trt_values

    results_df = pl.DataFrame(results_dict)

    # Detect outliers
    results_df, effective_threshold, n_multi_timepoint, inconsistent_usubjids = (
        detect_age_outliers(
            results_df=results_df,
            sample_id_col=sample_id_col,
            age_threshold=age_threshold,
            mad_threshold=mad_threshold,
            tukey_fence_multiplier=tukey_fence_multiplier,
            std_threshold=std_threshold,
        )
    )

    discordant = results_df.filter(pl.col("Is_Discordant"))
    n_discordant = len(discordant)
    discordant_rate = n_discordant / len(results_df) * 100

    logger.info(
        f"Discordant samples (|residual| > {effective_threshold:.1f} years): {n_discordant} ({discordant_rate:.1f}%)"
    )

    # Get top features sorted by coefficient magnitude
    # Return only protein IDs - names will be looked up in reporting for display
    all_top_features = sorted(
        [(feature_cols[i], coefs[i]) for i in selected_indices],
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    # Determine which features to plot
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
            logger.info(
                f"Using custom proteins for plotting: {[f[0] for f in top_features]}"
            )
    else:
        top_features = all_top_features[:5]

    # Mark discordant samples based on outlier detection
    n_both = 0
    n_pred_only = 0
    n_cons_only = 0

    if n_discordant > 0 or len(inconsistent_usubjids) > 0:
        # Categorize samples based on both checks
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

        # Mark as FAIL: outliers in both checks
        if n_both > 0:
            fail_ids = samples_both_outliers[sample_id_col].to_list()
            logger.warning(
                f"Marking {n_both} samples as FAIL (outliers in both prediction and consistency)"
            )
            qc_instance.failed_samples.extend(fail_ids)
            qc_instance._track_sample_qc(fail_ids, "FAIL_Age_Concordance")

        # Mark as WARN: outliers in only one check
        warn_ids = []
        if n_pred_only > 0:
            pred_warn_ids = samples_prediction_only[sample_id_col].to_list()
            warn_ids.extend(pred_warn_ids)
            logger.warning(
                f"Marking {n_pred_only} samples as WARN (prediction outliers only)"
            )
            qc_instance._track_sample_qc(pred_warn_ids, "WARN_Age_Prediction")

        if n_cons_only > 0:
            cons_warn_ids = samples_consistency_only[sample_id_col].to_list()
            warn_ids.extend(cons_warn_ids)
            logger.warning(
                f"Marking {n_cons_only} samples as WARN (from inconsistent participants only)"
            )
            qc_instance._track_sample_qc(cons_warn_ids, "WARN_Age_Consistency")

        if warn_ids:
            qc_instance.warned_samples.extend(warn_ids)

        qc_instance.qc_results["age_discordant_fail"] = n_both
        qc_instance.qc_results["age_discordant_warn"] = n_pred_only + n_cons_only
        qc_instance.qc_results["age_mae"] = float(mae)
        qc_instance.qc_results["age_r2"] = float(r2)

    # Calculate overlap with sex discordance
    sex_fail_col = "FAIL_Sex_Concordance"
    sex_discordant_ids = set()
    for qc_df in qc_instance.sample_qc_actions:
        if sex_fail_col in qc_df.columns:
            sex_discordant_ids = set(
                qc_df.filter(pl.col(sex_fail_col) == True)[sample_id_col].to_list()
            )
            break

    # Calculate overlap with age-failed samples
    age_failed_ids = set()
    if n_both > 0:
        age_failed_ids = set(samples_both_outliers[sample_id_col].to_list())
    overlap_age_sex = age_failed_ids & sex_discordant_ids
    n_overlap_age_sex = len(overlap_age_sex)

    # After age concordance completes, add sex concordance outliers to failed_samples
    # (so they are excluded from downstream analyses)
    if sex_discordant_ids:
        # Add to failed_samples if not already there
        new_failures = [
            sid for sid in sex_discordant_ids if sid not in qc_instance.failed_samples
        ]
        if new_failures:
            qc_instance.failed_samples.extend(new_failures)
            logger.info(
                f"Added {len(new_failures)} sex-discordant samples to failed_samples for exclusion from downstream analyses"
            )

    # Get unique groups for reporting
    unique_groups = np.unique(groups)
    n_unique_groups = len(unique_groups)

    return {
        "mae": mae,
        "r2": r2,
        "n_discordant": n_discordant,
        "n_selected": n_selected,
        "feature_cols": feature_cols,
        "optimal_alpha": optimal_alpha,
        "l1_ratio": l1_ratio,
        "n_splits": n_splits,
        "n_unique_groups": n_unique_groups,
        "n_both": n_both,
        "n_pred_only": n_pred_only,
        "n_cons_only": n_cons_only,
        "n_inconsistent_participants": len(inconsistent_usubjids),
        "n_overlap_age_sex": n_overlap_age_sex,
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
        "participant_stats": results_df.group_by("USUBJID")
        .agg(
            [
                pl.col("Visit_Adj_Age").median().alias("Median_Actual_Age"),
                pl.col("Predicted_Age").median().alias("Median_Predicted_Age"),
                pl.col("Predicted_Age").std().alias("Std_Predicted_Age"),
                pl.col("Predicted_Age").count().alias("N_Timepoints"),
                pl.col(sample_id_col).first().alias(f"Representative_{sample_id_col}"),
            ]
        )
        .filter(pl.col("N_Timepoints") > 1),
        "tukey_fence_multiplier": tukey_fence_multiplier,
        "protage_match_info": match_df,
        "n_protage_matched": len(protage_protein_ids),
        "sample_id_col": sample_id_col,
        "protein_id_col": protein_id_col,
    }
