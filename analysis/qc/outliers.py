"""
Outlier detection functions for proteomics QC.

Implements two independent outlier detection methods from UK Biobank:
1. PCA-based: Flags samples with extreme standardized principal components
2. Median/IQR-based: Flags samples with extreme protein expression distributions

This module is adapted from olink_qc to work with SomaScan data (SampleId, SomaId, log10_scaled_RFU).
"""

from __future__ import annotations

import gc
import logging
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .utils import impute_missing_values

logger = logging.getLogger(__name__)


def pca_outliers(
    qc_instance,
    n_components: int = 2,
    n_std: float = 5.0,
    random_seed: int = 0,
) -> tuple[pl.DataFrame, PCA, list[str]]:
    """
    PCA-based outlier detection (UK Biobank Method 1).

    Flags samples if |standardized PC1| > n_std OR |standardized PC2| > n_std.
    Tracked as FAIL_PCA_Outlier in QC summaries.

    Args:
        qc_instance: QC instance (SomaQC or OlinkQC)
        n_components: Number of principal components to compute (default 2)
        n_std: Number of standard deviations for outlier threshold (default 5.0)
        random_seed: Random seed for PCA reproducibility

    Returns:
        Tuple of (PCA DataFrame with columns: sample_id_col, PC1, PC2, PC1_std, PC2_std, Is_PCA_Outlier,
                  PCA object, list of PCA outlier sample IDs)
    """
    qc_instance._print_section("PCA-BASED OUTLIER DETECTION")

    # Get column names from instance
    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    data_col = qc_instance.data_col

    # Use qc_instance.samples which is already filtered to biological samples only
    sample_data = qc_instance.samples

    n_biological_samples = (
        sample_data.select(pl.col(sample_id_col).n_unique()).collect().item()
    )
    logger.info(
        f"Using {n_biological_samples} biological samples (SampleType='{qc_instance.sample_type}')"
    )

    if qc_instance.failed_samples:
        sample_data = sample_data.filter(
            ~pl.col(sample_id_col).is_in(qc_instance.failed_samples)
        )
        logger.info(f"excluding {len(qc_instance.failed_samples)} failed samples")

    total_proteins_before = sample_data.select(protein_id_col).unique().collect().height
    logger.info(f"total {protein_id_col}s before exclusion: {total_proteins_before}")

    # Exclude both FAIL and WARN proteins (only use PASS assays).
    failed_set = set(qc_instance.failed_proteins)
    warned_set = set(qc_instance.warned_proteins)
    excluded_proteins = list(failed_set | warned_set)

    if excluded_proteins:
        # WARN count should exclude proteins that are already FAIL
        warn_only_count = len(warned_set - failed_set)
        logger.info(
            f"Excluding {len(failed_set)} FAIL and {warn_only_count} WARN {protein_id_col}s "
            f"(total unique: {len(excluded_proteins)})"
        )
        sample_data = sample_data.filter(
            ~pl.col(protein_id_col).is_in(excluded_proteins)
        )
        proteins_after = sample_data.select(protein_id_col).unique().collect().height
        logger.info(
            f"{protein_id_col}s after exclusion: {proteins_after} (removed {total_proteins_before - proteins_after})"
        )

    # Aggregate by sample and protein ID to handle duplicates.
    sample_df = (
        sample_data.filter(pl.col(data_col).is_not_null() & ~pl.col(data_col).is_nan())
        .group_by([sample_id_col, protein_id_col])
        .agg(pl.col(data_col).median().alias(data_col))
        .collect()
    )

    n_samples, n_proteins = (
        sample_df[sample_id_col].n_unique(),
        sample_df[protein_id_col].n_unique(),
    )

    if n_samples < 3 or n_proteins < 2:
        logger.warning(
            f"Insufficient data: {n_samples} samples, {n_proteins} unique proteins"
        )
        empty_pca = PCA(n_components=2)
        return pl.DataFrame({sample_id_col: []}), empty_pca, []

    logger.info(f"retained {n_samples} samples × {n_proteins} unique proteins for PCA")

    # Pivot to wide format
    pivot = sample_df.pivot(on=protein_id_col, index=sample_id_col, values=data_col)

    if pivot.width <= 1:
        logger.warning("pivot failed")
        empty_pca = PCA(n_components=2)
        return pl.DataFrame({sample_id_col: []}), empty_pca, []

    # Prepare data
    all_ids = pivot[sample_id_col].to_list()
    X = pivot.drop(sample_id_col).to_numpy()

    # Impute missing values
    n_missing = np.sum(~np.isfinite(X))
    if n_missing > 0:
        n_samples_with_missing = np.sum(~np.all(np.isfinite(X), axis=1))
        logger.info(
            f"Imputing {n_missing:,} missing values across {n_samples_with_missing} samples"
        )
        X = impute_missing_values(X, random_seed=random_seed)

    # Store imputed data for reuse in median/IQR detection
    qc_instance._imputed_data_cache = X.copy()
    qc_instance._imputed_sample_ids = all_ids.copy()

    valid_ids = all_ids
    X_valid = X

    if len(valid_ids) < 3:
        logger.warning(f"Only {len(valid_ids)} valid samples")
        empty_pca = PCA(n_components=2)
        return pl.DataFrame({sample_id_col: []}), empty_pca, []

    # Run PCA
    X_scaled = StandardScaler().fit_transform(X_valid)
    pca = PCA(
        n_components=min(n_components, X_scaled.shape[1]), random_state=random_seed
    )
    coords = pca.fit_transform(X_scaled)
    del X_scaled
    n_pcs = coords.shape[1]

    pca_df = pl.DataFrame(
        {sample_id_col: valid_ids, **{f"PC{i+1}": coords[:, i] for i in range(n_pcs)}}
    )

    # Standardize PC1 and PC2 (PCA already centers at 0)
    pc1_std = coords[:, 0] / coords[:, 0].std()
    pc2_std = coords[:, 1] / coords[:, 1].std() if n_pcs > 1 else np.zeros_like(pc1_std)

    # Flag if |PC1| > n_std OR |PC2| > n_std
    pca_outlier_mask = (np.abs(pc1_std) > n_std) | (np.abs(pc2_std) > n_std)
    pca_outliers = [sid for sid, is_out in zip(valid_ids, pca_outlier_mask) if is_out]

    pca_df = pca_df.with_columns(
        [
            pl.Series("PC1_std", pc1_std),
            pl.Series("PC2_std", pc2_std),
            pl.Series("Is_PCA_Outlier", pca_outlier_mask),
        ]
    )

    # Track outliers
    if pca_outliers:
        qc_instance.failed_samples.extend(pca_outliers)
        qc_instance._track_sample_qc(pca_outliers, "FAIL_PCA_Outlier")

    var1 = pca.explained_variance_ratio_[0]
    var2 = (
        pca.explained_variance_ratio_[1]
        if len(pca.explained_variance_ratio_) > 1
        else 0
    )
    logger.info(f"PC1: {var1*100:.1f}% | PC2: {var2*100:.1f}%")
    logger.info(f"PCA outliers: {len(pca_outliers)}")

    qc_instance.qc_results.update(
        {
            "pca_outliers": len(pca_outliers),
            "pca_var_pc1": float(var1),
            "pca_var_pc2": float(var2),
        }
    )

    # Clean up
    try:
        del sample_df, X, pivot, coords
    except:
        pass
    gc.collect()

    return pca_df, pca, pca_outliers


def median_iqr_outliers(
    qc_instance,
    n_std: float = 5.0,
    random_seed: int = 0,
) -> tuple[pl.DataFrame, list[str]]:
    """
    Median/IQR-based outlier detection (UK Biobank Method 2).

    For each sample, computes median and IQR across all proteins.
    Flags samples if |z_median| > n_std OR |z_IQR| > n_std.
    Tracked as FAIL_Median_IQR_Outlier in QC summaries.

    Args:
        qc_instance: QC instance (SomaQC or OlinkQC)
        n_std: Number of standard deviations for outlier threshold (default 5.0)
        random_seed: Random seed for imputation reproducibility

    Returns:
        Tuple of (DataFrame with columns: sample_id_col, Z_Median, Z_IQR, Is_Median_IQR_Outlier,
                  list of Median/IQR outlier sample IDs)
    """
    qc_instance._print_section("MEDIAN/IQR-BASED OUTLIER DETECTION")

    # Get column names from instance
    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    data_col = qc_instance.data_col

    # Reuse imputed data if available from PCA detection
    if (
        hasattr(qc_instance, "_imputed_data_cache")
        and qc_instance._imputed_data_cache is not None
    ):
        logger.info("reusing imputed data from PCA detection")
        X_valid = qc_instance._imputed_data_cache
        valid_ids = qc_instance._imputed_sample_ids
    else:
        # Compute imputed data independently
        # Use qc_instance.samples which is already filtered to biological samples only
        sample_data = qc_instance.samples

        logger.info(
            f"Using biological samples only (SampleType='{qc_instance.sample_type}')"
        )

        if qc_instance.failed_samples:
            sample_data = sample_data.filter(
                ~pl.col(sample_id_col).is_in(qc_instance.failed_samples)
            )

        # Exclude both FAIL and WARN proteins (only use PASS assays)
        excluded_proteins = list(
            set(qc_instance.failed_proteins + qc_instance.warned_proteins)
        )
        if excluded_proteins:
            sample_data = sample_data.filter(
                ~pl.col(protein_id_col).is_in(excluded_proteins)
            )

        # Aggregate by sample and protein ID
        sample_df = (
            sample_data.filter(
                pl.col(data_col).is_not_null() & ~pl.col(data_col).is_nan()
            )
            .group_by([sample_id_col, protein_id_col])
            .agg(pl.col(data_col).median().alias(data_col))
            .collect()
        )

        if sample_df.height == 0:
            logger.warning("no valid data for median/IQR detection")
            return pl.DataFrame({sample_id_col: []}), []

        pivot = sample_df.pivot(on=protein_id_col, index=sample_id_col, values=data_col)
        valid_ids = pivot[sample_id_col].to_list()
        X = pivot.drop(sample_id_col).to_numpy()

        # Impute missing values
        n_missing = np.sum(~np.isfinite(X))
        if n_missing > 0:
            logger.info(f"imputing {n_missing:,} missing values")
            X = impute_missing_values(X, random_seed=random_seed)

        X_valid = X

    if len(valid_ids) < 3:
        logger.warning(f"Only {len(valid_ids)} valid samples")
        return pl.DataFrame({sample_id_col: []}), []

    # Create initial DataFrame
    outlier_df = pl.DataFrame({sample_id_col: valid_ids})

    # Compute sample statistics (median and IQR)
    sample_median = np.median(X_valid, axis=1)  # Median across proteins for each sample
    sample_q75 = np.percentile(X_valid, 75, axis=1)
    sample_q25 = np.percentile(X_valid, 25, axis=1)
    sample_iqr = sample_q75 - sample_q25

    # Z-score across samples
    median_mean = np.mean(sample_median)
    median_std = np.std(sample_median)
    z_median = (
        (sample_median - median_mean) / median_std
        if median_std > 0
        else np.zeros_like(sample_median)
    )

    iqr_mean = np.mean(sample_iqr)
    iqr_std = np.std(sample_iqr)
    z_iqr = (
        (sample_iqr - iqr_mean) / iqr_std if iqr_std > 0 else np.zeros_like(sample_iqr)
    )

    # Flag if |z_median| > n_std OR |z_IQR| > n_std
    median_iqr_outlier_mask = (np.abs(z_median) > n_std) | (np.abs(z_iqr) > n_std)
    median_iqr_outliers = [
        sid for sid, is_out in zip(valid_ids, median_iqr_outlier_mask) if is_out
    ]

    # Add to DataFrame
    outlier_df = outlier_df.with_columns(
        [
            pl.Series("Z_Median", z_median),
            pl.Series("Z_IQR", z_iqr),
            pl.Series("Is_Median_IQR_Outlier", median_iqr_outlier_mask),
        ]
    )

    # Track outliers (excluding those already in failed_samples)
    if median_iqr_outliers:
        # Only add samples not already in failed_samples
        unique_median_iqr = [
            s for s in median_iqr_outliers if s not in qc_instance.failed_samples
        ]
        if unique_median_iqr:
            qc_instance.failed_samples.extend(unique_median_iqr)

        # Always track median/IQR status
        qc_instance._track_sample_qc(median_iqr_outliers, "FAIL_Median_IQR_Outlier")

    logger.info(f"median/IQR outliers: {len(median_iqr_outliers)}")

    qc_instance.qc_results.update(
        {
            "median_iqr_outliers": len(median_iqr_outliers),
        }
    )

    # Clean up
    gc.collect()

    return outlier_df, median_iqr_outliers
