"""PCA and median/IQR sample outlier detection (the two UK Biobank methods)."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analysis.qc import detection
from analysis.qc.flags import FAIL_MEDIAN_IQR_OUTLIER, FAIL_PCA_OUTLIER
from analysis.qc.modeling import KNN_K, duplicate_assay_ids
from analysis.qc.utils import impute_missing_values, wide_matrix

logger = logging.getLogger(__name__)

# Defaults of TrialConfig ``qc_outlier_n_pcs`` and ``qc_outlier_n_std``.
OUTLIER_N_PCS: int = 2
OUTLIER_N_STD: float = 5.0


def _detection_weights(qc_instance, fit_ids: list[str], proteins: list[str]) -> np.ndarray:
    """Computes each assay's detection rate over the fit samples.

    The rate is the share of fit-sample rows with a value at or above the sample's LOD,
    among rows that have a LOD, as in the detection step. An assay with no LOD row on
    a fit sample gets weight 0 (the number is logged). The LOD comes from the
    instance's data, else its LOD results, else it is computed.

    Args:
        qc_instance: A SomaQC or OlinkQC instance.
        fit_ids: Sample ids of the fit samples.
        proteins: Assay ids; the weights follow this order.

    Returns:
        One weight per assay.

    Raises:
        ValueError: If every weight is 0.
    """
    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    data = qc_instance.get_data()
    if "LOD" not in data.collect_schema().names():
        data = getattr(qc_instance, "lod_results", None)
        if data is None:
            lod_fn = (
                detection.olink_lod if qc_instance.platform.name == "olink" else detection.soma_lod
            )
            data = lod_fn(qc_instance)

    # Study samples only: SomaScan buffers and calibrators share SampleIds.
    key = "_sample_idx" if "_sample_idx" in data.collect_schema().names() else sample_id_col
    data = data.join(qc_instance.samples.select(key).unique(), on=key, how="semi").filter(
        pl.col(sample_id_col).is_in(fit_ids) & pl.col(protein_id_col).is_in(proteins)
    )

    value = pl.col(qc_instance.detection_col)
    has_lod = pl.col("LOD").is_not_null() & pl.col("LOD").is_not_nan()
    detected = value.is_not_null() & value.is_not_nan() & (value >= pl.col("LOD"))
    rates = (
        data.group_by(protein_id_col)
        .agg(
            (detected & has_lod).sum().alias("_det"),
            has_lod.sum().alias("_tot"),
        )
        .collect()
    )
    rate = {p: d / t for p, d, t in rates.select(protein_id_col, "_det", "_tot").iter_rows() if t}
    n_no_lod = sum(p not in rate for p in proteins)
    if n_no_lod:
        logger.info(
            f"{n_no_lod} {protein_id_col}s have no LOD row on a fit sample: detection weight 0"
        )
    weights = np.array([rate.get(p, 0.0) for p in proteins])
    if not weights.any():
        raise ValueError(
            f"Every {protein_id_col} has detection weight 0 over the {len(fit_ids)} fit samples: "
            "no assay is detected above its LOD, so the outlier scores are undefined"
        )
    return weights


def _outlier_matrix(
    qc_instance,
    knn_k: int = KNN_K,
    detection_weighting: bool = True,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Builds the imputed sample x assay matrix shared by both outlier methods.

    Rows are the study samples without a FAIL flag, sorted by id; columns are every
    assay except ``FAIL_Duplicate_Assay`` copies, sorted. An assay with no value on
    any fit sample is dropped, since it can be neither imputed nor standardized.
    ``fit`` is False for samples that sample QC keeps out of the outlier fits (nulled
    blocks or high missing rate). Missing values are KNN-imputed from the ``knn_k``
    nearest fit samples only, so excluded samples cannot shape the reference; they are
    still scored from their imputed values.

    Args:
        qc_instance: A SomaQC or OlinkQC instance.
        knn_k: Neighbors of the KNN imputation.
        detection_weighting: Weights each assay by its detection rate over the fit
            samples (``_detection_weights``); False gives every assay weight 1.0.

    Returns:
        ``(ids, X, fit, weights)``.

    Raises:
        ValueError: If fewer than 3 samples or 2 assays remain.
    """
    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    data_col = qc_instance.data_col

    sample_data = qc_instance.samples

    n_biological_samples = sample_data.select(pl.col(sample_id_col).n_unique()).collect().item()
    logger.info(
        f"Using {n_biological_samples} biological samples (SampleType='{qc_instance.sample_type}')"
    )

    if qc_instance.failed_samples:
        sample_data = sample_data.filter(~pl.col(sample_id_col).is_in(qc_instance.failed_samples))
        logger.info(f"excluding {len(qc_instance.failed_samples)} failed samples")

    total_proteins_before = sample_data.select(protein_id_col).unique().collect().height
    logger.info(f"total {protein_id_col}s before exclusion: {total_proteins_before}")

    duplicates = duplicate_assay_ids(qc_instance)
    if duplicates:
        logger.info(f"Excluding {len(duplicates)} duplicate-assay {protein_id_col}s")
        sample_data = sample_data.filter(~pl.col(protein_id_col).is_in(duplicates))

    # Median over repeated sample and assay rows.
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
        raise ValueError(
            f"outlier detection needs at least 3 samples and 2 proteins; found {n_samples} "
            f"samples and {n_proteins} proteins"
        )

    logger.info(
        f"retained {n_samples} samples × {n_proteins} unique proteins for outlier detection"
    )

    # Sorted rows and columns for reproducibility.
    id_series, protein_cols, X = wide_matrix(
        sample_df, index=sample_id_col, on=protein_id_col, values=data_col, sort=True
    )
    ids = id_series.to_list()

    excluded = set(qc_instance._sample_qc_data["outlier_fit_exclude"])
    fit = np.array([s not in excluded for s in ids])
    logger.info(
        f"outlier fits use {int(fit.sum())} samples; {int((~fit).sum())} samples with "
        "nulled blocks or high missing rate are scored but not fitted"
    )

    observed = np.isfinite(X[fit]).any(axis=0)
    if not observed.all():
        logger.info(
            f"dropping {int((~observed).sum())} {protein_id_col}s with no value on any fit sample"
        )
        X = X[:, observed]
        protein_cols = [c for c, keep in zip(protein_cols, observed, strict=True) if keep]
    if X.shape[1] < 2:
        raise ValueError(
            f"outlier detection needs at least 2 assays with a value on a fit sample; "
            f"found {X.shape[1]}"
        )

    n_missing = np.sum(~np.isfinite(X))
    if n_missing > 0:
        n_samples_with_missing = np.sum(~np.all(np.isfinite(X), axis=1))
        logger.info(
            f"Imputing {n_missing:,} missing values across {n_samples_with_missing} samples "
            "(KNN, fit samples as donors)"
        )
        X = impute_missing_values(X, donors=fit, n_neighbors=knn_k)

    if detection_weighting:
        weights = _detection_weights(
            qc_instance,
            [s for s, in_fit in zip(ids, fit, strict=True) if in_fit],
            protein_cols,
        )
    else:
        weights = np.ones(len(protein_cols))
    return ids, X, fit, weights


def _weighted_quantile(X: np.ndarray, w: np.ndarray, q: float) -> np.ndarray:
    """Computes a row-wise weighted quantile with Hazen interpolation.

    Each value sits at the center of its weight mass, ``(cum_w - w / 2) / W``, and the
    quantile is read off by linear interpolation. With unit weights this equals
    ``np.percentile(..., method="hazen")``. Zero-weight columns are left out.

    Args:
        X: Samples x assays.
        w: One weight per column of ``X``.
        q: Quantile in [0, 1].

    Returns:
        One quantile per row of ``X``.
    """
    keep = w > 0
    X, w = X[:, keep], w[keep]
    order = np.argsort(X, axis=1, kind="stable")
    values = np.take_along_axis(X, order, axis=1)
    w_sorted = w[order]
    pos = (np.cumsum(w_sorted, axis=1) - w_sorted / 2) / w.sum()
    return np.array([np.interp(q, p, v) for p, v in zip(pos, values, strict=True)])


def pca_outliers(
    qc_instance,
    n_components: int = OUTLIER_N_PCS,
    n_std: float = OUTLIER_N_STD,
    random_seed: int = 0,
    knn_k: int = KNN_K,
    detection_weighting: bool = True,
) -> tuple[pl.DataFrame, PCA, list[str]]:
    """Flags samples with an extreme standardized principal component.

    A sample is an outlier if |standardized PCk| > ``n_std`` for any of the first
    ``n_components`` PCs. Outliers get ``FAIL_PCA_Outlier``. Each standardized assay
    column is scaled by the root of its detection rate. The scaler and PCA are fitted
    on the fit samples only (see ``_outlier_matrix``) and PCs are standardized by the
    fit samples' SD; every sample is then projected.

    Args:
        qc_instance: A SomaQC or OlinkQC instance.
        n_components: Number of principal components to compute and test.
        n_std: Outlier threshold, in standard deviations.
        random_seed: Seed of the PCA.
        knn_k: Neighbors of the KNN imputation of the sample x assay matrix.
        detection_weighting: Weights each assay by its detection rate; False gives
            unit weights.

    Returns:
        The PC table (sample id, ``PC1``..., ``PC1_std``..., ``Is_PCA_Outlier``,
        ``In_Outlier_Fit``), the PCA object and the outlier sample ids.

    Raises:
        ValueError: If the outlier fit holds fewer than 3 samples or 2 assays.
    """
    sample_id_col = qc_instance.sample_id_col
    matrix = _outlier_matrix(qc_instance, knn_k, detection_weighting)
    qc_instance._pca_matrix = matrix
    if matrix[2].sum() < 3:
        raise ValueError(f"the outlier fit needs at least 3 samples; found {int(matrix[2].sum())}")
    ids, X, fit, weights = matrix

    scaler = StandardScaler().fit(X[fit])
    Z = scaler.transform(X) * np.sqrt(weights)
    pca = PCA(n_components=min(n_components, X.shape[1]), random_state=random_seed)
    pca.fit(Z[fit])
    coords = pca.transform(Z)
    n_pcs = coords.shape[1]

    pca_df = pl.DataFrame(
        {sample_id_col: ids, **{f"PC{i + 1}": coords[:, i] for i in range(n_pcs)}}
    )

    # The fit is centered at 0, so the SD alone standardizes.
    pc_std = [coords[:, k] / coords[fit, k].std() for k in range(n_pcs)]
    pc1_std = pc_std[0]
    pc2_std = pc_std[1] if n_pcs > 1 else np.zeros_like(pc1_std)

    pca_outlier_mask = np.zeros(len(ids), dtype=bool)
    for std in pc_std:
        pca_outlier_mask |= np.abs(std) > n_std
    pca_outliers = [sid for sid, is_out in zip(ids, pca_outlier_mask, strict=True) if is_out]

    pca_df = pca_df.with_columns(
        [
            pl.Series("PC1_std", pc1_std),
            pl.Series("PC2_std", pc2_std),
            *(pl.Series(f"PC{k + 1}_std", pc_std[k]) for k in range(2, n_pcs)),
            pl.Series("Is_PCA_Outlier", pca_outlier_mask),
            pl.Series("In_Outlier_Fit", fit),
        ]
    )

    if pca_outliers:
        qc_instance._flag_samples(pca_outliers, FAIL_PCA_OUTLIER)

    var1 = pca.explained_variance_ratio_[0]
    var2 = pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0
    logger.info(f"PC1: {var1 * 100:.1f}% | PC2: {var2 * 100:.1f}%")
    logger.info(
        f"PCA outliers: {len(pca_outliers)} "
        f"({int(pca_outlier_mask[~fit].sum())} among samples outside the fit)"
    )

    qc_instance.qc_results.update(
        {
            "pca_outliers": len(pca_outliers),
            "pca_var_pc1": float(var1),
            "pca_var_pc2": float(var2),
        }
    )

    return pca_df, pca, pca_outliers


def median_iqr_outliers(
    qc_instance,
    n_std: float = OUTLIER_N_STD,
    knn_k: int = KNN_K,
    detection_weighting: bool = True,
) -> tuple[pl.DataFrame, list[str]]:
    """Flags samples with an extreme median or IQR across assays.

    Computes each sample's detection-weighted median and IQR across assays (Hazen
    interpolation, see ``_weighted_quantile``). A sample is an outlier if |z_median|
    or |z_IQR| exceeds ``n_std``, with the z-scores' mean and SD taken over the fit
    samples only. Outliers get ``FAIL_Median_IQR_Outlier``. The matrix built by
    ``pca_outliers`` is reused when present.

    Args:
        qc_instance: A SomaQC or OlinkQC instance.
        n_std: Outlier threshold, in standard deviations.
        knn_k: Neighbors of the KNN imputation, when the matrix is not already built.
        detection_weighting: As in ``pca_outliers``, when the matrix is not already
            built.

    Returns:
        The z-score table (sample id, ``Z_Median``, ``Z_IQR``,
        ``Is_Median_IQR_Outlier``) and the outlier sample ids. With too few data the
        table is empty.
    """
    sample_id_col = qc_instance.sample_id_col

    matrix = qc_instance._pca_matrix
    if matrix is not None:
        logger.info("reusing imputed data from PCA detection")
    else:
        matrix = _outlier_matrix(qc_instance, knn_k, detection_weighting)
    if matrix[2].sum() < 3:
        raise ValueError(f"the outlier fit needs at least 3 samples; found {int(matrix[2].sum())}")
    ids, X, fit, weights = matrix

    sample_median = _weighted_quantile(X, weights, 0.5)
    sample_iqr = _weighted_quantile(X, weights, 0.75) - _weighted_quantile(X, weights, 0.25)

    def z_score(values: np.ndarray) -> np.ndarray:
        """Z against the fit samples' mean and SD."""
        sd = values[fit].std()
        return (values - values[fit].mean()) / sd if sd > 0 else np.zeros_like(values)

    z_median = z_score(sample_median)
    z_iqr = z_score(sample_iqr)

    median_iqr_outlier_mask = (np.abs(z_median) > n_std) | (np.abs(z_iqr) > n_std)
    median_iqr_outliers = [
        sid for sid, is_out in zip(ids, median_iqr_outlier_mask, strict=True) if is_out
    ]

    outlier_df = pl.DataFrame(
        {
            sample_id_col: ids,
            "Z_Median": z_median,
            "Z_IQR": z_iqr,
            "Is_Median_IQR_Outlier": median_iqr_outlier_mask,
        }
    )

    if median_iqr_outliers:
        unique_median_iqr = [s for s in median_iqr_outliers if s not in qc_instance.failed_samples]
        if unique_median_iqr:
            qc_instance.failed_samples.extend(unique_median_iqr)

        # The status is tracked even for samples already failed.
        qc_instance._track_sample_qc(median_iqr_outliers, FAIL_MEDIAN_IQR_OUTLIER)

    logger.info(
        f"median/IQR outliers: {len(median_iqr_outliers)} "
        f"({int(median_iqr_outlier_mask[~fit].sum())} among samples outside the fit)"
    )

    qc_instance.qc_results.update(
        {
            "median_iqr_outliers": len(median_iqr_outliers),
        }
    )

    return outlier_df, median_iqr_outliers
