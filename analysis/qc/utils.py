"""Utility functions for proteomics QC."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

logger = logging.getLogger(__name__)


def ensure_lazy(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """
    Convert DataFrame to LazyFrame if needed.

    Args:
        df: Polars DataFrame or LazyFrame

    Returns:
        LazyFrame
    """
    if isinstance(df, pl.DataFrame):
        return df.lazy()
    return df


def ensure_date_column(df: pl.DataFrame | pl.LazyFrame, col_name: str) -> pl.Expr:
    """Convert column to date type if it's a string, or keep as-is if already date.

    Args:
        df: DataFrame or LazyFrame
        col_name: Name of the date column

    Returns:
        Polars expression for the date column
    """
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    if col_name not in schema.names():
        return pl.col(col_name)

    col_type = schema[col_name]
    if col_type in [pl.String, pl.Utf8]:
        return pl.col(col_name).str.to_date()
    return pl.col(col_name)


def save_qc_results(
    qc_instance,
    save_prefix: str | None = None,
    manifest: pl.LazyFrame | None = None,
    adsl: pl.DataFrame | pl.LazyFrame | None = None,
    usubjid_col: str = "USUBJID",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Generate and optionally save comprehensive QC results including summaries and data.

    Optionally saves:
    1. Sample summary TSV with all IDs (sample ID, AccessionNumber, USUBJID, STUDY, VISITNUM)
    2. Protein/assay summary TSV
    3. Long data parquet with QC information

    Args:
        qc_instance: SomaQC or OlinkQC instance
        save_prefix: Optional prefix for saving files
        manifest: Optional LazyFrame mapping sample ID to SUBJID and AccessionNumber
        adsl: Optional ADSL DataFrame with USUBJID information
        usubjid_col: Column name for unique participant ID

    Returns:
        Tuple of (sample_summary, protein_summary) DataFrames
    """

    def _add_qc_status(df, id_col, failed_list, warned_list):
        """Add QC status column to dataframe."""
        return df.with_columns(
            pl.when(pl.col(id_col).is_in(failed_list))
            .then(pl.lit("FAIL"))
            .when(pl.col(id_col).is_in(warned_list))
            .then(pl.lit("WARN"))
            .otherwise(pl.lit("PASS"))
            .alias("Status")
        )

    # Get column names from instance
    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    protein_name_col = qc_instance.protein_name_col
    data_col = qc_instance.data_col
    uniprot_col = qc_instance.uniprot_col

    # Get base sample data from biological samples only (exclude QC controls)
    sample_summary = qc_instance.samples.select(sample_id_col).unique().collect()
    logger.debug(f"sample_summary initial: {sample_summary.height} rows")

    # Get protein summary from analyte_metadata (where protein-level columns exist)
    # Exclude QC-specific columns like ColCheck that will be tracked via QC actions
    analyte_schema = qc_instance.analyte_metadata.collect_schema().names()
    protein_cols = [protein_id_col]
    if protein_name_col and protein_name_col in analyte_schema:
        protein_cols.append(protein_name_col)
    if uniprot_col and uniprot_col in analyte_schema:
        protein_cols.append(uniprot_col)
    # Add other useful metadata columns but exclude ColCheck (redundant with WARN_ColCheck)
    for col in analyte_schema:
        if col not in protein_cols and col not in [
            protein_id_col,
            "ColCheck",
            "RowCheck",
        ]:
            protein_cols.append(col)

    protein_summary = (
        qc_instance.analyte_metadata.select(protein_cols).unique().collect()
    )

    # Filter to AssayType == "assay" for platforms that have this column (Olink)
    if "AssayType" in protein_summary.columns:
        n_before = protein_summary.height
        protein_summary = protein_summary.filter(pl.col("AssayType") == "assay")
        n_after = protein_summary.height
        logger.info(
            f"Filtered protein summary to AssayType == 'assay': {n_before} -> {n_after} proteins"
        )

    # Verify uniqueness by protein_id_col
    n_proteins = protein_summary.height
    n_unique_proteins = protein_summary[protein_id_col].n_unique()
    if n_proteins != n_unique_proteins:
        logger.warning(
            f"Protein summary has duplicates: {n_proteins} rows but {n_unique_proteins} unique {protein_id_col}s"
        )
        # Force unique on protein_id_col
        protein_summary = protein_summary.unique(subset=[protein_id_col], keep="first")
        logger.info(f"Deduplicated protein summary to {protein_summary.height} rows")

    # Add USUBJID, STUDY, VISITNUM, and AccessionNumber from adsl_merged and manifest
    if qc_instance.adsl_merged is not None:
        # Get columns from adsl_merged (already has sample_id → USUBJID mapping)
        id_cols = [sample_id_col, usubjid_col]
        if "STUDY" in qc_instance.adsl_merged.columns:
            id_cols.append("STUDY")
        if "VISITNUM" in qc_instance.adsl_merged.columns:
            id_cols.append("VISITNUM")

        adsl_id_map = qc_instance.adsl_merged.select(id_cols).collect()

        conflict_cols = [col for col in id_cols if col != sample_id_col]
        conflicting_mappings = (
            adsl_id_map.group_by(sample_id_col)
            .agg(pl.struct(conflict_cols).n_unique().alias("_mapping_count"))
            .filter(pl.col("_mapping_count") > 1)
        )
        if conflicting_mappings.height > 0:
            conflicting_samples = conflicting_mappings[sample_id_col].to_list()
            sample_conflicts = adsl_id_map.filter(
                pl.col(sample_id_col).is_in(conflicting_samples)
            ).sort(
                [sample_id_col]
                + [col for col in conflict_cols if col in adsl_id_map.columns]
            )
            raise ValueError(
                "Conflicting sample-to-subject mappings found in adsl_merged for "
                f"{conflicting_mappings.height} {sample_id_col} value(s); examples: "
                f"{sample_conflicts.head(10).to_dicts()}"
            )

        sort_by: list[str] = ["_has_usubjid"]
        sort_descending = [True]
        sort_exprs: list[pl.Expr] = [
            pl.col(usubjid_col).is_not_null().alias("_has_usubjid")
        ]
        if "VISITNUM" in adsl_id_map.columns:
            sort_by.append("VISITNUM")
            sort_descending.append(False)
        if "ADT" in adsl_id_map.columns:
            sort_exprs.append(ensure_date_column(adsl_id_map, "ADT").alias("_adt_sort"))
            sort_by.append("_adt_sort")
            sort_descending.append(False)

        adsl_id_map = adsl_id_map.with_columns(sort_exprs)
        temp_cols = [c for c in sort_by if c.startswith("_")]
        id_map = (
            adsl_id_map.sort(
                by=sort_by + [sample_id_col],
                descending=sort_descending + [False],
                nulls_last=True,
            )
            .unique(subset=[sample_id_col], keep="first")
            .drop(temp_cols)
        )
        logger.debug(
            f"id_map: {id_map.height} rows, {id_map[sample_id_col].n_unique()} unique {sample_id_col}"
        )

        # Add AccessionNumber from manifest if available
        if manifest is not None:
            manifest_schema = ensure_lazy(manifest).collect_schema().names()
            if "AccessionNumber" in manifest_schema:
                accession_map = (
                    ensure_lazy(manifest)
                    .select([sample_id_col, "AccessionNumber"])
                    .collect()
                )
                id_map = id_map.join(accession_map, on=sample_id_col, how="left")

        # Cast VISITNUM to Int64 if present
        if "VISITNUM" in id_map.columns:
            id_map = id_map.with_columns(pl.col("VISITNUM").cast(pl.Int64))

        sample_summary = sample_summary.join(id_map, on=sample_id_col, how="left")
        logger.debug(f"sample_summary after id_map join: {sample_summary.height} rows")

        # Remove bridge samples (null USUBJID)
        n_bridge = sample_summary.filter(pl.col(usubjid_col).is_null()).height
        if n_bridge > 0:
            logger.info(
                f"removing {n_bridge} bridge samples (null {usubjid_col}) from QC summary"
            )
            sample_summary = sample_summary.filter(pl.col(usubjid_col).is_not_null())

    # Join QC actions and add status
    for qc_df in qc_instance.sample_qc_actions:
        sample_summary = sample_summary.join(
            qc_df, on=sample_id_col, how="left", coalesce=True
        )
        reason_col = [c for c in qc_df.columns if c != sample_id_col][0]
        logger.debug(
            f"sample_summary after '{reason_col}' join: {sample_summary.height} rows "
            f"(qc_df has {qc_df.height} rows, {qc_df[sample_id_col].n_unique()} unique)"
        )

    sample_summary = _add_qc_status(
        sample_summary,
        sample_id_col,
        qc_instance.failed_samples,
        qc_instance.warned_samples,
    )

    for qc_df in qc_instance.protein_qc_actions:
        protein_summary = protein_summary.join(
            qc_df, on=protein_id_col, how="left", coalesce=True
        )
    protein_summary = _add_qc_status(
        protein_summary,
        protein_id_col,
        qc_instance.failed_proteins,
        qc_instance.warned_proteins,
    )

    # Add sample QC metrics if available (RowCheck status, outlier fractions, etc.)
    if (
        hasattr(qc_instance, "_sample_qc_status")
        and qc_instance._sample_qc_status is not None
    ):
        sample_summary = sample_summary.join(
            qc_instance._sample_qc_status, on=sample_id_col, how="left", coalesce=True
        )
        logger.debug(
            f"sample_summary after _sample_qc_status join: {sample_summary.height} rows"
        )

    # Add PCA results if available (PC1_std, PC2_std, Z_Median, Z_IQR)
    if hasattr(qc_instance, "_pca_results") and qc_instance._pca_results is not None:
        n_pca_dups = (
            qc_instance._pca_results.height
            - qc_instance._pca_results.unique(subset=[sample_id_col]).height
        )
        if n_pca_dups > 0:
            logger.warning(
                f"_pca_results contains {n_pca_dups} duplicate rows per sample; "
                "keeping first occurrence for join"
            )
        pca_deduped = qc_instance._pca_results.unique(
            subset=[sample_id_col], keep="first"
        )
        sample_summary = sample_summary.join(pca_deduped, on=sample_id_col, how="left")

    # Add sex concordance results if available (Reported_Sex, predicted probabilities)
    if (
        hasattr(qc_instance, "_sex_concordance_results")
        and qc_instance._sex_concordance_results is not None
    ):
        n_sex_dups = (
            qc_instance._sex_concordance_results.height
            - qc_instance._sex_concordance_results.unique(subset=[sample_id_col]).height
        )
        if n_sex_dups > 0:
            logger.warning(
                f"_sex_concordance_results contains {n_sex_dups} duplicate rows per sample; "
                "keeping first occurrence for join"
            )
        sex_deduped = qc_instance._sex_concordance_results.unique(
            subset=[sample_id_col], keep="first"
        )
        sample_summary = sample_summary.join(sex_deduped, on=sample_id_col, how="left")

    # Add age concordance results if available (Visit_Adj_Age, Predicted_Age, Std_Predicted_Age)
    if (
        hasattr(qc_instance, "_age_concordance_results")
        and qc_instance._age_concordance_results is not None
    ):
        n_age_dups = (
            qc_instance._age_concordance_results.height
            - qc_instance._age_concordance_results.unique(subset=[sample_id_col]).height
        )
        if n_age_dups > 0:
            logger.warning(
                f"_age_concordance_results contains {n_age_dups} duplicate rows per sample; "
                "keeping first occurrence for join"
            )
        age_deduped = qc_instance._age_concordance_results.unique(
            subset=[sample_id_col], keep="first"
        )
        sample_summary = sample_summary.join(age_deduped, on=sample_id_col, how="left")

    # Debug: Check protein_summary size before joins
    n_proteins_initial = protein_summary.height
    logger.debug(f"Protein summary initial size: {n_proteins_initial} rows")

    # Add assay QC metrics if available (ColCheck status, Total_Samples)
    if (
        hasattr(qc_instance, "_assay_qc_status")
        and qc_instance._assay_qc_status is not None
    ):
        # Ensure _assay_qc_status has unique protein IDs
        assay_qc = qc_instance._assay_qc_status.unique(subset=[protein_id_col])
        n_assay_qc = assay_qc.height
        logger.debug(f"Joining assay QC status: {n_assay_qc} unique {protein_id_col}s")

        protein_summary = protein_summary.join(
            assay_qc, on=protein_id_col, how="left", coalesce=True
        )

        if protein_summary.height != n_proteins_initial:
            logger.warning(
                f"Protein summary size changed after assay QC join: {n_proteins_initial} -> {protein_summary.height}"
            )

    # Add CV stats if available (Intra_CV, Inter_CV)
    if hasattr(qc_instance, "_cv_stats") and qc_instance._cv_stats is not None:
        # Ensure _cv_stats has unique protein IDs
        cv_stats = qc_instance._cv_stats.unique(subset=[protein_id_col])
        n_cv = cv_stats.height
        logger.debug(f"Joining CV stats: {n_cv} unique {protein_id_col}s")

        protein_summary = protein_summary.join(
            cv_stats, on=protein_id_col, how="left", coalesce=True
        )

        if protein_summary.height != n_proteins_initial:
            logger.warning(
                f"Protein summary size changed after CV join: {n_proteins_initial} -> {protein_summary.height}"
            )

    # Add detection stats if available (Detection_Rate)
    if (
        hasattr(qc_instance, "_detection_stats")
        and qc_instance._detection_stats is not None
    ):
        # Ensure _detection_stats has unique protein IDs
        det_stats = qc_instance._detection_stats.unique(subset=[protein_id_col])
        n_det = det_stats.height
        logger.debug(f"Joining detection stats: {n_det} unique {protein_id_col}s")

        protein_summary = protein_summary.join(
            det_stats, on=protein_id_col, how="left", coalesce=True
        )

        if protein_summary.height != n_proteins_initial:
            logger.warning(
                f"Protein summary size changed after detection join: {n_proteins_initial} -> {protein_summary.height}"
            )

    logger.debug(f"Protein summary final size: {protein_summary.height} rows")

    # Reorder columns: ID cols, priority cols, Status, FAIL_*, WARN_*, analysis results in order
    def _reorder_cols(df, id_cols, priority_cols, is_protein=False):
        # Get base columns (ID, priority, Status)
        base = [c for c in id_cols + priority_cols + ["Status"] if c in df.columns]

        # Get QC flag columns (FAIL_* and WARN_*)
        fail_cols = sorted([c for c in df.columns if c.startswith("FAIL_")])
        warn_cols = sorted([c for c in df.columns if c.startswith("WARN_")])

        # Define analysis result columns in specific order
        if is_protein:
            # Protein/assay-specific result columns
            analysis_col_order = [
                "EntrezGeneSymbol",
                "Warn_Frac",
                "Missing_Frac",
                "LOD_Method",
            ]
            # Add CV columns (may be multiple: Intra_CV, Inter_CV, or grouped by study: Intra_CV_All, Intra_CV_{study})
            cv_cols = sorted(
                [
                    c
                    for c in df.columns
                    if c.startswith("Intra_CV") or c.startswith("Inter_CV")
                ]
            )
            analysis_col_order.extend(cv_cols)
            # Add Detection_Rate columns (may be multiple for different groups)
            detection_cols = sorted(
                [c for c in df.columns if c.startswith("Detection_Rate")]
            )
            analysis_col_order.extend(detection_cols)
        else:
            # Sample-specific result columns
            analysis_col_order = [
                "RFU_Outlier_Frac",
                "Warn_Frac",
                "Missing_Frac",
                "PC1_std",
                "PC2_std",
                "Z_Median",
                "Z_IQR",
                "Reported_Sex",
                "Predicted_Sex",
                "Predicted_Male_Probability",
                "Visit_Adj_Age",
                "Predicted_Age",
                "Std_Predicted_Age",
            ]

        # Collect analysis columns that exist (filter out detection_cols from order if already added)
        if is_protein:
            analysis_cols = [c for c in analysis_col_order if c in df.columns]
        else:
            analysis_cols = [c for c in analysis_col_order if c in df.columns]

        # Fill null values in boolean columns with False
        bool_cols = fail_cols + warn_cols
        return df.with_columns(
            [pl.col(c).fill_null(False) for c in bool_cols if c in df.columns]
        ).select(base + fail_cols + warn_cols + analysis_cols)

    # Build priority columns list dynamically
    sample_priority = ["AccessionNumber", usubjid_col, "STUDY", "VISITNUM"]

    protein_priority = []
    if protein_name_col:
        protein_priority.append(protein_name_col)
    if uniprot_col and uniprot_col in analyte_schema:
        protein_priority.append(uniprot_col)

    sample_summary = _reorder_cols(
        sample_summary,
        [sample_id_col],
        sample_priority,
        is_protein=False,
    ).sort(sample_id_col)

    protein_summary = _reorder_cols(
        protein_summary,
        [protein_id_col] + ([protein_name_col] if protein_name_col else []),
        protein_priority[1:] if protein_name_col else protein_priority,
        is_protein=True,
    ).sort(protein_name_col if protein_name_col else protein_id_col)

    # Save files if requested
    if save_prefix:
        # Replace spaces with underscores in column names
        sample_summary = sample_summary.rename(
            {col: col.replace(" ", "_") for col in sample_summary.columns}
        )
        protein_summary = protein_summary.rename(
            {col: col.replace(" ", "_") for col in protein_summary.columns}
        )

        sample_summary.write_csv(f"{save_prefix}.sample_qc_summary.tsv", separator="\t")
        protein_summary.write_csv(f"{save_prefix}.assay_qc_summary.tsv", separator="\t")
        logger.info(
            f"saved: {save_prefix}.sample_qc_summary.tsv, {save_prefix}.assay_qc_summary.tsv"
        )

        # Long format with QC status and LOD values (if available)
        # Only export biological samples (not QC controls)
        schema_names = qc_instance.samples.collect_schema().names()

        # For SomaScan, save RFU instead of log10_scaled_RFU (can derive later)
        # For Olink, save NPX
        if "RFU" in schema_names:
            measurement_col = "RFU"
            filename_suffix = "rfu_long.parquet"
        else:
            measurement_col = data_col
            filename_suffix = "npx_long.parquet"

        select_cols = [sample_id_col, protein_id_col, measurement_col]
        if "LOD" in schema_names:
            select_cols.append("LOD")

        long_data = qc_instance.samples.select(select_cols).collect()

        long_data = _add_qc_status(
            long_data,
            sample_id_col,
            qc_instance.failed_samples,
            qc_instance.warned_samples,
        ).rename({"Status": "SampleQC_Status"})
        long_data = _add_qc_status(
            long_data,
            protein_id_col,
            qc_instance.failed_proteins,
            qc_instance.warned_proteins,
        ).rename({"Status": "ProteinQC_Status"})

        output_file = f"{save_prefix}.{filename_suffix}"
        long_data.write_parquet(output_file)
        logger.info(f"saved: {output_file} ({long_data.height:,} rows)")

    return sample_summary, protein_summary


def impute_missing_values(
    X: np.ndarray,
    random_seed: int = 0,
    method: str = "knn",
    n_neighbors: int = 5,
) -> np.ndarray:
    """
    Impute missing values using various imputation strategies.

    Args:
        X: 2D numpy array with potential NaN values (samples × features)
        random_seed: Random seed for reproducibility (default: 0)
        method: Imputation method to use (default: "knn")
            - "iterative": MICE algorithm using BayesianRidge
            - "knn": K-nearest neighbors imputation
            - "mean": Simple mean imputation
            - "median": Simple median imputation
        n_neighbors: Number of neighbors for KNN imputation (default: 5)

    Returns:
        Array with NaN values imputed
    """
    if not np.any(np.isnan(X)):
        return X

    if method == "iterative":
        # For high-dimensional data, limit features used per imputation round
        imputer = IterativeImputer(
            random_state=random_seed,
            max_iter=10,
            tol=1e-3,
            n_nearest_features=100,
            imputation_order="ascending",
            verbose=0,
        )
    elif method == "knn":
        # Set numpy seed for deterministic tie-breaking in KNN
        np.random.seed(random_seed)
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    elif method == "mean":
        imputer = SimpleImputer(strategy="mean")
    elif method == "median":
        imputer = SimpleImputer(strategy="median")
    else:
        raise ValueError(
            f"Unknown imputation method: {method}. "
            f"Choose from: 'iterative', 'knn', 'mean', 'median'"
        )

    return imputer.fit_transform(X)
