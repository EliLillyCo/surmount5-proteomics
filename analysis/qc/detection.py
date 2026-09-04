"""Detection and limit of detection (LOD) calculations for proteomics data."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def soma_lod(
    qc_instance,
    buffer_sample_type: str = "Buffer",
) -> pl.LazyFrame:
    """Calculate estimated Limit of Detection (eLOD) for each SomaId.

    eLOD is calculated using buffer/blank samples following the SomaLogic methodology:
    1. For each SOMAmer, calculate median and adjusted MAD across buffer samples
       MAD_adjusted = 1.4826 * MAD
       (1.4826 adjusts MAD to be reflective of standard deviation of normal distribution)
    2. Calculate eLOD = median + 3 * MAD_adjusted

    This adds an "LOD" column to the long format dataframe.

    Args:
        qc_instance: SomaQC instance
        buffer_sample_type: SampleType value to use for buffer samples (default: "Buffer")

    Returns:
        Updated LazyFrame with LOD column added
    """
    logger.info(f"calculating eLOD using {buffer_sample_type} samples...")

    # Get column names from instance
    protein_id_col = qc_instance.protein_id_col  # SeqId for SomaScan
    sample_id_col = qc_instance.sample_id_col  # SampleId

    # SomaScan ADAT data has _sample_idx to preserve buffer replicates
    # (multiple replicates can share same SampleId but have unique _sample_idx)
    join_on = "_sample_idx"
    unique_id_col = "_sample_idx"

    # Filter sample metadata to buffer samples first, then semi-join with data
    buffer_sample_ids = qc_instance.sample_metadata.filter(
        pl.col("SampleType") == buffer_sample_type
    ).select([join_on])

    # Check if we have buffer samples
    n_buffer = (
        buffer_sample_ids.select(pl.col(unique_id_col).n_unique()).collect().item()
    )
    if n_buffer == 0:
        logger.warning(
            f"No buffer samples found with SampleType = '{buffer_sample_type}'"
        )
        logger.warning("skipping eLOD calculation - LOD column will be null")
        return qc_instance.get_data().with_columns(
            [pl.lit(None).cast(pl.Float64).alias("LOD")]
        )

    logger.info(f"found {n_buffer} buffer samples for eLOD calculation")

    # Get buffer samples by semi-joining with filtered metadata (more efficient)
    buffer_samples = qc_instance.get_data().join(
        buffer_sample_ids,
        on=join_on,
        how="semi",
    )

    # Calculate median and MAD for each protein using buffer samples (on raw RFU values)
    # Formula: eLOD = median(RFU) + 3 * 1.4826 * MAD(RFU)
    lod_stats = (
        buffer_samples.group_by(protein_id_col)
        .agg(
            [
                pl.col("RFU").median().alias("buffer_median"),
                (pl.col("RFU") - pl.col("RFU").median())
                .abs()
                .median()
                .alias("buffer_mad"),
            ]
        )
        .with_columns(
            [
                # Adjusted MAD (1.4826 makes MAD ~ SD for normal distribution).
                (1.4826 * pl.col("buffer_mad")).alias("buffer_mad_adjusted"),
            ]
        )
        .with_columns(
            [
                # eLOD = median + 3 * MAD_adjusted.
                (pl.col("buffer_median") + 3 * pl.col("buffer_mad_adjusted")).alias(
                    "LOD"
                ),
            ]
        )
        .select([protein_id_col, "LOD"])
    )

    result = qc_instance.get_data().join(lod_stats, on=protein_id_col, how="left")

    n_analytes = lod_stats.select(pl.len()).collect().item()
    logger.info(f"✓ calculated eLOD for {n_analytes} analytes")

    return result


def olink_lod(
    qc_instance,
    min_count_threshold: int = 150,
    min_num_nc: int = 10,
) -> pl.LazyFrame:
    """
    Calculate the limit of detection for Olink data from negative controls.

    Args:
        qc_instance: OlinkQC instance
        min_count_threshold: Minimum count threshold for selecting the LOD
            calculation method (default: 150)
        min_num_nc: Minimum number of negative controls required (default: 10)

    Returns:
        LazyFrame with updated data including LOD columns
    """
    logger.info("calculating Olink LOD (NCLOD)...")

    data = qc_instance.get_data()
    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    data_col = qc_instance.data_col

    # Filter for passing negative controls
    lod_data = data.filter(
        (pl.col("SampleType") == "NEGATIVE_CONTROL")
        & (pl.col("SampleQC") == "PASS")
        & (pl.col("AssayType") == "assay")
        & pl.col(data_col).is_not_null()
        & ~pl.col(data_col).is_nan()
    )

    # Check we have enough NCs
    n_nc = (
        lod_data.select(pl.col(sample_id_col).n_unique())
        .collect(engine="streaming")
        .item()
    )
    if n_nc < min_num_nc:
        raise ValueError(
            f"At least {min_num_nc} Negative Controls are required to calculate LOD from Negative Controls. Found {n_nc}."
        )

    # Calculate LOD per assay and DataAnalysisRefID
    lod_stats = (
        lod_data.group_by(["OlinkID", "DataAnalysisRefID"])
        .agg(
            [
                pl.col("Count").max().alias("MaxCount"),
                pl.col("PCNormalizedNPX").median().alias("MedianNPX"),
                pl.col("PCNormalizedNPX").std().alias("StdNPX"),
            ]
        )
        .with_columns(
            [
                # LODNPX: median + max(0.2, 3*SD)
                (
                    pl.col("MedianNPX")
                    + pl.max_horizontal(pl.lit(0.2), pl.col("StdNPX").mul(3.0))
                ).alias("LODNPX"),
                # LODCount: max(min_count_threshold, 2*MaxCount)
                pl.max_horizontal(
                    pl.lit(min_count_threshold), pl.col("MaxCount").mul(2)
                ).alias("LODCount"),
                # LODMethod: "lod_npx" if MaxCount > min_count_threshold, else "lod_count"
                pl.when(pl.col("MaxCount") > min_count_threshold)
                .then(pl.lit("lod_npx"))
                .otherwise(pl.lit("lod_count"))
                .alias("LODMethod"),
            ]
        )
        .select(["OlinkID", "DataAnalysisRefID", "LODNPX", "LODCount", "LODMethod"])
    )

    # Calculate PC median and extension control counts
    pc_median = (
        data.filter(
            (pl.col("SampleType") == "PLATE_CONTROL")
            & pl.col("ExtNPX").is_not_null()
            & ~pl.col("ExtNPX").is_nan()
        )
        .group_by([protein_id_col, "PlateID"])
        .agg(pl.col("ExtNPX").median().alias("PCMedian"))
    )

    ext_count = data.filter(pl.col("AssayType") == "ext_ctrl").select(
        [
            sample_id_col,
            "WellID",
            "Panel",
            "Block",
            "SampleType",
            "PlateID",
            pl.col("Count").alias("ExtCount"),
        ]
    )

    n_pc = pc_median.select(pl.len()).collect(engine="streaming").item()
    if n_pc == 0:
        raise ValueError(
            "Insufficient Plate Control data for normalization of LOD."
        )

    # Join LOD stats, PC medians, and ext counts; compute plate-control-normalized LOD
    result = (
        data.join(lod_stats, on=[protein_id_col, "DataAnalysisRefID"], how="left")
        .join(pc_median, on=[protein_id_col, "PlateID"], how="left", coalesce=True)
        .join(
            ext_count,
            on=[sample_id_col, "WellID", "Panel", "Block", "SampleType", "PlateID"],
            how="left",
            coalesce=True,
        )
        .with_columns(
            [
                # Calculate PCNormalizedLOD (plate control normalized LOD)
                pl.when(pl.col(data_col).is_null() | pl.col(data_col).is_nan())
                .then(None)
                .when(
                    (pl.col(data_col).is_not_null())
                    & ~pl.col(data_col).is_nan()
                    & (pl.col("LODMethod") == "lod_npx")
                )
                .then(pl.col("LODNPX"))
                .otherwise(
                    (pl.col("LODCount") / pl.col("ExtCount")).log(2)
                    - pl.col("PCMedian")
                )
                .alias("PCNormalizedLOD"),
            ]
        )
    )

    # Adjust LOD for intensity-normalized plates (subtract plate median NPX)
    has_intensity = (
        result.select(pl.col("Normalization").eq("Intensity").any())
        .collect(engine="streaming")
        .item()
    )

    if has_intensity:
        # Calculate plate median NPX for intensity normalization from SAMPLE rows
        plate_median_npx = (
            result.filter(
                (pl.col("SampleType") == "SAMPLE")
                & pl.col("PCNormalizedNPX").is_not_null()
                & ~pl.col("PCNormalizedNPX").is_nan()
            )
            .group_by([protein_id_col, "PlateID"])
            .agg(pl.col("PCNormalizedNPX").median().alias("PlateMedianNPX"))
        )

        # Join and calculate final LOD adjusted for normalization method
        result = result.join(
            plate_median_npx,
            on=[protein_id_col, "PlateID"],
            how="left",
            coalesce=True,
        ).with_columns(
            [
                pl.when(pl.col("Normalization") == "Intensity")
                .then(pl.col("PCNormalizedLOD") - pl.col("PlateMedianNPX"))
                .when(pl.col("Normalization") == "Plate control")
                .then(pl.col("PCNormalizedLOD"))
                .otherwise(None)
                .alias("LOD"),
            ]
        )
    else:
        # No intensity normalization - LOD = PCNormalizedLOD for plate control samples
        result = result.with_columns(
            [
                pl.when(pl.col("Normalization") == "Plate control")
                .then(pl.col("PCNormalizedLOD"))
                .otherwise(None)
                .alias("LOD"),
            ]
        )

    # Filter to SAMPLE rows with assay type only and keep essential columns
    # Keep only SAMPLE/assay rows and essential columns.
    result = result.filter(
        (pl.col("SampleType") == "SAMPLE") & (pl.col("AssayType") == "assay")
    ).select([sample_id_col, protein_id_col, data_col, "LOD"])

    n_analytes = (
        lod_stats.select(pl.col("OlinkID").n_unique())
        .collect(engine="streaming")
        .item()
    )
    logger.info(f"✓ calculated LOD for {n_analytes} analytes using negative-control LOD (NCLOD)")
    logger.info(
        "✓ filtered to SAMPLE rows only (removed controls) and kept essential columns"
    )

    return result


def assay_cv(
    qc_instance,
    qc_sample_type: str | None = None,
    intra_threshold: float = 25.0,
    inter_threshold: float = 25.0,
    filter_high_cv: bool = True,
) -> pl.DataFrame | None:
    """Calculate intra/inter-plate CV.

    Platform-agnostic function that works for both SomaScan and Olink data.
    Platform is inferred from qc_instance.protein_id_col:
    - SeqId -> uses standard CV formula (σ/μ × 100%), PlateId column, and QC sample type
    - OlinkID -> uses Olink CV formula (sqrt(exp((ln(2)*σ)^2)-1) × 100%), PlateID column, and SAMPLE_CONTROL sample type

    Args:
        qc_instance: SomaQC or OlinkQC instance
        qc_sample_type: SampleType value for QC samples (default: auto-detected from platform)
        intra_threshold: Threshold for intra-plate CV (proteins above this are flagged)
        inter_threshold: Threshold for inter-plate CV (proteins above this are flagged)
        filter_high_cv: If True, mark proteins with Intra_CV > intra_threshold as FAIL

    Returns:
        Dictionary with CV statistics per protein
    """
    # Get column names from instance attributes
    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    plate_id_col = qc_instance.plate_id_col

    # Use instance qc_sample_type if not provided
    if qc_sample_type is None:
        qc_sample_type = qc_instance.qc_sample_type

    qc_instance._print_section("COEFFICIENT OF VARIATION")
    logger.info(f"calculating CV using {qc_sample_type} samples...")

    # For CV calculations, use raw measurement values (RFU/NPX), NOT scaled values
    # SomaScan: use RFU (not log10_scaled_RFU which is already normalized)
    if protein_id_col == "SeqId":
        data_col = "RFU"
        cv_formula = "standard"
    elif protein_id_col == "OlinkID":
        data_col = qc_instance.data_col  # NPX
        cv_formula = "olink"
    else:
        raise ValueError(
            f"Cannot infer platform from protein_id_col='{protein_id_col}'. "
            "Expected 'SeqId' (SomaScan) or 'OlinkID' (Olink)."
        )

    # Get QC samples by joining with sample metadata

    # Check if _sample_idx exists (SomaScan ADAT data) or not (Olink parquet data)
    has_sample_idx = "_sample_idx" in qc_instance.get_data().collect_schema().names()

    if has_sample_idx:
        # SomaScan: Join on _sample_idx to preserve replicates (QC, Calibrator, Buffer share SampleId)
        join_cols = ["_sample_idx", sample_id_col, "SampleType", plate_id_col]
        join_on = "_sample_idx"
    else:
        # Olink: Join on SampleId (data already in proper long format)
        join_cols = [sample_id_col, "SampleType", plate_id_col]
        join_on = sample_id_col

    qc = (
        qc_instance.get_data()
        .join(
            qc_instance.sample_metadata.select(join_cols),
            on=join_on,
            how="left",
        )
        .filter(pl.col("SampleType") == qc_sample_type)
    )

    if qc.select(pl.len()).collect().item() == 0:
        logger.warning(f"No QC samples found with SampleType = '{qc_sample_type}'")
        return None

    # Filter out failed samples and proteins (platform-agnostic QC filtering)
    if qc_instance.failed_samples:
        qc = qc.filter(~pl.col(sample_id_col).is_in(qc_instance.failed_samples))
    if qc_instance.failed_proteins:
        qc = qc.filter(~pl.col(protein_id_col).is_in(qc_instance.failed_proteins))

    # Filter out null/NaN data values
    qc_clean = qc.filter(pl.col(data_col).is_not_null() & ~pl.col(data_col).is_nan())

    # Use appropriate unique identifier based on platform
    unique_id_col = "_sample_idx" if has_sample_idx else sample_id_col

    # Get counts in single query
    qc_counts = (
        qc_clean.select(
            [
                pl.col(unique_id_col).n_unique().alias("n_qc_samples"),
                pl.col(plate_id_col).n_unique().alias("n_plates"),
            ]
        )
        .collect()
        .row(0, named=True)
    )
    n_qc_samples = qc_counts["n_qc_samples"]
    n_plates = qc_counts["n_plates"]

    logger.info(
        f"found {n_qc_samples} {qc_sample_type} samples across {n_plates} plates"
    )

    if n_plates < 2:
        logger.warning("need ≥2 plates for inter-plate CV calculation")
        # Can still calculate intra-plate CV if we have multiple samples per plate

    # Check if STUDY column exists for stratified CV calculation
    qc_schema = qc_clean.collect_schema().names()
    use_study_stratified = "STUDY" in qc_schema
    if use_study_stratified:
        # Already have counts, just get n_studies in same query if needed
        study_check = qc_clean.select(pl.col("STUDY").n_unique()).collect().item()
        if study_check > 1:
            logger.info(f"Calculating CVs stratified by {study_check} studies")
        else:
            use_study_stratified = False
    else:
        logger.info(
            "STUDY column not found in QC data - skipping per-study CV calculation"
        )

    # Choose CV calculation method based on formula
    ln2 = np.log(2) if cv_formula == "olink" else None

    # Intra-plate CV: Calculate CV from replicates within each plate, then take median across plates
    # Note: We use ALL replicate measurements (count), not unique sample IDs
    if use_study_stratified:
        # Calculate per-study intra-CV
        if cv_formula == "standard":
            intra_per_study = (
                qc_clean.group_by([protein_id_col, "STUDY", plate_id_col])
                .agg(
                    [
                        pl.col(data_col).mean().alias("mean_data"),
                        pl.col(data_col).std().alias("sd_data"),
                        pl.col(data_col).count().alias("n_replicates"),
                    ]
                )
                .filter(
                    pl.col("sd_data").is_not_null()
                    & ~pl.col("sd_data").is_nan()
                    & (pl.col("sd_data") > 0)
                    & (pl.col("mean_data") > 0)
                    & (pl.col("n_replicates") >= 2)
                )
                .with_columns(
                    [
                        ((pl.col("sd_data") / pl.col("mean_data")) * 100).alias(
                            "plate_cv"
                        )
                    ]
                )
                .group_by([protein_id_col, "STUDY"])
                .agg(
                    [
                        pl.col("plate_cv").median().alias("Intra_CV"),
                        pl.col("plate_cv").count().alias("n_plates"),
                    ]
                )
                .filter(pl.col("n_plates") >= 1)
            )
        else:  # olink
            intra_per_study = (
                qc_clean.group_by([protein_id_col, "STUDY", plate_id_col])
                .agg(
                    [
                        pl.col(data_col).std().alias("sd_data"),
                        pl.col(data_col).count().alias("n_replicates"),
                    ]
                )
                .filter(
                    pl.col("sd_data").is_not_null()
                    & ~pl.col("sd_data").is_nan()
                    & (pl.col("sd_data") > 0)
                    & (pl.col("n_replicates") >= 2)
                )
                .with_columns(
                    [
                        (((ln2 * pl.col("sd_data")).pow(2)).exp() - 1)
                        .sqrt()
                        .mul(100)
                        .alias("plate_cv")
                    ]
                )
                .group_by([protein_id_col, "STUDY"])
                .agg(
                    [
                        pl.col("plate_cv").median().alias("Intra_CV"),
                        pl.col("plate_cv").count().alias("n_plates"),
                    ]
                )
                .filter(pl.col("n_plates") >= 1)
            )

        # Overall (non-stratified) intra-CV used for pass/fail filtering
        if cv_formula == "standard":
            intra = (
                qc_clean.group_by([protein_id_col, plate_id_col])
                .agg(
                    [
                        pl.col(data_col).mean().alias("mean_data"),
                        pl.col(data_col).std().alias("sd_data"),
                        pl.col(data_col).count().alias("n_replicates"),
                    ]
                )
                .filter(
                    pl.col("sd_data").is_not_null()
                    & ~pl.col("sd_data").is_nan()
                    & (pl.col("sd_data") > 0)
                    & (pl.col("mean_data") > 0)
                    & (pl.col("n_replicates") >= 2)
                )
                .with_columns(
                    [
                        ((pl.col("sd_data") / pl.col("mean_data")) * 100).alias(
                            "plate_cv"
                        )
                    ]
                )
                .group_by(protein_id_col)
                .agg(
                    [
                        pl.col("plate_cv").median().alias("Intra_CV"),
                        pl.col("plate_cv").count().alias("n_plates"),
                    ]
                )
                .filter(pl.col("n_plates") >= 1)
            )
        else:  # olink
            intra = (
                qc_clean.group_by([protein_id_col, plate_id_col])
                .agg(
                    [
                        pl.col(data_col).std().alias("sd_data"),
                        pl.col(data_col).count().alias("n_replicates"),
                    ]
                )
                .filter(
                    pl.col("sd_data").is_not_null()
                    & ~pl.col("sd_data").is_nan()
                    & (pl.col("sd_data") > 0)
                    & (pl.col("n_replicates") >= 2)
                )
                .with_columns(
                    [
                        (((ln2 * pl.col("sd_data")).pow(2)).exp() - 1)
                        .sqrt()
                        .mul(100)
                        .alias("plate_cv")
                    ]
                )
                .group_by(protein_id_col)
                .agg(
                    [
                        pl.col("plate_cv").median().alias("Intra_CV"),
                        pl.col("plate_cv").count().alias("n_plates"),
                    ]
                )
                .filter(pl.col("n_plates") >= 1)
            )
    else:
        intra_per_study = None
        if cv_formula == "standard":
            # Standard formula: CV = (σ/μ) × 100%
            intra = (
                qc_clean.group_by([protein_id_col, plate_id_col])
                .agg(
                    [
                        pl.col(data_col).mean().alias("mean_data"),
                        pl.col(data_col).std().alias("sd_data"),
                        pl.col(data_col).count().alias("n_replicates"),
                    ]
                )
                .filter(
                    pl.col("sd_data").is_not_null()
                    & ~pl.col("sd_data").is_nan()
                    & (pl.col("sd_data") > 0)
                    & (pl.col("mean_data") > 0)
                    & (pl.col("n_replicates") >= 2)
                )
                .with_columns(
                    [
                        ((pl.col("sd_data") / pl.col("mean_data")) * 100).alias(
                            "plate_cv"
                        )
                    ]
                )
                .group_by(protein_id_col)
                .agg(
                    [
                        pl.col("plate_cv").median().alias("Intra_CV"),
                        pl.col("plate_cv").count().alias("n_plates"),
                    ]
                )
                .filter(pl.col("n_plates") >= 1)
            )
        else:  # olink
            # Olink formula: CV = sqrt(exp((ln(2)*σ)^2)-1) × 100%
            intra = (
                qc_clean.group_by([protein_id_col, plate_id_col])
                .agg(
                    [
                        pl.col(data_col).std().alias("sd_data"),
                        pl.col(data_col).count().alias("n_replicates"),
                    ]
                )
                .filter(
                    pl.col("sd_data").is_not_null()
                    & ~pl.col("sd_data").is_nan()
                    & (pl.col("sd_data") > 0)
                    & (pl.col("n_replicates") >= 2)
                )
                .with_columns(
                    [
                        (((ln2 * pl.col("sd_data")).pow(2)).exp() - 1)
                        .sqrt()
                        .mul(100)
                        .alias("plate_cv")
                    ]
                )
                .group_by(protein_id_col)
                .agg(
                    [
                        pl.col("plate_cv").median().alias("Intra_CV"),
                        pl.col("plate_cv").count().alias("n_plates"),
                    ]
                )
                .filter(pl.col("n_plates") >= 1)
            )

    # Inter-plate CV: Calculate mean of replicates per plate, then CV across plates
    if n_plates >= 2:
        if use_study_stratified:
            # Calculate per-study inter-CV
            if cv_formula == "standard":
                inter_per_study = (
                    qc_clean.group_by([protein_id_col, "STUDY", plate_id_col])
                    .agg(
                        [
                            pl.col(data_col).mean().alias("plate_mean"),
                            pl.col(data_col).count().alias("n_replicates"),
                        ]
                    )
                    .filter(pl.col("n_replicates") >= 1)
                    .group_by([protein_id_col, "STUDY"])
                    .agg(
                        [
                            pl.col("plate_mean").mean().alias("grand_mean"),
                            pl.col("plate_mean").std().alias("sd_plate_means"),
                            pl.col("plate_mean").count().alias("n_plates"),
                        ]
                    )
                    .filter(
                        pl.col("sd_plate_means").is_not_null()
                        & ~pl.col("sd_plate_means").is_nan()
                        & (pl.col("sd_plate_means") > 0)
                        & (pl.col("grand_mean") > 0)
                        & (pl.col("n_plates") >= 2)
                    )
                    .with_columns(
                        [
                            (
                                (pl.col("sd_plate_means") / pl.col("grand_mean")) * 100
                            ).alias("Inter_CV")
                        ]
                    )
                )
            else:  # olink
                inter_per_study = (
                    qc_clean.group_by([protein_id_col, "STUDY", plate_id_col])
                    .agg(
                        [
                            pl.col(data_col).mean().alias("plate_mean"),
                            pl.col(data_col).count().alias("n_replicates"),
                        ]
                    )
                    .filter(pl.col("n_replicates") >= 1)
                    .group_by([protein_id_col, "STUDY"])
                    .agg(
                        [
                            pl.col("plate_mean").std().alias("sd_plate_means"),
                            pl.col("plate_mean").count().alias("n_plates"),
                        ]
                    )
                    .filter(
                        pl.col("sd_plate_means").is_not_null()
                        & ~pl.col("sd_plate_means").is_nan()
                        & (pl.col("sd_plate_means") > 0)
                        & (pl.col("n_plates") >= 2)
                    )
                    .with_columns(
                        [
                            (((ln2 * pl.col("sd_plate_means")).pow(2)).exp() - 1)
                            .sqrt()
                            .mul(100)
                            .alias("Inter_CV")
                        ]
                    )
                )

            # Overall (non-stratified) inter-CV used for filtering
            if cv_formula == "standard":
                inter = (
                    qc_clean.group_by([protein_id_col, plate_id_col])
                    .agg(
                        [
                            pl.col(data_col).mean().alias("plate_mean"),
                            pl.col(data_col).count().alias("n_replicates"),
                        ]
                    )
                    .filter(pl.col("n_replicates") >= 1)
                    .group_by(protein_id_col)
                    .agg(
                        [
                            pl.col("plate_mean").mean().alias("grand_mean"),
                            pl.col("plate_mean").std().alias("sd_plate_means"),
                            pl.col("plate_mean").count().alias("n_plates"),
                        ]
                    )
                    .filter(
                        pl.col("sd_plate_means").is_not_null()
                        & ~pl.col("sd_plate_means").is_nan()
                        & (pl.col("sd_plate_means") > 0)
                        & (pl.col("grand_mean") > 0)
                        & (pl.col("n_plates") >= 2)
                    )
                    .with_columns(
                        [
                            (
                                (pl.col("sd_plate_means") / pl.col("grand_mean")) * 100
                            ).alias("Inter_CV")
                        ]
                    )
                    .select([protein_id_col, "Inter_CV"])
                )
            else:  # olink
                inter = (
                    qc_clean.group_by([protein_id_col, plate_id_col])
                    .agg(
                        [
                            pl.col(data_col).mean().alias("plate_mean"),
                            pl.col(data_col).count().alias("n_replicates"),
                        ]
                    )
                    .filter(pl.col("n_replicates") >= 1)
                    .group_by(protein_id_col)
                    .agg(
                        [
                            pl.col("plate_mean").std().alias("sd_plate_means"),
                            pl.col("plate_mean").count().alias("n_plates"),
                        ]
                    )
                    .filter(
                        pl.col("sd_plate_means").is_not_null()
                        & ~pl.col("sd_plate_means").is_nan()
                        & (pl.col("sd_plate_means") > 0)
                        & (pl.col("n_plates") >= 2)
                    )
                    .with_columns(
                        [
                            (((ln2 * pl.col("sd_plate_means")).pow(2)).exp() - 1)
                            .sqrt()
                            .mul(100)
                            .alias("Inter_CV")
                        ]
                    )
                    .select([protein_id_col, "Inter_CV"])
                )
        else:
            inter_per_study = None
            if cv_formula == "standard":
                inter = (
                    qc_clean.group_by([protein_id_col, plate_id_col])
                    .agg(
                        [
                            pl.col(data_col).mean().alias("plate_mean"),
                            pl.col(data_col).count().alias("n_replicates"),
                        ]
                    )
                    .filter(pl.col("n_replicates") >= 1)
                    .group_by(protein_id_col)
                    .agg(
                        [
                            pl.col("plate_mean").mean().alias("grand_mean"),
                            pl.col("plate_mean").std().alias("sd_plate_means"),
                            pl.col("plate_mean").count().alias("n_plates"),
                        ]
                    )
                    .filter(
                        pl.col("sd_plate_means").is_not_null()
                        & ~pl.col("sd_plate_means").is_nan()
                        & (pl.col("sd_plate_means") > 0)
                        & (pl.col("grand_mean") > 0)
                        & (pl.col("n_plates") >= 2)
                    )
                    .with_columns(
                        [
                            (
                                (pl.col("sd_plate_means") / pl.col("grand_mean")) * 100
                            ).alias("Inter_CV")
                        ]
                    )
                    .select([protein_id_col, "Inter_CV"])
                )
            else:  # olink
                inter = (
                    qc_clean.group_by([protein_id_col, plate_id_col])
                    .agg(
                        [
                            pl.col(data_col).mean().alias("plate_mean"),
                            pl.col(data_col).count().alias("n_replicates"),
                        ]
                    )
                    .filter(pl.col("n_replicates") >= 1)
                    .group_by(protein_id_col)
                    .agg(
                        [
                            pl.col("plate_mean").std().alias("sd_plate_means"),
                            pl.col("plate_mean").count().alias("n_plates"),
                        ]
                    )
                    .filter(
                        pl.col("sd_plate_means").is_not_null()
                        & ~pl.col("sd_plate_means").is_nan()
                        & (pl.col("sd_plate_means") > 0)
                        & (pl.col("n_plates") >= 2)
                    )
                    .with_columns(
                        [
                            (((ln2 * pl.col("sd_plate_means")).pow(2)).exp() - 1)
                            .sqrt()
                            .mul(100)
                            .alias("Inter_CV")
                        ]
                    )
                    .select([protein_id_col, "Inter_CV"])
                )
    else:
        inter = None
        inter_per_study = None

    # Join intra and inter CV
    if inter is not None:
        cv_df = intra.join(inter, on=protein_id_col, how="left").collect()
    else:
        cv_df = intra.with_columns(
            [pl.lit(None).cast(pl.Float64).alias("Inter_CV")]
        ).collect()

    if len(cv_df) == 0:
        logger.warning("no CV data calculated")
        return None

    # Summary statistics
    med_intra = cv_df["Intra_CV"].median()
    med_inter = cv_df["Inter_CV"].median() if inter is not None else None
    high_intra = (cv_df["Intra_CV"] > intra_threshold).sum()
    high_inter = (cv_df["Inter_CV"] > inter_threshold).sum() if inter is not None else 0

    logger.info(
        f"Median CV: Intra={med_intra:.2f}% | Inter={(med_inter if med_inter is not None else 0):.2f}% | "
        f"High: {high_intra}/{high_inter}"
    )

    # Store CV data for later use
    qc_instance.cv_data = cv_df

    # Store per-study CV stats if available
    if use_study_stratified:
        cv_df_per_study = intra_per_study.join(
            inter_per_study.select([protein_id_col, "STUDY", "Inter_CV"]),
            on=[protein_id_col, "STUDY"],
            how="left",
        ).collect()

        # Pivot per-study CVs into separate columns for assay summary
        unique_studies = cv_df_per_study["STUDY"].unique().sort().to_list()

        # Start with overall CVs renamed to "All" when studies exist
        cv_stats_summary = cv_df.select(
            [
                protein_id_col,
                pl.col("Intra_CV").alias("Intra_CV_All"),
                pl.col("Inter_CV").alias("Inter_CV_All"),
            ]
        )

        for study in unique_studies:
            study_cv = cv_df_per_study.filter(pl.col("STUDY") == study).select(
                [
                    protein_id_col,
                    pl.col("Intra_CV").alias(f"Intra_CV_{study}"),
                    pl.col("Inter_CV").alias(f"Inter_CV_{study}"),
                ]
            )
            cv_stats_summary = cv_stats_summary.join(
                study_cv, on=protein_id_col, how="left"
            )

        qc_instance._cv_stats = cv_stats_summary

        # Log per-study medians
        for study in unique_studies:
            study_intra = cv_df_per_study.filter(pl.col("STUDY") == study)[
                "Intra_CV"
            ].median()
            study_inter = cv_df_per_study.filter(pl.col("STUDY") == study)[
                "Inter_CV"
            ].median()
            logger.info(
                f"  {study}: Intra={study_intra:.2f}% | Inter={study_inter:.2f}%"
            )
    else:
        cv_df_per_study = None
        qc_instance._cv_stats = cv_df.select([protein_id_col, "Intra_CV", "Inter_CV"])

    # Determine CV pass/fail status for combined filtering with detection
    cv_pass_proteins = cv_df.filter(pl.col("Intra_CV") <= intra_threshold)[
        protein_id_col
    ].to_list()
    cv_fail_proteins = cv_df.filter(pl.col("Intra_CV") > intra_threshold)[
        protein_id_col
    ].to_list()

    logger.info(
        f"CV status: {len(cv_pass_proteins)} proteins ≤{intra_threshold}% (pass), "
        f"{len(cv_fail_proteins)} proteins >{intra_threshold}% (fail)"
    )

    # Store CV pass/fail status for combined filtering (don't mark as FAIL yet)
    qc_instance._cv_pass_proteins = cv_pass_proteins
    qc_instance._cv_fail_proteins = cv_fail_proteins

    return {
        "cv_df": cv_df,
        "med_intra": med_intra,
        "med_inter": med_inter,
        "high_intra": high_intra,
        "high_inter": high_inter,
        "intra_threshold": intra_threshold,
        "inter_threshold": inter_threshold,
        "qc_sample_type": qc_sample_type,
        "cv_pass_proteins": len(cv_pass_proteins),
        "cv_fail_proteins": len(cv_fail_proteins),
    }


def assay_detection(
    qc_instance,
    sample_threshold: float = 0.2,
    use_group_criteria: bool = True,
) -> pl.DataFrame:
    """Remove proteins detected in <sample_threshold of samples (data≥LOD).

    Platform-agnostic function that works for both SomaScan and Olink data.
    If LOD has not been calculated, it will be computed automatically using platform-specific methods.
    Platform is inferred from qc_instance.protein_id_col ("SeqId" -> SomaScan, "OlinkID" -> Olink).

    If groups are available and use_group_criteria=True, implements two-tier criteria:
      - Criterion 1: >sample_threshold of ALL samples detected across all groups
      - Criterion 2: ≥sample_threshold of samples detected in AT LEAST ONE group
      - Protein fails only if BOTH criteria fail

    Args:
        qc_instance: SomaQC or OlinkQC instance
        sample_threshold: Minimum detection rate to keep protein (default: 0.2)
        use_group_criteria: Use two-tier group-aware criteria if groups available

    Returns:
        DataFrame with detection statistics
    """
    qc_instance._print_section("DETECTION RATE ANALYSIS")
    logger.info("calculating detection rates...")

    # Get column names from instance
    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col

    # For detection, use the same data column as LOD calculation
    # SomaScan LOD is calculated in RFU units, so compare RFU >= LOD
    # Olink LOD is calculated in NPX units, so compare NPX >= LOD
    if protein_id_col == "SeqId":
        data_col = "RFU"
    elif protein_id_col == "OlinkID":
        data_col = qc_instance.data_col
    else:
        # Fallback to instance default
        data_col = qc_instance.data_col

    # Check if LOD has been calculated
    # For both platforms, LOD should be in data_full
    if not qc_instance._has_column("LOD"):
        if protein_id_col == "OlinkID":
            result = olink_lod(qc_instance)
            qc_instance.data_full = result
            qc_instance._schema = result.collect_schema()
        elif protein_id_col == "SeqId":
            result = soma_lod(qc_instance)
            qc_instance.data_full = result
            qc_instance._schema = result.collect_schema()

    # Check if group-aware analysis is possible
    use_groups = False
    if (
        qc_instance.has_groups
        and use_group_criteria
        and qc_instance.group_data is not None
    ):
        # Check if group column has >1 unique value
        n_unique = (
            qc_instance.group_data.filter(pl.col(qc_instance.group_col).is_not_null())
            .select(pl.col(qc_instance.group_col).n_unique())
            .collect(engine="streaming")
            .item()
        )

        if n_unique > 1:
            use_groups = True
            logger.info(f"using group-aware detection with {n_unique} groups")

    # Get samples data - with groups if needed
    if use_groups:
        samples_data = qc_instance.get_data().join(
            qc_instance.group_data, on=sample_id_col, how="left"
        )
    else:
        samples_data = qc_instance.get_data()

    # Filter out failed samples and proteins (platform-agnostic QC filtering)
    # Use tracked failures instead of relying on specific QC columns
    if qc_instance.failed_samples:
        samples_data = samples_data.filter(
            ~pl.col(sample_id_col).is_in(qc_instance.failed_samples)
        )
    if qc_instance.failed_proteins:
        samples_data = samples_data.filter(
            ~pl.col(protein_id_col).is_in(qc_instance.failed_proteins)
        )

    # LOD is already in data_full for both platforms (calculated during olink_lod/soma_lod)
    # For SomaScan: LOD is in RFU scale, so compare RFU >= LOD (not log10_scaled_RFU)
    # For Olink: LOD and NPX are both in log scale, so compare NPX >= LOD
    detection_col = "RFU" if protein_id_col == "SeqId" else data_col
    base_query = samples_data.with_columns(
        [(pl.col(detection_col) >= pl.col("LOD")).alias("det")]
    )

    # Group by columns
    if use_groups:
        # Label samples with null group as "bridge_samples" (if applicable)
        base_query = base_query.with_columns(
            [
                pl.when(pl.col(qc_instance.group_col).is_null())
                .then(pl.lit("bridge_samples"))
                .otherwise(pl.col(qc_instance.group_col))
                .alias("group_for_stats")
            ]
        )
        group_by_cols = [protein_id_col, "group_for_stats"]
    else:
        group_by_cols = [protein_id_col]

    # Calculate detection statistics
    # Include median LOD for visualization (works for both Olink and SomaScan)
    agg_exprs = [
        pl.col("det").sum().alias("Det"),
        pl.len().alias("Tot"),
    ]

    # Add median LOD if available
    if qc_instance._has_column("LOD"):
        agg_exprs.append(
            pl.col("LOD")
            .filter(pl.col("LOD").is_not_null() & ~pl.col("LOD").is_nan())
            .median()
            .alias("Med_LOD")
        )

    det_stats = (
        base_query.group_by(group_by_cols)
        .agg(agg_exprs)
        .with_columns([(pl.col("Det") / pl.col("Tot")).alias("Det_Rate")])
        .collect(engine="streaming")
    )

    # Add protein name for visualization (join from analyte metadata)
    protein_names = qc_instance.analyte_metadata.select(
        [protein_id_col, qc_instance.protein_name_col]
    ).collect(engine="streaming")
    det_stats = det_stats.join(protein_names, on=protein_id_col, how="left")

    # If using groups, add overall detection rate column (across all non-bridge samples)
    if use_groups and "group_for_stats" in det_stats.columns:
        overall_det_rate = (
            base_query.filter(pl.col("group_for_stats") != "bridge_samples")
            .group_by([protein_id_col])
            .agg(
                [
                    pl.col("det").sum().alias("Det_Overall"),
                    pl.len().alias("Tot_Overall"),
                ]
            )
            .with_columns(
                [
                    (pl.col("Det_Overall") / pl.col("Tot_Overall")).alias(
                        "Overall_Det_Rate"
                    )
                ]
            )
            .select([protein_id_col, "Overall_Det_Rate"])
            .collect(engine="streaming")
        )
        det_stats = det_stats.join(overall_det_rate, on=protein_id_col, how="left")

    # Apply filtering criteria
    if use_groups:
        # Group-aware filtering: PASS if ≥1 group passes
        det_stats_for_filtering = det_stats.filter(
            pl.col("group_for_stats") != "bridge_samples"
        )

        proteins_passing_any_group = set(
            det_stats_for_filtering.filter(pl.col("Det_Rate") >= sample_threshold)[
                protein_id_col
            ]
            .unique()
            .to_list()
        )

        all_proteins = set(det_stats[protein_id_col].unique().to_list())
        to_remove = list(all_proteins - proteins_passing_any_group)

        logger.info(
            f"Detection status (group-aware): {len(proteins_passing_any_group)} proteins "
            f"≥{sample_threshold*100:.0f}% in ≥1 group (pass), {len(to_remove)} proteins "
            f"<{sample_threshold*100:.0f}% in all groups (fail)"
        )
    else:
        # Standard single-threshold filtering
        proteins_passing = set(
            det_stats.filter(pl.col("Det_Rate") >= sample_threshold)[
                protein_id_col
            ].to_list()
        )
        all_proteins = set(det_stats[protein_id_col].unique().to_list())
        to_remove = list(all_proteins - proteins_passing)

        logger.info(
            f"Detection status: {len(proteins_passing)} proteins ≥{sample_threshold*100:.0f}% (pass), "
            f"{len(to_remove)} proteins <{sample_threshold*100:.0f}% (fail)"
        )

    # Store detection pass/fail status
    detection_pass_proteins = list(all_proteins - set(to_remove))
    detection_fail_proteins = to_remove

    # Store detection fail set for use by finalize_assay_flags_soma
    qc_instance._detection_fail_proteins = set(detection_fail_proteins)

    # Apply combined CV + Detection filtering logic
    cv_pass = set(getattr(qc_instance, "_cv_pass_proteins", []))
    cv_fail = set(getattr(qc_instance, "_cv_fail_proteins", []))
    has_cv_data = len(cv_pass) > 0 or len(cv_fail) > 0

    det_pass = set(detection_pass_proteins)
    det_fail = set(detection_fail_proteins)

    if has_cv_data:
        # Combined filtering: PASS = both pass, WARN = one passes, FAIL = both fail
        proteins_pass_both = cv_pass & det_pass
        proteins_fail_both = cv_fail & det_fail
        proteins_warn = (cv_pass & det_fail) | (cv_fail & det_pass)

        logger.info(
            f"combined CV + detection: {len(proteins_pass_both)} pass, {len(proteins_warn)} warn, {len(proteins_fail_both)} fail"
        )

        # Mark proteins appropriately
        if proteins_fail_both:
            qc_instance.failed_proteins.extend(list(proteins_fail_both))
            qc_instance._track_protein_qc(
                list(proteins_fail_both), "FAIL_Intra_CV_and_Detection_Rate"
            )

        if proteins_warn:
            qc_instance.warned_proteins.extend(list(proteins_warn))
            # Track specific reason for warning
            warn_cv_pass_det_fail = list(cv_pass & det_fail)
            warn_cv_fail_det_pass = list(cv_fail & det_pass)
            if warn_cv_pass_det_fail:
                qc_instance._track_protein_qc(
                    warn_cv_pass_det_fail, "WARN_Detection_Rate"
                )
            if warn_cv_fail_det_pass:
                qc_instance._track_protein_qc(warn_cv_fail_det_pass, "WARN_Intra_CV")

        n_overlap_cv_detection = len(proteins_fail_both)
        # Store counts for reporting
        combined_counts = {
            "n_pass_both": len(proteins_pass_both),
            "n_warn_combined": len(proteins_warn),
            "n_fail_both": len(proteins_fail_both),
        }
    else:
        # No CV data - just mark detection failures
        if to_remove:
            logger.warning(f"marking {len(to_remove)} proteins as FAIL (low detection)")
            qc_instance.failed_proteins.extend(to_remove)
            qc_instance._track_protein_qc(to_remove, "FAIL_Low_Detection")

        n_overlap_cv_detection = 0
        combined_counts = {}

    # Prepare CV-detection join data if available
    cv_det = None
    if qc_instance.cv_data is not None and len(qc_instance.cv_data) > 0:
        # When using groups, use Overall_Det_Rate (all samples); otherwise use Det_Rate
        det_rate_col = "Overall_Det_Rate" if use_groups else "Det_Rate"

        # Get unique protein detection rates (one row per protein)
        det_for_cv = det_stats.select(
            [protein_id_col, qc_instance.protein_name_col, det_rate_col]
        ).unique(subset=[protein_id_col])

        # Rename to Det_Rate for consistent plotting
        if det_rate_col == "Overall_Det_Rate":
            det_for_cv = det_for_cv.rename({"Overall_Det_Rate": "Det_Rate"})

        cv_det = qc_instance.cv_data.join(
            det_for_cv,
            on=protein_id_col,
            how="inner",
        )
        if len(cv_det) == 0:
            cv_det = None

    # Store detection stats for summary export.
    # If groups are used, pivot detection rate by group into separate columns
    if use_groups and "group_for_stats" in det_stats.columns:
        # Filter out bridge samples for the summary
        det_stats_for_pivot = det_stats.filter(
            pl.col("group_for_stats") != "bridge_samples"
        )

        # Calculate overall detection rate (across all non-bridge samples)
        det_stats_overall = (
            base_query.filter(pl.col("group_for_stats") != "bridge_samples")
            .group_by([protein_id_col])
            .agg(
                [
                    pl.col("det").sum().alias("Det"),
                    pl.len().alias("Tot"),
                ]
            )
            .with_columns([(pl.col("Det") / pl.col("Tot")).alias("Det_Rate")])
            .select([protein_id_col, pl.col("Det_Rate").alias("Detection_Rate_All")])
            .collect()
        )

        unique_groups = det_stats_for_pivot["group_for_stats"].unique().sort().to_list()

        if unique_groups:
            # Start with overall detection rate
            det_stats_summary = det_stats_overall

            # Add per-group detection rates
            for group in unique_groups:
                group_det = det_stats_for_pivot.filter(
                    pl.col("group_for_stats") == group
                ).select(
                    [
                        protein_id_col,
                        pl.col("Det_Rate").alias(f"Detection_Rate_{group}"),
                    ]
                )
                det_stats_summary = det_stats_summary.join(
                    group_det, on=protein_id_col, how="left"
                )

            qc_instance._detection_stats = det_stats_summary
        else:
            # No groups after filtering, rename to Detection_Rate
            qc_instance._detection_stats = det_stats_overall.rename(
                {"Detection_Rate_All": "Detection_Rate"}
            )
    else:
        # No groups, use simple format
        qc_instance._detection_stats = det_stats.select(
            [protein_id_col, pl.col("Det_Rate").alias("Detection_Rate")]
        )

    return {
        "det_stats": det_stats,
        "to_remove": list(detection_fail_proteins),
        "detection_pass_proteins": len(detection_pass_proteins),
        "detection_fail_proteins": len(detection_fail_proteins),
        "sample_threshold": sample_threshold,
        "use_groups": use_groups,
        "n_overlap_cv_detection": n_overlap_cv_detection,
        "cv_det": cv_det,
        "has_combined_filtering": has_cv_data,
        **combined_counts,  # Unpack n_pass_both, n_warn_combined, n_fail_both
    }
