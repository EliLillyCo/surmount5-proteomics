"""Limit of detection (LOD), CV and detection-rate calculations for Olink and SomaScan."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from analysis.qc.flags import (
    FAIL_INTRA_CV_AND_DETECTION_RATE,
    FAIL_LOW_DETECTION,
    WARN_DETECTION_RATE,
    WARN_INTRA_CV,
)
from analysis.qc.platform import (
    OLINK_LOD_IN_MEMORY,
    OLINK_NPX,
    OLINK_NPX_LOD,
    OLINK_PC_LOD,
    OLINK_PC_NPX,
    OLINK_VALUE_COLUMNS,
)
from analysis.qc.quality_control import DELIVERED_RFU_COL

logger = logging.getLogger(__name__)


def soma_lod(
    qc_instance,
    buffer_sample_type: str = "Buffer",
) -> pl.LazyFrame:
    """Adds the estimated limit of detection (eLOD) of each SomaId to the long data.

    Following the SomaLogic method, per SOMAmer the eLOD is the median of the buffer
    samples plus 3 times their adjusted MAD, where the adjusted MAD is 1.4826 * MAD
    (which makes it reflect the standard deviation of a normal distribution).

    The buffer wells are never normalized, so the eLOD is on the delivered RFU scale;
    detection compares it with the study tubes' delivered RFU (``RFU_delivered``).

    Args:
        qc_instance: SomaQC instance.
        buffer_sample_type: SampleType value of the buffer samples.

    Returns:
        The long data with an ``LOD`` column added.

    Raises:
        ValueError: If there are no buffer samples.
    """
    logger.info(f"calculating eLOD using {buffer_sample_type} samples...")

    protein_id_col = qc_instance.protein_id_col

    # Buffer replicates can share a SampleId, so key on the unique _sample_idx.
    join_on = "_sample_idx"
    unique_id_col = "_sample_idx"

    buffer_sample_ids = qc_instance.sample_metadata.filter(
        pl.col("SampleType") == buffer_sample_type
    ).select([join_on])

    n_buffer = buffer_sample_ids.select(pl.col(unique_id_col).n_unique()).collect().item()
    if n_buffer == 0:
        raise ValueError(
            f"No buffer samples found with SampleType = '{buffer_sample_type}': the eLOD needs buffer wells"
        )

    logger.info(f"found {n_buffer} buffer samples for eLOD calculation")

    buffer_samples = qc_instance.get_data().join(
        buffer_sample_ids,
        on=join_on,
        how="semi",
    )

    # On the delivered RFU, the scale detection compares on.
    lod_stats = (
        buffer_samples.group_by(protein_id_col)
        .agg(
            [
                pl.col(DELIVERED_RFU_COL).median().alias("buffer_median"),
                (pl.col(DELIVERED_RFU_COL) - pl.col(DELIVERED_RFU_COL).median())
                .abs()
                .median()
                .alias("buffer_mad"),
            ]
        )
        .with_columns(
            [
                # 1.4826 makes MAD estimate the SD of a normal distribution.
                (1.4826 * pl.col("buffer_mad")).alias("buffer_mad_adjusted"),
            ]
        )
        .with_columns(
            [
                (pl.col("buffer_median") + 3 * pl.col("buffer_mad_adjusted")).alias("LOD"),
            ]
        )
        .select([protein_id_col, "LOD"])
    )

    result = qc_instance.get_data().join(lod_stats, on=protein_id_col, how="left")

    n_analytes = lod_stats.select(pl.len()).collect().item()
    logger.info(f"calculated eLOD for {n_analytes} analytes")

    return result


# Group keys of the negative-control LOD summary.
NC_LOD_KEYS: tuple[str, ...] = ("OlinkID", "DataAnalysisRefID")


def nc_lod_stats(
    lod_data: pl.LazyFrame, sample_id_col: str, min_count_threshold: int
) -> pl.LazyFrame:
    """LOD per assay and DataAnalysisRefID from the negative controls.

    As OlinkAnalyze ``olink_nc_lod``: ``LODNPX`` is the median plus the larger of 0.2
    and 3 SD of the controls' PCNormalizedNPX, ``LODCount`` the larger of
    ``min_count_threshold`` and twice their highest count, and ``LODMethod`` says
    which one applies. The controls are sorted by assay, plate, well and value before
    the aggregation and the summary by assay, so its values and its row order are the
    same whatever order the controls arrive in.

    Args:
        lod_data: Passing negative-control assay rows with ``OlinkID``,
            ``DataAnalysisRefID``, ``PlateID``, ``sample_id_col``, ``Count`` and
            ``PCNormalizedNPX``.
        sample_id_col: Sample ID column (the control's well label).
        min_count_threshold: Count threshold that decides between the NPX-based and
            the count-based LOD.

    Returns:
        One row per ``NC_LOD_KEYS`` pair, sorted by them, with ``LODNPX``,
        ``LODCount`` and ``LODMethod``.
    """
    keys = list(NC_LOD_KEYS)
    # The SD is a floating-point sum whose last bits follow the order it adds the rows
    # in, and a hash group_by returns the groups in a different order on every run;
    # sorting the input and keeping its order makes both a function of the data.
    return (
        lod_data.sort([*keys, "PlateID", sample_id_col, "PCNormalizedNPX"])
        .group_by(keys, maintain_order=True)
        .agg(
            [
                pl.col("Count").max().alias("MaxCount"),
                pl.col("PCNormalizedNPX").median().alias("MedianNPX"),
                pl.col("PCNormalizedNPX").std().alias("StdNPX"),
            ]
        )
        .with_columns(
            [
                # LODNPX is the median plus the larger of 0.2 and 3 SD.
                (
                    pl.col("MedianNPX") + pl.max_horizontal(pl.lit(0.2), pl.col("StdNPX").mul(3.0))
                ).alias("LODNPX"),
                # LODCount is the larger of the threshold and twice MaxCount.
                pl.max_horizontal(pl.lit(min_count_threshold), pl.col("MaxCount").mul(2)).alias(
                    "LODCount"
                ),
                # LODMethod is "lod_npx" above the count threshold, else "lod_count".
                pl.when(pl.col("MaxCount") > min_count_threshold)
                .then(pl.lit("lod_npx"))
                .otherwise(pl.lit("lod_count"))
                .alias("LODMethod"),
            ]
        )
        .select([*keys, "LODNPX", "LODCount", "LODMethod"])
        .sort(keys)
    )


def olink_lod(
    qc_instance,
    lod_method: str = "NCLOD",
    lod_file_path: str | None = None,
    min_count_threshold: int = 150,
    min_num_nc: int = 10,
) -> pl.LazyFrame:
    """Calculates the limit of detection for Olink data.

    Follows OlinkAnalyze ``olink_lod_internal``: LODNPX and LODCount come from the
    negative controls (``NCLOD``) or the Olink fixed-LOD file (``FixedLOD``), are
    converted to ``PCNormalizedLOD``, and to the ``NPX``-scale LOD (``NPXLOD``,
    written as ``LOD`` in the output files). Both are produced whatever
    ``qc_instance.data_col`` is; ``LOD`` holds the one that pairs with ``data_col``,
    which is what detection compares with.

    Args:
        qc_instance: OlinkQC instance.
        lod_method: ``"NCLOD"`` or ``"FixedLOD"`` (``"Both"`` is not implemented).
        lod_file_path: Path to the fixed-LOD CSV file (required for ``FixedLOD``).
        min_count_threshold: Count threshold that decides between the NPX-based and
            the count-based LOD.
        min_num_nc: Minimum number of negative controls required for ``NCLOD``.

    Returns:
        LazyFrame of SAMPLE assay rows with the sample ID, protein ID, ``PlateID``,
            ``data_col``, ``LOD`` (on the ``data_col`` scale), ``NPX``,
            ``PCNormalizedNPX``, ``NPXLOD`` (on the ``NPX`` scale) and
            ``PCNormalizedLOD`` (on the ``PCNormalizedNPX`` scale).

    Raises:
        ValueError: If ``lod_method`` is unknown, a required input is missing, or
            there are too few negative controls or plate controls.
        NotImplementedError: If ``lod_method`` is ``"Both"``.
    """
    logger.info(f"calculating Olink LOD ({lod_method})...")

    valid_methods = ["NCLOD", "FixedLOD", "Both"]
    if lod_method not in valid_methods:
        raise ValueError(f"lod_method must be one of {valid_methods}")
    if lod_method == "Both":
        raise NotImplementedError(
            "lod_method='Both' is not implemented: OlinkAnalyze returns separate "
            "NCLOD and FixedLOD columns for it, and detection here reads a single "
            "LOD column. Use 'NCLOD' or 'FixedLOD'."
        )

    if lod_method == "FixedLOD" and lod_file_path is None:
        raise ValueError(f"lod_file_path is required when lod_method is '{lod_method}'")

    data = qc_instance.get_data()
    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    data_col = qc_instance.data_col
    if data_col not in OLINK_VALUE_COLUMNS:
        raise ValueError(
            f"Olink LOD is defined for data_col {' or '.join(map(repr, OLINK_VALUE_COLUMNS))}, got '{data_col}'"
        )
    # The output has NPX and PCNormalizedNPX, each with its own LOD. The NPX-scale LOD needs
    # the Normalization column; data without it raise an error.
    missing = [
        c
        for c in (OLINK_NPX, OLINK_PC_NPX, "Normalization")
        if c not in data.collect_schema().names()
    ]
    if missing:
        raise ValueError(
            f"Olink data lacks the column(s) {missing}: the LOD is written on both the NPX and the PCNormalizedNPX scale"
        )

    if lod_method == "NCLOD":
        # Passing negative controls only.
        lod_data = data.filter(
            (pl.col("SampleType") == "NEGATIVE_CONTROL")
            & (pl.col("SampleQC") == "PASS")
            & (pl.col("AssayType") == "assay")
            & pl.col(data_col).is_not_null()
            & ~pl.col(data_col).is_nan()
        )

        # Count unique (sample ID, PlateID) pairs.
        n_nc = (
            lod_data.select([sample_id_col, "PlateID"])
            .unique()
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        )
        if n_nc < min_num_nc:
            raise ValueError(
                f"At least {min_num_nc} Negative Controls are required to calculate LOD from Negative Controls. Found {n_nc}."
            )

        lod_stats = nc_lod_stats(lod_data, sample_id_col, min_count_threshold)
    else:  # FixedLOD, as OlinkAnalyze ``olink_fixed_lod``
        lod_file = pl.read_csv(lod_file_path, separator=";")
        dar_ids = (
            data.select("DataAnalysisRefID")
            .unique()
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )
        lod_stats = (
            lod_file.filter(pl.col("DataAnalysisRefID").is_in(dar_ids))
            .select(["OlinkID", "DataAnalysisRefID", "LODNPX", "LODCount", "LODMethod"])
            .lazy()
        )

    # Plate-control median and extension-control counts, as OlinkAnalyze
    # ``pc_norm_count``.
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
        raise ValueError("Insufficient Plate Control data for normalization of LOD.")

    # PCNormalizedLOD is on the PCNormalizedNPX scale.
    result = (
        data.join(
            lod_stats,
            on=[protein_id_col, "DataAnalysisRefID"],
            how="left",
            validate="m:1",
        )
        .join(
            pc_median,
            on=[protein_id_col, "PlateID"],
            how="left",
            coalesce=True,
            validate="m:1",
        )
        .join(
            ext_count,
            on=[sample_id_col, "WellID", "Panel", "Block", "SampleType", "PlateID"],
            how="left",
            coalesce=True,
        )
        .with_columns(
            [
                pl.when(pl.col(data_col).is_null() | pl.col(data_col).is_nan())
                .then(None)
                .when(pl.col("LODMethod") == "lod_npx")
                .then(pl.col("LODNPX"))
                .otherwise((pl.col("LODCount") / pl.col("ExtCount")).log(2) - pl.col("PCMedian"))
                .alias(OLINK_PC_LOD),
            ]
        )
    )

    # The LOD on both scales, whatever data_col is. OlinkAnalyze returns two LODs:
    #   - PCNormalizedLOD, on the PCNormalizedNPX scale for every Normalization value,
    #     so it pairs with PCNormalizedNPX;
    #   - LOD, which pairs with NPX and is NPXLOD here: for Normalization ==
    #     "Intensity" it is PCNormalizedLOD minus the plate median of SAMPLE
    #     PCNormalizedNPX (intensity normalization subtracts that median from NPX);
    #     for "Plate control" it is PCNormalizedLOD; otherwise NA.
    # ``LOD`` is the one that pairs with data_col, the column detection compares with.
    plate_median_npx = (
        result.filter(
            (pl.col("SampleType") == "SAMPLE")
            & pl.col(OLINK_PC_NPX).is_not_null()
            & ~pl.col(OLINK_PC_NPX).is_nan()
        )
        .group_by([protein_id_col, "PlateID"])
        .agg(pl.col(OLINK_PC_NPX).median().alias("PlateMedianNPX"))
    )
    result = (
        result.join(
            plate_median_npx,
            on=[protein_id_col, "PlateID"],
            how="left",
            coalesce=True,
            validate="m:1",
        )
        .with_columns(
            pl.when(pl.col("Normalization") == "Intensity")
            .then(pl.col(OLINK_PC_LOD) - pl.col("PlateMedianNPX"))
            .when(pl.col("Normalization") == "Plate control")
            .then(pl.col(OLINK_PC_LOD))
            .otherwise(None)
            .alias(OLINK_NPX_LOD)
        )
        .with_columns(pl.col(OLINK_LOD_IN_MEMORY[data_col]).alias("LOD"))
    )

    # Keep SAMPLE rows of assay type and the essential columns only.
    result = result.filter(
        (pl.col("SampleType") == "SAMPLE") & (pl.col("AssayType") == "assay")
    ).select(
        list(
            dict.fromkeys(
                [
                    sample_id_col,
                    protein_id_col,
                    "PlateID",
                    data_col,
                    "LOD",
                    OLINK_NPX,
                    OLINK_PC_NPX,
                    OLINK_NPX_LOD,
                    OLINK_PC_LOD,
                ]
            )
        )
    )

    n_analytes = lod_stats.select(pl.col("OlinkID").n_unique()).collect(engine="streaming").item()
    source = lod_file_path if lod_method == "FixedLOD" else "negative controls"
    logger.info(
        f"calculated LOD for {n_analytes} analytes using {lod_method} ({source}) "
        f"on both the NPX and PCNormalizedNPX scales (detection reads the {data_col} one)"
    )
    logger.info("filtered to SAMPLE rows only (removed controls) and kept essential columns")

    return qc_instance._checkpoint(result, "lod")


def plate_cv(
    qc_clean: pl.LazyFrame,
    protein_id_col: str,
    plate_id_col: str,
    data_col: str,
    formula: str,
) -> pl.DataFrame:
    """Computes the intra- and inter-plate CV per protein.

    Intra-plate CV is the median across plates of each plate's replicate CV.
    Inter-plate CV is the CV (same formula) of per-plate means across plates; it is
    null when there is only one plate.

    The float aggregates (mean, sd) run over rows in a fixed order, so the result is
    bit-identical for any input row order and any thread count. The datapoints are
    sorted on (protein, plate, value) and aggregated per plate eagerly; a lazy sort in
    front of an order-insensitive aggregate is dropped by the optimizer. The per-plate
    frame is in (protein, plate) order, and the aggregates across plates read it in
    that order.

    Args:
        qc_clean: Control datapoints with no null or NaN values.
        protein_id_col: Protein ID column.
        plate_id_col: Plate ID column.
        data_col: Measurement column.
        formula: ``"standard"`` (sd/mean) or ``"olink"`` (log2 formula).

    Returns:
        One row per protein with ``Intra_CV``, ``n_plates`` and ``Inter_CV``.
    """
    ln2 = np.log(2) if formula == "olink" else None

    plate_stats = (
        qc_clean.sort([protein_id_col, plate_id_col, data_col])
        .collect()
        .group_by([protein_id_col, plate_id_col], maintain_order=True)
        .agg(
            [
                pl.col(data_col).mean().alias("mean_data"),
                pl.col(data_col).std().alias("sd_data"),
                pl.col(data_col).count().alias("n_replicates"),
            ]
        )
        .lazy()
    )

    if formula == "standard":
        intra = (
            plate_stats.filter(
                pl.col("sd_data").is_not_null()
                & ~pl.col("sd_data").is_nan()
                & (pl.col("sd_data") > 0)
                & (pl.col("mean_data") > 0)
                & (pl.col("n_replicates") >= 2)
            )
            .with_columns([((pl.col("sd_data") / pl.col("mean_data")) * 100).alias("plate_cv")])
            .group_by(protein_id_col, maintain_order=True)
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
            plate_stats.filter(
                pl.col("sd_data").is_not_null()
                & ~pl.col("sd_data").is_nan()
                & (pl.col("sd_data") > 0)
                & (pl.col("n_replicates") >= 2)
            )
            .with_columns(
                [(((ln2 * pl.col("sd_data")).pow(2)).exp() - 1).sqrt().mul(100).alias("plate_cv")]
            )
            .group_by(protein_id_col, maintain_order=True)
            .agg(
                [
                    pl.col("plate_cv").median().alias("Intra_CV"),
                    pl.col("plate_cv").count().alias("n_plates"),
                ]
            )
            .filter(pl.col("n_plates") >= 1)
        )

    n_plates = qc_clean.select(pl.col(plate_id_col).n_unique()).collect().item()

    if n_plates >= 2:
        if formula == "standard":
            inter = (
                plate_stats.filter(pl.col("n_replicates") >= 1)
                .group_by(protein_id_col, maintain_order=True)
                .agg(
                    [
                        pl.col("mean_data").mean().alias("grand_mean"),
                        pl.col("mean_data").std().alias("sd_plate_means"),
                        pl.col("mean_data").count().alias("n_plates"),
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
                    [((pl.col("sd_plate_means") / pl.col("grand_mean")) * 100).alias("Inter_CV")]
                )
                .select([protein_id_col, "Inter_CV"])
            )
        else:  # olink
            inter = (
                plate_stats.filter(pl.col("n_replicates") >= 1)
                .group_by(protein_id_col, maintain_order=True)
                .agg(
                    [
                        pl.col("mean_data").std().alias("sd_plate_means"),
                        pl.col("mean_data").count().alias("n_plates"),
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
        return intra.join(inter, on=protein_id_col, how="left").collect()

    return intra.with_columns([pl.lit(None).cast(pl.Float64).alias("Inter_CV")]).collect()


def assay_cv(
    qc_instance,
    qc_sample_type: str | None = None,
    intra_threshold: float = 25.0,
    inter_threshold: float = 25.0,
) -> dict:
    """Calculates the intra- and inter-plate CV of the QC samples.

    Works for both SomaScan and Olink; the platform sets the CV formula, the plate
    column and the QC sample type. SomaScan uses the standard formula (sd/mean * 100).
    Olink uses its log2 formula (sqrt(exp((ln(2) * sd)^2) - 1) * 100) on the
    ``SAMPLE_CONTROL`` samples, scoring only real assays (``AssayType == "assay"``) and
    PASS control datapoints.

    Sets ``cv_data``, ``_cv_stats``, ``_cv_pass_proteins`` and ``_cv_fail_proteins``
    on the instance; a protein fails when its intra-plate CV exceeds the threshold.
    Nothing is flagged here, since detection combines the two.

    Args:
        qc_instance: SomaQC or OlinkQC instance.
        qc_sample_type: SampleType value for QC samples (default: the instance's).
        intra_threshold: Intra-plate CV above which a protein fails.
        inter_threshold: Inter-plate CV above which a protein counts as high.

    Returns:
        Dictionary of CV statistics (``cv_df``, medians, high counts, thresholds and
            pass/fail counts).

    Raises:
        ValueError: If there are no control samples or no assay has a CV.
    """
    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    plate_id_col = qc_instance.plate_id_col

    if qc_sample_type is None:
        qc_sample_type = qc_instance.qc_sample_type

    logger.info(f"calculating CV using {qc_sample_type} samples...")

    # CV on the native scale (``raw_col``): SomaScan RFU; Olink, the QC data column.
    data_col = qc_instance.raw_col
    cv_formula = qc_instance.platform.cv_formula

    # SomaScan ADAT data has _sample_idx; Olink parquet data does not.
    has_sample_idx = "_sample_idx" in qc_instance.get_data().collect_schema().names()

    if has_sample_idx:
        # SomaScan: QC, Calibrator and Buffer replicates share a SampleId, so join on
        # _sample_idx.
        join_cols = ["_sample_idx", sample_id_col, "SampleType", plate_id_col]
        join_on = "_sample_idx"
    else:
        # Olink: join on the sample ID.
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
        raise ValueError(f"the CV step found no control samples with SampleType = '{qc_sample_type}'")

    # Olink: SampleQC is read per datapoint (sample and block); only PASS control
    # datapoints enter the CV, as for the negative controls of the LOD. The internal
    # control assays (ext_ctrl, inc_ctrl, amp_ctrl) are not proteins and stay out.
    if qc_instance.platform.name == "olink":
        qc = qc.filter((pl.col("SampleQC") == "PASS") & (pl.col("AssayType") == "assay"))

    if qc_instance.failed_samples:
        qc = qc.filter(~pl.col(sample_id_col).is_in(qc_instance.failed_samples))
    if qc_instance.failed_proteins:
        qc = qc.filter(~pl.col(protein_id_col).is_in(qc_instance.failed_proteins))

    # Drop null/NaN values, then collect once for the counts and plate_cv below.
    qc_clean = (
        qc.filter(pl.col(data_col).is_not_null() & ~pl.col(data_col).is_nan()).collect().lazy()
    )

    unique_id_col = "_sample_idx" if has_sample_idx else sample_id_col

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

    logger.info(f"found {n_qc_samples} {qc_sample_type} samples across {n_plates} plates")

    if n_plates < 2:
        # Intra-plate CV is still computed with several samples per plate.
        logger.warning("need ≥2 plates for inter-plate CV calculation")

    cv_df = plate_cv(qc_clean, protein_id_col, plate_id_col, data_col, cv_formula)

    if len(cv_df) == 0:
        raise ValueError("the CV step computed no CV: no assay has two replicate controls on a plate")

    med_intra = cv_df["Intra_CV"].median()
    med_inter = cv_df["Inter_CV"].median() if n_plates >= 2 else None
    high_intra = (cv_df["Intra_CV"] > intra_threshold).sum()
    high_inter = (cv_df["Inter_CV"] > inter_threshold).sum() if n_plates >= 2 else 0

    logger.info(
        f"Median CV: Intra={med_intra:.2f}% | Inter={(med_inter if med_inter is not None else 0):.2f}% | "
        f"High: {high_intra}/{high_inter}"
    )

    qc_instance.cv_data = cv_df

    # CV pass/fail is combined with detection later.
    cv_pass_proteins = cv_df.filter(pl.col("Intra_CV") <= intra_threshold)[protein_id_col].to_list()
    cv_fail_proteins = cv_df.filter(pl.col("Intra_CV") > intra_threshold)[protein_id_col].to_list()

    logger.info(
        f"CV status: {len(cv_pass_proteins)} proteins ≤{intra_threshold}% (pass), "
        f"{len(cv_fail_proteins)} proteins >{intra_threshold}% (fail)"
    )

    qc_instance._cv_pass_proteins = cv_pass_proteins
    qc_instance._cv_fail_proteins = cv_fail_proteins

    qc_instance._cv_stats = cv_df.select([protein_id_col, "Intra_CV", "Inter_CV"])

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


def _det_rate(det: str, tot: str) -> pl.Expr:
    """Builds ``det / tot``, null when no row had a LOD (a null rate fails)."""
    return pl.when(pl.col(tot) > 0).then(pl.col(det) / pl.col(tot)).otherwise(None)


def assay_detection(
    qc_instance,
    sample_threshold: float = 0.2,
    use_group_criteria: bool = True,
) -> pl.DataFrame:
    """Flags proteins detected in fewer than ``sample_threshold`` of samples.

    A sample detects a protein when its value is at least the LOD. Works for both
    SomaScan and Olink; the LOD is computed first if it is not yet in the data. Rows
    without a LOD are left out of the denominator.

    With groups available and ``use_group_criteria``, a protein passes when its
    detection rate reaches the threshold in at least one group (bridge samples
    excluded). Otherwise the rate is taken over all samples.

    On Olink, a detection failure is combined with the CV result: both failing is
    ``FAIL_Intra_CV_and_Detection_Rate``, one failing is a WARN, and a detection
    failure with no CV row is ``FAIL_Low_Detection``. SomaScan only records the sets.

    Args:
        qc_instance: SomaQC or OlinkQC instance.
        sample_threshold: Minimum detection rate to keep a protein.
        use_group_criteria: Whether to use the group-aware rule when groups exist.

    Returns:
        Dictionary with the detection statistics (``det_stats``), the proteins to
            remove, pass/fail counts, the CV-detection join and the combined counts.
    """
    logger.info("calculating detection rates...")

    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col

    # Compute the LOD if it is not in the data yet, reusing a cached result.
    if not qc_instance._has_column("LOD"):
        result = getattr(qc_instance, "lod_results", None)
        if result is None:
            if qc_instance.platform.name == "olink":
                result = olink_lod(qc_instance)
            else:
                result = qc_instance._checkpoint(soma_lod(qc_instance), "lod")
        qc_instance.data_full = result
        qc_instance._schema = result.collect_schema()

    use_groups = False
    if qc_instance.has_groups and use_group_criteria and qc_instance.group_data is not None:
        n_unique = (
            qc_instance.group_data.filter(pl.col(qc_instance.group_col).is_not_null())
            .select(pl.col(qc_instance.group_col).n_unique())
            .collect(engine="streaming")
            .item()
        )

        if n_unique > 1:
            use_groups = True
            logger.info(f"using group-aware detection with {n_unique} groups")

    # Restrict to study samples. On Olink, data_full is already SAMPLE rows after
    # olink_lod; on SomaScan it still holds Buffer, Calibrator and QC rows, which
    # share SampleIds among themselves, so join on _sample_idx when present.
    samples_data = qc_instance.get_data()
    sample_key = (
        "_sample_idx" if "_sample_idx" in samples_data.collect_schema().names() else sample_id_col
    )
    samples_data = samples_data.join(
        qc_instance.samples.select(sample_key).unique(), on=sample_key, how="semi"
    )

    if use_groups:
        samples_data = samples_data.join(qc_instance.group_data, on=sample_id_col, how="left")

    if qc_instance.failed_samples:
        samples_data = samples_data.filter(~pl.col(sample_id_col).is_in(qc_instance.failed_samples))
    if qc_instance.failed_proteins:
        samples_data = samples_data.filter(
            ~pl.col(protein_id_col).is_in(qc_instance.failed_proteins)
        )

    # Detection compares ``detection_col`` with the LOD: SomaScan, the delivered non-ANML RFU
    # with the buffer-well eLOD; Olink, the QC data column with its paired LOD. A NaN value is not detected
    # (NaN >= LOD is True in Polars). Rows without a LOD (null or NaN) cannot be judged and are
    # left out of the denominator.
    detection_col = qc_instance.detection_col
    has_lod = pl.col("LOD").is_not_null() & pl.col("LOD").is_not_nan()
    base_query = samples_data.with_columns(
        [
            (
                pl.col(detection_col).is_not_null()
                & pl.col(detection_col).is_not_nan()
                & (pl.col(detection_col) >= pl.col("LOD"))
            ).alias("det"),
            has_lod.alias("has_lod"),
        ]
    )
    n_no_lod = base_query.select((~pl.col("has_lod")).sum()).collect(engine="streaming").item()
    if n_no_lod:
        logger.info(
            f"{n_no_lod} sample-assay rows have no LOD and are excluded from detection rates"
        )

    if use_groups:
        # Bridge samples come from the manifest's BridgeSample flag, with a fallback for
        # USUBJID == "BRIDGE_*". A null group is not a bridge indicator.
        bridge_sample_ids = qc_instance.bridge_sample_ids

        base_query = base_query.with_columns(
            [
                pl.when(pl.col(sample_id_col).cast(pl.Utf8).is_in(bridge_sample_ids))
                .then(pl.lit("bridge_samples"))
                .when(pl.col(qc_instance.group_col).is_null())
                .then(pl.lit("Unknown"))
                .otherwise(pl.col(qc_instance.group_col))
                .alias("group_for_stats")
            ]
        )
        group_by_cols = [protein_id_col, "group_for_stats"]
    else:
        group_by_cols = [protein_id_col]

    agg_exprs = [
        pl.col("det").filter(pl.col("has_lod")).sum().alias("Det"),
        pl.col("has_lod").sum().alias("Tot"),
    ]

    # The median LOD per assay, kept in the detection statistics.
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
        .with_columns([_det_rate("Det", "Tot").alias("Det_Rate")])
        .collect(engine="streaming")
    )

    protein_names = qc_instance.analyte_metadata.select(
        [protein_id_col, qc_instance.protein_name_col]
    ).collect(engine="streaming")
    det_stats = det_stats.join(protein_names, on=protein_id_col, how="left")

    # With groups, add the overall detection rate over all non-bridge samples.
    if use_groups and "group_for_stats" in det_stats.columns:
        overall_det_rate = (
            base_query.filter(pl.col("group_for_stats") != "bridge_samples")
            .group_by([protein_id_col])
            .agg(
                [
                    pl.col("det").filter(pl.col("has_lod")).sum().alias("Det_Overall"),
                    pl.col("has_lod").sum().alias("Tot_Overall"),
                ]
            )
            .with_columns([_det_rate("Det_Overall", "Tot_Overall").alias("Overall_Det_Rate")])
            .select([protein_id_col, "Overall_Det_Rate"])
            .collect(engine="streaming")
        )
        det_stats = det_stats.join(overall_det_rate, on=protein_id_col, how="left")

    if use_groups:
        # Group-aware: PASS if at least one group passes.
        det_stats_for_filtering = det_stats.filter(pl.col("group_for_stats") != "bridge_samples")

        proteins_passing_any_group = set(
            det_stats_for_filtering.filter(pl.col("Det_Rate") >= sample_threshold)[protein_id_col]
            .unique()
            .to_list()
        )

        all_proteins = set(det_stats[protein_id_col].unique().to_list())
        to_remove = list(all_proteins - proteins_passing_any_group)

        logger.info(
            f"Detection status (group-aware): {len(proteins_passing_any_group)} proteins "
            f"≥{sample_threshold * 100:.0f}% in ≥1 group (pass), {len(to_remove)} proteins "
            f"<{sample_threshold * 100:.0f}% in all groups (fail)"
        )
    else:
        proteins_passing = set(
            det_stats.filter(pl.col("Det_Rate") >= sample_threshold)[protein_id_col].to_list()
        )
        all_proteins = set(det_stats[protein_id_col].unique().to_list())
        to_remove = list(all_proteins - proteins_passing)

        logger.info(
            f"Detection status: {len(proteins_passing)} proteins ≥{sample_threshold * 100:.0f}% (pass), "
            f"{len(to_remove)} proteins <{sample_threshold * 100:.0f}% (fail)"
        )

    detection_pass_proteins = list(all_proteins - set(to_remove))
    detection_fail_proteins = to_remove

    # Read by finalize_assay_flags_soma.
    qc_instance._detection_fail_proteins = set(detection_fail_proteins)

    cv_pass = set(getattr(qc_instance, "_cv_pass_proteins", []))
    cv_fail = set(getattr(qc_instance, "_cv_fail_proteins", []))
    has_cv_data = len(cv_pass) > 0 or len(cv_fail) > 0
    olink = qc_instance.platform.name == "olink"

    det_pass = set(detection_pass_proteins)
    det_fail = set(detection_fail_proteins)

    if has_cv_data:
        # PASS = both pass, WARN = one passes, FAIL = both fail.
        proteins_pass_both = cv_pass & det_pass
        proteins_fail_both = cv_fail & det_fail
        proteins_warn = (cv_pass & det_fail) | (cv_fail & det_pass)

        logger.info(
            f"combined CV + detection: {len(proteins_pass_both)} pass, {len(proteins_warn)} warn, {len(proteins_fail_both)} fail"
        )

        # Olink only: the SomaScan consolidation rebuilds every protein flag from the
        # check sets in finalize_assay_flags_soma.
        if olink:
            if proteins_fail_both:
                qc_instance._flag_proteins(
                    list(proteins_fail_both), FAIL_INTRA_CV_AND_DETECTION_RATE
                )

            if proteins_warn:
                qc_instance.warned_proteins.extend(list(proteins_warn))
                warn_cv_pass_det_fail = list(cv_pass & det_fail)
                warn_cv_fail_det_pass = list(cv_fail & det_pass)
                if warn_cv_pass_det_fail:
                    qc_instance._track_protein_qc(warn_cv_pass_det_fail, WARN_DETECTION_RATE)
                if warn_cv_fail_det_pass:
                    qc_instance._track_protein_qc(warn_cv_fail_det_pass, WARN_INTRA_CV)

        # A detection-failing protein with no CV row (null control values, fewer than
        # two replicates, zero SD) has no CV verdict to combine with: per protein, as
        # with no CV data.
        no_cv_det_fail = det_fail - cv_pass - cv_fail
        if olink and no_cv_det_fail:
            logger.warning(
                f"marking {len(no_cv_det_fail)} proteins with no CV row as FAIL (low detection)"
            )
            qc_instance._flag_proteins(sorted(no_cv_det_fail), FAIL_LOW_DETECTION)

        n_overlap_cv_detection = len(proteins_fail_both)
        combined_counts = {
            "n_pass_both": len(proteins_pass_both),
            "n_warn_combined": len(proteins_warn),
            "n_fail_both": len(proteins_fail_both),
        }
    else:
        # No CV data: just mark detection failures.
        if olink and to_remove:
            logger.warning(f"marking {len(to_remove)} proteins as FAIL (low detection)")
            qc_instance._flag_proteins(to_remove, FAIL_LOW_DETECTION)

        n_overlap_cv_detection = 0
        combined_counts = {}

    cv_det = None
    if qc_instance.cv_data is not None and len(qc_instance.cv_data) > 0:
        # With groups, the overall rate over all samples; otherwise Det_Rate.
        det_rate_col = "Overall_Det_Rate" if use_groups else "Det_Rate"

        # One row per protein.
        det_for_cv = det_stats.select(
            [protein_id_col, qc_instance.protein_name_col, det_rate_col]
        ).unique(subset=[protein_id_col])

        if det_rate_col == "Overall_Det_Rate":
            det_for_cv = det_for_cv.rename({"Overall_Det_Rate": "Det_Rate"})

        cv_det = qc_instance.cv_data.join(
            det_for_cv,
            on=protein_id_col,
            how="inner",
        )
        if len(cv_det) == 0:
            cv_det = None

    # Summary export: with groups, one detection-rate column per group.
    if use_groups and "group_for_stats" in det_stats.columns:
        det_stats_for_pivot = det_stats.filter(pl.col("group_for_stats") != "bridge_samples")

        det_stats_overall = (
            base_query.filter(pl.col("group_for_stats") != "bridge_samples")
            .group_by([protein_id_col])
            .agg(
                [
                    pl.col("det").filter(pl.col("has_lod")).sum().alias("Det"),
                    pl.col("has_lod").sum().alias("Tot"),
                ]
            )
            .with_columns([_det_rate("Det", "Tot").alias("Det_Rate")])
            .select([protein_id_col, pl.col("Det_Rate").alias("Detection_Rate_All")])
            .collect()
        )

        unique_groups = det_stats_for_pivot["group_for_stats"].unique().sort().to_list()

        if unique_groups:
            det_stats_summary = det_stats_overall

            for group in unique_groups:
                group_det = det_stats_for_pivot.filter(pl.col("group_for_stats") == group).select(
                    [
                        protein_id_col,
                        pl.col("Det_Rate").alias(f"Detection_Rate_{group}"),
                    ]
                )
                det_stats_summary = det_stats_summary.join(group_det, on=protein_id_col, how="left")

            qc_instance._detection_stats = det_stats_summary
        else:
            # No groups left after dropping bridge samples.
            qc_instance._detection_stats = det_stats_overall.rename(
                {"Detection_Rate_All": "Detection_Rate"}
            )
    else:
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
        **combined_counts,
    }
