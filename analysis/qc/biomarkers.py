"""Biomarker correlation analysis for validating proteomic measurements."""

from __future__ import annotations

# Standard library
import logging

# Third-party
import polars as pl
from scipy.stats import linregress, spearmanr

# Local imports
from .utils import ensure_lazy

logger = logging.getLogger(__name__)


def _extract_visitnum_from_avisit(avisit_col: pl.Expr) -> pl.Expr:
    """Extract visit number from AVISIT column.

    Handles visit strings like:
    - "BASELINE" → 0
    - "VISIT 6 (WEEK 12)" → 6
    - "UNSCHEDULED" → -1
    - "CORE MAXIMUM" → -2
    - "CORE_EXTENSION MAXIMUM" → -3

    Args:
        avisit_col: Polars expression for AVISIT column

    Returns:
        Polars expression with extracted visit numbers
    """
    return avisit_col.str.to_lowercase().map_elements(
        lambda x: _parse_avisit_to_visitnum(x) if x else None, return_dtype=pl.Int64
    )


def _parse_avisit_to_visitnum(avisit_str: str) -> int:
    """Parse AVISIT string to visit number.

    Args:
        avisit_str: AVISIT value (lowercase)

    Returns:
        Visit number as integer
    """
    avisit_lower = avisit_str.lower().strip()

    # Handle BASELINE
    if "baseline" in avisit_lower:
        return 0

    # Handle "VISIT X" pattern (e.g., "VISIT 6 (WEEK 12)")
    import re

    match = re.search(r"visit\s+(\d+)", avisit_lower)
    if match:
        return int(match.group(1))

    # Handle special visit types
    if "unscheduled" in avisit_lower:
        return -1
    if "core maximum" in avisit_lower and "extension" not in avisit_lower:
        return -2
    if "core_extension maximum" in avisit_lower or "extension maximum" in avisit_lower:
        return -3

    # If no pattern matches, return None
    return None


def assay_biomarker_correlation(
    qc_instance,
    lab_data: pl.DataFrame | pl.LazyFrame,
    protein_biomarker_map: dict[str, str | list[str]] | str,
    lab_test_name: str | None = None,
    visit_col: str = "VISITNUM",
    lab_test_col: str = "LBTEST",
    lab_value_col: str = "LBSTRESN",
    lab_unit_col: str = "LBSTRESU",
) -> dict:
    """
    Check correlation between protein levels and measured biomarkers at different timepoints.

    Platform-agnostic function that works for both SomaScan and Olink data.
    Uses qc_instance attributes to infer platform-specific column names.

    Args:
        qc_instance: SomaQC or OlinkQC instance
        lab_data: DataFrame or LazyFrame with lab measurements (e.g., adlb)
        protein_biomarker_map: Either:
            - Dict mapping protein names to lab test name(s) (e.g., {"CST3": "Cystatin C", "INS-CPEPTIDE": ["C-peptide", "Insulin"]})
            - Single protein name as string (requires lab_test_name parameter)
        lab_test_name: Lab test name (required only if protein_biomarker_map is a string)
        visit_col: Column name for visit number in both adsl_merged and lab_data
        lab_test_col: Column name for lab test name in lab_data
        lab_value_col: Column name for lab test value in lab_data
        lab_unit_col: Column name for lab test unit in lab_data

    Note:
        Uses qc_instance.adsl_merged which includes VISITNUM and ADT from manifest merged with ADSL data
        Uses qc_instance.usubjid_col for the subject ID column name

    Returns:
        Dictionary with correlation statistics and plots (keyed by protein name if multiple pairs)
    """
    sample_id_col = qc_instance.sample_id_col
    protein_id_col = qc_instance.protein_id_col
    data_col = qc_instance.data_col

    # Infer platform from protein_id_col.
    if protein_id_col == "SeqId":
        is_somascan = True
        protein_name_col = "Target"
    elif protein_id_col == "OlinkID":
        is_somascan = False
        protein_name_col = "Assay"
    else:
        raise ValueError(
            f"Unsupported platform: protein_id_col='{protein_id_col}'. Expected 'SeqId' or 'OlinkID'."
        )

    # Allow a single protein passed as a string (with lab_test_name).
    if isinstance(protein_biomarker_map, str):
        if lab_test_name is None:
            raise ValueError(
                "lab_test_name must be provided when protein_biomarker_map is a string"
            )
        protein_biomarker_map = {protein_biomarker_map: lab_test_name}

    # Normalize all values to lists for consistent handling.
    normalized_map = {}
    for protein_id, lab_tests in protein_biomarker_map.items():
        if isinstance(lab_tests, str):
            normalized_map[protein_id] = [lab_tests]
        else:
            normalized_map[protein_id] = lab_tests

    # Expand to individual protein-lab test pairs for separate analysis.
    expanded_pairs = []
    for protein_id, lab_tests in normalized_map.items():
        for lab_test in lab_tests:
            expanded_pairs.append((protein_id, lab_test))

    # If single pair, return single result; if multiple, return dict of results.
    is_single_pair = len(expanded_pairs) == 1

    # Print section header once for all pairs.
    pair_names = ", ".join([f"{p} vs {b}" for p, b in expanded_pairs])
    qc_instance._print_section(f"BIOMARKER CORRELATION: {pair_names}")

    # Create a mapping from protein_id to protein_name for display purposes
    protein_id_to_name = (
        qc_instance.analyte_metadata.select([protein_id_col, protein_name_col])
        .unique()
        .collect()
        .to_dict(as_series=False)
    )
    protein_id_to_name_dict = dict(
        zip(protein_id_to_name[protein_id_col], protein_id_to_name[protein_name_col])
    )

    # Check that adsl_merged is available
    if qc_instance.adsl_merged is None:
        raise ValueError(
            "adsl_merged not available. Initialize QC instance with adsl and manifest parameters."
        )

    lab_lazy = ensure_lazy(lab_data)

    # Detect available columns and use fallbacks if defaults don't exist
    lab_schema = lab_lazy.collect_schema().names()

    # Check lab test column
    if lab_test_col not in lab_schema:
        if "PARAM" in lab_schema:
            logger.info(f"Column '{lab_test_col}' not found, using 'PARAM' instead")
            lab_test_col = "PARAM"
        else:
            raise ValueError(
                f"Cannot find '{lab_test_col}' or 'PARAM' in lab_data columns"
            )

    # Check lab value column
    if lab_value_col not in lab_schema:
        if "AVAL" in lab_schema:
            logger.info(f"Column '{lab_value_col}' not found, using 'AVAL' instead")
            lab_value_col = "AVAL"
        else:
            raise ValueError(
                f"Cannot find '{lab_value_col}' or 'AVAL' in lab_data columns"
            )

    # Check lab unit column - extract from PARAM if needed
    if lab_unit_col not in lab_schema:
        if "PARAM" in lab_schema:
            logger.info(
                f"Column '{lab_unit_col}' not found, will extract units from 'PARAM'"
            )
            lab_unit_col = None  # Signal that we'll extract from PARAM
        else:
            logger.warning(
                f"Cannot find '{lab_unit_col}', 'LBSTRESU' or 'PARAM' - units will be empty"
            )
            lab_unit_col = None

    all_biomarker_names = [lab_test for _, lab_test in expanded_pairs]

    logger.info(
        f"Filtering and collecting lab data for {len(all_biomarker_names)} biomarkers"
    )

    # Read ADLB files once with early filters
    all_protein_ids = list(
        dict.fromkeys([protein_id for protein_id, _ in expanded_pairs])
    )
    all_lab_tests = list(dict.fromkeys([lab for _, lab in expanded_pairs]))

    # Check if visit_col exists in adsl_merged, otherwise use AVISIT
    adsl_schema = qc_instance.adsl_merged.collect_schema().names()
    if visit_col not in adsl_schema:
        if "AVISIT" in adsl_schema:
            logger.info(
                f"Column '{visit_col}' not found in adsl_merged, using 'AVISIT' instead"
            )
            visit_col = "AVISIT"
            use_avisit = True
        else:
            raise ValueError(
                f"Cannot find '{visit_col}' or 'AVISIT' in adsl_merged columns"
            )
    else:
        use_avisit = False

    # Collect adsl_merged with visit information
    adsl_merged_with_visit = qc_instance.adsl_merged.select(
        [sample_id_col, qc_instance.usubjid_col, visit_col]
    ).collect()

    # If using AVISIT, extract visit numbers
    if use_avisit:
        adsl_merged_with_visit = adsl_merged_with_visit.with_columns(
            _extract_visitnum_from_avisit(pl.col("AVISIT")).alias("VISITNUM")
        )
        visit_col = "VISITNUM"
    else:
        adsl_merged_with_visit = adsl_merged_with_visit.with_columns(
            [pl.col(visit_col).cast(pl.Int64)]
        )

    # Collect ADLB with only necessary filters (minimize SAS read)
    # Check if visit_col exists in lab_data, fallback to AVISIT if not
    lab_visit_col = visit_col
    if visit_col not in lab_schema:
        if "AVISIT" in lab_schema:
            logger.info(
                f"Column '{visit_col}' not found in lab_data, using 'AVISIT' instead"
            )
            lab_visit_col = "AVISIT"
            use_avisit = True
        else:
            raise ValueError(
                f"Cannot find '{visit_col}' or 'AVISIT' in lab_data columns"
            )

    select_cols = [
        qc_instance.usubjid_col,
        lab_visit_col,
        lab_test_col,
        lab_value_col,
    ]

    # Add unit column to select if available, otherwise add PARAM for extraction
    if lab_unit_col is not None:
        select_cols.append(lab_unit_col)
    elif "PARAM" in lab_schema and "PARAM" not in select_cols:
        select_cols.append("PARAM")

    lab_data_collected = (
        lab_lazy.filter(pl.col(lab_test_col).is_in(all_biomarker_names))
        .select(select_cols)
        .collect(engine="streaming")  # Stream large SAS file
    )

    # If using AVISIT in lab_data, extract visit numbers
    if use_avisit and lab_visit_col == "AVISIT":
        lab_data_collected = lab_data_collected.with_columns(
            _extract_visitnum_from_avisit(pl.col("AVISIT")).alias("VISITNUM")
        ).drop("AVISIT")
        lab_visit_col = "VISITNUM"
    else:
        lab_data_collected = lab_data_collected.with_columns(
            [pl.col(lab_visit_col).cast(pl.Int64)]
        )

    # Extract units from PARAM if needed
    if lab_unit_col is None and "PARAM" in lab_data_collected.columns:
        lab_data_collected = lab_data_collected.with_columns(
            pl.col("PARAM").str.extract(r"\(([^)]+)\)$", 1).alias("LBSTRESU")
        )
        lab_unit_col = "LBSTRESU"
        logger.info("Extracted units from PARAM column")

    logger.info(f"collected {lab_data_collected.height} lab measurements from ADLB")

    # Filter to relevant patient-visits and aggregate (in-memory, fast)
    relevant_patient_visits = adsl_merged_with_visit.select(
        [qc_instance.usubjid_col, visit_col]
    ).unique()

    lab_data_collected = (
        lab_data_collected.join(
            relevant_patient_visits,
            on=[qc_instance.usubjid_col, lab_visit_col],
            how="inner",
        )
        .group_by([qc_instance.usubjid_col, lab_visit_col, lab_test_col, lab_unit_col])
        .agg(pl.col(lab_value_col).mean().alias(lab_value_col))
    )

    logger.info(f"filtered to {lab_data_collected.height} relevant lab measurements")

    logger.info(f"preparing protein data for {len(all_protein_ids)} proteins")

    # Collect protein data once - use data_full to include all proteins even if filtered by QC
    # We want to check biomarker correlations for specific proteins regardless of QC status
    # But filter to biological samples only (not controls)
    # Join with analyte metadata to get protein names (Target column)
    protein_data_base = qc_instance.samples.join(
        qc_instance.analyte_metadata.select([protein_id_col, protein_name_col]),
        on=protein_id_col,
        how="left",
    ).filter(pl.col(protein_id_col).is_in(all_protein_ids))

    if qc_instance.failed_samples:
        protein_data_base = protein_data_base.filter(
            ~pl.col(sample_id_col).is_in(qc_instance.failed_samples)
        )
        logger.info(f"excluding {len(qc_instance.failed_samples)} failed samples")

    protein_data_collected = (
        protein_data_base.select(
            [sample_id_col, protein_id_col, protein_name_col, data_col]
        )
        .join(
            adsl_merged_with_visit.select(
                [sample_id_col, qc_instance.usubjid_col, visit_col]
            ).lazy(),
            on=sample_id_col,
            how="inner",
        )
        .collect()
    )

    logger.info(f"collected {protein_data_collected.height} protein measurements")

    logger.info(f"analyzing {len(expanded_pairs)} protein-biomarker pairs")

    all_results = {}
    all_pair_summaries = []

    for i, (protein_id, biomarker_name) in enumerate(expanded_pairs):
        result_key = f"{protein_id}_vs_{biomarker_name}"
        protein_name = protein_id_to_name_dict.get(protein_id, protein_id)

        # Filter pre-collected data - much faster than lazy collect
        logger.debug(f"Processing {protein_id} ({protein_name}) vs {biomarker_name}")
        protein_subset = protein_data_collected.filter(
            pl.col(protein_id_col) == protein_id
        )
        lab_subset = lab_data_collected.filter(pl.col(lab_test_col) == biomarker_name)

        result = _check_single_biomarker_correlation(
            qc_instance=qc_instance,
            lab_data_prefiltered=lab_subset,
            protein_data_prefiltered=protein_subset,
            protein_id=protein_id,
            protein_name=protein_name,
            lab_test_name=biomarker_name,
            sample_id_col=sample_id_col,
            protein_id_col=protein_id_col,
            protein_name_col=protein_name_col,
            data_col=data_col,
            usubjid_col=qc_instance.usubjid_col,
            visit_col=lab_visit_col,
            lab_test_col=lab_test_col,
            lab_value_col=lab_value_col,
            lab_unit_col=lab_unit_col,
            add_to_report=False,
        )
        all_results[result_key] = result

        if result["n_matched_observations"] > 0 and result.get("plot_data") is not None:
            all_pair_summaries.append(
                {
                    "protein_id": protein_id,
                    "protein_name": protein_name,
                    "biomarker": biomarker_name,
                    "n_matched_observations": result["n_matched_observations"],
                    "n_unique_samples": result["n_unique_samples"],
                    "n_unique_participants": result["n_unique_participants"],
                    "spearman_r": result["spearman_r"],
                    "spearman_p": result["spearman_p"],
                    "slope": result["slope"],
                    "intercept": result["intercept"],
                }
            )

    return {
        "all_results": all_results,
        "all_pair_summaries": all_pair_summaries,
        "is_single_pair": is_single_pair,
    }


def _check_single_biomarker_correlation(
    qc_instance,
    lab_data_prefiltered: pl.DataFrame,
    protein_data_prefiltered: pl.DataFrame,
    protein_id: str,
    protein_name: str,
    lab_test_name: str,
    sample_id_col: str,
    protein_id_col: str,
    protein_name_col: str,
    data_col: str,
    usubjid_col: str,
    visit_col: str,
    lab_test_col: str,
    lab_value_col: str,
    lab_unit_col: str,
    add_to_report: bool = True,
) -> dict:
    """Internal function to check correlation for a single protein-biomarker pair.

    Optimized version that accepts pre-filtered/pre-joined data to avoid redundant operations.
    """
    if add_to_report:
        qc_instance._print_section(
            f"BIOMARKER CORRELATION: {protein_id} ({protein_name}) vs {lab_test_name}"
        )
    else:
        logger.info(f"analyzing {protein_id} ({protein_name}) vs {lab_test_name}")

    protein_data = protein_data_prefiltered.filter(pl.col(protein_id_col) == protein_id)

    if protein_data.height == 0:
        logger.warning(
            f"Protein '{protein_id}' ({protein_name}) not found in proteomics data - skipping"
        )
        return {
            "n_matched_observations": 0,
            "n_unique_samples": 0,
            "n_unique_participants": 0,
            "spearman_r": None,
            "spearman_p": None,
            "r_squared": None,
            "slope": None,
            "intercept": None,
            "merged_data": pl.DataFrame(),
            "plot_data": None,
        }

    lab_filtered = lab_data_prefiltered.filter(pl.col(lab_test_col) == lab_test_name)
    if lab_filtered.height == 0:
        logger.warning(f"Biomarker '{lab_test_name}' not found in lab data - skipping")
        return {
            "n_matched_observations": 0,
            "n_unique_samples": 0,
            "n_unique_participants": 0,
            "spearman_r": None,
            "spearman_p": None,
            "r_squared": None,
            "slope": None,
            "intercept": None,
            "merged_data": pl.DataFrame(),
            "plot_data": None,
        }

    merged_data = protein_data.join(
        lab_filtered,
        on=[usubjid_col, visit_col],
        how="inner",
    ).drop_nulls([lab_value_col, data_col])

    if merged_data.height == 0:
        logger.warning(
            f"No matching samples found between {protein_id} ({protein_name}) and {lab_test_name}"
        )
        return {
            "n_matched_observations": 0,
            "n_unique_samples": 0,
            "n_unique_participants": 0,
            "spearman_r": None,
            "spearman_p": None,
            "r_squared": None,
            "slope": None,
            "intercept": None,
            "merged_data": pl.DataFrame(),
            "plot_data": None,
        }

    # Filter out NaN values and ensure positive lab values for log transformation
    valid_data = merged_data.filter(
        ~pl.col(data_col).is_nan()
        & ~pl.col(lab_value_col).is_nan()
        & (pl.col(lab_value_col) > 0)
    )

    n_valid = valid_data.height

    if n_valid < 10:
        logger.warning(
            f"Insufficient valid paired values for correlation: {n_valid}/{merged_data.height}"
        )
        return {
            "n_matched_observations": merged_data.height,
            "n_unique_samples": (
                merged_data[sample_id_col].n_unique() if merged_data.height > 0 else 0
            ),
            "n_unique_participants": (
                merged_data[usubjid_col].n_unique() if merged_data.height > 0 else 0
            ),
            "spearman_r": None,
            "spearman_p": None,
            "r_squared": None,
            "slope": None,
            "intercept": None,
            "merged_data": merged_data,
            "plot_data": None,
        }

    valid_data = valid_data.with_columns(
        pl.col(lab_value_col).log(base=10).alias("lab_log")
    )

    protein_valid = valid_data[data_col].to_numpy()
    lab_valid_log = valid_data["lab_log"].to_numpy()

    spearman_r, spearman_p = spearmanr(protein_valid, lab_valid_log)
    slope, intercept, r_value, p_value, std_err = linregress(
        protein_valid, lab_valid_log
    )

    units = merged_data[lab_unit_col].unique().to_list()
    unit_str = units[0] if len(units) == 1 else "/".join([str(u) for u in units])

    data_label = data_col

    plot_data = {
        "protein_id": protein_id,
        "protein_name": protein_name,
        "lab_test_name": lab_test_name,
        "merged_data": merged_data,
        "protein_valid": protein_valid,
        "lab_valid_log": lab_valid_log,
        "slope": slope,
        "intercept": intercept,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "lab_value_col": lab_value_col,
        "unit_str": unit_str,
        "data_col": data_col,  # Include for axis labels
        "sample_id_col": sample_id_col,
        "data_label": data_label,  # Include for display labels
    }

    if add_to_report:
        logger.info(f"spearman ρ={spearman_r:.3f} (p={spearman_p:.2e})")
        logger.info(
            f"Matched: {merged_data[usubjid_col].n_unique()} participants, {merged_data[sample_id_col].n_unique()} samples"
        )
    else:
        logger.info(
            f"{protein_id} ({protein_name}): Spearman ρ={spearman_r:.3f}, {merged_data[usubjid_col].n_unique()} participants, {merged_data[sample_id_col].n_unique()} samples"
        )

    return {
        "n_matched_observations": merged_data.height,
        "n_unique_samples": merged_data[sample_id_col].n_unique(),
        "n_unique_participants": merged_data[usubjid_col].n_unique(),
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "r_squared": r_value**2,
        "slope": slope,
        "intercept": intercept,
        "merged_data": merged_data,
        "plot_data": plot_data,
    }
