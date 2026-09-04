"""Core Proteomics QC classes for SomaScan and Olink data processing."""

from __future__ import annotations

import glob
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from . import quality_control
from . import detection
from . import outliers
from . import sex_concordance
from . import age_concordance
from . import biomarkers
from . import clinical
from . import utils

warnings.filterwarnings("ignore")

# Configure logging to show info messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProteomicsQC:
    """
    Base QC pipeline for proteomics data (SomaScan and Olink).

    This abstract base class provides common functionality for both platforms,
    including clinical data merging, QC tracking, outlier detection, concordance checks,
    and report generation.

    Platform-specific implementations (SomaQC, OlinkQC) should:
    1. Set platform-specific column mappings (sample_id_col, protein_id_col, data_col, etc.)
    2. Load data and create sample_metadata + analyte_metadata tables
    3. Bind platform-specific QC methods
    """

    COLORS = {
        "primary": "#D31710",
        "blue": "#003A6C",
        "azure": "#86B3F2",
        "coral": "#FD9485",
        "gold": "#F4C003",
        "green": "#00422C",
        "grey": "#818C94",
        "black": "#000000",
        "rose": "#F9EEED",
        "cream": "#F9F0E0",
    }

    def __init__(
        self,
        project_name: str,
        save_prefix: str | None,
        adsl: pl.DataFrame | pl.LazyFrame | None,
        manifest: pl.DataFrame | pl.LazyFrame | None,
        usubjid_col: str,
        clinical_vars: list[str] | None,
        force_categorical: list[str] | None,
        group_col: str | list[str] | None,
        random_seed: int,
        data_col: str | None = None,
    ):
        """
        Initialize the base QC pipeline.

        This base __init__ should be called by subclasses after they have:
        1. Set platform-specific column mappings
        2. Loaded data into self.data_full
        3. Created self.sample_metadata and self.analyte_metadata

        Args:
            project_name: Name of the project for reporting
            save_prefix: Prefix for saving intermediate result files
            adsl: Phenotype/clinical data for concordance checks and analysis
            manifest: Sample ID mapping manifest
            usubjid_col: Column name for unique subject ID in manifest and adsl
            clinical_vars: List of column names from adsl/manifest to include
            force_categorical: List of column names to treat as categorical
            group_col: Column name(s) for grouping
            random_seed: Random seed for reproducibility
            data_col: Override the default data column name (e.g., 'PCNormalizedNPX' for Olink,
                      'RFU' for SomaScan). If None, uses the platform default.
        """
        self.project_name = project_name
        self.save_prefix = save_prefix
        self.random_seed = random_seed

        # Override data_col if provided
        if data_col is not None:
            self.data_col = data_col

        # Phenotype/clinical configuration (set by subclass before calling super().__init__)
        self.adsl = adsl
        self.manifest = manifest
        self.usubjid_col = usubjid_col
        self.clinical_vars = clinical_vars
        self.force_categorical = force_categorical

        # Cache for imputed data
        self._imputed_data_cache = None
        self._imputed_sample_ids = None

        # These must be set by subclass BEFORE calling super().__init__:
        # - self.data_full (LazyFrame)
        # - self.sample_metadata (LazyFrame)
        # - self.analyte_metadata (LazyFrame)
        # - self.sample_id_col (str)
        # - self.protein_id_col (str)
        # - self.protein_name_col (str)
        # - self.data_col (str)
        # - self.uniprot_col (str)
        # - self.sample_type (str)

        # Get schema
        self._schema = self.data_full.collect_schema()
        logger.info(
            f"Data shape: {self.data_full.select(pl.len()).collect().item():,} rows"
        )

        # Partition data by sample type
        self._partition_by_sample_type()

        # Pre-merge ADSL data with sample IDs
        self._merge_clinical_data()

        # Store group information
        self._setup_groups(group_col)

        # QC tracking lists
        self.failed_samples: list[str] = []
        self.failed_proteins: list[str] = []
        self.warned_samples: list[str] = []
        self.warned_proteins: list[str] = []

        # Efficient QC tracking using DataFrames
        self.sample_qc_actions: list[pl.DataFrame] = []
        self.protein_qc_actions: list[pl.DataFrame] = []

        # Report components
        self.plots: list[Any] = []
        self.text_sections: list[str] = []
        self.section_plot_counts: list[int] = []

        # Get summary counts
        counts = self._get_counts()
        self.qc_results: dict[str, Any] = {
            k: counts[k]
            for k in [
                "n_samples",
                "n_proteins",
                "n_plates",
                "n_blocks",
                "n_calibrator",
                "n_qc",
                "n_buffer",
                "n_pc",
                "n_sc",
                "n_nc",
            ]
            if k in counts
        }

        # Cached data for later use
        self.cv_data: pl.DataFrame | None = None
        self.pca_data: Any | None = None

        # Add header and print summary
        self._add_header(counts)

    def _partition_by_sample_type(self):
        """Partition data by sample type (platform-agnostic)."""
        metadata_schema = self.sample_metadata.collect_schema()
        analyte_schema = self.analyte_metadata.collect_schema()

        if "SampleType" in metadata_schema.names():
            # Get unique sample IDs for the target sample type only
            sample_type_ids = (
                self.sample_metadata.filter(pl.col("SampleType") == self.sample_type)
                .select(self.sample_id_col)
                .unique()
            )

            # Semi-join to keep only rows matching these sample IDs
            self.samples = self.data_full.join(
                sample_type_ids,
                on=self.sample_id_col,
                how="semi",
            )
            logger.debug(
                f"Partitioned {self.sample_type} samples from data_full using semi-join"
            )
        elif "SampleType" in self._schema.names():
            # SampleType already in data_full
            self.samples = self.data_full.filter(
                pl.col("SampleType") == self.sample_type
            )
            logger.debug(f"Partitioned {self.sample_type} samples from data_full")
        else:
            # If no SampleType column, assume all rows are samples
            self.samples = self.data_full
            logger.debug("No SampleType column found - treating all data as samples")

        # Additionally filter to AssayType == "assay" for platforms that have this column (Olink)
        if "AssayType" in analyte_schema.names():
            assay_ids = (
                self.analyte_metadata.filter(pl.col("AssayType") == "assay")
                .select(self.protein_id_col)
                .unique()
            )
            self.samples = self.samples.join(
                assay_ids,
                on=self.protein_id_col,
                how="semi",
            )
            logger.debug(
                f"Further filtered samples to AssayType == 'assay' using semi-join"
            )

    def _merge_clinical_data(self):
        """Pre-merge ADSL data with sample IDs."""
        self.adsl_merged = None
        self.clinical_vars_safe = {}

        if self.adsl is None or self.manifest is None:
            return

        adsl_lazy = self._ensure_lazy(self.adsl)
        manifest_lazy = self._ensure_lazy(self.manifest)

        # Filter ADSL columns to only those needed
        if self.clinical_vars is not None:
            adsl_cols_needed = [self.usubjid_col]
            for col_name in self.clinical_vars:
                if col_name not in adsl_cols_needed:
                    adsl_cols_needed.append(col_name)

            adsl_cols_present = adsl_lazy.collect_schema().names()
            adsl_cols_to_select = [
                c for c in adsl_cols_needed if c in adsl_cols_present
            ]
            adsl_lazy = adsl_lazy.select(adsl_cols_to_select)

        # Join manifest with ADSL via subject ID mapping
        manifest_schema = manifest_lazy.collect_schema()
        manifest_cols = [self.sample_id_col, self.usubjid_col]

        if "VISITNUM" in manifest_schema.names():
            manifest_cols.append("VISITNUM")
        if "ADT" in manifest_schema.names():
            manifest_cols.append("ADT")

        adsl_with_samples = manifest_lazy.select(manifest_cols).join(
            adsl_lazy,
            on=self.usubjid_col,
            how="left",
        )

        # Select and rename clinical variables if specified
        if self.clinical_vars is not None:
            adsl_schema = adsl_with_samples.collect_schema()
            select_exprs = [self.sample_id_col, self.usubjid_col]

            # Always include visit columns from manifest if present
            if "VISITNUM" in adsl_schema.names():
                select_exprs.append("VISITNUM")
            if "ADT" in adsl_schema.names():
                select_exprs.append("ADT")

            for col_name in self.clinical_vars:
                self.clinical_vars_safe[col_name] = col_name

                # Check if column exists in adsl first, then manifest
                if col_name in adsl_schema.names():
                    select_exprs.append(pl.col(col_name))
                elif col_name in manifest_schema.names():
                    logger.info(
                        f"Clinical variable '{col_name}' found in manifest (not in adsl)"
                    )
                    # Need to re-join with manifest to get this column
                    before_rejoin = (
                        adsl_with_samples.select(self.sample_id_col).collect().height
                    )
                    adsl_with_samples = adsl_with_samples.join(
                        manifest_lazy.select([self.sample_id_col, col_name]),
                        on=self.sample_id_col,
                        how="left",
                    )
                    after_rejoin = (
                        adsl_with_samples.select(self.sample_id_col).collect().height
                    )
                    if after_rejoin != before_rejoin:
                        logger.warning(
                            f"Re-join for '{col_name}' changed row count: {before_rejoin} -> {after_rejoin}"
                        )
                    select_exprs.append(pl.col(col_name))
                else:
                    logger.warning(
                        f"Clinical variable '{col_name}' not found in adsl or manifest, using null"
                    )
                    select_exprs.append(pl.lit(None).alias(col_name))

            self.adsl_merged = adsl_with_samples.select(select_exprs).cache()
            logger.info(
                f"Clinical variables merged: {len(self.clinical_vars_safe)} variables"
            )
        else:
            # No clinical vars - select essential columns plus visit info
            adsl_schema = adsl_with_samples.collect_schema()
            select_exprs = [self.sample_id_col, self.usubjid_col]

            # Include standard visit columns from manifest if present
            if "VISITNUM" in adsl_schema.names():
                select_exprs.append("VISITNUM")
            if "ADT" in adsl_schema.names():
                select_exprs.append("ADT")

            self.adsl_merged = adsl_with_samples.select(select_exprs).cache()
            logger.info("ADSL data merged: essential columns only")

    def _standardize_group_name(self, value: str) -> str:
        """
        Standardize group/treatment names to consistent format.

        Converts spaces to hyphens, handles dose numbers by:
        - Replacing '/' with '_'
        - Removing spaces between dose components and units
        - Replacing spaces after dose units with underscores

        Examples:
        - "LAA 3/6/9 mg" -> "LAA-3_6_9mg"
        - "LAA 9 mg" -> "LAA-9mg"
        - "TZP 10mg" -> "TZP-10mg"
        - "SEMA 2.4mg or MTD" -> "SEMA-2.4mg_or_MTD"
        - "Placebo" -> "Placebo"
        """
        import re

        if value is None or value == "":
            return value

        # Replace "/" with "_" in dose ranges (e.g., "3/6/9" -> "3_6_9")
        value = value.replace("/", "_")

        # Replace spaces followed by digits with hyphen + digits (e.g., " 3" -> "-3")
        # This handles "LAA 9 mg" -> "LAA-9mg"
        value = re.sub(r"\s+(\d)", r"-\1", value)

        # Replace spaces after dose units (mg, mcg, ug, g) with underscores
        # This handles "2.4mg or MTD" -> "2.4mg_or_MTD"
        value = re.sub(r"(mg|mcg|ug|g)\s+", r"\1_", value)

        # Remove any remaining spaces
        value = value.replace(" ", "")

        return value

    def _setup_groups(self, group_col):
        """Setup group information (platform-agnostic)."""
        self.group_col_input = group_col
        self.group_col = None
        self.group_data = None
        self.has_groups = False

        if group_col is None or self.adsl_merged is None:
            return

        # Convert to list if string
        group_cols = [group_col] if isinstance(group_col, str) else group_col

        # Check all columns exist in adsl_merged
        schema_names = self.adsl_merged.collect_schema().names()
        missing_cols = [col for col in group_cols if col not in schema_names]

        if missing_cols:
            logger.warning(
                f"Group column(s) {missing_cols} not found in adsl/clinical_vars. Skipping group analysis."
            )
        else:
            if len(group_cols) == 1:
                # Single column - standardize group names
                self.group_col = group_cols[0]
                self.group_data = self.adsl_merged.select(
                    [
                        self.sample_id_col,
                        pl.col(self.group_col)
                        .cast(pl.Utf8)
                        .map_elements(
                            self._standardize_group_name, return_dtype=pl.Utf8
                        )
                        .alias(self.group_col),
                    ]
                )
                self.has_groups = True
                logger.info(f"Group column loaded: {self.group_col}")
            else:
                # Multiple columns - create composite column
                self.group_col = "_".join(group_cols)

                # Create composite column by concatenating with underscore separator
                self.group_data = self.adsl_merged.select(
                    [
                        self.sample_id_col,
                        pl.concat_str(
                            [
                                pl.col(c)
                                .cast(pl.Utf8)
                                .map_elements(
                                    self._standardize_group_name, return_dtype=pl.Utf8
                                )
                                for c in group_cols
                            ],
                            separator="_",
                            ignore_nulls=False,
                        ).alias(self.group_col),
                    ]
                )
                self.has_groups = True
                logger.info(
                    f"Group columns combined: {' + '.join(group_cols)} -> {self.group_col}"
                )

    def _ensure_lazy(self, df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
        """Convert DataFrame to LazyFrame if needed."""
        if isinstance(df, pl.DataFrame):
            return df.lazy()
        return df

    def _get_counts(self) -> dict[str, int]:
        """Collect all counts in one operation."""
        # Check if _sample_idx exists (SomaScan ADAT format)
        data_schema = self.data_full.collect_schema().names()
        has_sample_idx = "_sample_idx" in data_schema
        sample_schema = self.sample_metadata.collect_schema().names()

        # Check if we have AssayType column (Olink)
        has_assay_type = "AssayType" in self.analyte_metadata.collect_schema().names()

        # Use platform-specific plate column if set, otherwise detect
        plate_col = getattr(self, "plate_id_col", None)
        if plate_col and plate_col not in sample_schema:
            plate_col = None

        # Get all unique SampleType values if available from data_full or sample_metadata
        if "SampleType" in data_schema:
            # Olink: SampleType is in data_full
            sample_types = (
                self.data_full.select("SampleType")
                .unique()
                .collect()["SampleType"]
                .to_list()
            )
            # Filter out the main sample type and None
            control_types = [st for st in sample_types if st and st != self.sample_type]

            # Create mapping for Olink control types to abbreviated keys
            control_type_map = {
                "PLATE_CONTROL": "pc",
                "SAMPLE_CONTROL": "sc",
                "NEGATIVE_CONTROL": "nc",
            }
        elif "SampleType" in sample_schema:
            # SomaScan: SampleType is in sample_metadata
            sample_types = (
                self.sample_metadata.select("SampleType")
                .unique()
                .collect()["SampleType"]
                .to_list()
            )
            # Filter out the main sample type and None
            control_types = [st for st in sample_types if st and st != self.sample_type]
            control_type_map = {}
        else:
            control_types = []
            control_type_map = {}

        # For SomaScan with _sample_idx, count from data_full joined with sample_metadata
        if has_sample_idx and "SampleType" in sample_schema:
            # Join data_full with sample_metadata to get SampleType for each _sample_idx
            data_with_type = (
                self.data_full.select(["_sample_idx", self.sample_id_col])
                .unique()
                .join(
                    self.sample_metadata.select(
                        ["_sample_idx", self.sample_id_col, "SampleType"]
                        + ([plate_col] if plate_col else [])
                    ),
                    on="_sample_idx",
                    how="left",
                )
            )

            # Build count expressions
            count_exprs = [
                # Primary samples - count unique _sample_idx
                pl.col("_sample_idx")
                .filter(pl.col("SampleType") == self.sample_type)
                .n_unique()
                .alias("n_samples"),
            ]

            # For each control type, count unique (_sample_idx, PlateId) pairs
            for control_type in control_types:
                # Use mapped abbreviation if available, otherwise create from name
                if control_type in control_type_map:
                    key = f"n_{control_type_map[control_type]}"
                else:
                    # Create safe key name (lowercase, replace spaces/hyphens)
                    key = (
                        f"n_{control_type.lower().replace(' ', '_').replace('-', '_')}"
                    )

                if plate_col:
                    count_exprs.append(
                        pl.struct(["_sample_idx", plate_col])
                        .filter(pl.col("SampleType") == control_type)
                        .n_unique()
                        .alias(key)
                    )
                else:
                    count_exprs.append(
                        pl.col("_sample_idx")
                        .filter(pl.col("SampleType") == control_type)
                        .n_unique()
                        .alias(key)
                    )

            # Add plate count
            if plate_col:
                count_exprs.append(pl.col(plate_col).n_unique().alias("n_plates"))
            else:
                count_exprs.append(pl.lit(0).alias("n_plates"))

            # Add protein count (filter for AssayType=="assay" if available)
            if has_assay_type:
                protein_count_data = self.data_full.join(
                    self.analyte_metadata.select([self.protein_id_col, "AssayType"]),
                    on=self.protein_id_col,
                    how="left",
                ).filter(pl.col("AssayType") == "assay")
            else:
                protein_count_data = self.data_full

            combined_counts = (
                data_with_type.join(
                    protein_count_data.select([self.protein_id_col]).unique(),
                    how="cross",
                )
                .select(
                    count_exprs
                    + [pl.col(self.protein_id_col).n_unique().alias("n_proteins")]
                )
                .collect()
                .row(0, named=True)
            )
            counts = combined_counts
            counts["n_blocks"] = 0  # SomaScan doesn't use blocks
        else:
            # For Olink, count from data_full directly (not just sample_metadata)
            # This ensures we count all controls that appear in the actual data

            # Count samples by type from data_full
            count_exprs = []

            # Count biological samples
            count_exprs.append(
                pl.col(self.sample_id_col)
                .filter(pl.col("SampleType") == self.sample_type)
                .n_unique()
                .alias("n_samples")
            )

            # For each control type, count unique (SampleID, PlateID) pairs from data_full
            for control_type in control_types:
                # Use mapped abbreviation if available, otherwise create from name
                if control_type in control_type_map:
                    key = f"n_{control_type_map[control_type]}"
                else:
                    # Create safe key name (lowercase, replace spaces/hyphens)
                    key = (
                        f"n_{control_type.lower().replace(' ', '_').replace('-', '_')}"
                    )

                if plate_col:
                    count_exprs.append(
                        pl.struct([self.sample_id_col, plate_col])
                        .filter(pl.col("SampleType") == control_type)
                        .n_unique()
                        .alias(key)
                    )
                else:
                    count_exprs.append(
                        pl.col(self.sample_id_col)
                        .filter(pl.col("SampleType") == control_type)
                        .n_unique()
                        .alias(key)
                    )

            # Add plate and block counts from data_full
            if plate_col:
                count_exprs.append(pl.col(plate_col).n_unique().alias("n_plates"))
            else:
                count_exprs.append(pl.lit(0).alias("n_plates"))

            if "Block" in data_schema:
                count_exprs.append(pl.col("Block").n_unique().alias("n_blocks"))
            else:
                count_exprs.append(pl.lit(0).alias("n_blocks"))

            # Filter proteins by AssayType=="assay" if available
            if has_assay_type:
                protein_data = self.data_full.join(
                    self.analyte_metadata.select([self.protein_id_col, "AssayType"]),
                    on=self.protein_id_col,
                    how="left",
                ).filter(pl.col("AssayType") == "assay")
            else:
                protein_data = self.data_full

            # Combine all counts in single query
            combined_counts = (
                self.data_full.select(count_exprs)
                .join(protein_data.select([self.protein_id_col]).unique(), how="cross")
                .with_columns(
                    pl.col(self.protein_id_col).n_unique().alias("n_proteins")
                )
                .collect()
                .row(0, named=True)
            )
            counts = combined_counts

        return counts

    def _add_header(self, counts: dict):
        """Log the dataset summary."""
        logger.info(
            "%s: %s samples, %s proteins, %s plates, %s blocks",
            self.project_name,
            counts.get("n_samples", 0),
            counts.get("n_proteins", 0),
            counts.get("n_plates", 0),
            counts.get("n_blocks", 0),
        )

    def _print_section(self, title: str):
        """Standard section header for console output."""
        logger.info("")
        logger.info("=" * 80)
        logger.info(title)
        logger.info("=" * 80)

    def _compute_visitage(self):
        """
        Lazily compute visit-adjusted age (visitage) from manifest ADT.
        Computes as: baseline_age + (visit_date - baseline_date) / 365
        where baseline is the visit with lowest VISITNUM for each participant.
        """
        if self.adsl_merged is None:
            logger.warning("Cannot compute visitage: adsl_merged not available")
            return self.adsl_merged

        # Check required columns
        schema_names = self.adsl_merged.collect_schema().names()
        if "AGE" not in schema_names:
            logger.warning("Cannot compute visitage: 'AGE' column not in adsl_merged")
            return self.adsl_merged

        if "ADT" not in schema_names:
            logger.warning(
                "Cannot compute visitage: 'ADT' column not in adsl_merged (from manifest)"
            )
            return self.adsl_merged

        if "VISITNUM" not in schema_names:
            logger.warning(
                "Cannot compute visitage: 'VISITNUM' column not in adsl_merged"
            )
            return self.adsl_merged

        logger.info("Computing visit-adjusted age (visitage) from manifest ADT")

        # Get baseline visit (lowest VISITNUM) and ADT for each participant
        baseline_visits = self.adsl_merged.group_by(self.usubjid_col).agg(
            [
                pl.col("VISITNUM").min().alias("baseline_visitnum"),
            ]
        )

        # Join to get baseline ADT
        baseline_adt = (
            self.adsl_merged.join(
                baseline_visits,
                on=self.usubjid_col,
                how="inner",
            )
            .filter(pl.col("VISITNUM") == pl.col("baseline_visitnum"))
            .select([self.usubjid_col, pl.col("ADT").alias("baseline_adt")])
            .unique(subset=[self.usubjid_col], keep="first")
        )

        # Compute visitage: baseline_age + (current_adt - baseline_adt) / 365
        result = self.adsl_merged.join(
            baseline_adt, on=self.usubjid_col, how="left"
        ).with_columns(
            pl.when(pl.col("baseline_adt").is_not_null() & pl.col("ADT").is_not_null())
            .then(
                pl.col("AGE")
                + (
                    pl.col("ADT").cast(pl.Date) - pl.col("baseline_adt").cast(pl.Date)
                ).dt.total_days()
                / 365.0
            )
            .otherwise(pl.col("AGE"))
            .alias("VISITAGE")
        )

        logger.info("Computed visitage (lazy evaluation)")
        return result

    def _has_column(self, col: str) -> bool:
        """Check if column exists in schema. Refreshes schema if needed."""
        if col in self._schema.names():
            return True
        # Schema might be stale, refresh once
        try:
            self._schema = self.data_full.collect_schema()
            return col in self._schema.names()
        except Exception:
            return False

    def _track_sample_qc(self, sample_ids: list[str], reason: str):
        """Track QC action for samples."""
        if sample_ids:
            qc_df = pl.DataFrame({self.sample_id_col: sample_ids, reason: True})
            self.sample_qc_actions.append(qc_df)

    def _track_protein_qc(self, protein_ids: list[str], reason: str):
        """Track QC action for proteins."""
        if protein_ids:
            qc_df = pl.DataFrame({self.protein_id_col: protein_ids, reason: True})
            self.protein_qc_actions.append(qc_df)

    def get_data(self) -> pl.LazyFrame:
        """Get the current state of the data."""
        return self.data_full

    def collect(self) -> pl.DataFrame:
        """Collect the LazyFrame into a DataFrame."""
        return self.data_full.collect()

    def get_filtered_data(
        self,
        remove_failed_samples: bool = True,
        remove_failed_proteins: bool = True,
    ) -> pl.LazyFrame:
        """
        Get filtered dataset after QC.

        Args:
            remove_failed_samples: Remove failed samples from output
            remove_failed_proteins: Remove failed proteins from output

        Returns:
            Filtered LazyFrame
        """
        filtered = self.data_full

        if remove_failed_samples and self.failed_samples:
            filtered = filtered.filter(
                ~pl.col(self.sample_id_col).is_in(self.failed_samples)
            )

        if remove_failed_proteins and self.failed_proteins:
            filtered = filtered.filter(
                ~pl.col(self.protein_id_col).is_in(self.failed_proteins)
            )

        return filtered

    # Shared QC methods (platform-agnostic)
    def filter_assay_cv(self, **kwargs):
        """Filter assays by coefficient of variation."""
        cv_results = detection.assay_cv(self, **kwargs)
        self.cv_results = cv_results
        return cv_results

    def filter_assay_detection(self, **kwargs):
        """Filter assays by detection rate."""
        det_results = detection.assay_detection(self, **kwargs)
        self.detection_results = det_results
        return self

    def remove_sample_outliers(self, **kwargs):
        """Remove outlier samples using PCA and median/IQR methods."""
        if "random_seed" not in kwargs:
            kwargs["random_seed"] = self.random_seed

        pca_df, pca_obj, pca_outliers = outliers.pca_outliers(self, **kwargs)

        if pca_df.is_empty():
            logger.warning("PCA outlier detection returned no data")
            self.pca_data = (pca_df, pca_obj, None)
            return pca_df, pca_obj, []

        median_iqr_kwargs = {
            k: v for k, v in kwargs.items() if k in ["n_std", "random_seed"]
        }
        median_iqr_df, median_iqr_outliers = outliers.median_iqr_outliers(
            self, **median_iqr_kwargs
        )

        if not median_iqr_df.is_empty():
            pca_df = pca_df.join(
                median_iqr_df.select(
                    [self.sample_id_col, "Z_Median", "Z_IQR", "Is_Median_IQR_Outlier"]
                ),
                on=self.sample_id_col,
                how="left",
            )
        else:
            pca_df = pca_df.with_columns(
                [
                    pl.lit(None).cast(pl.Float64).alias("Z_Median"),
                    pl.lit(None).cast(pl.Float64).alias("Z_IQR"),
                    pl.lit(False).alias("Is_Median_IQR_Outlier"),
                ]
            )

        all_outliers = list(set(pca_outliers) | set(median_iqr_outliers))
        pca_df = pca_df.with_columns(
            pl.Series("Is_Outlier", pca_df[self.sample_id_col].is_in(all_outliers))
        )

        self.qc_results.update({"total_outliers": len(all_outliers)})

        # Join with sample_metadata to get plate ID for coloring PCA plot
        plate_col = getattr(self, "plate_id_col", None)
        if plate_col and plate_col in self.sample_metadata.collect_schema().names():
            plate_data = self.sample_metadata.select(
                [self.sample_id_col, plate_col]
            ).collect()
            pca_df = pca_df.join(
                plate_data,
                on=self.sample_id_col,
                how="left",
                coalesce=True,
            )

        if self.adsl_merged is not None and self.clinical_vars_safe:
            pca_df = pca_df.join(
                self.adsl_merged.collect(),
                on=self.sample_id_col,
                how="left",
                coalesce=True,
            )

        if "force_categorical" not in kwargs:
            kwargs["force_categorical"] = self.force_categorical
        if kwargs["force_categorical"] and self.clinical_vars_safe:
            force_categorical_safe = []
            for col in kwargs["force_categorical"]:
                if col in self.clinical_vars_safe:
                    force_categorical_safe.append(self.clinical_vars_safe[col])
                else:
                    force_categorical_safe.append(col)
            kwargs["force_categorical"] = force_categorical_safe

        self.pca_data = (pca_df, pca_obj, None)

        # Store PCA results for summary export.
        pca_cols = [self.sample_id_col]
        if "PC1_std" in pca_df.columns:
            pca_cols.append("PC1_std")
        if "PC2_std" in pca_df.columns:
            pca_cols.append("PC2_std")
        if "Z_Median" in pca_df.columns:
            pca_cols.append("Z_Median")
        if "Z_IQR" in pca_df.columns:
            pca_cols.append("Z_IQR")
        self._pca_results = pca_df.select(pca_cols)

        logger.info(f"Outlier detection complete: {len(all_outliers)} total outliers")
        return pca_df, pca_obj, all_outliers

    def remove_sample_sex_concordance(self, **kwargs):
        """Check sex concordance between reported and predicted sex."""
        sex_results = sex_concordance.sample_sex_concordance(self, **kwargs)
        self.sex_concordance_results = sex_results

        # Store sex concordance results for summary export.
        if sex_results and "results_df" in sex_results:
            results_df = sex_results["results_df"]
            sex_col = kwargs.get("sex_col", "SEX")
            cols = [self.sample_id_col]
            if sex_col in results_df.columns:
                cols.append(sex_col)
            if "Predicted_Male_Probability" in results_df.columns:
                cols.append("Predicted_Male_Probability")
            self._sex_concordance_results = results_df.select(cols)
            if sex_col in cols:
                self._sex_concordance_results = self._sex_concordance_results.rename(
                    {sex_col: "Reported_Sex"}
                )

        return sex_results

    def remove_sample_age_concordance(self, **kwargs):
        """Check age concordance between reported and predicted age."""
        age_results = age_concordance.sample_age_concordance(self, **kwargs)
        self.age_concordance_results = age_results

        # Store age concordance results for summary export.
        if age_results and "results_df" in age_results:
            results_df = age_results["results_df"]
            cols = [self.sample_id_col]
            age_col = kwargs.get("age_col", "AGE")
            # Use original age column name as it appears in results_df
            if age_col in results_df.columns:
                cols.append(age_col)
            if "Predicted_Age" in results_df.columns:
                cols.append("Predicted_Age")
            if "Std_Predicted_Age" in results_df.columns:
                cols.append("Std_Predicted_Age")
            self._age_concordance_results = results_df.select(cols)
            if age_col in cols and age_col != "Visit_Adj_Age":
                # Rename to Visit_Adj_Age for consistency
                self._age_concordance_results = self._age_concordance_results.rename(
                    {age_col: "Visit_Adj_Age"}
                )

        return age_results

    def remove_sample_unsuitable(self, **kwargs):
        """Remove samples unsuitable for analysis (e.g., post-treatment)."""
        return clinical.sample_unsuitable(self, **kwargs)

    def check_assay_biomarker_correlation(self, **kwargs):
        """Check correlation between protein measurements and clinical biomarkers.

        Args:
            protein_biomarker_map: Dict mapping protein IDs to lab test names.
                                   If not provided, uses DEFAULT_PROTEIN_BIOMARKER_MAP.
            **kwargs: Additional arguments passed to assay_biomarker_correlation
        """
        # Use default protein_biomarker_map if not provided
        if "protein_biomarker_map" not in kwargs:
            if hasattr(self, "DEFAULT_PROTEIN_BIOMARKER_MAP"):
                kwargs["protein_biomarker_map"] = self.DEFAULT_PROTEIN_BIOMARKER_MAP
                logger.info(
                    f"Using default protein_biomarker_map with {len(kwargs['protein_biomarker_map'])} protein-biomarker pairs"
                )
            else:
                raise ValueError(
                    "protein_biomarker_map must be provided or DEFAULT_PROTEIN_BIOMARKER_MAP must be defined"
                )

        biomarker_results = biomarkers.assay_biomarker_correlation(self, **kwargs)
        self.biomarker_correlation_results = biomarker_results
        return biomarker_results

    def generate_report(self, save_prefix: str | None = None):
        """Report generation is intentionally omitted from the public QC runner."""
        raise NotImplementedError(
            "HTML QC report generation is not included in this publication pipeline."
        )

    def save_results(self, save_prefix: str | None = None):
        """Save QC results to disk."""
        utils.save_qc_results(
            self,
            save_prefix or self.save_prefix,
        )

    def __repr__(self) -> str:
        """String representation of the QC object."""
        n_rows = self.data_full.select(pl.len()).collect().item()
        n_cols = len(self._schema)
        return f"{self.__class__.__name__}(project='{self.project_name}', n_rows={n_rows:,}, n_cols={n_cols})"


def _scan_file(path: str | Path) -> pl.LazyFrame:
    """
    Scan a file based on its extension (supports csv, tsv, parquet, and gzipped variants).

    Args:
        path: Path to the file

    Returns:
        LazyFrame with the file contents

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file extension is not supported
    """
    import os

    path_str = str(path)

    if not os.path.exists(path_str):
        raise FileNotFoundError(f"File not found: {path_str}")

    ext = os.path.splitext(path_str)[1].lower()

    if ext == ".parquet":
        return pl.scan_parquet(path_str)
    elif ext == ".csv":
        return pl.scan_csv(path_str, separator=",", has_header=True)
    elif ext == ".tsv":
        return pl.scan_csv(path_str, separator="\t", has_header=True)
    elif ext == ".gz":
        base_ext = os.path.splitext(os.path.splitext(path_str)[0])[1].lower()
        if base_ext == ".csv":
            return pl.scan_csv(path_str, separator=",", has_header=True)
        elif base_ext == ".tsv":
            return pl.scan_csv(path_str, separator="\t", has_header=True)
        else:
            raise ValueError(f"Unsupported compressed file type: {path_str}")
    else:
        raise ValueError(
            f"Unsupported file extension: {ext}. Supported: .parquet, .csv, .tsv, .csv.gz, .tsv.gz"
        )


class OlinkQC(ProteomicsQC):
    """
    QC pipeline for Olink Explore HT proteomics data.

    This class loads Olink NPX data files (parquet, csv, or tsv) and performs QC analysis including:
    - Sample/Assay QC flag filtering
    - Limit of Detection (LOD) calculation
    - CV filtering
    - Detection rate filtering
    - Duplicate OlinkID correlation
    - Bridge normalization (for multi-study)
    - PCA outlier detection
    - Sex/Age concordance checks
    - Biomarker correlation

    Expected columns in data file:
        - SampleID: Sample identifier
        - OlinkID: Protein identifier (unique per reagent)
        - NPX: Normalized Protein eXpression value
        - Count: Read count (for LOD calculation)
        - PCNormalizedNPX: Plate control normalized NPX (for LOD calculation)
        - ExtNPX: Extension control normalized NPX (for LOD calculation)
        - Normalization: Normalization method (Intensity or Plate control, for LOD calculation)
        - SampleType: SAMPLE, PLATE_CONTROL, SAMPLE_CONTROL, NEGATIVE_CONTROL, etc.
        - AssayType: assay, inc_ctrl, amp_ctrl
        - SampleQC: PASS/WARN/FAIL
        - AssayQC: PASS/WARN/FAIL
        - PlateID: Plate identifier
        - Block: Block identifier
        - WellID: Well identifier
        - Assay: Protein name
        - UniProt: UniProt ID
        - Panel: Panel name
        - DataAnalysisRefID: Reagent lot identifier (for LOD calculation)
    """

    # Default protein-biomarker mappings for common clinical analytes
    DEFAULT_PROTEIN_BIOMARKER_MAP = {
        "OID43357": "Alkaline Phosphatase",
        "OID45304": "Amylase",
        "OID45135": "Aspartate Aminotransferase",
        "OID43443": "Calcitonin",
        "OID45334": "Choriogonadotropin Beta",
        "OID45445": "Complement C3a",
        "OID45447": "Complement C5a",
        "OID45073": "Creatine Kinase",
        "OID44543": "Creatine Kinase",
        "OID45345": "Cystatin C",
        "OID43729": "Follicle Stimulating Hormone",
        "OID45132": "Gamma Glutamyl Transferase",
        "OID41280": "Hemoglobin",
        "OID43793": "Hemoglobin",
        "OID44698": "Hemoglobin",
        "OID42585": "Interferon Gamma",
        "OID42608": "Interleukin 1 Beta",
        "OID42595": "Interleukin 10",
        "OID43842": "Interleukin 12",
        "OID43843": "Interleukin 12",
        "OID42600": "Interleukin 13",
        "OID42610": "Interleukin 2",
        "OID42627": "Interleukin 4",
        "OID43856": "Interleukin 6",
        "OID43586": "Interleukin 8",
        "OID44722": ["Insulin", "C-peptide"],
        "OID45398": "Lipase",
        "OID45001": "Thyrotropin",
        "OID44356": "Tryptase",
        "OID43204": "Tumor Necrosis Factor",
    }

    def __init__(
        self,
        npx_path: str | Path | None = None,
        data: pl.DataFrame | pl.LazyFrame | None = None,
        project_name: str = "Olink Explore HT QC",
        save_prefix: str | None = None,
        group_col: str | list[str] | None = None,
        adsl: pl.DataFrame | pl.LazyFrame | None = None,
        manifest: pl.DataFrame | pl.LazyFrame | None = None,
        usubjid_col: str = "USUBJID",
        clinical_vars: list[str] | None = None,
        force_categorical: list[str] | None = None,
        random_seed: int = 0,
        data_col: str | None = None,
    ):
        """
        Initialize the Olink QC pipeline.

        Args:
            npx_path: Path to Olink NPX parquet file (mutually exclusive with data)
            data: Pre-loaded Polars DataFrame/LazyFrame (mutually exclusive with npx_path)
            project_name: Name of the project for reporting
            save_prefix: Prefix for saving intermediate result files
            group_col: Column name(s) for grouping
            adsl: Phenotype/clinical data
            manifest: Sample ID mapping manifest
            usubjid_col: Column name for unique subject ID
            clinical_vars: List of column names from adsl/manifest to include
            force_categorical: List of column names to treat as categorical
            random_seed: Random seed for reproducibility
            data_col: Override the default data column ('NPX'). Use 'PCNormalizedNPX'
                      for plate-control normalized values.
        """
        if npx_path is None and data is None:
            raise ValueError("Either npx_path or data must be provided")
        if npx_path is not None and data is not None:
            raise ValueError("Only one of npx_path or data can be provided")

        # Set platform-specific column mappings
        self.sample_id_col = "SampleID"
        self.protein_id_col = "OlinkID"
        self.protein_name_col = "Assay"
        self.data_col = "NPX"
        self.uniprot_col = "UniProt"
        self.plate_id_col = "PlateID"
        self.sample_type = "SAMPLE"
        self.qc_sample_type = "SAMPLE_CONTROL"

        # Load data
        if npx_path is not None:
            logger.info(f"Loading Olink data from: {npx_path}")
            # Handle glob patterns
            if "*" in str(npx_path):
                matched_files = glob.glob(str(npx_path))
                if not matched_files:
                    raise FileNotFoundError(
                        f"No files found matching pattern: {npx_path}"
                    )
                npx_path = matched_files[0]
                logger.info(f"Matched file: {npx_path}")

            raw_data = _scan_file(npx_path)
        else:
            raw_data = (
                self._ensure_lazy(data) if not isinstance(data, pl.LazyFrame) else data
            )

        # Extract sample_metadata and analyte_metadata to reduce memory
        logger.info("Extracting sample and analyte metadata...")

        # Determine which columns go where
        all_cols = set(raw_data.collect_schema().names())

        # Columns that must stay in data_full (measurements and normalization values)
        data_cols = {self.sample_id_col, self.protein_id_col, self.data_col}
        # Add measurement-related columns
        for col in ["Count", "PCNormalizedNPX", "ExtNPX", "Normalization"]:
            if col in all_cols:
                data_cols.add(col)
        # Keep join keys in data_full (SampleID alone is not unique - need PlateID, Block)
        for col in ["PlateID", "Block", "WellID", "Panel", "SampleType"]:
            if col in all_cols:
                data_cols.add(col)
        # Keep DataAnalysisRefID and AssayType for LOD calculation and filtering
        for col in ["DataAnalysisRefID", "AssayType", "SampleQC"]:
            if col in all_cols:
                data_cols.add(col)

        # Columns for sample_metadata (sample-level attributes)
        sample_cols = {self.sample_id_col}
        for col in ["SampleType", "PlateID", "Block", "SampleQC", "WellID"]:
            if col in all_cols:
                sample_cols.add(col)

        # Columns for analyte_metadata (protein-level attributes)
        analyte_cols = {self.protein_id_col}
        for col in [
            self.protein_name_col,
            self.uniprot_col,
            "AssayType",
            "Panel",
            "AssayQC",
            "DataAnalysisRefID",
        ]:
            if col in all_cols:
                analyte_cols.add(col)

        # Extract metadata tables
        # Otherwise we get multiple rows per SampleID/OlinkID if other attributes differ
        self.sample_metadata = raw_data.select(list(sample_cols)).unique(
            subset=[self.sample_id_col]
        )
        self.analyte_metadata = raw_data.select(list(analyte_cols)).unique(
            subset=[self.protein_id_col]
        )

        # Create minimal data_full with only measurement columns
        self.data_full = raw_data.select(list(data_cols))

        # Reclassify generic "CONTROL" SampleType based on SampleID prefix.
        # Some datasets label all controls as "CONTROL" instead of the expected
        # NEGATIVE_CONTROL / PLATE_CONTROL / SAMPLE_CONTROL values.
        if "SampleType" in sample_cols:
            reclassify_expr = (
                pl.when(
                    (pl.col("SampleType") == "CONTROL")
                    & pl.col(self.sample_id_col).str.starts_with("NEG_CTRL")
                )
                .then(pl.lit("NEGATIVE_CONTROL"))
                .when(
                    (pl.col("SampleType") == "CONTROL")
                    & pl.col(self.sample_id_col).str.starts_with("PLATE_CTRL")
                )
                .then(pl.lit("PLATE_CONTROL"))
                .when(
                    (pl.col("SampleType") == "CONTROL")
                    & pl.col(self.sample_id_col).str.starts_with("SAMPLE_CTRL")
                )
                .then(pl.lit("SAMPLE_CONTROL"))
                .otherwise(pl.col("SampleType"))
                .alias("SampleType")
            )
            self.sample_metadata = self.sample_metadata.with_columns(reclassify_expr)

            data_full_cols = self.data_full.collect_schema().names()
            if "SampleType" in data_full_cols and self.sample_id_col in data_full_cols:
                sample_type_mapping = self.sample_metadata.select(
                    [self.sample_id_col, "SampleType"]
                ).unique(subset=[self.sample_id_col], keep="first")
                self.data_full = (
                    self.data_full.drop("SampleType")
                    .join(sample_type_mapping, on=self.sample_id_col, how="left")
                    .select(data_full_cols)
                )

        logger.info(
            f"Data partitioned: {len(data_cols)} measurement cols, "
            f"{len(sample_cols)} sample metadata cols, "
            f"{len(analyte_cols)} analyte metadata cols"
        )

        # Now call parent __init__ to set up QC infrastructure
        super().__init__(
            project_name=project_name,
            save_prefix=save_prefix,
            adsl=adsl,
            manifest=manifest,
            usubjid_col=usubjid_col,
            clinical_vars=clinical_vars,
            force_categorical=force_categorical,
            group_col=group_col,
            random_seed=random_seed,
            data_col=data_col,
        )

        # Partition by assay type for Olink (not in base class)
        logger.debug("Partitioning by assay type...")
        self.protein_assays = self.data_full.join(
            self.analyte_metadata.filter(pl.col("AssayType") == "assay").select(
                self.protein_id_col
            ),
            on=self.protein_id_col,
            how="semi",
        )
        self.inc_controls = self.data_full.join(
            self.analyte_metadata.filter(pl.col("AssayType") == "inc_ctrl").select(
                self.protein_id_col
            ),
            on=self.protein_id_col,
            how="semi",
        )
        self.amp_controls = self.data_full.join(
            self.analyte_metadata.filter(pl.col("AssayType") == "amp_ctrl").select(
                self.protein_id_col
            ),
            on=self.protein_id_col,
            how="semi",
        )

        # Partition by sample type for Olink controls
        self.plate_controls = self.data_full.join(
            self.sample_metadata.filter(pl.col("SampleType") == "PLATE_CONTROL").select(
                self.sample_id_col
            ),
            on=self.sample_id_col,
            how="semi",
        )
        self.sample_controls = self.data_full.join(
            self.sample_metadata.filter(
                pl.col("SampleType") == "SAMPLE_CONTROL"
            ).select(self.sample_id_col),
            on=self.sample_id_col,
            how="semi",
        )
        self.negative_controls = self.data_full.join(
            self.sample_metadata.filter(
                pl.col("SampleType") == "NEGATIVE_CONTROL"
            ).select(self.sample_id_col),
            on=self.sample_id_col,
            how="semi",
        )

        logger.info("OLINK EXPLORE HT QC PIPELINE")
        logger.info(
            f"Dataset: {self.qc_results['n_samples']} samples, "
            f"{self.qc_results['n_proteins']} proteins, "
            f"{self.qc_results['n_plates']} plates"
        )

    # Olink-specific QC methods
    def remove_sample_qc(self, **kwargs):
        """Run sample QC checks for Olink (SampleQC flags)."""
        sample_qc_status = quality_control.sample_qc_olink(self, **kwargs)
        self.sample_qc_status = sample_qc_status
        return sample_qc_status

    def filter_assay_qc(self, **kwargs):
        """Run assay QC checks for Olink (AssayQC flags) and generate report section."""
        assay_results = quality_control.assay_qc_olink(self, **kwargs)
        self.assay_qc_results = assay_results
        return assay_results

    def olink_lod(self, **kwargs):
        """Calculate Limit of Detection (LOD) for Olink assays."""
        from . import detection as det_module

        lod_results = det_module.olink_lod(self, **kwargs)
        self.lod_results = lod_results
        return lod_results

    def filter_assay_duplicates(self, **kwargs):
        """Identify and correlate duplicate assays (same protein, different OlinkID)."""
        raise NotImplementedError(
            "Duplicate-assay review is not part of the minimal SURMOUNT-5 QC pipeline."
        )


class SomaQC(ProteomicsQC):
    """
    QC pipeline for SomaScan proteomics data.

    This class reads ADAT files using the somadata package and converts them
    to a long-format Polars DataFrame for QC analysis. It also supports
    pre-processed data files in parquet, csv, or tsv format.

    Expected long format columns:
        - SampleId: Sample identifier
        - SeqId: SOMAmer sequence identifier (primary protein ID)
        - SomaId: SomaLogic target identifier
        - Target: Short target name
        - UniProt: UniProt identifier
        - RFU: Relative Fluorescence Units
        - log10_scaled_RFU: Log10-transformed and scaled RFU
    """

    # Default protein-biomarker mappings for common clinical analytes
    DEFAULT_PROTEIN_BIOMARKER_MAP = {
        "18380-78": "Albumin",
        "16015-19": "Alanine Aminotransferase",
        "10463-23": "Alkaline Phosphatase",
        "16926-44": "Alkaline Phosphatase",
        "6715-63": "Alkaline Phosphatase",
        "7813-6": "Alkaline Phosphatase",
        "7918-114": "Amylase",
        "18917-53": "Amylase",
        "10439-57": "Amylase",
        "15556-49": "Amylase",
        "4912-17": "Aspartate Aminotransferase",
        "23903-3": "Aspartate Aminotransferase",
        "29468-2": "Calcitonin",
        "29467-12": "Choriogonadotropin Beta",
        "4900-8": "Complement C3a",
        "2755-8": "Complement C3a",
        "2851-63": "Complement C5a",
        "2670-67": "Creatine Kinase",
        "3714-49": "Creatine Kinase",
        "2609-59": "Cystatin C",
        "3032-11": "Follicle Stimulating Hormone",
        "6334-9": "Gamma Glutamyl Transferase",
        "21548-20": "Gamma Glutamyl Transferase",
        "4915-64": "Hemoglobin",
        "2989-6": "Interferon Gamma",
        "2989-17": "Interferon Gamma",
        "15346-31": "Interferon Gamma",
        "3037-62": "Interleukin 1 Beta",
        "2773-50": "Interleukin 10",
        "13723-6": "Interleukin 10",
        "10367-62": "Interleukin 12",
        "14085-28": "Interleukin 13",
        "3070-1": "Interleukin 2",
        "2906-55": "Interleukin 4",
        "13663-2": "Interleukin 4",
        "2573-20": "Interleukin 6",
        "4673-13": "Interleukin 6",
        "3447-64": "Interleukin 8",
        "4883-56": "Insulin",
        "15613-16": "Lipase",
        "31546-1": "Thyrotropin",
        "3521-16": "Thyrotropin",
        "9409-11": "Tryptase",
        "3403-1": "Tryptase",
        "14696-45": "Tryptase",
        "5692-79": "Tumor Necrosis Factor",
        "5936-53": "Tumor Necrosis Factor",
    }

    def __init__(
        self,
        adat_path: str | Path | None = None,
        data: pl.DataFrame | pl.LazyFrame | None = None,
        project_name: str = "SomaScan QC",
        save_prefix: str | None = None,
        group_col: str | list[str] | None = None,
        adsl: pl.DataFrame | pl.LazyFrame | None = None,
        manifest: pl.DataFrame | pl.LazyFrame | None = None,
        usubjid_col: str = "USUBJID",
        clinical_vars: list[str] | None = None,
        force_categorical: list[str] | None = None,
        random_seed: int = 0,
        data_col: str | None = None,
    ):
        """
        Initialize the SomaScan QC pipeline.

        Args:
            adat_path: Path to data file - can be ADAT file (.adat) or pre-processed
                       file (parquet, csv, tsv). Mutually exclusive with data.
            data: Pre-processed Polars DataFrame/LazyFrame (mutually exclusive with adat_path)
            project_name: Name of the project for reporting
            save_prefix: Prefix for saving intermediate result files
            group_col: Column name(s) for grouping
            adsl: Phenotype/clinical data
            manifest: Sample ID mapping manifest
            usubjid_col: Column name for unique subject ID
            clinical_vars: List of column names from adsl/manifest to include
            force_categorical: List of column names to treat as categorical
            random_seed: Random seed for reproducibility
            data_col: Override the default data column ('log10_scaled_RFU'). Use 'RFU'
                      for raw relative fluorescence units.
        """
        if adat_path is None and data is None:
            raise ValueError("Either adat_path or data must be provided")
        if adat_path is not None and data is not None:
            raise ValueError("Only one of adat_path or data can be provided")

        # Set platform-specific column mappings
        self.sample_id_col = "SampleId"
        self.protein_id_col = "SeqId"  # Use SeqId as primary protein identifier
        self.protein_name_col = "Target"
        self.data_col = "log10_scaled_RFU"
        self.uniprot_col = "UniProt"
        self.plate_id_col = "PlateId"
        self.sample_type = "Sample"
        self.qc_sample_type = "QC"

        # Load or use provided data
        if adat_path is not None:
            import time
            import os

            path_str = str(adat_path)
            ext = os.path.splitext(path_str)[1].lower()

            # Check if this is an ADAT file or a pre-processed file (parquet/csv/tsv)
            if ext == ".adat":
                import somadata

                logger.info(f"Reading ADAT file: {adat_path}")
                t0 = time.time()
                self.adat_raw = somadata.read_adat(path_str)
                logger.info(
                    f"Loaded ADAT file in {time.time() - t0:.2f}s - shape: {self.adat_raw.shape}"
                )

                t1 = time.time()
                self.data_full, self.sample_metadata, self.analyte_metadata = (
                    self._adat_to_long_format(self.adat_raw)
                )
                logger.info(f"Converted to long format in {time.time() - t1:.2f}s")
            else:
                # Load from pre-processed file (parquet, csv, tsv)
                logger.info(
                    f"Loading SomaScan data from pre-processed file: {adat_path}"
                )
                self.adat_raw = None
                self.data_full = _scan_file(adat_path)
                logger.info(
                    "Loaded pre-processed data - sample_metadata and analyte_metadata will be minimal"
                )
                self.sample_metadata = self.data_full.select(
                    self.sample_id_col
                ).unique()
                self.analyte_metadata = self.data_full.select(
                    self.protein_id_col
                ).unique()
        else:
            self.adat_raw = None
            self.data_full = (
                self._ensure_lazy(data) if not isinstance(data, pl.LazyFrame) else data
            )
            logger.warning(
                "Using pre-provided data - sample_metadata and analyte_metadata will be minimal"
            )
            self.sample_metadata = (
                self.data_full.select(self.sample_id_col).unique().lazy()
            )
            self.analyte_metadata = (
                self.data_full.select(self.protein_id_col).unique().lazy()
            )

        # Call parent __init__ to set up QC infrastructure
        super().__init__(
            project_name=project_name,
            save_prefix=save_prefix,
            adsl=adsl,
            manifest=manifest,
            usubjid_col=usubjid_col,
            clinical_vars=clinical_vars,
            force_categorical=force_categorical,
            group_col=group_col,
            random_seed=random_seed,
            data_col=data_col,
        )

        logger.info("SOMASCAN QC PIPELINE")
        logger.info(
            f"Dataset: {self.qc_results['n_samples']} samples, "
            f"{self.qc_results['n_proteins']} proteins, "
            f"{self.qc_results['n_plates']} plates"
        )

    def _adat_to_long_format(
        self, adat: somadata.Adat
    ) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
        """
        Convert ADAT object to three separate tables for optimized storage.

        Returns:
            Tuple of (measurements_lazy, sample_metadata_lazy, analyte_metadata_lazy)
        """
        # Extract sample metadata (from ADAT index)
        sample_meta_df = adat.index.to_frame(index=False).reset_index(drop=True)

        # Extract analyte metadata (from ADAT columns)
        analyte_meta_df = adat.columns.to_frame(index=False).reset_index(drop=True)

        # Filter for Type == "Protein" & Organism == "Human"
        if "Type" in analyte_meta_df.columns and "Organism" in analyte_meta_df.columns:
            filter_mask = (analyte_meta_df["Type"] == "Protein") & (
                analyte_meta_df["Organism"] == "Human"
            )
            filtered_indices = analyte_meta_df.index[filter_mask].tolist()
            analyte_meta_df = analyte_meta_df[filter_mask].reset_index(drop=True)
            rfu_values_filtered = adat.values[:, filtered_indices]
        else:
            logger.warning(
                "Type or Organism columns not found in analyte metadata - skipping filter"
            )
            rfu_values_filtered = adat.values

        # Get dimensions
        n_samples = len(sample_meta_df)
        n_analytes = len(analyte_meta_df)
        logger.info(
            f"Converting ADAT to long format: {n_samples} samples × {n_analytes} analytes"
        )

        # Convert pandas dataframes to polars
        sample_meta_pl = pl.from_pandas(sample_meta_df)
        analyte_meta_pl = pl.from_pandas(analyte_meta_df)

        # Drop Type and Organism from analyte metadata after filtering
        cols_to_drop = []
        if "Type" in analyte_meta_pl.columns:
            cols_to_drop.append("Type")
        if "Organism" in analyte_meta_pl.columns:
            cols_to_drop.append("Organism")
        if cols_to_drop:
            analyte_meta_pl = analyte_meta_pl.drop(cols_to_drop)

        # Keep only essential columns for analyte metadata
        essential_analyte_cols = [
            self.protein_id_col,  # SeqId
            "SomaId",
            self.protein_name_col,  # Target
            self.uniprot_col,
            "EntrezGeneSymbol",
            "ColCheck",  # QC flag for assay quality
        ]
        analyte_cols_to_keep = [
            c for c in essential_analyte_cols if c in analyte_meta_pl.columns
        ]
        analyte_meta_pl = analyte_meta_pl.select(analyte_cols_to_keep)

        logger.debug(
            f"Kept {len(analyte_cols_to_keep)} analyte metadata columns: {', '.join(analyte_cols_to_keep)}"
        )

        # Add index columns for joining
        sample_meta_pl = sample_meta_pl.with_row_index("_sample_idx")
        analyte_meta_pl = analyte_meta_pl.with_row_index("_analyte_idx")

        # Convert RFU values to Polars DataFrame (wide format)
        rfu_wide = pl.DataFrame(
            rfu_values_filtered, schema=[f"analyte_{i}" for i in range(n_analytes)]
        ).with_row_index("_sample_idx")

        # Compute log10-transformed and scaled RFU in wide format
        analyte_cols = [f"analyte_{i}" for i in range(n_analytes)]

        # Log10 transform all RFU values at once
        log10_rfu_wide = rfu_wide.select(
            [
                pl.col("_sample_idx"),
                *[pl.col(col).log10().alias(f"log10_{col}") for col in analyte_cols],
            ]
        )

        # Center and scale per analyte: (value - col_mean) / col_std
        log10_scaled_wide = log10_rfu_wide.select(
            [
                pl.col("_sample_idx"),
                *[
                    (
                        (
                            pl.col(f"log10_analyte_{i}")
                            - pl.col(f"log10_analyte_{i}").mean()
                        )
                        / pl.col(f"log10_analyte_{i}").std()
                    ).alias(f"scaled_analyte_{i}")
                    for i in range(n_analytes)
                ],
            ]
        )

        # Melt to long format
        rfu_long = rfu_wide.unpivot(
            index="_sample_idx", variable_name="_analyte_col", value_name="RFU"
        )

        log10_scaled_long = log10_scaled_wide.unpivot(
            index="_sample_idx",
            variable_name="_analyte_col_scaled",
            value_name="log10_scaled_RFU",
        )

        # Extract analyte index from column names
        rfu_long = rfu_long.with_columns(
            pl.col("_analyte_col")
            .str.extract(r"analyte_(\d+)", 1)
            .cast(pl.UInt32)
            .alias("_analyte_idx")
        ).drop("_analyte_col")

        log10_scaled_long = log10_scaled_long.with_columns(
            pl.col("_analyte_col_scaled")
            .str.extract(r"scaled_analyte_(\d+)", 1)
            .cast(pl.UInt32)
            .alias("_analyte_idx")
        ).drop("_analyte_col_scaled")

        measurements = rfu_long.join(
            log10_scaled_long.select(
                ["_sample_idx", "_analyte_idx", "log10_scaled_RFU"]
            ),
            on=["_sample_idx", "_analyte_idx"],
            how="left",
        )

        # Add SampleId and SeqId to measurements by joining with metadata
        sample_ids = sample_meta_pl.select(["_sample_idx", self.sample_id_col])
        analyte_ids = analyte_meta_pl.select(["_analyte_idx", self.protein_id_col])

        measurements = (
            measurements.join(sample_ids, on="_sample_idx", how="left")
            .join(analyte_ids, on="_analyte_idx", how="left")
            .select(
                [
                    "_sample_idx",
                    self.sample_id_col,
                    self.protein_id_col,
                    "RFU",
                    "log10_scaled_RFU",
                ]
            )
        )

        # Prepare sample metadata (KEEP the index column to preserve replicates)
        sample_metadata = sample_meta_pl

        # Prepare analyte metadata (drop the index column)
        analyte_metadata = analyte_meta_pl.drop("_analyte_idx")

        logger.info(
            f"Conversion complete: {measurements.height:,} measurements, "
            f"{sample_metadata.height:,} samples, {analyte_metadata.height:,} analytes"
        )

        return measurements.lazy(), sample_metadata.lazy(), analyte_metadata.lazy()

    def save_long_format(self, output_path: str | None = None):
        """Save the long-format data to a parquet file."""
        if output_path is None:
            if self.save_prefix is None:
                raise ValueError("Either output_path or save_prefix must be set")
            output_path = f"{self.save_prefix}.rfu_long.parquet"

        self.data_full.collect().write_parquet(output_path)
        logger.info(f"Saved long-format data to {output_path}")

    # SomaScan-specific QC methods
    def remove_sample_qc(self, **kwargs):
        """Run sample QC checks for SomaScan (RowCheck flags)."""
        sample_qc_status = quality_control.sample_qc_soma(self, **kwargs)
        self.sample_qc_status = sample_qc_status
        return sample_qc_status

    def filter_assay_qc(self):
        """Run assay QC checks for SomaScan (ColCheck flags) and generate report section."""
        assay_results = quality_control.assay_qc_soma(self)
        self.assay_qc_results = assay_results
        return assay_results

    def filter_assay_cv(self, **kwargs):
        """Filter assays by CV — reporting deferred to finalize_assay_flags."""
        cv_results = detection.assay_cv(self, **kwargs)
        self.cv_results = cv_results
        return cv_results

    def filter_assay_detection(self, **kwargs):
        """Filter assays by detection rate — reporting deferred to finalize_assay_flags."""
        det_results = detection.assay_detection(self, **kwargs)
        self.detection_results = det_results
        return self

    def filter_assay_saturation(self, **kwargs):
        """Flag analytes where more than the configured sample fraction exceeds the saturation RFU threshold (default: 50%; override with ``sample_frac`` in ``**kwargs``)."""
        saturation_results = quality_control.assay_saturation_soma(self, **kwargs)
        self.saturation_results = saturation_results
        return saturation_results

    def filter_assay_sb_ratio(self, **kwargs):
        """Flag analytes with low signal-to-background ratio (median sample / median buffer)."""
        sb_results = quality_control.assay_sb_ratio_soma(self, **kwargs)
        self.sb_ratio_results = sb_results
        return sb_results

    def finalize_assay_flags(self):
        """Apply unified WARN/FAIL logic across all assay QC checks and generate report.

        ≥1 flag (ColCheck, Intra-CV, Detection Rate, Saturation, SB Ratio) → WARN
        ≥2 flags → FAIL (excluded from downstream analysis)

        Must be called after filter_assay_qc, filter_assay_cv, filter_assay_detection,
        filter_assay_saturation, and filter_assay_sb_ratio.
        """
        finalize_results = quality_control.finalize_assay_flags_soma(self)
        self.finalize_results = finalize_results
        return finalize_results
