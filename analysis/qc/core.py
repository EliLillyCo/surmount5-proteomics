"""Holds the ProteomicsQC pipeline base class and its Olink and SomaScan subclasses."""

from __future__ import annotations

import glob
import inspect
import logging
import os
import re
import tempfile
import time
import weakref
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import somadata

from analysis.qc import (
    age_concordance,
    clinical,
    detection,
    duplication,
    outliers,
    quality_control,
    sex_concordance,
    utils,
)
from analysis.qc.config import TrialConfig
from analysis.qc.corrections import (
    EXPECTED_SEX_COL,
    SEX_EXCEPTION_COL,
    with_expected_sex,
)
from analysis.qc.flags import (
    WARN_AGE_CONCORDANCE,
    WARN_AGE_CONSISTENCY,
    WARN_AGE_PREDICTION,
    WARN_SEX_CONCORDANCE,
)
from analysis.qc.io import scan_file
from analysis.qc.platform import OLINK, OLINK_NPX, OLINK_PC_NPX, SOMA, get_platform

logger = logging.getLogger(__name__)


_BRIDGE_FLAG_STRINGS = {"y": True, "true": True, "n": False, "false": False}
# Manifest columns carried per tube.
_MANIFEST_VISIT_COLS = ("VISITNUM", "ADT", "ADT_SOURCE")


def validate_manifest_schema(
    manifest: pl.LazyFrame,
    sample_id_col: str = "SampleID",
    usubjid_col: str = "USUBJID",
) -> pl.LazyFrame:
    """Validates the manifest and returns it with ``BridgeSample`` as Boolean.

    Control wells (no subject) may repeat across plates. Plate and well
    are not required: they come from the assay file.

    Args:
        manifest: Sample manifest.
        sample_id_col: Manifest column holding the sample ID.
        usubjid_col: Manifest column holding the subject ID.

    Returns:
        The manifest with ``BridgeSample`` cast to Boolean.

    Raises:
        ValueError: If a required column is absent, if ``BridgeSample`` is neither
            Boolean nor a Y/N/True/False string (any case), or if a subject or bridge
            SampleID appears on more than one row.
    """
    required = {
        sample_id_col,
        usubjid_col,
        "VISITNUM",
        "ADT",
        "AccessionNumber",
        "BridgeSample",
    }
    schema = manifest.collect_schema()
    missing = required - set(schema.names())
    if missing:
        raise ValueError(f"Manifest missing required columns: {', '.join(sorted(missing))}")

    flag_dtype = schema["BridgeSample"]
    if flag_dtype == pl.String:
        values = set(
            manifest.select(pl.col("BridgeSample").drop_nulls().unique())
            .collect()
            .to_series()
            .to_list()
        )
        bad = sorted(v for v in values if v.strip().lower() not in _BRIDGE_FLAG_STRINGS)
        if bad:
            raise ValueError(f"BridgeSample has values other than Y/N/True/False: {', '.join(bad)}")
        manifest = manifest.with_columns(
            pl.col("BridgeSample")
            .str.strip_chars()
            .str.to_lowercase()
            .replace_strict(_BRIDGE_FLAG_STRINGS, return_dtype=pl.Boolean)
        )
    elif flag_dtype != pl.Boolean:
        raise ValueError(
            f"BridgeSample must be Boolean or a Y/N/True/False string; got {flag_dtype}"
        )

    duplicated = (
        manifest.filter(pl.col(usubjid_col).is_not_null() | pl.col("BridgeSample").fill_null(False))
        .group_by(sample_id_col)
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .sort(sample_id_col)
        .collect()[sample_id_col]
        .cast(pl.Utf8)
        .to_list()
    )
    if duplicated:
        raise ValueError(
            f"Manifest has {len(duplicated)} {sample_id_col}s on more than one row: "
            f"{', '.join(duplicated[:10])}"
        )
    return manifest


class ProteomicsQC:
    """A QC pipeline run on one trial's proteomics data, shared by SomaScan and Olink.

    Holds clinical data merging, QC tracking, outlier detection, concordance checks
    and report generation. A platform subclass (``SomaQC``, ``OlinkQC``) loads the
    data and, before calling ``super().__init__``, sets ``data_full``,
    ``sample_metadata`` and ``analyte_metadata`` (LazyFrames) and the column names
    ``sample_id_col``, ``protein_id_col``, ``protein_name_col``, ``data_col``,
    ``uniprot_col`` and ``sample_type``.

    Args:
        project_name: Project name used in the report.
        save_prefix: Prefix for intermediate result files.
        adsl: Subject-level clinical data for concordance checks and analysis.
        manifest: Sample manifest mapping sample IDs to subjects and visits.
        usubjid_col: Subject ID column in both the manifest and ADSL.
        clinical_vars: ADSL or manifest columns to carry into the analysis.
        force_categorical: Columns to treat as categorical.
        group_col: Grouping column or columns.
        random_seed: Seed for every stochastic step.
        data_col: Data column to QC instead of the platform default (for example
            ``PCNormalizedNPX`` for Olink or ``RFU`` for SomaScan).

    Attributes:
        expected_sex: Per-subject expected-sex overrides (USUBJID,
            Expected_Protein_Sex) from the config, or None.
    """

    expected_sex: pl.DataFrame | None = None

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
        """Validates the subclass's data and runs the platform-agnostic setup."""
        self._workdir = tempfile.TemporaryDirectory(prefix="qc_")
        # The directory's own finalizer warns (ResourceWarning) on implicit cleanup.
        weakref.finalize(self, self._workdir.cleanup)
        self.project_name = project_name
        self.save_prefix = save_prefix
        self.random_seed = random_seed

        if data_col is not None:
            self.data_col = data_col

        self.adsl = adsl
        self.manifest = manifest
        self.usubjid_col = usubjid_col
        self.clinical_vars = clinical_vars
        self.force_categorical = force_categorical

        # PCA matrix cache, reused by the median/IQR outlier detection.
        self._pca_matrix = None

        # The subclass has set the data and column attributes (see the class
        # docstring). The QC data column and the identity layer must both exist.
        self._schema = self.data_full.collect_schema()
        for role, col in (
            ("data column", self.data_col),
            ("identity layer", self.platform.identity_layer),
        ):
            if col not in self._schema.names():
                raise ValueError(
                    f"{self.platform.name} {role} '{col}' not found in the proteomics data"
                )
        self._warn_if_data_col_off_identity_layer(self.data_col)
        logger.info(f"Data shape: {self.data_full.select(pl.len()).collect().item():,} rows")

        self._partition_by_sample_type()

        self._merge_clinical_data()

        self._setup_groups(group_col)

        self.failed_samples: list[str] = []
        self.failed_proteins: list[str] = []
        self.warned_samples: list[str] = []
        self.warned_proteins: list[str] = []

        self.sample_qc_actions: list[pl.DataFrame] = []
        self.protein_qc_actions: list[pl.DataFrame] = []

        # Samples with no manifest row.
        self._exclude_no_manifest_match_samples()

        # Tubes an ``exclude`` correction removed.
        self._exclude_correction_samples()

        # Samples with no arm are failed; non-study samples are exempt.
        self._exclude_null_group_samples()

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

        self.cv_data: pl.DataFrame | None = None
        self.pca_data: Any | None = None

        # Set by identity.identity_check.
        self.identity_events: pl.DataFrame | None = None
        self.identity_proposals: pl.DataFrame | None = None
        # Identity annotation of technically failed tubes only.
        self.unfit_identity: pl.DataFrame | None = None
        # The cut, in SDs, remove_sample_outliers judged PCk_std, Z_Median and Z_IQR at.
        self.outlier_n_std: float | None = None
        # SampleID and VISITNUM_ANALYSIS from clinical.sample_unsuitable.
        self.analysis_visits: pl.DataFrame | None = None
        # Until sample QC replaces this, no sample is kept out of the outlier fits.
        self._sample_qc_data: dict = {"outlier_fit_exclude": []}
        # SampleID, TRTEDT, EOTSTT, TRT_WINDOW_END and days since first and last dose.
        self.dose_timing: pl.DataFrame | None = None

        # Out-of-fold predictions for identity evidence, set by sex/age concordance.
        self.sex_oof_predictions: pl.DataFrame | None = None
        self.age_oof_predictions: pl.DataFrame | None = None

    def _resolve_config(self, config: TrialConfig | None) -> TrialConfig:
        """Returns the trial config, defaulting to this platform's.

        Raises:
            ValueError: If the config names another platform.
        """
        if config is None:
            return TrialConfig(platform=self.platform.name)
        if get_platform(config.platform) is not self.platform:
            raise ValueError(
                f"TrialConfig platform {config.platform!r} does not match "
                f"{type(self).__name__} ({self.platform.name})"
            )
        return config

    def _apply_corrections(self, manifest: pl.LazyFrame, adsl, usubjid_col: str) -> pl.LazyFrame:
        """Applies the config's corrections and returns the corrected ADSL.

        ``subject_set`` entries change ADSL here. ``expected_sex`` entries leave it
        alone and are resolved into ``self.expected_sex`` for the sex checks. The other
        types apply in the manifest build.

        Raises:
            ValueError: If build-time corrections exist but the manifest was not built
                (it has no ``MANIFEST_EXCLUDE``), so they would be dropped silently.
        """
        from analysis.qc.corrections import (
            BUILD_TIME_TYPES,
            apply_subject_sets,
            resolve_expected_sex,
        )

        build_time = sorted({c.type for c in self.config.corrections if c.type in BUILD_TIME_TYPES})
        if build_time and "MANIFEST_EXCLUDE" not in manifest.collect_schema().names():
            raise ValueError(
                f"Corrections of type {', '.join(build_time)} apply in the manifest build; "
                "this manifest has no MANIFEST_EXCLUDE column, so it was not built from "
                "the config's [manifest]"
            )
        corrected = apply_subject_sets(
            self.config.corrections, self._ensure_lazy(adsl), usubjid_col
        )
        self.expected_sex = resolve_expected_sex(self.config.corrections, corrected, usubjid_col)
        return corrected

    def _partition_by_sample_type(self):
        """Sets ``self.samples`` to the study-sample rows of the assay analytes."""
        metadata_schema = self.sample_metadata.collect_schema()
        analyte_schema = self.analyte_metadata.collect_schema()

        if "SampleType" in metadata_schema.names():
            sample_type_ids = (
                self.sample_metadata.filter(pl.col("SampleType") == self.sample_type)
                .select(self.sample_id_col)
                .unique()
            )

            self.samples = self.data_full.join(
                sample_type_ids,
                on=self.sample_id_col,
                how="semi",
            )
            logger.debug(f"Partitioned {self.sample_type} samples from data_full using semi-join")
        elif "SampleType" in self._schema.names():
            self.samples = self.data_full.filter(pl.col("SampleType") == self.sample_type)
            logger.debug(f"Partitioned {self.sample_type} samples from data_full")
        else:
            self.samples = self.data_full
            logger.debug("No SampleType column found - treating all data as samples")

        # Olink only: keep the assay analytes.
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
            logger.debug("Further filtered samples to AssayType == 'assay' using semi-join")

    def _merge_clinical_data(self):
        """Joins ADSL to the manifest's sample IDs into ``self.adsl_merged``."""
        self.adsl_merged = None
        self.clinical_vars_safe = {}
        # Clinical columns joined from ADSL by subject (re-joined on reassignment).
        self.adsl_derived_cols: list[str] = []

        if self.adsl is None or self.manifest is None:
            return

        adsl_lazy = self._ensure_lazy(self.adsl)
        manifest_lazy = self._ensure_lazy(self.manifest)
        adsl_cols_present = adsl_lazy.collect_schema().names()

        if self.clinical_vars is not None:
            adsl_cols_needed = [self.usubjid_col]
            for col_name in self.clinical_vars:
                if col_name not in adsl_cols_needed:
                    adsl_cols_needed.append(col_name)

            adsl_cols_to_select = [c for c in adsl_cols_needed if c in adsl_cols_present]
            adsl_lazy = adsl_lazy.select(adsl_cols_to_select)

        manifest_schema = manifest_lazy.collect_schema()
        # Visit and draw-date columns travel with the tube, whatever the subject.
        visit_cols = [c for c in _MANIFEST_VISIT_COLS if c in manifest_schema.names()]
        manifest_cols = [self.sample_id_col, self.usubjid_col, *visit_cols]
        # Clinical variables only the manifest has also travel with the tube:
        # one row per manifest row, so repeated control wells keep their own values.
        manifest_only_vars = [
            c
            for c in (self.clinical_vars or [])
            if c in manifest_schema.names()
            and c not in adsl_cols_present
            and c not in manifest_cols
        ]
        manifest_cols += manifest_only_vars

        adsl_with_samples = manifest_lazy.select(manifest_cols).join(
            adsl_lazy,
            on=self.usubjid_col,
            how="left",
            validate="m:1",
        )

        if self.clinical_vars is not None:
            adsl_schema = adsl_with_samples.collect_schema()
            select_exprs = [self.sample_id_col, self.usubjid_col, *visit_cols]

            for col_name in self.clinical_vars:
                self.clinical_vars_safe[col_name] = col_name

                if col_name in manifest_only_vars:
                    logger.info(f"Clinical variable '{col_name}' found in manifest (not in adsl)")
                    select_exprs.append(pl.col(col_name))
                elif col_name in adsl_schema.names():
                    select_exprs.append(pl.col(col_name))
                    if col_name in adsl_cols_present:
                        self.adsl_derived_cols.append(col_name)
                else:
                    raise ValueError(
                        f"Clinical variable '{col_name}' not found in adsl or manifest"
                    )

            self.adsl_merged = adsl_with_samples.select(select_exprs).cache()
            logger.info(f"Clinical variables merged: {len(self.clinical_vars_safe)} variables")
        else:
            select_exprs = [self.sample_id_col, self.usubjid_col, *visit_cols]
            self.adsl_merged = adsl_with_samples.select(select_exprs).cache()
            logger.info("ADSL data merged: essential columns only")

    def _standardize_group_name(self, value: str) -> str:
        """Returns a treatment name with spaces and slashes normalized.

        A slash becomes ``_``, a space before a digit becomes ``-``, a space after a
        dose unit becomes ``_`` and other spaces are dropped. For example,
        ``"DRUG 3/6/9 mg"`` becomes ``"DRUG-3_6_9mg"`` and ``"Placebo"`` is unchanged. None and ``""`` come
        back as they are.
        """
        import re

        if value is None or value == "":
            return value

        value = value.replace("/", "_")
        value = re.sub(r"\s+(\d)", r"-\1", value)
        value = re.sub(r"(mg|mcg|ug|g)\s+", r"\1_", value)
        value = value.replace(" ", "")

        return value

    def _setup_groups(self, group_col):
        """Sets ``group_col`` and ``group_data``, joining several columns with ``_``."""
        self.group_col_input = group_col
        self.group_col = None
        self.group_data = None
        self.has_groups = False

        if group_col is None or self.adsl_merged is None:
            return

        group_cols = [group_col] if isinstance(group_col, str) else group_col

        schema_names = self.adsl_merged.collect_schema().names()
        missing_cols = [col for col in group_cols if col not in schema_names]

        if missing_cols:
            raise ValueError(f"Group column(s) {missing_cols} not found in adsl/clinical_vars")
        else:
            if len(group_cols) == 1:
                self.group_col = group_cols[0]
                self.group_data = self.adsl_merged.select(
                    [
                        self.sample_id_col,
                        pl.col(self.group_col)
                        .cast(pl.Utf8)
                        .map_elements(self._standardize_group_name, return_dtype=pl.Utf8)
                        .alias(self.group_col),
                    ]
                )
                self.has_groups = True
                logger.info(f"Group column loaded: {self.group_col}")
            else:
                self.group_col = "_".join(group_cols)
                self.group_data = self.adsl_merged.select(
                    [
                        self.sample_id_col,
                        pl.concat_str(
                            [
                                pl.col(c)
                                .cast(pl.Utf8)
                                .map_elements(self._standardize_group_name, return_dtype=pl.Utf8)
                                for c in group_cols
                            ],
                            separator="_",
                            ignore_nulls=False,
                        ).alias(self.group_col),
                    ]
                )
                self.has_groups = True
                logger.info(f"Group columns combined: {' + '.join(group_cols)} -> {self.group_col}")

    def _ensure_lazy(self, df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
        """Returns ``df`` as a LazyFrame."""
        if isinstance(df, pl.DataFrame):
            return df.lazy()
        return df

    def _checkpoint(self, frame: pl.LazyFrame, name: str) -> pl.LazyFrame:
        """Streams ``frame`` to parquet in the private work directory and rescans it."""
        path = Path(self._workdir.name) / f"{name}.parquet"
        frame.sink_parquet(path)
        return pl.scan_parquet(path)

    def close(self) -> None:
        """Deletes the private work directory.

        Checkpointed frames are unreadable afterwards.
        """
        self._workdir.cleanup()

    @property
    def raw_col(self) -> str:
        """The column on the assay's native scale.

        RFU on SomaScan, the QC data column on Olink.
        """
        return "RFU" if self.platform.name == "soma" else self.data_col

    @property
    def detection_col(self) -> str:
        """The column detection compares with the LOD.

        SomaScan: the delivered non-ANML RFU. Olink: the QC data column.
        """
        return quality_control.DELIVERED_RFU_COL if self.platform.name == "soma" else self.raw_col

    @cached_property
    def manifest_bridge_sample_ids(self) -> set[str]:
        """Samples the manifest flags as bridge samples (``BridgeSample == True``)."""
        if self.manifest is None:
            return set()
        manifest = self._ensure_lazy(self.manifest)
        if "BridgeSample" not in manifest.collect_schema().names():
            return set()
        return set(
            manifest.filter(pl.col("BridgeSample") == True)  # noqa: E712
            .select(pl.col(self.sample_id_col).cast(pl.Utf8))
            .collect()
            .to_series()
            .to_list()
        )

    @cached_property
    def bridge_sample_ids(self) -> set[str]:
        """Bridge samples.

        The manifest's flagged samples plus samples whose USUBJID carries the bridge
        prefix.
        """
        ids = set(self.manifest_bridge_sample_ids)
        if (
            self.adsl_merged is not None
            and self.usubjid_col in self.adsl_merged.collect_schema().names()
        ):
            ids |= set(
                self.adsl_merged.filter(pl.col(self.usubjid_col).str.starts_with("BRIDGE_"))
                .select(pl.col(self.sample_id_col).cast(pl.Utf8))
                .collect()
                .to_series()
                .to_list()
            )
        return ids

    def _exclude_null_group_samples(self):
        """Fails study samples with no value in the group column.

        Bridge samples and lab controls (outside ``self.samples``) are exempt.
        """
        if not self.has_groups or self.group_data is None:
            return

        real_sample_ids = set(
            self.samples.select(self.sample_id_col)
            .unique()
            .collect()[self.sample_id_col]
            .cast(pl.Utf8)
            .to_list()
        )

        null_group_samples = (
            self.group_data.filter(pl.col(self.group_col).is_null())
            .select(self.sample_id_col)
            .collect()[self.sample_id_col]
            .cast(pl.Utf8)
            .to_list()
        )
        null_group_samples = [s for s in null_group_samples if s in real_sample_ids]
        if not null_group_samples:
            return

        bridge_sample_ids = self.bridge_sample_ids
        no_treatment_samples = [s for s in null_group_samples if s not in bridge_sample_ids]
        if no_treatment_samples:
            logger.info(
                f"Marking {len(no_treatment_samples)} samples with no value in "
                f"'{self.group_col}' as FAIL_No_Treatment_Arm"
            )
            self._flag_samples(no_treatment_samples, clinical.FAIL_NO_TREATMENT_ARM)

    def _exclude_no_manifest_match_samples(self):
        """Fails study samples that have no manifest row at all.

        Such a sample never reaches ``adsl_merged`` or ``group_data``, so no null check
        there can catch it; it only turns null when later code left-joins from the
        proteomics data.
        """
        if self.manifest is None:
            return

        manifest_lazy = self._ensure_lazy(self.manifest)
        manifest_sample_ids = set(
            manifest_lazy.select(self.sample_id_col)
            .unique()
            .collect()[self.sample_id_col]
            .cast(pl.Utf8)
            .to_list()
        )

        data_sample_ids = set(
            self.samples.select(self.sample_id_col)
            .unique()
            .collect()[self.sample_id_col]
            .cast(pl.Utf8)
            .to_list()
        )

        already_failed = set(self.failed_samples)
        no_manifest_match_samples = [
            s for s in data_sample_ids if s not in manifest_sample_ids and s not in already_failed
        ]
        if no_manifest_match_samples:
            logger.info(
                f"Marking {len(no_manifest_match_samples)} samples with no "
                "manifest row as FAIL_No_Manifest_Match"
            )
            self._flag_samples(no_manifest_match_samples, clinical.FAIL_NO_MANIFEST_MATCH)

    def _exclude_correction_samples(self):
        """Fails samples that the manifest build excluded by an ``exclude`` correction.

        Such a sample has a non-null ``MANIFEST_EXCLUDE`` (the correction's reason) and
        is flagged ``FAIL_Correction_Exclude``. Manifests without the column are left
        alone.
        """
        if self.manifest is None:
            return
        manifest = self._ensure_lazy(self.manifest)
        if "MANIFEST_EXCLUDE" not in manifest.collect_schema().names():
            return
        excluded = (
            manifest.filter(pl.col("MANIFEST_EXCLUDE").is_not_null())
            .select(pl.col(self.sample_id_col).cast(pl.Utf8))
            .join(
                self.samples.select(pl.col(self.sample_id_col).cast(pl.Utf8)).unique(),
                on=self.sample_id_col,
                how="semi",
            )
            .unique(maintain_order=True)
            .collect()[self.sample_id_col]
            .to_list()
        )
        logger.info(f"Marking {len(excluded)} samples with FAIL_Correction_Exclude")
        self._flag_samples(excluded, clinical.CORRECTION_EXCLUDE_FLAG)

    def _get_counts(self) -> dict[str, int]:
        """Returns the sample, analyte, plate and control-well counts."""
        # SomaScan ADAT data carries _sample_idx.
        data_schema = self.data_full.collect_schema().names()
        has_sample_idx = "_sample_idx" in data_schema
        sample_schema = self.sample_metadata.collect_schema().names()

        has_assay_type = "AssayType" in self.analyte_metadata.collect_schema().names()

        plate_col = getattr(self, "plate_id_col", None)
        if plate_col and plate_col not in sample_schema:
            plate_col = None

        if "SampleType" in data_schema:
            # Olink: SampleType is in data_full.
            sample_types = (
                self.data_full.select("SampleType").unique().collect()["SampleType"].to_list()
            )
            control_types = [st for st in sample_types if st and st != self.sample_type]
            control_type_map = {
                "PLATE_CONTROL": "pc",
                "SAMPLE_CONTROL": "sc",
                "NEGATIVE_CONTROL": "nc",
            }
        elif "SampleType" in sample_schema:
            # SomaScan: SampleType is in sample_metadata.
            sample_types = (
                self.sample_metadata.select("SampleType").unique().collect()["SampleType"].to_list()
            )
            control_types = [st for st in sample_types if st and st != self.sample_type]
            control_type_map = {}
        else:
            control_types = []
            control_type_map = {}

        if has_sample_idx and "SampleType" in sample_schema:
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

            count_exprs = [
                pl.col("_sample_idx")
                .filter(pl.col("SampleType") == self.sample_type)
                .n_unique()
                .alias("n_samples"),
            ]

            # A control well is counted once per plate it sits on.
            for control_type in control_types:
                if control_type in control_type_map:
                    key = f"n_{control_type_map[control_type]}"
                else:
                    key = f"n_{control_type.lower().replace(' ', '_').replace('-', '_')}"

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

            if plate_col:
                count_exprs.append(pl.col(plate_col).n_unique().alias("n_plates"))
            else:
                count_exprs.append(pl.lit(0).alias("n_plates"))

            # Counted from analyte_metadata to avoid a cross join.
            if has_assay_type:
                n_proteins = (
                    self.analyte_metadata.filter(pl.col("AssayType") == "assay")
                    .select(pl.col(self.protein_id_col).n_unique())
                    .collect()
                    .item()
                )
            else:
                n_proteins = (
                    self.analyte_metadata.select(pl.col(self.protein_id_col).n_unique())
                    .collect()
                    .item()
                )

            combined_counts = data_with_type.select(count_exprs).collect().row(0, named=True)
            counts = combined_counts
            counts["n_proteins"] = n_proteins
            counts["n_blocks"] = 0  # SomaScan has no blocks
        else:
            # Olink counts from data_full, so every control in the data is counted.
            count_exprs = []
            count_exprs.append(
                pl.col(self.sample_id_col)
                .filter(pl.col("SampleType") == self.sample_type)
                .n_unique()
                .alias("n_samples")
            )

            # A control well is counted once per plate it sits on.
            for control_type in control_types:
                if control_type in control_type_map:
                    key = f"n_{control_type_map[control_type]}"
                else:
                    key = f"n_{control_type.lower().replace(' ', '_').replace('-', '_')}"

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

            if plate_col:
                count_exprs.append(pl.col(plate_col).n_unique().alias("n_plates"))
            else:
                count_exprs.append(pl.lit(0).alias("n_plates"))

            if "Block" in data_schema:
                count_exprs.append(pl.col("Block").n_unique().alias("n_blocks"))
            else:
                count_exprs.append(pl.lit(0).alias("n_blocks"))

            # Counted from analyte_metadata to avoid a cross join.
            if has_assay_type:
                n_proteins = (
                    self.analyte_metadata.filter(pl.col("AssayType") == "assay")
                    .select(pl.col(self.protein_id_col).n_unique())
                    .collect()
                    .item()
                )
            else:
                n_proteins = (
                    self.analyte_metadata.select(pl.col(self.protein_id_col).n_unique())
                    .collect()
                    .item()
                )

            combined_counts = self.data_full.select(count_exprs).collect().row(0, named=True)
            counts = combined_counts
            counts["n_proteins"] = n_proteins

        return counts

    def _compute_visitage(self):
        """Returns ``adsl_merged`` with the visit-adjusted age, lazily.

        ``VISITAGE = AGE + (visit date - baseline date) / 365``, where the baseline is
        the participant's draw at the config's ``[visits]`` baseline, or its earliest
        draw when it has none there. Without ``[visits]`` it is the draw at the
        participant's lowest VISITNUM. Returns ``adsl_merged`` unchanged, with a
        warning, when it lacks AGE, ADT or VISITNUM.
        """
        if self.adsl_merged is None:
            logger.warning("Cannot compute visitage: adsl_merged not available")
            return self.adsl_merged

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
            logger.warning("Cannot compute visitage: 'VISITNUM' column not in adsl_merged")
            return self.adsl_merged

        logger.info("Computing visit-adjusted age (visitage) from manifest ADT")

        result = visitage_frame(self.adsl_merged, self.usubjid_col, self.config.visits_baseline)
        logger.info("Computed visitage (lazy evaluation)")
        return result

    def _has_column(self, col: str) -> bool:
        """Checks whether ``col`` is in the current schema of ``data_full``."""
        return col in self.data_full.collect_schema().names()

    def _track_sample_qc(self, sample_ids: list[str], reason: str):
        """Records a sample QC action."""
        if sample_ids:
            qc_df = pl.DataFrame({self.sample_id_col: sample_ids, reason: True})
            self.sample_qc_actions.append(qc_df)

    def _track_protein_qc(self, protein_ids: list[str], reason: str):
        """Records a protein QC action.

        Raises:
            ValueError: If ``reason`` is outside the platform's flag set.
        """
        if reason not in clinical.PROTEIN_QC_FLAGS[self.platform.name]:
            raise ValueError(f"protein flag {reason} is not in the {self.platform.name} flag set")
        if protein_ids:
            qc_df = pl.DataFrame({self.protein_id_col: protein_ids, reason: True})
            self.protein_qc_actions.append(qc_df)

    def _flag_samples(self, sample_ids: list[str], reason: str):
        """Records a sample QC action.

        FAIL_ and WARN_ reasons also extend ``failed_samples`` or ``warned_samples``.
        """
        if not sample_ids:
            return
        if reason.startswith("FAIL_"):
            self.failed_samples.extend(sample_ids)
        elif reason.startswith("WARN_"):
            self.warned_samples.extend(sample_ids)
        self._track_sample_qc(sample_ids, reason)

    def _flag_proteins(self, protein_ids: list[str], reason: str):
        """Records a protein QC action.

        FAIL_ and WARN_ reasons also extend ``failed_proteins`` or ``warned_proteins``.

        Raises:
            ValueError: If ``reason`` is outside the platform's flag set.
        """
        if reason not in clinical.PROTEIN_QC_FLAGS[self.platform.name]:
            raise ValueError(f"protein flag {reason} is not in the {self.platform.name} flag set")
        if not protein_ids:
            return
        if reason.startswith("FAIL_"):
            self.failed_proteins.extend(protein_ids)
        elif reason.startswith("WARN_"):
            self.warned_proteins.extend(protein_ids)
        self._track_protein_qc(protein_ids, reason)

    def _replace_sample_flags(self, sample_ids, flags: dict[str, set]) -> None:
        """Resets each reason in ``flags`` for ``sample_ids``.

        The samples' old entries for the reason are dropped and the new set added.
        Samples outside ``sample_ids`` keep their flags. ``failed_samples`` and
        ``warned_samples`` are rebuilt for ``sample_ids`` from the FAIL_ and WARN_
        actions that remain.
        """
        ids = set(sample_ids)
        sid = self.sample_id_col
        id_list = list(ids)
        actions = []
        kept: dict[str, pl.DataFrame] = {}
        for df in self.sample_qc_actions:
            reasons = [reason for reason in flags if reason in df.columns]
            if reasons:
                df = df.filter(~pl.col(sid).is_in(id_list))
                if df.height == 0:
                    continue
                kept[reasons[0]] = df
            actions.append(df)
        self.sample_qc_actions = actions
        for reason, flagged in flags.items():
            new_ids = sorted(set(flagged) & ids)
            if reason not in kept:
                self._track_sample_qc(new_ids, reason)
            elif new_ids:
                # One frame per reason, so the summary keeps one column for it (a
                # second frame would become ``<reason>_right``).
                position = next(i for i, df in enumerate(actions) if df is kept[reason])
                added = pl.DataFrame({sid: new_ids, reason: True}).cast(kept[reason].schema)
                actions[position] = pl.concat([kept[reason], added])

        def still(prefix: str) -> list:
            return sorted(
                {
                    s
                    for df in self.sample_qc_actions
                    for reason in df.columns
                    if reason.startswith(prefix)
                    for s in df.filter(pl.col(reason) & pl.col(sid).is_in(list(ids)))[sid].to_list()
                }
            )

        still_failed, still_warned = still("FAIL_"), still("WARN_")
        self.failed_samples = [s for s in self.failed_samples if s not in ids] + still_failed
        self.warned_samples = [s for s in self.warned_samples if s not in ids] + still_warned

    def get_data(self) -> pl.LazyFrame:
        """Returns the current data."""
        return self.data_full

    # ── Shared QC steps ──
    def filter_assay_cv(self, **kwargs):
        """Flags assays by coefficient of variation."""
        return detection.assay_cv(self, **kwargs)

    def filter_assay_detection(self, **kwargs):
        """Flags assays by detection rate."""
        det_results = detection.assay_detection(self, **kwargs)
        self.detection_results = det_results
        return self

    def remove_sample_outliers(self, **kwargs):
        """Flags outlier samples by PCA and median/IQR.

        Flagged samples are not removed. Config values fill any argument the caller
        does not pass.

        Returns:
            The PCA frame, the fitted PCA object and the outlier sample IDs.
        """
        if "random_seed" not in kwargs:
            kwargs["random_seed"] = self.random_seed
        kwargs.setdefault("n_components", self.config.qc_outlier_n_pcs)
        kwargs.setdefault("n_std", self.config.qc_outlier_n_std)
        kwargs.setdefault("knn_k", self.config.qc_outlier_knn_k)
        kwargs.setdefault("detection_weighting", self.config.qc_outlier_detection_weighting)
        self.outlier_n_std = float(kwargs["n_std"])

        pca_df, pca_obj, pca_outliers = outliers.pca_outliers(self, **kwargs)

        if pca_df.is_empty():
            logger.warning("PCA outlier detection returned no data")
            self.pca_data = (pca_df, pca_obj, None)
            return pca_df, pca_obj, []

        median_iqr_kwargs = {
            k: v for k, v in kwargs.items() if k in ("n_std", "knn_k", "detection_weighting")
        }
        median_iqr_df, median_iqr_outliers = outliers.median_iqr_outliers(self, **median_iqr_kwargs)

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

        # For the summary export and the unfit-tube annotation: taken before
        # the plate join, which fans a SomaScan sample ID out to one row per replicate
        # tube, and with every PCk_std the cut was judged on.
        pca_cols = [self.sample_id_col]
        pca_cols += [c for c in pca_df.columns if re.fullmatch(r"PC\d+_std", c)]
        if "Z_Median" in pca_df.columns:
            pca_cols.append("Z_Median")
        if "Z_IQR" in pca_df.columns:
            pca_cols.append("Z_IQR")
        if "In_Outlier_Fit" in pca_df.columns:
            pca_cols.append("In_Outlier_Fit")
        self._pca_results = pca_df.select(pca_cols)

        # The plate of each sample, carried in the returned PCA frame.
        plate_col = getattr(self, "plate_id_col", None)
        if plate_col and plate_col in self.sample_metadata.collect_schema().names():
            plate_data = self.sample_metadata.select([self.sample_id_col, plate_col]).collect()
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

        logger.info(f"Outlier detection complete: {len(all_outliers)} total outliers")
        return pca_df, pca_obj, all_outliers

    @contextmanager
    def _models_on_identity_layer(self):
        """Points ``data_col`` at the identity layer while the sex/age models run."""
        data_col = self.data_col
        self.data_col = self.platform.identity_layer
        try:
            yield
        finally:
            self.data_col = data_col

    def _warn_if_data_col_off_identity_layer(self, data_col: str) -> None:
        if data_col != self.platform.identity_layer:
            logger.warning(
                "QC data column is '%s', but the sex/age models and the identity check "
                "stay on the identity layer '%s'",
                data_col,
                self.platform.identity_layer,
            )

    def remove_sample_sex_concordance(self, **kwargs):
        """Checks reported against predicted sex on the identity layer."""
        kwargs.setdefault("sex_model", self.config.sex_model)
        kwargs.setdefault("probability_threshold", self.config.sex_warn_prob)
        kwargs.setdefault("C", self.config.sex_c)
        kwargs.setdefault("cv_folds", self.config.model_cv_folds)
        kwargs.setdefault("knn_k", self.config.model_knn_k)
        kwargs.setdefault("max_missing", self.config.model_max_missing)
        kwargs.setdefault("feature_clip", self.config.sex_feature_clip)
        kwargs.setdefault("fit_exclude", self._sample_qc_data["outlier_fit_exclude"])
        kwargs.setdefault("random_seed", self.random_seed)
        with self._models_on_identity_layer():
            sex_results = sex_concordance.sample_sex_concordance(self, **kwargs)
            if "error" in sex_results:
                raise ValueError(f"sex concordance model failed: {sex_results['error']}")
            self.sex_concordance_results = sex_results

            if sex_results and "results_df" in sex_results:
                results_df = sex_results["results_df"]
                cols = [self.sample_id_col]
                if "Reported_Sex" in results_df.columns:
                    cols.append("Reported_Sex")
                # Always present; Sex_Exception is false without an exception.
                cols += [EXPECTED_SEX_COL, SEX_EXCEPTION_COL]
                if "Predicted_Male_Probability" in results_df.columns:
                    cols.append("Predicted_Male_Probability")
                self._sex_concordance_results = results_df.select(cols)
        return sex_results

    def remove_sample_age_concordance(self, **kwargs):
        """Checks reported against predicted age on the identity layer."""
        if self.config.age_model != "elasticnet":
            kwargs.setdefault("age_model", self.config.age_model)
        kwargs.setdefault("alpha", self.config.age_enet_alpha)
        kwargs.setdefault("l1_ratio", self.config.age_enet_l1_ratio)
        kwargs.setdefault("cv_folds", self.config.model_cv_folds)
        kwargs.setdefault("knn_k", self.config.model_knn_k)
        kwargs.setdefault("max_missing", self.config.model_max_missing)
        kwargs.setdefault("feature_clip", self.config.age_feature_clip)
        kwargs.setdefault("fit_exclude", self._sample_qc_data["outlier_fit_exclude"])
        kwargs.setdefault("random_seed", self.random_seed)
        defaults = inspect.signature(age_concordance.sample_age_concordance).parameters
        # Kept so reassignment can re-evaluate the age flags with the same settings.
        self._age_concordance_params = {
            k: kwargs.get(k, defaults[k].default) for k in ("age_col", "std_threshold")
        }
        with self._models_on_identity_layer():
            age_results = age_concordance.sample_age_concordance(self, **kwargs)
            if "error" in age_results:
                raise ValueError(f"age concordance model failed: {age_results['error']}")
            self.age_concordance_results = age_results

            if age_results and "results_df" in age_results:
                results_df = age_results["results_df"]
                # Visit_Adj_Age is the model target.
                cols = [self.sample_id_col]
                for col in (
                    "Visit_Adj_Age",
                    "Predicted_Age",
                    "Std_Predicted_Age",
                    "Age_Residual_Corrected",
                ):
                    if col in results_df.columns:
                        cols.append(col)
                self._age_concordance_results = results_df.select(cols)
        return age_results

    def remove_sample_unsuitable(self, **kwargs):
        """Flags samples unsuitable for analysis, such as post-treatment draws.

        Flagged samples are not removed.
        """
        return clinical.sample_unsuitable(self, **kwargs)

    def exclusion_summary(self, save_prefix: str | None = None, **kwargs):
        """Writes the exclusion table by arm, reason and visit.

        Args:
            save_prefix: Output prefix; defaults to the pipeline's.
            **kwargs: The arguments given to ``remove_sample_unsuitable``; pass the
                same ones.
        """
        return clinical.exclusion_summary(self, save_prefix or self.save_prefix, **kwargs)

    def save_results(self, save_prefix: str | None = None):
        """Writes the QC results to disk."""
        utils.save_qc_results(self, save_prefix or self.save_prefix)

    def __repr__(self) -> str:
        """Returns the class, project and data shape."""
        n_rows = self.data_full.select(pl.len()).collect().item()
        n_cols = len(self._schema)
        return f"{self.__class__.__name__}(project='{self.project_name}', n_rows={n_rows:,}, n_cols={n_cols})"


def _scan_file(path: str | Path) -> pl.LazyFrame:
    """Scans a parquet, CSV or TSV file (CSV and TSV may be gzipped) by its extension.

    ID columns come back as String (see ``proteomics_qc.io.scan_file``).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the extension is not supported.
    """
    return scan_file(path)


class OlinkQC(ProteomicsQC):
    """A QC pipeline run on Olink Explore HT NPX data.

    Reads a long-format NPX file (parquet, CSV or TSV) with the columns SampleID,
    OlinkID (one per reagent), NPX, SampleType (``SAMPLE``, ``PLATE_CONTROL``,
    ``SAMPLE_CONTROL``, ``NEGATIVE_CONTROL``, ...), AssayType (``assay``,
    ``inc_ctrl``, ``amp_ctrl``), SampleQC and AssayQC (PASS/WARN/FAIL), PlateID,
    Block, WellID, Assay (protein name), UniProt and Panel. The LOD calculation also
    uses Count, PCNormalizedNPX, ExtNPX, Normalization (Intensity or Plate control)
    and DataAnalysisRefID (reagent lot). QC reads the value column the config's
    ``[olink] value_column`` names (PCNormalizedNPX by default, or NPX).

    Args:
        npx_path: NPX file, or a glob that matches exactly one file. Exactly one of
            ``npx_path`` and ``data`` is given.
        data: NPX data already loaded.
        project_name: Project name, used in log lines.
        save_prefix: Prefix of the output files.
        group_col: Grouping column or columns.
        adsl: Subject-level clinical data.
        manifest: Sample manifest.
        usubjid_col: Subject ID column.
        clinical_vars: ADSL or manifest columns to carry into the analysis.
        force_categorical: Columns to treat as categorical.
        random_seed: Seed for every stochastic step.
        data_col: Data column to QC instead of the config's ``[olink]
            value_column``; the identity layer stays on that value column.
        config: Trial config for corrections and thresholds; defaults to the Olink
            defaults.

    Raises:
        ValueError: If both or neither of ``npx_path`` and ``data`` are given.
        FileNotFoundError: If ``npx_path`` matches no file.
        ValueError: If a wildcard ``npx_path`` matches more than one file.
    """

    platform = OLINK

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
        config=None,
    ):
        """Loads the NPX data, splits off the metadata and runs the base setup."""
        if npx_path is None and data is None:
            raise ValueError("Either npx_path or data must be provided")
        if npx_path is not None and data is not None:
            raise ValueError("Only one of npx_path or data can be provided")

        self.config = self._resolve_config(config)
        # The config's [olink] value_column sets the data column and the identity
        # layer for this run; the class keeps the default platform.
        self.platform = self.config.qc_platform

        self.sample_id_col = "SampleID"
        self.protein_id_col = "OlinkID"
        self.protein_name_col = "Assay"
        self.data_col = self.platform.default_data_col
        self.uniprot_col = "UniProt"
        self.plate_id_col = "PlateID"
        self.sample_type = "SAMPLE"
        self.qc_sample_type = "SAMPLE_CONTROL"

        if npx_path is not None:
            logger.info(f"Loading Olink data from: {npx_path}")
            if "*" in str(npx_path):
                matched_files = glob.glob(str(npx_path))
                if not matched_files:
                    raise FileNotFoundError(f"No files found matching pattern: {npx_path}")
                if len(matched_files) != 1:
                    raise ValueError(
                        f"Pattern {npx_path} must match exactly one file; "
                        f"it matches {len(matched_files)}: {sorted(matched_files)}"
                    )
                npx_path = matched_files[0]
                logger.info(f"Matched file: {npx_path}")

            raw_data = _scan_file(npx_path)
        else:
            raw_data = self._ensure_lazy(data) if not isinstance(data, pl.LazyFrame) else data

        # Sample and analyte attributes move to metadata tables to save memory.
        logger.info("Extracting sample and analyte metadata...")
        all_cols = set(raw_data.collect_schema().names())

        # A missing data column is reported by the base init's column check.
        data_cols = {self.sample_id_col, self.protein_id_col} | ({self.data_col} & all_cols)
        # Both value columns stay: the LOD needs PCNormalizedNPX and the output carries
        # both.
        for col in ["Count", OLINK_NPX, OLINK_PC_NPX, "ExtNPX", "Normalization"]:
            if col in all_cols:
                data_cols.add(col)
        # PlateID and Block complete the composite join keys.
        for col in ["PlateID", "Block", "WellID", "Panel", "SampleType"]:
            if col in all_cols:
                data_cols.add(col)
        # For the LOD calculation and filtering.
        for col in ["DataAnalysisRefID", "AssayType", "SampleQC", "AssayQC"]:
            if col in all_cols:
                data_cols.add(col)

        sample_cols = {self.sample_id_col}
        for col in ["SampleType", "PlateID", "Block", "SampleQC", "WellID"]:
            if col in all_cols:
                sample_cols.add(col)

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

        # Materialized once so later uses do not re-scan the file. One row per ID,
        # chosen by a full-row sort (lowest PlateID, then Block, ...) so the kept row
        # does not depend on input order.
        sample_order = [
            self.sample_id_col,
            *sorted(sample_cols & {self.plate_id_col}),
            *sorted(sample_cols - {self.sample_id_col, self.plate_id_col}),
        ]
        analyte_order = [
            self.protein_id_col,
            *sorted(analyte_cols - {self.protein_id_col}),
        ]
        self.sample_metadata = (
            raw_data.select(sample_order)
            .collect()
            .sort(sample_order, nulls_last=True)
            .unique(subset=[self.sample_id_col], keep="first", maintain_order=True)
            .lazy()
        )
        self.analyte_metadata = (
            raw_data.select(analyte_order)
            .collect()
            .sort(analyte_order, nulls_last=True)
            .unique(subset=[self.protein_id_col], keep="first", maintain_order=True)
            .lazy()
        )

        self.data_full = raw_data.select(list(data_cols))

        # Resolves generic CONTROL rows by SampleID prefix.
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
                self.data_full = self.data_full.with_columns(reclassify_expr)

        logger.info(
            f"Data partitioned: {len(data_cols)} measurement cols, "
            f"{len(sample_cols)} sample metadata cols, "
            f"{len(analyte_cols)} analyte metadata cols"
        )

        if manifest is not None:
            manifest = validate_manifest_schema(
                self._ensure_lazy(manifest), self.sample_id_col, usubjid_col
            )
        if self.config.corrections and manifest is not None and adsl is not None:
            adsl = self._apply_corrections(manifest, adsl, usubjid_col)

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

        logger.debug("Partitioning by assay type...")
        self.protein_assays = self.data_full.join(
            self.analyte_metadata.filter(pl.col("AssayType") == "assay").select(
                self.protein_id_col
            ),
            on=self.protein_id_col,
            how="semi",
        )

        logger.info("OLINK EXPLORE HT QC PIPELINE")
        logger.info(
            f"Dataset: {self.qc_results['n_samples']} samples, "
            f"{self.qc_results['n_proteins']} proteins, "
            f"{self.qc_results['n_plates']} plates"
        )

    # ── Olink QC steps ──
    def remove_sample_qc(self, **kwargs):
        """Flags samples by the vendor's SampleQC."""
        sample_qc_status = quality_control.sample_qc_olink(self, **kwargs)
        self.sample_qc_status = sample_qc_status
        return sample_qc_status

    def filter_assay_qc(self, **kwargs):
        """Flags assays by the vendor's AssayQC."""
        assay_results = quality_control.assay_qc_olink(self, **kwargs)
        return assay_results

    def olink_lod(self, **kwargs):
        """Computes the limit of detection (LOD) of each assay."""
        lod_results = detection.olink_lod(self, **kwargs)
        self.lod_results = lod_results
        return lod_results

    def filter_assay_duplicates(self, **kwargs):
        """Correlates duplicate assays (one protein, several OlinkIDs).

        Raises:
            ValueError: If there is no OlinkID column. The duplicate assay flags feed the
                identity matrix, so the check cannot be skipped.
        """
        if not self._has_column("OlinkID"):
            raise ValueError(
                "OlinkID column not found: the duplicate assay flags feed the identity "
                "matrix, so the check cannot be skipped"
            )
        return duplication.assay_duplicates(self, **kwargs)


def adat_process_steps(path: str | Path) -> str | None:
    """Returns the ``!ProcessSteps`` value of an ADAT header, or None if absent.

    Reads header lines only, stopping at the first ``^`` section after ``^HEADER``.
    """
    with open(path) as f:
        for line in f:
            if line.startswith("^") and not line.startswith("^HEADER"):
                return None
            key, _, value = line.rstrip("\r\n").partition("\t")
            if key == "!ProcessSteps":
                return value
    return None


def anml_steps(process_steps: str) -> list[str]:
    """Returns the study samples' ANML steps in a ``!ProcessSteps`` value.

    These are the comma-separated steps starting with ``anml`` except ``anmlQC``,
    which applies ANML to the QC samples only.
    """
    steps = [s.strip() for s in process_steps.split(",")]
    return [s for s in steps if s.lower().startswith("anml") and s.lower() != "anmlqc"]


def _require_non_anml(path: str, allow_anml: bool) -> None:
    """Checks that the ADAT at ``path`` is not ANML-normalized.

    Raises:
        ValueError: If ``allow_anml`` is False and the file is ANML-normalized or its header
            states no ``!ProcessSteps``.
    """
    _check_non_anml(adat_process_steps(path), path, allow_anml)


def _check_non_anml(steps: str | None, path: str, allow_anml: bool) -> None:
    """Runs the ANML check of ``_require_non_anml`` on a ``!ProcessSteps`` value.

    ``steps`` is None when the source states none; ``path`` names the source in
    messages.
    """
    if steps is None:
        if not allow_anml:
            raise ValueError(
                f"{path}: no !ProcessSteps in the ADAT header, so ANML normalization cannot be ruled out. "
                "Use an ADAT that states its steps, or set allow_anml = true in the trial TOML."
            )
        return
    anml = anml_steps(steps)
    if anml and not allow_anml:
        raise ValueError(
            f"{path} is ANML-normalized (!ProcessSteps: {steps}; ANML steps "
            f"{anml}).  Use the non-ANML ADAT, or set allow_anml = true in the "
            "trial TOML."
        )
    if anml:
        logger.warning("%s: ANML-normalized ADAT accepted (allow_anml = true)", path)


# The ADAT analyte column holding the dilution group (20, 0.5, 0.005; 0 for hyb
# controls).
DILUTION_COL = "Dilution"


class SomaQC(ProteomicsQC):
    """A QC pipeline run on SomaScan data.

    Reads an ADAT (through ``somadata``), a bundle directory written by
    ``write_bundle``, or a long-format parquet, CSV or TSV file. The long tables
    have the columns SampleId, SeqId (the protein ID), SomaId, Target (short target
    name), UniProt, RFU and log10_scaled_RFU, plus the delivered RFU detection uses.

    Args:
        adat_path: ADAT file, bundle directory or long-format file. Exactly one of
            ``adat_path`` and ``data`` is given.
        data: Long-format data already loaded; its metadata tables hold only the
            IDs.
        project_name: Project name, used in log lines.
        save_prefix: Prefix of the output files.
        group_col: Grouping column or columns.
        adsl: Subject-level clinical data.
        manifest: Sample manifest.
        usubjid_col: Subject ID column.
        clinical_vars: ADSL or manifest columns to carry into the analysis.
        force_categorical: Columns to treat as categorical.
        random_seed: Seed for every stochastic step.
        data_col: Data column to QC instead of ``log10_scaled_RFU`` (for example
            ``RFU``).
        config: Trial config for corrections and thresholds; defaults to the
            SomaScan defaults.

    Raises:
        ValueError: If both or neither of ``adat_path`` and ``data`` are given, if
            an ADAT is ANML-normalized when that is not allowed, or if the data lacks
            the delivered RFU column.
    """

    platform = SOMA

    # Column names of the long tables, the same on the ADAT and bundle paths.
    SAMPLE_ID_COL = "SampleId"
    PROTEIN_ID_COL = "SeqId"
    PROTEIN_NAME_COL = "Target"
    UNIPROT_COL = "UniProt"
    STUDY_SAMPLE_TYPE = "Sample"
    BUNDLE_FILES = (
        "measurements.parquet",
        "sample_metadata.parquet",
        "analyte_metadata.parquet",
    )
    # The ADAT's !ProcessSteps, when it states one.
    BUNDLE_STEPS_FILE = "process_steps.txt"

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
        config=None,
    ):
        """Loads the data into long tables and runs the base setup."""
        if adat_path is None and data is None:
            raise ValueError("Either adat_path or data must be provided")
        if adat_path is not None and data is not None:
            raise ValueError("Only one of adat_path or data can be provided")

        self.config = self._resolve_config(config)

        self.sample_id_col = self.SAMPLE_ID_COL
        self.protein_id_col = self.PROTEIN_ID_COL
        self.protein_name_col = self.PROTEIN_NAME_COL
        self.data_col = self.platform.default_data_col
        self.uniprot_col = self.UNIPROT_COL
        self.plate_id_col = "PlateId"
        self.sample_type = self.STUDY_SAMPLE_TYPE
        self.qc_sample_type = "QC"

        if adat_path is not None:
            path_str = str(adat_path)
            ext = os.path.splitext(path_str)[1].lower()

            if os.path.isdir(path_str):
                logger.info(f"Reading SomaScan bundle: {adat_path}")
                self.data_full, self.sample_metadata, self.analyte_metadata = (
                    self._bundle_to_long_format(Path(path_str))
                )
            elif ext == ".adat":
                _require_non_anml(path_str, self.config.allow_anml)
                logger.info(f"Reading ADAT file: {adat_path}")
                t0 = time.time()
                adat_raw = somadata.read_adat(path_str)
                logger.info(
                    f"Loaded ADAT file in {time.time() - t0:.2f}s - shape: {adat_raw.shape}"
                )

                t1 = time.time()
                self.data_full, self.sample_metadata, self.analyte_metadata = (
                    self._adat_to_long_format(adat_raw)
                )
                logger.info(f"Converted to long format in {time.time() - t1:.2f}s")
            else:
                logger.info(f"Loading SomaScan data from pre-processed file: {adat_path}")
                self.data_full = _scan_file(adat_path)
                logger.info(
                    "Loaded pre-processed data - sample_metadata and analyte_metadata will be minimal"
                )
                self.sample_metadata = self.data_full.select(self.sample_id_col).unique()
                self.analyte_metadata = self.data_full.select(self.protein_id_col).unique()
        else:
            self.data_full = self._ensure_lazy(data) if not isinstance(data, pl.LazyFrame) else data
            logger.warning(
                "Using pre-provided data - sample_metadata and analyte_metadata will be minimal"
            )
            self.sample_metadata = self.data_full.select(self.sample_id_col).unique().lazy()
            self.analyte_metadata = self.data_full.select(self.protein_id_col).unique().lazy()

        # Detection compares the delivered RFU with the eLOD; fail before any
        # QC step runs.
        if quality_control.DELIVERED_RFU_COL not in self.data_full.collect_schema().names():
            raise ValueError(
                f"SomaScan data lacks the {quality_control.DELIVERED_RFU_COL!r} column (detection compares the "
                "delivered non-ANML RFU with the buffer-well eLOD). Read the ADAT or a bundle, or add "
                f"{quality_control.DELIVERED_RFU_COL!r} (the delivered non-ANML RFU) to the long data"
            )

        if manifest is not None:
            manifest = validate_manifest_schema(
                self._ensure_lazy(manifest), self.sample_id_col, usubjid_col
            )
        if self.config.corrections and manifest is not None and adsl is not None:
            adsl = self._apply_corrections(manifest, adsl, usubjid_col)

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

        # The long export needs PlateId on samples; the ADAT keeps it in the sample
        # index (sample_metadata), not the measurements.
        if self.plate_id_col not in self.samples.collect_schema().names():
            sm_names = self.sample_metadata.collect_schema().names()
            samples_names = self.samples.collect_schema().names()
            if self.plate_id_col in sm_names:
                join_key = (
                    "_sample_idx"
                    if "_sample_idx" in sm_names and "_sample_idx" in samples_names
                    else self.sample_id_col
                )
                plate_map = self.sample_metadata.select([join_key, self.plate_id_col]).unique(
                    subset=[join_key]
                )
                self.samples = self.samples.join(plate_map, on=join_key, how="left")
                logger.info(f"Joined {self.plate_id_col} from sample_metadata into samples")
            elif self.manifest is not None:
                manifest_lazy = self._ensure_lazy(self.manifest)
                if self.plate_id_col in manifest_lazy.collect_schema().names():
                    plate_map = manifest_lazy.select(
                        [self.sample_id_col, self.plate_id_col]
                    ).unique(maintain_order=True)
                    plates_per_sample = plate_map.group_by(self.sample_id_col).len().collect()
                    if (plates_per_sample["len"] > 1).any():
                        raise ValueError("the manifest names more than one plate for a sample")
                    self.samples = self.samples.join(plate_map, on=self.sample_id_col, how="left")
                    logger.info(f"Joined {self.plate_id_col} from manifest into samples")

        logger.info("SOMASCAN QC PIPELINE")
        logger.info(
            f"Dataset: {self.qc_results['n_samples']} samples, "
            f"{self.qc_results['n_proteins']} proteins, "
            f"{self.qc_results['n_plates']} plates"
        )

    def _adat_to_long_format(
        self, adat: somadata.Adat
    ) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
        """Returns the measurements, sample metadata and analyte metadata of an ADAT."""
        return self._long_tables(*self._adat_tables(adat))

    @classmethod
    def _adat_tables(cls, adat: somadata.Adat) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray]:
        """Returns the ADAT's sample metadata, analyte metadata and RFU matrix.

        The sample metadata gains ``_sample_idx``. The analytes are the human
        proteins, in ADAT column order; the matrix is samples x analytes.

        Raises:
            ValueError: If the analyte metadata lacks the Type or Organism column.
        """
        sample_meta_df = adat.index.to_frame(index=False).reset_index(drop=True)
        analyte_meta_df = adat.columns.to_frame(index=False).reset_index(drop=True)

        missing = [c for c in ("Type", "Organism") if c not in analyte_meta_df.columns]
        if missing:
            raise ValueError(
                f"analyte metadata lacks {' and '.join(missing)}: the Type and Organism "
                "columns are needed to drop controls and non-human analytes"
            )
        filter_mask = (analyte_meta_df["Type"] == "Protein") & (
            analyte_meta_df["Organism"] == "Human"
        )
        filtered_indices = analyte_meta_df.index[filter_mask].tolist()
        analyte_meta_df = analyte_meta_df[filter_mask].reset_index(drop=True)
        rfu_values_filtered = adat.values[:, filtered_indices]

        sample_meta_pl = pl.from_pandas(sample_meta_df)
        analyte_meta_pl = pl.from_pandas(analyte_meta_df)

        analyte_meta_pl = analyte_meta_pl.drop(["Type", "Organism"])

        essential_analyte_cols = [
            cls.PROTEIN_ID_COL,
            "SomaId",
            cls.PROTEIN_NAME_COL,
            cls.UNIPROT_COL,
            "EntrezGeneSymbol",
            "ColCheck",
            # The dilution group of each analyte.
            DILUTION_COL,
        ]
        analyte_cols_to_keep = [c for c in essential_analyte_cols if c in analyte_meta_pl.columns]
        analyte_meta_pl = analyte_meta_pl.select(analyte_cols_to_keep)

        logger.debug(
            f"Kept {len(analyte_cols_to_keep)} analyte metadata columns: {', '.join(analyte_cols_to_keep)}"
        )

        # The row index tells replicate tubes of one SampleId apart.
        sample_meta_pl = sample_meta_pl.with_row_index("_sample_idx")
        return (
            sample_meta_pl,
            analyte_meta_pl,
            np.asarray(rfu_values_filtered, dtype=np.float64),
        )

    @classmethod
    def _long_tables(
        cls,
        sample_meta_pl: pl.DataFrame,
        analyte_meta_pl: pl.DataFrame,
        rfu: np.ndarray,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
        """Returns the long measurements and the two metadata tables as lazy frames.

        The ADAT path and the bundle path both come through here. ``RFU`` and
        ``RFU_delivered`` both hold the delivered non-ANML values, the ones the buffer
        wells' eLOD is on. ``log10_scaled_RFU`` is z-scored per analyte over the
        biological samples. The ``Dilution`` column stays in the analyte metadata, and
        the assay summary writes it.

        Args:
            sample_meta_pl: Sample metadata with ``_sample_idx``.
            analyte_meta_pl: Analyte metadata.
            rfu: Samples x analytes, in the row orders of the two metadata tables.

        Raises:
            ValueError: If no sample is a study sample.
        """
        n_samples = len(sample_meta_pl)
        n_analytes = len(analyte_meta_pl)
        logger.info(f"Converting to long format: {n_samples} samples × {n_analytes} analytes")
        # Mean and SD per analyte come from the biological samples only; buffer,
        # calibrator and QC wells are scaled with them but never enter them.
        is_study = (
            (sample_meta_pl["SampleType"] == cls.STUDY_SAMPLE_TYPE).fill_null(False).to_numpy()
        )
        if not is_study.any():
            raise ValueError(f"The data has no sample with SampleType {cls.STUDY_SAMPLE_TYPE!r}")
        # Row-major ravel pairs each sample with every analyte in order, matching
        # np.repeat and np.tile.
        log10 = np.log10(rfu)
        study = log10[is_study]
        scaled = (log10 - study.mean(axis=0)) / study.std(axis=0, ddof=1)
        measurements = pl.DataFrame(
            {
                "_sample_idx": np.repeat(sample_meta_pl["_sample_idx"].to_numpy(), n_analytes),
                cls.SAMPLE_ID_COL: np.repeat(
                    sample_meta_pl[cls.SAMPLE_ID_COL].to_numpy(), n_analytes
                ),
                cls.PROTEIN_ID_COL: np.tile(
                    analyte_meta_pl[cls.PROTEIN_ID_COL].to_numpy(), n_samples
                ),
                "RFU": rfu.ravel(),
                quality_control.DELIVERED_RFU_COL: rfu.ravel(),
                "log10_scaled_RFU": scaled.ravel(),
            }
        )

        logger.info(
            f"Conversion complete: {measurements.height:,} measurements, "
            f"{n_samples:,} samples, {n_analytes:,} analytes"
        )

        return measurements.lazy(), sample_meta_pl.lazy(), analyte_meta_pl.lazy()

    @classmethod
    def write_bundle(cls, adat_path: str | Path, out_dir: str | Path) -> Path:
        """Writes the ADAT's long tables to ``out_dir`` as a bundle ``SomaQC`` can read.

        ``measurements.parquet`` holds ``_sample_idx``, SampleId, SeqId and RFU; the two
        metadata tables are the ones the ADAT path builds; ``process_steps.txt`` holds
        the ADAT's ``!ProcessSteps`` when it has one. A bundle run gives the same
        outputs as the ADAT run, and a bundle can be edited (values, rows) where an
        ADAT cannot.

        Returns:
            The bundle directory.
        """
        adat_path, out = Path(adat_path), Path(out_dir)
        sample_meta, analyte_meta, rfu = cls._adat_tables(somadata.read_adat(str(adat_path)))
        out.mkdir(parents=True, exist_ok=True)
        n_analytes = len(analyte_meta)
        pl.DataFrame(
            {
                "_sample_idx": np.repeat(sample_meta["_sample_idx"].to_numpy(), n_analytes),
                cls.SAMPLE_ID_COL: np.repeat(sample_meta[cls.SAMPLE_ID_COL].to_numpy(), n_analytes),
                cls.PROTEIN_ID_COL: np.tile(
                    analyte_meta[cls.PROTEIN_ID_COL].to_numpy(), len(sample_meta)
                ),
                "RFU": rfu.ravel(),
            }
        ).write_parquet(out / cls.BUNDLE_FILES[0])
        sample_meta.write_parquet(out / cls.BUNDLE_FILES[1])
        analyte_meta.write_parquet(out / cls.BUNDLE_FILES[2])
        steps = adat_process_steps(adat_path)
        if steps is not None:
            (out / cls.BUNDLE_STEPS_FILE).write_text(steps)
        return out

    def _bundle_to_long_format(
        self, directory: Path
    ) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
        """Returns the three long tables from a bundle directory (see ``write_bundle``).

        RFU is placed in the matrix by ``_sample_idx`` and SeqId, so row order in the
        files does not matter; the sample and analyte order of the metadata tables
        fixes the matrix order, as the ADAT's does.

        Raises:
            FileNotFoundError: If a bundle file is missing.
            ValueError: If a table lacks a column or has duplicate keys, if the
                measurements are not exactly one RFU per sample row and analyte, if a
                SampleId disagrees with the sample metadata, or if the bundle is
                ANML-normalized and the config does not allow it.
        """
        names = self.BUNDLE_FILES
        missing = [n for n in names if not (directory / n).is_file()]
        if missing:
            raise FileNotFoundError(f"SomaScan bundle {directory} lacks {missing}")
        steps_file = directory / self.BUNDLE_STEPS_FILE
        _check_non_anml(
            steps_file.read_text() if steps_file.is_file() else None,
            str(directory),
            self.config.allow_anml,
        )
        measurements = pl.read_parquet(directory / names[0])
        sample_meta = pl.read_parquet(directory / names[1])
        analyte_meta = pl.read_parquet(directory / names[2])
        sid, pid = self.sample_id_col, self.protein_id_col
        for name, frame, cols in (
            (names[0], measurements, ("_sample_idx", sid, pid, "RFU")),
            (names[1], sample_meta, ("_sample_idx", sid, "SampleType")),
            (names[2], analyte_meta, (pid,)),
        ):
            absent = [c for c in cols if c not in frame.columns]
            if absent:
                raise ValueError(f"{directory / name} lacks columns {absent}")
        for name, frame, key in (
            (names[1], sample_meta, "_sample_idx"),
            (names[2], analyte_meta, pid),
        ):
            if frame[key].is_duplicated().any():
                raise ValueError(f"{directory / name}: duplicate {key}")
        n_samples, n_analytes = sample_meta.height, analyte_meta.height
        placed = measurements.join(
            sample_meta.select(
                "_sample_idx",
                pl.col(sid).alias("_sid"),
                pl.int_range(pl.len()).alias("_i"),
            ),
            on="_sample_idx",
            how="inner",
        ).join(
            analyte_meta.select(pid, pl.int_range(pl.len()).alias("_j")),
            on=pid,
            how="inner",
        )
        if (
            placed.height != measurements.height
            or placed.height != n_samples * n_analytes
            or placed.select(pl.struct("_i", "_j").is_duplicated().any()).item()
        ):
            raise ValueError(
                f"{directory / names[0]} must hold one RFU per sample row and analyte "
                f"({n_samples} x {n_analytes}); it has {measurements.height} rows, "
                f"{placed.height} of them placed"
            )
        if (placed[sid] != placed["_sid"]).any():
            raise ValueError(f"{directory / names[0]}: SampleId disagrees with the sample metadata")
        rfu = np.empty((n_samples, n_analytes), dtype=np.float64)
        rfu[placed["_i"].to_numpy(), placed["_j"].to_numpy()] = placed["RFU"].to_numpy()
        return self._long_tables(sample_meta, analyte_meta, rfu)

    # ── SomaScan QC steps ──
    def remove_sample_qc(self, **kwargs):
        """Flags samples by the ADAT's RowCheck."""
        sample_qc_status = quality_control.sample_qc_soma(self, **kwargs)
        self.sample_qc_status = sample_qc_status
        return sample_qc_status

    def filter_assay_qc(self):
        """Flags assays by the ADAT's ColCheck."""
        return quality_control.assay_qc_soma(self)

    def filter_assay_cv(self, **kwargs):
        """Flags assays by CV; ``finalize_assay_flags`` combines the flags."""
        return detection.assay_cv(self, **kwargs)

    def filter_assay_detection(self, **kwargs):
        """Flags assays by detection rate; ``finalize_assay_flags`` combines the flags."""
        det_results = detection.assay_detection(self, **kwargs)
        self.detection_results = det_results
        return self

    def filter_assay_saturation(self, **kwargs):
        """Flags analytes saturated in more than a fraction of samples.

        An analyte is flagged when more than ``sample_frac`` (default 50%) of the
        samples exceed the saturation RFU threshold.
        """
        saturation_results = quality_control.assay_saturation_soma(self, **kwargs)
        self.saturation_results = saturation_results
        return saturation_results

    def filter_assay_sb_ratio(self, **kwargs):
        """Flags analytes with a low signal-to-background ratio.

        The ratio is the median over samples divided by the median over buffer wells.
        """
        sb_results = quality_control.assay_sb_ratio_soma(self, **kwargs)
        self.sb_ratio_results = sb_results
        return sb_results

    def finalize_assay_flags(self):
        """Consolidates individual assay-check results on the QC instance.

        Call after ``filter_assay_qc``, ``filter_assay_cv``,
        ``filter_assay_detection``, ``filter_assay_saturation`` and
        ``filter_assay_sb_ratio``.
        """
        return quality_control.finalize_assay_flags_soma(self)


def visitage_frame(merged: pl.LazyFrame, uid: str, baseline_visit: float | None) -> pl.LazyFrame:
    """Adds ``VISITAGE`` to ``merged``: AGE plus the days since the anchor draw / 365.

    The anchor is the participant's draw at ``baseline_visit``, or its earliest draw
    when it has none there; with no ``baseline_visit`` it is the draw at the
    participant's lowest VISITNUM.

    Args:
        merged: Rows with ``uid``, ``VISITNUM``, ``ADT`` and ``AGE``.
        uid: Subject ID column.
        baseline_visit: The config's ``[visits]`` baseline, or None.

    Raises:
        ValueError: If a subject has a dated draw but no dated anchor.
    """
    # With a baseline, a participant with no sample there anchors on all of its
    # samples, so the earliest draw wins.
    visit = pl.col("VISITNUM").cast(pl.Float64)
    if baseline_visit is None:
        is_anchor = visit == visit.min().over(uid)
    else:
        at_baseline = (visit == baseline_visit).fill_null(False)
        is_anchor = at_baseline | ~at_baseline.any().over(uid)

    baseline_adt = (
        merged.filter(is_anchor)
        .sort([uid, "ADT"], nulls_last=True)
        .select([uid, pl.col("ADT").alias("baseline_adt")])
        .unique(subset=[uid], keep="first", maintain_order=True)
    )

    # Such a subject would silently keep its unadjusted AGE.
    n_undated = (
        merged.filter(pl.col(uid).is_not_null() & pl.col("ADT").is_not_null())
        .select(uid)
        .unique()
        .join(
            baseline_adt.filter(pl.col("baseline_adt").is_null()),
            on=uid,
            how="semi",
        )
        .collect()
        .height
    )
    if n_undated:
        raise ValueError(
            f"{n_undated} subject(s) have a dated draw but a null ADT on the draw "
            "that anchors VISITAGE; the age cannot be visit-adjusted"
        )

    result = (
        merged.join(baseline_adt, on=uid, how="left")
        .with_columns(
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
        .drop("baseline_adt")
    )

    return result


# ── Identity reassignment ────────────────────────────────────────────────────

_AGE_WARN_REASONS = (
    WARN_AGE_CONCORDANCE,
    WARN_AGE_PREDICTION,
    WARN_AGE_CONSISTENCY,
)


def reassigned_frame(
    merged: pl.DataFrame,
    adsl: pl.DataFrame,
    moves: pl.DataFrame,
    *,
    sid: str,
    uid: str,
    adsl_derived_cols: list[str],
    visits_baseline: float | None,
) -> pl.DataFrame:
    """Returns ``merged`` with moved tubes placed on their inferred subject and visit.

    A pure function: ``apply_reassignments`` stores its result and refreshes the rest
    of the QC state. The result keeps ``merged``'s columns and row order, with:

    - the inferred USUBJID and VISITNUM, and the labeled values in
      ``USUBJID_LABELED`` / ``VISITNUM_LABELED`` (added when absent, null for a tube
      that did not move);
    - every ADSL-derived column re-joined on the new USUBJID (columns taken from the
      manifest stay with the tube);
    - the tube's own ADT, unchanged, and ``ADT_SOURCE`` set to
      ``reassigned`` for a moved tube when ``merged`` carries it;
    - ``VISITAGE`` recomputed against ``visits_baseline`` when ``merged`` had it.

    Args:
        merged: ``adsl_merged``, collected.
        adsl: ADSL with ``uid`` and every ``adsl_derived_cols`` column.
        moves: One row per moved tube, with ``sid``, ``USUBJID_INFERRED`` and
            ``VISITNUM_INFERRED``.
        sid: Sample ID column.
        uid: Subject ID column.
        adsl_derived_cols: Columns ``merged`` took from ADSL by subject.
        visits_baseline: The config's ``[visits]`` baseline, or None.
    """
    visit_dtype = merged.schema["VISITNUM"]
    moves = moves.with_columns(pl.col("VISITNUM_INFERRED").cast(visit_dtype))

    columns = merged.columns
    had_visitage = "VISITAGE" in columns
    moved = pl.col("USUBJID_INFERRED").is_not_null()
    label_cols = {
        "USUBJID_LABELED": (uid, pl.Utf8),
        "VISITNUM_LABELED": ("VISITNUM", visit_dtype),
    }
    label_exprs = [
        pl.when(moved)
        .then(pl.col(src))
        .otherwise(pl.col(name) if name in columns else pl.lit(None, dtype=dtype))
        .alias(name)
        for name, (src, dtype) in label_cols.items()
    ]
    if "ADT_SOURCE" in columns:
        label_exprs.append(
            pl.when(moved)
            .then(pl.lit("reassigned"))
            .otherwise(pl.col("ADT_SOURCE"))
            .alias("ADT_SOURCE")
        )
    merged = (
        merged.join(moves, on=sid, how="left", validate="m:1")
        .with_columns(label_exprs)
        .with_columns(
            pl.coalesce("USUBJID_INFERRED", uid).alias(uid),
            pl.coalesce("VISITNUM_INFERRED", "VISITNUM").alias("VISITNUM"),
        )
        .drop("USUBJID_INFERRED", "VISITNUM_INFERRED", *adsl_derived_cols)
        .drop("VISITAGE", strict=False)
    )
    if adsl_derived_cols:
        merged = merged.join(
            adsl.select([uid, *adsl_derived_cols]), on=uid, how="left", validate="m:1"
        )
    out_cols = [c for c in columns if c != "VISITAGE"] + [c for c in label_cols if c not in columns]
    merged = merged.select(out_cols)
    if had_visitage:
        merged = (
            visitage_frame(merged.lazy(), uid, visits_baseline)
            .select(columns + [c for c in label_cols if c not in columns])
            .collect()
        )
    return merged


def apply_reassignments(qc: ProteomicsQC, events: pl.DataFrame) -> None:
    """Moves reassigned samples to their inferred subject and visit.

    ``adsl_merged`` is rebuilt by ``reassigned_frame``. ``group_data`` is rebuilt
    and ``FAIL_No_Treatment_Arm`` re-judged from it for the moved samples. Sex and
    age WARN flags, ``Reported_Sex`` and ``Visit_Adj_Age`` are re-evaluated against
    the inferred subject from the stored out-of-fold predictions; the models are not
    re-fitted.

    Args:
        qc: The pipeline whose state is updated in place.
        events: Identity events; the ``action == "reassigned"`` rows, with
            ``SampleID``, ``USUBJID_INFERRED`` and ``VISITNUM_INFERRED``, are moved.

    Raises:
        ValueError: If a reassigned sample is not in ``adsl_merged``.
    """
    sid, uid = qc.sample_id_col, qc.usubjid_col
    moves = events.filter(pl.col("action") == "reassigned").select(
        pl.col("SampleID").alias(sid), "USUBJID_INFERRED", "VISITNUM_INFERRED"
    )
    if moves.height == 0:
        return

    merged = qc.adsl_merged.collect()
    unknown = set(moves[sid]) - set(merged[sid])
    if unknown:
        raise ValueError(f"Reassigned samples not in adsl_merged: {', '.join(sorted(unknown))}")
    # Subjects whose sample sets change: the labeled and the inferred ones.
    touched = set(merged.filter(pl.col(sid).is_in(moves[sid].to_list()))[uid].to_list()) | set(
        moves["USUBJID_INFERRED"].to_list()
    )
    qc.adsl_merged = reassigned_frame(
        merged,
        qc._ensure_lazy(qc.adsl).select([uid, *qc.adsl_derived_cols]).collect(),
        moves,
        sid=sid,
        uid=uid,
        adsl_derived_cols=qc.adsl_derived_cols,
        visits_baseline=qc.config.visits_baseline,
    ).lazy()
    qc.__dict__.pop("bridge_sample_ids", None)  # cached on the old USUBJIDs
    qc._setup_groups(qc.group_col_input)
    logger.info(f"Applied {moves.height} identity reassignments to adsl_merged")

    _reevaluate_arm_flags(qc, moves[sid].to_list())
    _reevaluate_sex_flags(qc, moves[sid].to_list())
    _reevaluate_age_flags(qc, touched)


def _reevaluate_arm_flags(qc: ProteomicsQC, sample_ids: list[str]) -> None:
    """Re-judges FAIL_No_Treatment_Arm for moved samples from the new ``group_data``.

    A tube labeled with a subject that has no arm and moved to one that has an arm
    loses the flag; the reverse gains it. FAIL_No_Manifest_Match needs no re-judging:
    such a tube never reaches identity, which joins on the manifest.
    """
    if not qc.has_groups or qc.group_data is None:
        return
    sid = qc.sample_id_col
    no_arm = set(
        qc.group_data.filter(
            pl.col(sid).cast(pl.Utf8).is_in(sample_ids) & pl.col(qc.group_col).is_null()
        )
        .select(pl.col(sid).cast(pl.Utf8))
        .collect()[sid]
        .to_list()
    )
    qc._replace_sample_flags(sample_ids, {clinical.FAIL_NO_TREATMENT_ARM: no_arm})
    logger.info(
        f"Arm re-judged for {len(sample_ids)} reassigned samples: {len(no_arm)} with no arm"
    )


def _reevaluate_sex_flags(qc: ProteomicsQC, sample_ids: list[str]) -> None:
    """Re-checks WARN_Sex_Concordance and Reported_Sex against the current subject.

    The WARN compares with the new subject's expected protein sex; Reported_Sex is the new subject's recorded sex,
    and the two provenance columns follow the new subject.
    """
    results = getattr(qc, "sex_concordance_results", None)
    if qc.sex_oof_predictions is None or not results or "results_df" not in results:
        return
    sid, uid = qc.sample_id_col, qc.usubjid_col
    sex_col = results["sex_col"]
    labels = with_expected_sex(
        qc.adsl_merged.filter(pl.col(sid).is_in(sample_ids)).select(sid, uid, sex_col).collect(),
        qc.expected_sex,
        uid,
        sex_col,
        EXPECTED_SEX_COL,
    )
    preds = qc.sex_oof_predictions.join(labels, on=sid, how="inner", validate="1:1")
    if preds.height == 0:
        return
    preds, _ = sex_concordance.detect_sex_discordance(
        preds, results["probability_threshold"], EXPECTED_SEX_COL
    )
    discordant = set(preds.filter(pl.col("Is_Discordant").fill_null(False))[sid].to_list())
    qc._replace_sample_flags(preds[sid].to_list(), {WARN_SEX_CONCORDANCE: discordant})
    resolved = sorted(set(preds[sid].to_list()) - discordant)
    logger.info(
        f"Sex re-evaluated for {preds.height} reassigned samples: "
        f"{len(discordant)} still discordant, {len(resolved)} concordant"
    )
    new_sex = dict(preds.select(sid, sex_col).iter_rows())
    updated = [
        pl.col(sid)
        .replace_strict(new_sex, default=pl.col("Reported_Sex"), return_dtype=pl.Utf8)
        .alias("Reported_Sex")
    ]
    new_expected = dict(preds.select(sid, EXPECTED_SEX_COL).iter_rows())
    excepted = set() if qc.expected_sex is None else set(qc.expected_sex[uid].to_list())
    new_exception = {s: u in excepted for s, u in preds.select(sid, uid).iter_rows()}
    updated += [
        pl.col(sid)
        .replace_strict(new_expected, default=pl.col(EXPECTED_SEX_COL), return_dtype=pl.Utf8)
        .alias(EXPECTED_SEX_COL),
        pl.col(sid)
        .replace_strict(new_exception, default=pl.col(SEX_EXCEPTION_COL), return_dtype=pl.Boolean)
        .alias(SEX_EXCEPTION_COL),
    ]
    qc._sex_concordance_results = qc._sex_concordance_results.with_columns(updated)


def _reevaluate_age_flags(qc: ProteomicsQC, subjects: set) -> None:
    """Re-checks the age WARN flags and Visit_Adj_Age for every sample of ``subjects``.

    Uses the stored out-of-fold Predicted_Age, the effective residual threshold
    and the consistency settings of the original run; residuals and the
    per-subject consistency are recomputed on the current labels.
    """
    results = getattr(qc, "age_concordance_results", None)
    if qc.age_oof_predictions is None or not results or "results_df" not in results:
        return
    sid = qc.sample_id_col
    age_col = qc._age_concordance_params["age_col"]
    labels = qc.adsl_merged.select(
        sid,
        pl.col(qc.usubjid_col).alias("USUBJID"),
        pl.col(age_col).alias("Visit_Adj_Age"),
    ).collect()
    old = results["results_df"]
    current = (
        old.drop(
            "USUBJID",
            "Visit_Adj_Age",
            "Residual",
            "Abs_Residual",
            "Is_Discordant",
            "From_Inconsistent_Participant",
        )
        .drop("Std_Predicted_Age", strict=False)
        .join(
            labels.filter(pl.col(sid).is_in(old[sid].to_list())),
            on=sid,
            how="inner",
            validate="1:1",
        )
        .with_columns((pl.col("Predicted_Age") - pl.col("Visit_Adj_Age")).alias("Residual"))
        .with_columns(pl.col("Residual").abs().alias("Abs_Residual"))
    )
    current, _, _, _ = age_concordance.detect_age_outliers(
        results_df=current,
        sample_id_col=sid,
        age_threshold=results["effective_threshold"],
        tukey_fence_multiplier=results["tukey_fence_multiplier"],
        std_threshold=qc._age_concordance_params["std_threshold"],
    )
    # After the move every sample that was or is in ``subjects`` is in one of them.
    affected = current.filter(pl.col("USUBJID").is_in(list(subjects)))
    pred, cons = pl.col("Is_Discordant"), pl.col("From_Inconsistent_Participant")
    flags = {
        reason: set(affected.filter(expr)[sid].to_list())
        for reason, expr in zip(
            _AGE_WARN_REASONS, (pred & cons, pred & ~cons, ~pred & cons), strict=True
        )
    }
    qc._replace_sample_flags(affected[sid].to_list(), flags)
    logger.info(
        f"Age re-evaluated for {affected.height} samples of {len(subjects)} subjects "
        f"touched by reassignment: {sum(len(v) for v in flags.values())} flagged"
    )
    # Std_Predicted_Age is per subject (null under 3 timepoints): refresh it with
    # Visit_Adj_Age for every sample of the touched subjects.
    refreshed = ["Visit_Adj_Age"]
    if "Std_Predicted_Age" in qc._age_concordance_results.columns:
        refreshed.append("Std_Predicted_Age")
        if "Std_Predicted_Age" not in affected.columns:  # no subject has 3+ timepoints
            affected = affected.with_columns(pl.lit(None, pl.Float64).alias("Std_Predicted_Age"))
    # Age_Residual_Corrected subtracts a per-fold line in the fitted age, which
    # is not kept: a sample whose Visit_Adj_Age changed gets null, not the
    # value fitted against its labeled subject.
    if "Age_Residual_Corrected" in qc._age_concordance_results.columns:
        refreshed.append("Age_Residual_Corrected")
        fitted = old.select(sid, pl.col("Visit_Adj_Age").alias("_fitted_age"))
        affected = (
            affected.join(fitted, on=sid, how="left", validate="1:1")
            .with_columns(
                pl.when(pl.col("Visit_Adj_Age") == pl.col("_fitted_age"))
                .then(pl.col("Age_Residual_Corrected"))
                .alias("Age_Residual_Corrected")
            )
            .drop("_fitted_age")
        )
    new_vals = affected.select(sid, *refreshed)
    qc._age_concordance_results = (
        qc._age_concordance_results.join(new_vals, on=sid, how="left", suffix="_new")
        .with_columns(
            pl.when(pl.col(sid).is_in(new_vals[sid].implode()))
            .then(pl.col(f"{c}_new"))
            .otherwise(pl.col(c))
            .alias(c)
            for c in refreshed
        )
        .drop(f"{c}_new" for c in refreshed)
    )
