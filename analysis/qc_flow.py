"""Readers of the QC exclusion-flow artifact the study-design figure draws.

``analysis/outputs/qc_exclusion_breakdown.parquet`` (written by
``analysis/pipelines/qc_exclusion_breakdown.py``) is tidy long form, one fact per row:
``platform``, ``stage``, ``order``, ``reason``, ``label``, ``arm``, ``n_samples``,
``n_subjects``. These helpers unpack it into the per-platform counts the CONSORT figure
(``figures/supp_fig_study_design.py``, laid out by ``figures/_internal/_qc_breakdown.py``)
draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from analysis.output_layout import QC_BREAKDOWN
from analysis.qc_contract import IDENTITY, LONGITUDINAL, REASSIGNED, TECHNICAL

# Breakdown stages beside the flow: the identity-check outcomes (one per input sample),
# the samples each correction type moved, and the samples whose participant
# has no genotype whatever their identity outcome.
IDENTITY_CHECK = "identity_check"
IDENTITY_CHECKED = "checked"
CORRECTION = "correction"
REASSIGN = "reassign"
BRIDGE_SWAP = "bridge_swap"
GENOTYPE_COVERAGE = "genotype_coverage"
NOT_GENOTYPED = "not_genotyped"
FINAL_VISIT = "final_visit"

QC_BREAKDOWN_PATH = QC_BREAKDOWN


@dataclass(frozen=True)
class PlatformQC:
    """Per-platform QC-flow counts unpacked from the breakdown artifact.

    Attributes:
        rand_subjects: Randomized subjects in the analyzed arms.
        rand_arm: Randomized subjects by arm (``TREAT_VALUE`` / ``CONTROL_VALUE``).
        cohort_subjects: Subjects entering the proteomics cohort.
        cohort_arm: Proteomics cohort subjects by arm.
        input_samples: Non-bridge samples entering technical QC.
        technical: Technical-exclusion bullets as ``(label, count)`` in display order.
        identity: Identity-exclusion bullets as ``(label, count)`` in display order.
        longitudinal: Longitudinal-exclusion bullets as ``(label, count)`` in display order.
        reassigned: Tubes moved to their genotype's participant and retained.
        well_swaps: Retained samples counted under the bridge_swap correction.
        final_samples: Samples retained after QC.
        final_subjects: Subjects retained after QC.
        final_arm: Retained subjects by arm.
    """

    rand_subjects: int
    rand_arm: dict[str, int]
    cohort_subjects: int
    cohort_arm: dict[str, int]
    input_samples: int
    technical: list[tuple[str, int]]
    identity: list[tuple[str, int]]
    longitudinal: list[tuple[str, int]]
    reassigned: int
    well_swaps: int
    final_samples: int
    final_subjects: int
    final_arm: dict[str, int]

    @property
    def technical_n(self) -> int:
        """Samples excluded by the technical gates."""
        return sum(n for _, n in self.technical)

    @property
    def identity_n(self) -> int:
        """Samples excluded by the identity checks and the corrections."""
        return sum(n for _, n in self.identity)

    @property
    def longitudinal_n(self) -> int:
        """Samples excluded by the longitudinal-eligibility gates."""
        return sum(n for _, n in self.longitudinal)


def load_breakdown() -> pl.DataFrame:
    """The QC exclusion-flow artifact.

    Returns:
        The tidy long-form breakdown table.
    """
    return pl.read_parquet(QC_BREAKDOWN_PATH)


def arm_subjects(platform: str, stage: str, df: pl.DataFrame) -> dict[str, int]:
    """Arm label -> subject count for one platform at one flow stage.

    Args:
        platform: ``olink`` or ``soma``.
        stage: Flow stage name in the breakdown's ``stage`` column.
        df: The breakdown table from ``load_breakdown``.

    Returns:
        Subject count per arm, in the stage's ``order``; the overall (null-arm) row is
        excluded.
    """
    rows = df.filter(
        (pl.col("platform") == platform) & (pl.col("stage") == stage) & pl.col("arm").is_not_null()
    ).sort("order")
    return dict(zip(rows["arm"], rows["n_subjects"], strict=True))


def _count(platform: str, stage: str, reason: str, df: pl.DataFrame) -> int:
    return df.filter(
        (pl.col("platform") == platform) & (pl.col("stage") == stage) & (pl.col("reason") == reason)
    )["n_samples"].item()


def load_platform_qc(platform: str, df: pl.DataFrame) -> PlatformQC:
    """Build a :class:`PlatformQC` for ``olink`` / ``soma`` from the artifact.

    Args:
        platform: ``olink`` or ``soma``.
        df: The breakdown table from ``load_breakdown``.

    Returns:
        The platform's flow counts.
    """
    pdf = df.filter(pl.col("platform") == platform)

    def _bullets(stage: str) -> list[tuple[str, int]]:
        rows = pdf.filter(pl.col("stage") == stage).sort("order")
        return list(zip(rows["label"], rows["n_samples"], strict=True))

    def _overall(stage: str, col: str) -> int:
        return pdf.filter((pl.col("stage") == stage) & pl.col("arm").is_null())[col].item()

    return PlatformQC(
        rand_subjects=_overall("randomized", "n_subjects"),
        rand_arm=arm_subjects(platform, "randomized", df),
        cohort_subjects=_overall("cohort", "n_subjects"),
        cohort_arm=arm_subjects(platform, "cohort", df),
        input_samples=_overall("input", "n_samples"),
        technical=_bullets(TECHNICAL),
        identity=_bullets(IDENTITY),
        longitudinal=_bullets(LONGITUDINAL),
        reassigned=_overall(REASSIGNED, "n_samples"),
        well_swaps=_count(platform, CORRECTION, BRIDGE_SWAP, df),
        final_samples=_overall("final", "n_samples"),
        final_subjects=_overall("final", "n_subjects"),
        final_arm=arm_subjects(platform, "final", df),
    )


def visit_samples(platform: str, df: pl.DataFrame) -> dict[str, list[tuple[str, int]]]:
    """Analysis-ready samples per arm and week for one platform.

    Args:
        platform: ``olink`` or ``soma``.
        df: The breakdown table from ``load_breakdown``.

    Returns:
        Arm -> ``(week label, samples)`` in week order, arms in the breakdown's order.
    """
    rows = df.filter((pl.col("platform") == platform) & (pl.col("stage") == FINAL_VISIT))
    arms = rows["arm"].unique(maintain_order=True).to_list()
    return {
        arm: list(zip(sub["label"], sub["n_samples"], strict=True))
        for arm in arms
        for sub in [rows.filter(pl.col("arm") == arm).sort("order")]
    }
