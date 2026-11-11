"""Presentation of the QC exclusion flow for the study-design (CONSORT) figure.

The counts come from ``analysis.qc_flow``, the reader of
``analysis/outputs/qc_exclusion_breakdown.parquet``; this module only arranges them for the
figure: the arm labels, the trial-level randomized card, the analysis-ready grid and the
callout lines. ``ARM_DISPLAY`` is the single place the breakdown's arm tokens map to the
figure's arm labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from analysis.design import ARM_LABELS, CONTROL_VALUE, TREAT_VALUE
from analysis.qc_contract import PLATFORMS
from analysis.qc_flow import PlatformQC, load_platform_qc, visit_samples

# Breakdown arm token (the analysis TRT01A value) -> figure arm label, treatment first.
ARM_DISPLAY: dict[str, str] = {TREAT_VALUE: ARM_LABELS["TZP"], CONTROL_VALUE: ARM_LABELS["SEMA"]}
ARM_ORDER: tuple[str, ...] = tuple(ARM_DISPLAY)
MINUS = "\N{MINUS SIGN}"
GRID_TOTAL = "Total"


def randomized(df: pl.DataFrame) -> tuple[int, dict[str, int]]:
    """Trial-level randomized participants (total, by arm), the same on every platform.

    Raises:
        ValueError: When the platforms disagree.
    """
    per_platform = {}
    for platform in PLATFORMS:
        qc = load_platform_qc(platform, df)
        per_platform[platform] = (qc.rand_subjects, {arm: qc.rand_arm[arm] for arm in ARM_ORDER})
    values = list(per_platform.values())
    if any(v != values[0] for v in values[1:]):
        raise ValueError(f"randomized counts differ across platforms: {per_platform}")
    return values[0]


def grid_columns(df: pl.DataFrame) -> list[str]:
    """Column headers of the analysis-ready grid: participants, then each visit week."""
    weeks = [label for label, _ in visit_samples(PLATFORMS[0], df)[ARM_ORDER[0]]]
    for platform in PLATFORMS:
        for arm, cells in visit_samples(platform, df).items():
            if [label for label, _ in cells] != weeks:
                raise ValueError(f"{platform}/{arm} visit weeks differ from {weeks}")
    return ["Participants", *weeks]


@dataclass(frozen=True)
class GridRow:
    """One row of the analysis-ready grid; ``arm`` is None on a platform's total row."""

    platform: str
    arm: str | None
    counts: list[int]
    total: bool


def grid_rows(df: pl.DataFrame) -> list[GridRow]:
    """Each platform's arms (participants, samples per visit), then the platform total."""
    rows: list[GridRow] = []
    for platform in PLATFORMS:
        participants = load_platform_qc(platform, df).final_arm
        visits = visit_samples(platform, df)
        arms = [[participants[arm], *(n for _, n in visits[arm])] for arm in ARM_ORDER]
        rows += [GridRow(platform, arm, counts, False) for arm, counts in zip(ARM_ORDER, arms, strict=True)]
        rows.append(GridRow(platform, None, [sum(col) for col in zip(*arms, strict=True)], True))
    return rows


def removed_lines(reasons: list[tuple[str, int]]) -> list[tuple[str, str]]:
    """Callout lines for removed samples: each reason with its count after a minus sign."""
    return [(label, f"{MINUS}{n:,}") for label, n in reasons]


def identity_lines(qc: PlatformQC) -> list[tuple[str, str]]:
    """Identity callout: removed samples, then a retained-corrections line when its count is positive.

    Only the removed lines sum to the callout's total.
    """
    lines = removed_lines(qc.identity)
    if qc.well_swaps > 0:
        swaps = "Well swap" if qc.well_swaps == 1 else "Well swaps"
        lines.append((f"{swaps} corrected, retained", f"{qc.well_swaps:,}"))
    return lines
