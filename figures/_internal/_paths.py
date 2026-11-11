"""Shared path constants for the figure scripts; ``_common`` re-exports every name."""
# ruff: noqa: F401

from __future__ import annotations

from pathlib import Path

from paths import (
    PHENO_DIR,
    mmrm_dir,
)

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "analysis"
FIGURES_DIR = ROOT / "figures"
