"""Shared path constants for manuscript figure scripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "analysis"
FIGURES_DIR = ROOT / "figures"

sys.path.insert(0, str(ROOT))
from paths import (  # noqa: E402
    COVAR,
    MMRM_OLINK_DIR,
    MMRM_ROOT,
    MMRM_SOMA_DIR,
    PHENO_DIR,
    QC_OLINK_DIR,
    QC_SOMA_DIR,
    mmrm_dir,
)

RESULTS_DIR = MMRM_ROOT
