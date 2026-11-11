"""Repo-relative locations of the generated analysis outputs (``analysis/outputs/``).

Imports nothing beyond ``pathlib``, so the stage scripts, the analysis modules, the figure and
table builders and the tools can share one definition of where each stage writes without
loading the mount configuration. Mounted inputs and the MMRM, pathway and mediation outputs
are configured separately in ``paths.py`` (``paths.yaml``).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_OUTPUTS = ROOT / "analysis" / "outputs"
# Directory for generated reports.
REPORTS_OUT = ANALYSIS_OUTPUTS / "reports"
# One provenance record per stage, figure build and table build.
PROVENANCE_DIR = ANALYSIS_OUTPUTS / "provenance"
# Default home of the figure renders that the figure-baseline commands write.
RENDER_ROOT = ANALYSIS_OUTPUTS / "_publication_manifest"

# Stage outputs that a writer and its readers share.
UNIPROT_MAP = ANALYSIS_OUTPUTS / "uniprot_map.parquet"
QC_BREAKDOWN = ANALYSIS_OUTPUTS / "qc_exclusion_breakdown.parquet"
PLATFORM_CONCORDANCE = ANALYSIS_OUTPUTS / "platform_concordance.parquet"
MARETTY_STEP = ANALYSIS_OUTPUTS / "maretty_step_data.parquet"
E354Q_FRAME = ANALYSIS_OUTPUTS / "genotype_E354Q_frame.parquet"
E354Q_CONTRASTS = ANALYSIS_OUTPUTS / "genotype_E354Q_contrasts.parquet"
PCA_SCORES = ANALYSIS_OUTPUTS / "proteome_pca_scores.parquet"
PCA_VARIANCE = ANALYSIS_OUTPUTS / "proteome_pca_variance.parquet"
PCA_PARTITION = ANALYSIS_OUTPUTS / "proteome_pca_partition.parquet"


def pca_model_input(platform: str) -> Path:
    """Complete-case clinical and technical model input passed to R as parquet."""
    return ANALYSIS_OUTPUTS / f"proteome_pca_model_{platform}.parquet"


def trajectory_path(platform: str) -> Path:
    """The trajectory-class table of ``platform`` (``olink`` or ``soma``).

    Args:
        platform: Platform key.

    Returns:
        ``analysis/outputs/trajectory_<platform>.parquet``.
    """
    return ANALYSIS_OUTPUTS / f"trajectory_{platform}.parquet"
