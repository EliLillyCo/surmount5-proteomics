"""Shared figure helpers used by manuscript figure scripts: one facade over the private modules.

Every name is re-exported for the figure scripts, so unused-import checks are off for this file.
"""
# ruff: noqa: F401

from __future__ import annotations

from ._annotations import (
    BANNER_EDGE_INSET_IN,
    BANNER_GAP_IN,
    BANNER_HEIGHT_IN,
    PANEL_DX_IN,
    PANEL_DY_IN,
    TRAJ_PANEL_ASPECT,
    draw_banner,
    draw_segments_left_aligned,
    place_panel_letter,
)
from analysis.marker_genes import (
    UNIPROT_MAP_PATH,
    build_marker_genes,
    load_marker_gene_pairs,
    load_marker_to_gene,
)
from analysis.protein_labels import marker_labels
from ._output import _resolve_output_formats, _resolve_output_path, save_figure
from ._paths import (
    ANALYSIS_DIR,
    FIGURES_DIR,
    PHENO_DIR,
    ROOT,
    mmrm_dir,
)
from ._sizing import (
    NAT_HMAX,
    NAT_W1,
    NAT_W15A,
    NAT_W15B,
    NAT_W2,
    grid_boxes,
    nature_figsize,
    panel_figure,
    panel_grid,
)
from ._text import deterministic_adjust_text, separate_overlapping_labels
from ._theme import (
    BETWEEN_TREATMENT_FDR_KEY,
    ARM_COLORS,
    ARM_LABELS,
    COLOR_DOWN,
    COLOR_NS,
    COLOR_UP,
    FDR_STAR_KEY,
    FDR_THRESHOLD,
    FS_AUX,
    FS_AXLABEL,
    FS_BANNER_BOLD,
    FS_BANNER_TAIL,
    FS_BODY,
    FS_DIRECTION,
    FS_EMPH,
    FS_FOREST_YLABEL,
    FS_GENE_LABEL,
    FS_LEGEND,
    FS_LEGEND_TITLE,
    FS_MIN,
    FS_NARR,
    FS_PANEL,
    FS_PILL,
    FS_TICK,
    LEADER_GRAY,
    TRAJECTORY_COLORS,
    c,
    fdr_stars,
    init_figure_theme,
)
