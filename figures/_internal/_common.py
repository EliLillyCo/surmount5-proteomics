"""Shared figure helpers used by manuscript figure scripts."""

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
    draw_segments_right_aligned,
    place_panel_letter,
)
from ._lookups import (
    _UNIPROT_MAP_PATH,
    build_gene_lookup,
    build_marker_genes,
    load_marker_gene_pairs,
    load_marker_to_gene,
)
from ._output import _resolve_output_formats, _resolve_output_path, save_figure
from ._paths import (
    ANALYSIS_DIR,
    COVAR,
    FIGURES_DIR,
    MMRM_OLINK_DIR,
    MMRM_ROOT,
    MMRM_SOMA_DIR,
    PHENO_DIR,
    QC_OLINK_DIR,
    QC_SOMA_DIR,
    RESULTS_DIR,
    ROOT,
    mmrm_dir,
)
from ._sizing import NAT_HMAX, NAT_W1, NAT_W15A, NAT_W15B, NAT_W2, nature_figsize
from ._text import deterministic_adjust_text
from ._theme import (
    ARM_COLORS,
    ARM_LABELS,
    COLOR_DOWN,
    COLOR_NS,
    COLOR_UP,
    FDR_THRESHOLD,
    FS_AUX,
    FS_AXLABEL,
    FS_BANNER_BOLD,
    FS_BANNER_TAIL,
    FS_BODY,
    FS_COUNTS,
    FS_DIRECTION,
    FS_EMPH,
    FS_FOREST_YLABEL,
    FS_GENE_LABEL,
    FS_LEGEND,
    FS_LEGEND_TITLE,
    FS_MIN,
    FS_NARR,
    FS_PANEL,
    FS_PANEL_LABEL,
    FS_PILL,
    FS_TICK,
    LILLY_COLORS,
    PALETTE_VARIANT,
    TRAJECTORY_COLORS,
    TRAJECTORY_ORDER,
    apply_lilly_theme,
    build_cmap,
    c,
    init_figure_theme,
)
