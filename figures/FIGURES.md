# Figures Developer Guide

Reference for creating or modifying publication figures in this directory.

## Directory Structure

```
figures/
├── fig1_trajectory.py       Figure 1: trajectory heatmap + line panels
├── fig2_volcano.py          Figure 2: across-treatment volcano plots
├── fig3_pathway.py          Figure 3: cameraPR pathway enrichment dot plots
├── fig4_mediation.py        Figure 4: mediation scatter + forest plots
├── fig5_ppy.py              Figure 5: PPY/REG4 trajectory + pharmacogenomics
├── build_main_figures.py    Public wrapper: render a combined main-figure PDF
├── build_supplementary_figures.py   Public wrapper: render combined Supplementary Figures PDF
│
├── supp_fig1_study_design.py           Supp Fig 1: study design schematic
├── supp_fig2_concordance.py            Supp Fig 2: cross-platform GMM histogram
├── supp_fig3_biomarker_correlation.py  Supp Fig 3: biomarker correlation scatter
├── supp_fig4_trajectory_lines.py       Supp Fig 4: platform-specific trajectory exemplars
├── supp_fig5_tzp_induced.py            Supp Fig 5: TZP-induced trajectory panels
├── supp_fig6_tzp_suppressed.py         Supp Fig 6: TZP-suppressed trajectory panels
├── supp_fig7_exocrine_pancreas.py      Supp Fig 7: exocrine pancreas enzyme trajectories
├── supp_fig8_tzp_opposing.py           Supp Fig 8: opposing-trajectory panels
├── supp_fig9_cross_study.py            Supp Fig 9: STEP 1/2 cross-study scatter
├── supp_fig10_cross_study_pcbl.py      Supp Fig 10: STEP 1/2 %CFB scatter
├── supp_fig11_pathway_ora.py           Supp Fig 11: ORA pathway dot plots (reuses `_pathway_figure`)
├── supp_fig12_mediation_week24.py      Supp Fig 12: mediation week-24 overview (reuses `_mediation_figure`)
├── supp_fig13_mediation_weight_mediated.py  Supp Fig 13: weight-mediated mediation forests (reuses `_mediation_figure`)
├── supp_fig14_mediation_heterogeneous.py    Supp Fig 14: heterogeneous mediation forests (reuses `_mediation_figure`)
├── supp_fig15_gipr_glp1r_celltype.py   Supp Fig 15: GIPR vs GLP1R cell-type specificity in pancreas (Hormone Cell Atlas dot plot)
├── supp_fig16_reg4.py                  Supp Fig 16: REG4 trajectory + pharmacogenomics (reuses `_ppy_figure`)
│
├── _internal/               Private implementation package for figure code
│   ├── __init__.py
│   ├── _common.py           Shared helper re-exports for figure scripts
│   ├── _annotations.py      Internal: banners, panel letters, text placement
│   ├── _figure_bundle.py    Internal: shared PDF bundle/layout helpers
│   ├── _gip_panel.py        Internal: Hormone Cell Atlas GIP panel renderer
│   ├── _lookups.py          Internal: UniProt marker/gene lookup loaders
│   ├── _maretty_step.py     Internal: shared reader for prepared STEP/Maretty parquet
│   ├── _mediation_figure.py Internal: Figure 4 mediation family implementation
│   ├── _output.py           Internal: save/output override helpers
│   ├── _paths.py            Internal: repo/data path constants for figures
│   ├── _pathway_figure.py   Internal: Figure 3 pathway family implementation
│   ├── _ppy_figure.py       Internal: Figure 5 PPY/REG4 family implementation
│   ├── _sizing.py           Internal: Nature width/height helpers
│   ├── _text.py             Internal: deterministic text-adjust helpers
│   ├── _theme.py            Internal: Lilly theme/style/font constants
│   ├── _trajectory_heatmap.py Internal: Figure 1 trajectory heatmap implementation
│   ├── _trajectory_lines.py Internal: treatment-arm line plot renderer
│   ├── _volcano_figure.py   Internal: Figure 2 volcano family implementation
│   ├── _pathway_labels.py   Internal: pathway term display-name mapping
│   └── lilly_theme.py       Internal: Lilly brand theme and colormap registration
│
└── configs/                 Curated term/gene CSVs and legend text
    └── supplementary_legends.yaml      Legend text for the combined Supplementary Figures PDF
```

**Naming conventions:**
- `fig{N}_*.py` — main manuscript figures (numbered)
- `supp_fig{N}_*.py` — supplemental figures (numbered in manuscript order)
- `build_main_figures.py` — bulk wrapper that renders the main figures and assembles a single `main_figures.pdf`
- `build_supplementary_figures.py` — bulk wrapper that renders the supplementary figures and assembles a single `supplementary_figures.pdf`
- `_internal/` — private implementation package (not standalone figures)

The numbered main-figure wrappers are intentionally thin public entrypoints.
Shared family implementations live in the matching internal modules above, while
the public `fig*.py` filenames remain the stable paths for manuscript renders.
Some supplements also import those internal family modules directly when they
reuse nontrivial layout or plotting internals.
External STEP/Maretty data are prepared once via
`analysis/scripts/maretty_step_data.py`; figure code reads the canonical
`analysis/outputs/maretty_step_data.parquet` output through
`_internal/_maretty_step.py`.
Figure scripts resolve mounted roots through `paths.py`. All file references
use the `surmount5_*` naming convention.

## Figure bundle wrappers

To build a single main-figure PDF with one labeled manuscript figure per page:

```bash
uv run python figures/build_main_figures.py
```

This wrapper renders the existing public `fig*.py` scripts into temporary PDFs,
then assembles `figures/main_figures.pdf` with simple `Figure 1` through
`Figure 5` labels above each page. Use `--figure fig3_pathway` (repeatable) to
build a subset while iterating.

To build a single Supplementary Figures PDF with one supplementary figure and its
full-width legend on each A4 page:

```bash
uv run python figures/build_supplementary_figures.py
```

This wrapper renders the existing public `supp_fig*.py` scripts into temporary
PDFs, then assembles `figures/supplementary_figures.pdf` using the legend text in
`configs/supplementary_legends.yaml`. The combined supplement uses fixed A4 pages
with a standard 0.5 inch page margin on all sides, then lays out figures and
legends inside that inner content box. Supplementary legend text is treated as
document text, not figure microtype: the wrapper uses a larger 9 pt bold title
and 8 pt body text with measured wrapping to the usable page width. Smaller
adjacent supplementary figures are packed onto the same page only when they still
fit cleanly inside that margin box, so the exact grouped pairs may shift as the
supplement typography or page margins are refined, while figure numbering order
is preserved. Use `--figure supp_fig9_cross_study` (repeatable) to build a subset
while iterating.

## Quick Start (New Figure Script)

```python
"""Brief description of what this figure shows."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _internal._common import (
    init_figure_theme,
    save_figure,
    # add other imports as needed (see below)
)

def make_figure():
    init_figure_theme()
    # ... build figure ...
    return fig

if __name__ == "__main__":
    fig = make_figure()
    out_dir = Path(__file__).resolve().parent
    save_figure(fig, str(out_dir / "output_name"), formats=("pdf",))
```

## Render Regression Workflow

The public figure wrappers keep their default manuscript behavior (`pdf` output in
`figures/`), but the shared `save_figure()` helper also honors two harness
environment variables used by `analysis/tools/publication_manifest.py`:

- `FIGURE_OUTPUT_DIR` — redirect outputs into an isolated render directory
- `FIGURE_OUTPUT_FORMATS` — override the save formats (for example `pdf,png`)

That lets you regression-test refactors without editing each figure script:

```bash
uv run python analysis/tools/publication_manifest.py record-figure-baseline \
  --baseline figures/figure_regression_baseline.json

uv run python analysis/tools/publication_manifest.py compare-figure-baseline \
  --baseline figures/figure_regression_baseline.json
```

The baseline stores PNG dimensions plus exact and renderer-tolerant PNG
fingerprints, not PDF byte-for-byte checksums. Use the comparison command after
structural refactors to confirm the rerendered figures are unchanged.

For no-data validation, run:

```bash
uv run pytest analysis/tests/test_figure_regression.py -v
uv run python analysis/tools/publication_manifest.py audit-figures
```

## `_internal/_common.py` Exports

`_internal/_common.py` is the shared helper surface for figure scripts.
Internally, a few low-risk helper clusters live in smaller sibling modules
(`_paths.py`, `_sizing.py`, `_theme.py`, `_output.py`, `_annotations.py`,
`_lookups.py`, `_text.py`) and the main figure families live in dedicated
internal modules (`_trajectory_heatmap.py`, `_volcano_figure.py`,
`_pathway_figure.py`, `_mediation_figure.py`, `_ppy_figure.py`). Public figure
code should usually import from `_internal._common` unless there is a clear
reason to depend on one of the narrower internal modules directly. The main
exception is the small set of supplements that intentionally reuse
family-specific pathway, mediation, or PPY/REG4 internals.

### Path Constants

| Constant | Value | Use |
|----------|-------|-----|
| `ROOT` | repo root (`figures/..`) | Base for all paths |
| `RESULTS_DIR` | `MMRM_ROOT` | MMRM outputs |
| `ANALYSIS_DIR` | `ROOT / "analysis"` | Analysis pipeline outputs |
| `FIGURES_DIR` | `ROOT / "figures"` | This directory |
| `COVAR` | `"covar_AGE_SEX"` | Covariate subdirectory key |

### Colors

**Directional (volcano/scatter/forest):**

| Constant | Hex | Meaning |
|----------|-----|---------|
| `COLOR_UP` | `#D31710` | Higher in TZP (red) |
| `COLOR_DOWN` | `#0F3A85` | Higher in SEMA (blue) |
| `COLOR_NS` | `#D3D3D3` | Not significant (light gray) |

**Threshold:**

| Constant | Value |
|----------|-------|
| `FDR_THRESHOLD` | `0.05` |

**Treatment arms:**

| Constant | Value |
|----------|-------|
| `ARM_COLORS` | `{"TZP": red, "SEMA": bold_blue}` |
| `ARM_LABELS` | `{"TZP": "TZP 15 mg", "SEMA": "SEMA 2.4 mg"}` |

### Lilly Brand Palette (`LILLY_COLORS[PALETTE_VARIANT]`)

The manuscript figures use `PALETTE_VARIANT = "web"` in `_internal/_common.py`
so submitted PDFs remain RGB. Access via:

```python
from _internal._common import LILLY_COLORS, PALETTE_VARIANT
c = LILLY_COLORS[PALETTE_VARIANT]
```

| Key | Hex | Typical use |
|-----|-----|-------------|
| `red` | `#E1251B` | TZP arm, upregulation |
| `bold_blue` | `#0F3A85` | SEMA arm, downregulation |
| `bold_green` | `#144B2D` | ORA significance dots |
| `bold_brown` | `#521207` | Reversal ↑↓ trajectory |
| `bold_grey` | `#8A969E` | Discordant distributions |
| `black` | `#212121` | Threshold lines, text |
| `vibrant_azure` | `#99BFE5` | Light blue (partial mediation) |
| `vibrant_coral` | `#F58E7D` | Light red (partial mediation) |
| `vibrant_gold` | `#FFC709` | Transient ↑ trajectory |
| `neutral_stone` | `#E4EBF1` | Late-onset ↓ trajectory |
| `neutral_orange` | `#FDD1B0` | Late-onset ↑ trajectory |
| `neutral_cream` | `#FFF0D8` | Background accents |
| `neutral_sage` | `#C6DCD8` | Background accents |
| `neutral_rose` | `#FDE8E5` | Background accents |
| `pink` | `#FBCFC8` | Highlight accents |
| `white` | `#FFFFFF` | — |

### Trajectory Constants

```python
TRAJECTORY_ORDER   # 10 canonical trajectory classes in display order
TRAJECTORY_COLORS  # trajectory name → Lilly brand hex
```

Order: Late-onset ↓, Progressive ↓, Sustained ↓, Transient ↓, Reversal ↓↑, Reversal ↑↓, Transient ↑, Sustained ↑, Progressive ↑, Late-onset ↑

### Gene Lookup Functions

All use `analysis/outputs/uniprot_map.parquet` as the authoritative source.

| Function | Returns | Use case |
|----------|---------|----------|
| `load_marker_gene_pairs()` | `pl.DataFrame[marker, gene_symbol]` | Base: all marker↔gene pairs (both platforms) |
| `build_gene_lookup()` | `dict[str, str]` | Quick marker → first gene symbol |
| `build_marker_genes()` | `dict[str, list[str]]` | Marker → all gene symbols (heterodimers) |
| `load_marker_to_gene(platform)` | `dict[str, str]` | Single-platform lookup (`"olink"` or `"soma"`) |

### Other Re-exports

| Name | From | Purpose |
|------|------|---------|
| `apply_lilly_theme()` | lilly_theme | Lower-level theme application; prefer `init_figure_theme()` in scripts |
| `save_figure(fig, path, formats, **kwargs)` | _common wrapper | Save final figure files; also supports harness-controlled output dir/format overrides |
| `build_cmap(name)` | lilly_theme | Build registered Lilly colormaps |
| `LILLY_COLORS` | lilly_theme | Full palette dict |

### Text Placement Helpers

```python
draw_segments_right_aligned(fig, x_right, y, segments, **kwargs)
draw_segments_left_aligned(fig, x_left, y, segments, **kwargs)
```

Place multi-colored text segments at exact positions in figure coordinates. `segments` is a list of `(text, color)` tuples.

## Figure Sizing

UltraPlot `journal` presets set Nature-compliant column widths:

| Preset | Width | Typical use |
|--------|-------|-------------|
| `"nat1"` | 1 column (~89 mm) | Single-panel supplementals |
| `"nat2"` | 2 columns (~183 mm) | Multi-panel supplementals |

For complex multi-panel figures, use explicit sizes:

```python
fig, axs = uplt.subplots(figwidth=15.0, figheight=8.5, ...)  # inches
```

Fig 1 uses raw matplotlib/subfigures because PyComplexHeatmap requires it. Several high-density supplemental trajectory grids also use explicit inch/fraction layout rather than UltraPlot so that platform banners, legends, and panel aspect ratios can be controlled exactly.

## Font Size Hierarchy (Nature compliance: 5–7 pt body)

Consistent across all figures; defined as `FS_*` constants in `_internal/_common.py`:

| Size (pt) | Tier | Elements |
|-----------|------|----------|
| 8.0 | NAVIGATION | Panel letters (a, b, c…) |
| 7.0 | EMPHASIS | Direction cues, banner bold names |
| 6.5 | NARRATIVE | Axis labels, Week pill, legend text, counts |
| 6.0 | BODY | Gene labels (scatter + forest y-axis) |
| 5.5 | AUXILIARY | Tick labels, banner tagline/subtitle |
| 5.0 | MICRO | Dense heatmap/forest labels where needed |

Half-point steps keep tiers visually distinct without violating Nature's minimum.

## Recurring Visual Patterns

### Banners

Light-gray horizontal strips above panel pairs that label platform or theme:

```python
_draw_banner(fig, ax_left, ax_right, bold_text="Theme", tail_text="· subtitle")
```

Constants: `BANNER_HEIGHT_IN = 0.20`, `BANNER_GAP_IN = 0.04` (inches).

### Direction Cues

In-axes "← higher in SEMA" / "higher in TZP →" text anchored at bottom corners:

```python
ax.text(0.03, 0.02, "← higher in SEMA", transform=ax.transAxes,
        fontsize=6, color=COLOR_DOWN, ha="left", va="bottom")
ax.text(0.97, 0.02, "higher in TZP →", transform=ax.transAxes,
        fontsize=6, color=COLOR_UP, ha="right", va="bottom")
```

### Week Pills

Rounded-rect badge at top-center of each panel:

```python
ax.text(0.5, 0.97, "Week 24", transform=ax.transAxes, fontsize=6,
        fontweight="bold", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0F0F0",
                  edgecolor="#CCCCCC", linewidth=0.4))
```

### Gene Labels with Halo

White-stroke halo for legibility over dense scatter:

```python
from matplotlib.patheffects import withStroke
halo = [withStroke(linewidth=3, foreground="white")]
ax.text(x, y, gene, fontsize=6, path_effects=halo)
```

Use `adjustText.adjust_text()` for automatic label repulsion.

### Spearman ρ Annotation

Upper-right corner, single line, superscript exponent:

```python
ax.text(0.97, 0.97, f"ρ = {r:.3f} · {p_str}",
        transform=ax.transAxes, fontsize=5.5, ha="right", va="top", color="#444444")
```

## Style Rules

1. **PDF only** — always `save_figure(fig, path, formats=("pdf",))` for final manuscript outputs.
2. **Solid lines only** — differentiate by weight/color/alpha, never `ls="--"` or `":"`.
3. **No dashed reference lines** — `axhline`/`axvline` use `lw=0.5` solid at `#AAAAAA`.
4. **Inter font** — registered via `init_figure_theme()`. Fallback chain: Inter → Arial → Liberation Sans → DejaVu Sans.
5. **No abc formatting from UltraPlot** — always `axs.format(abc=False)` then place panel letters manually with `fig.text()` for precise control.
6. **Shared axis limits** — compute across panels, then apply uniformly per row/column.
7. **White edge on colored scatter dots** — `edgecolors="white", linewidths=0.2` for visual separation at small sizes.
8. **8 pt panel labels** — use `FS_PANEL` and lowercase labels.
9. **Black/neutral legend text** — encode categories with markers/swatches, not colored text.
10. **Measure tight text-dependent layouts** — when matching panel width to a subtitle/legend, measure rendered text with `get_window_extent() / dpi` rather than guessing.

## Cross-Script Imports

Some supplemental figures import reusable visualization logic from the internal
family modules rather than from the thin public `fig*.py` wrappers:

| Supplement | Imports from |
|------------|-------------|
| `supp_fig11_pathway_ora.py` | `_pathway_figure` (pathway helpers, constants, layout logic) |
| `supp_fig12_mediation_week24.py` | `_mediation_figure` (shared loaders, scatter helpers, mediation layout constants) |
| `supp_fig13_mediation_weight_mediated.py` | `_mediation_figure` (forest plot helpers, banners, theme definitions) |
| `supp_fig14_mediation_heterogeneous.py` | `_mediation_figure` (forest plot helpers, banners, theme definitions) |
| `supp_fig16_reg4.py` | `_ppy_figure` (layout constants + drawing primitives) |

This avoids duplicating complex rendering logic while keeping each script
self-contained for execution. The public wrappers remain the stable manuscript
entrypoints; the internal modules are the shared implementation layer.

## Running

All render commands below require mounted protected inputs configured via
`paths.yaml`. The no-data checks are the figure regression test file and
`publication_manifest.py audit-figures`.

```bash
# Main figures
uv run python figures/fig1_trajectory.py
uv run python figures/fig2_volcano.py
uv run python figures/fig3_pathway.py
uv run python figures/fig4_mediation.py
uv run python figures/fig5_ppy.py

# Supplementals
uv run python figures/supp_fig1_study_design.py
uv run python figures/supp_fig2_concordance.py
uv run python figures/supp_fig3_biomarker_correlation.py
uv run python figures/supp_fig4_trajectory_lines.py
uv run python figures/supp_fig5_tzp_induced.py
uv run python figures/supp_fig6_tzp_suppressed.py
uv run python figures/supp_fig7_exocrine_pancreas.py
uv run python figures/supp_fig8_tzp_opposing.py
uv run python figures/supp_fig9_cross_study.py
uv run python figures/supp_fig10_cross_study_pcbl.py
uv run python figures/supp_fig11_pathway_ora.py
uv run python figures/supp_fig12_mediation_week24.py
uv run python figures/supp_fig13_mediation_weight_mediated.py
uv run python figures/supp_fig14_mediation_heterogeneous.py
uv run python figures/supp_fig15_gipr_glp1r_celltype.py
uv run python figures/supp_fig16_reg4.py
```

Most scripts expose `make_figure()` or `render()` plus a `__main__` block that
writes to `figures/`. `analysis/manifest.yaml` is the central command map for
public reproduction commands.
