# Figures Developer Guide

Reference for creating or modifying publication figures in this directory.

## Directory Structure

```
figures/
├── fig1_trajectory.py       Figure 1: class-count butterflies + trajectory panels of the configured genes
├── fig2_volcano.py          Figure 2: between-treatment volcano plots
├── fig3_pathway.py          Figure 3: cameraPR pathway dot plots (a) over the HPA disease-signature reversal skyline (b)
├── fig4_mediation.py        Figure 4: mediation scatter + forest plots
├── fig5_ppy.py              Figure 5: PPY GIP cells + trajectory + STEP forest
├── build_figures.py         Registry (MAIN / EXTENDED / SUPP) + bundle builder: main_figures.pdf,
│                            extended_data_figures.pdf, supplementary_figures.pdf
├── __init__.py              Makes `figures` a package (imports resolve from the repo root via PYTHONPATH)
│
│   Extended Data (7 figures; registry position = ED number)
├── supp_fig_study_design.py           ED 1: study design + dual-platform QC flow (CONSORT; reads only qc_exclusion_breakdown.parquet)
├── supp_fig_concordance.py            ED 2: cross-platform GMM histogram
├── supp_fig_biomarker_correlation.py  ED 3: biomarker correlation scatter
├── supp_fig_exocrine_pancreas.py      ED 4: exocrine pancreas enzyme trajectories (`--part lead`)
├── supp_fig_cross_study.py            ED 5: STEP 1/2 cross-study scatter
├── supp_fig_mediation_week24.py       ED 6: mediation week-24 overview (reuses `_mediation_figure`)
├── supp_fig_gipr_glp1r_celltype.py    ED 7: GIPR vs GLP1R cell-type specificity (Hormone Cell Atlas)
│
│   Supplementary (11 figures; registry position = Supp number)
├── supp_fig_proteome_pca.py           Supp 1: pooled PCA and mixed-model variance attribution
├── supp_fig_tzp_induced.py            Supp 2: TZP-induced trajectory panels (week-72 rule, `analysis/trajectory_panels.py`)
├── supp_fig_tzp_suppressed.py         Supp 3: TZP-suppressed trajectory panels (same rule)
├── supp_fig_exocrine_pancreas.py      Supp 4: exocrine enzymes beyond ED 4 (`--part rest`)
├── supp_fig_sema_greater.py           Supp 5: proteins with a larger change under SEMA (same rule)
├── supp_fig_tzp_opposing.py           Supp 6: opposing-trajectory panels
├── supp_fig_cross_study_pcbl.py       Supp 7: STEP 1/2 %CFB scatter
├── supp_fig_pathway_ora.py            Supp 8: ORA pathway dot plots (reuses `_pathway_figure`)
├── supp_fig_mediation_weight_mediated.py  Supp 9: mediation theme forests (`_mediation_forest`)
├── supp_fig_mediation_heterogeneous.py    Supp 10: mediation theme forests (`_mediation_forest`)
├── supp_fig_e354q.py                  Supp 11: GIPR E354Q genotype x arm, PPY (uses `_ppy_figure.draw_genotype_row`)
│
├── _internal/               Private implementation package for figure code
│   ├── __init__.py
│   ├── _common.py           Shared helper re-exports for figure scripts
│   ├── _annotations.py      Internal: banners (`draw_banner`), panel letters, text placement
│   ├── _figure_bundle.py    Internal: journal box checks, A4 content-box fitting, page stacking
│   ├── _qc_breakdown.py     Internal: CONSORT layout of the `analysis.qc_flow` counts (arm labels, grid, callout lines)
│   ├── _mediation_forest.py Internal: three-glyph mediation forest (total / indirect / direct)
│   ├── _gip_panel.py        Internal: Hormone Cell Atlas GIP panel renderer
│   ├── _maretty_step.py     Internal: shared reader for prepared STEP/Maretty parquet
│   ├── _mediation_figure.py Internal: Figure 4 mediation family implementation
│   ├── _output.py           Internal: save/output override helpers, `save_figure_stack` (same-width figures stacked into one page)
│   ├── _paths.py            Internal: repo/data path constants for figures
│   ├── _pathway_figure.py   Internal: Figure 3 pathway family implementation and the Fig 3 composite (`make_figure3`: panel letters, 247 mm check)
│   ├── _disease_reversal_figure.py Internal: Figure 3b disease-signature reversal skyline
│   ├── _ppy_figure.py       Internal: Figure 5 (GIP cells, trajectory, STEP forest) + the E354Q genotype row (`draw_genotype_row`, drawn only in Supp 11)
│   ├── _sizing.py           Internal: journal width/height helpers, `panel_figure` + `panel_grid` (inch-geometry panel grids)
│   ├── _text.py             Internal: deterministic text-adjust helpers
│   ├── _theme.py            Internal: `init_figure_theme` (Inter, `FIGURE_RC`, tick-label scales), web palette `c`, font ladder, colors, trajectory colors, base `save_figure`
│   ├── _trajectory_data.py  Internal: the trajectory panels' readers (within-arm %CFB, between-treatment FDR, classes)
│   ├── _trajectory_lines.py Internal: the one trajectory line panel (`draw_lines`, `draw_gene_panel`, `draw_marker_panel`), the TZP | SEMA class banner, and the stacked-platform layout, platform banner, y-label and bottom strip of the supplementary trajectory figures
│   └── _volcano_figure.py   Internal: Figure 2 volcano family implementation
│
└── configs/                 Curated term CSVs
    ├── pathway_figure_terms.csv        Fig 3a curated cameraPR terms and themes (21 terms, 6 themes)
    └── pathway_ora_figure_terms.csv    Supp 8 curated ORA terms and themes (18 terms, 7 themes)
```

**Naming conventions**:
- `fig{N}_*.py` — Figures 1–5 (slot number in the name)
- `supp_fig_<topic>.py` — Extended Data and Supplementary figures; the name carries no
  number, so a figure can move between bundles or slots without a rename
- `build_figures.py` — the registry and bundle builder; a figure's number is its position
  in `MAIN`, `EXTENDED` or `SUPP`, and its output stem carries bundle and slot
  (`fig1_`, `ed_fig1_`, `supp_fig1_`)
- `_internal/` — private implementation package (not standalone figures)

The numbered `fig*.py` wrappers are thin entrypoints.
Shared family implementations live in the matching internal modules above, while
the `fig*.py` filenames remain the stable paths for the figure renders.
Some supplements also import those internal family modules directly when they
reuse nontrivial layout or plotting internals.
External STEP/Maretty data are prepared once via
`analysis/pipelines/maretty_step_data.py`; figure code reads the canonical
`analysis/outputs/maretty_step_data.parquet` output through
`_internal/_maretty_step.py`.
Fig 3b reads `analysis/outputs/disease_reversal_olink_wk72.parquet`, written by
`analysis/pipelines/run_disease_reversal.py` from the HPA blood disease atlas (fetched by
`analysis/external/hpa_disease_atlas.py`).
MMRM, pathway and mediation file names are built in `analysis/mmrm_outputs.py` and
`analysis/mediation/results.py`; figure scripts never spell them out.

## Figure bundles

```bash
pixi run python figures/build_figures.py --which all     # or main | extended | supp
pixi run python figures/build_figures.py --list-figures  # slot, stem, script
```

Each registered script runs with the figure harness pointed at a scratch directory
(`FIGURE_OUTPUT_DIR`, PDF only); its PDF is copied to `figures/output/<stem>.pdf`. Every
bundle page is the figure's page with a label strip above it ("Figure N", "Extended Data
Fig. N", "Supplementary Fig. N"), sized to the figure. A main
figure must fit 183 x 247 mm and an Extended Data figure 180 x 170 mm as drawn, and one that
does not is refused, never scaled; a Supplementary figure larger than the A4 content box
(190.7 x 271.6 mm) is scaled down to it. A failed render is reported, the others keep their
fixed numbers, and the build exits nonzero. Setting `FIGURE_OUTPUT_DIR` for the build itself
writes the bundles to that directory and the single-figure PDFs to its `output/`.

**Adding a figure** is one registry line in `build_figures.py` plus the matching
`analysis/manifest.yaml` line, which `publication_manifest.py check` enforces.

## Quick Start (New Figure Script)

```python
"""Brief description of what this figure shows."""
from __future__ import annotations

from pathlib import Path

from figures._internal._common import (
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

The figure wrappers keep their default behavior (`pdf` output in
`figures/`), but the shared `save_figure()` helper also honors two harness
environment variables used by `analysis/tools/publication_manifest.py`:

- `FIGURE_OUTPUT_DIR` — redirect outputs into an isolated render directory
- `FIGURE_OUTPUT_FORMATS` — override the save formats (for example `pdf,png`)

That lets you regression-test refactors without editing each figure script. Record a
baseline first, then compare later renders with it:

```bash
pixi run python analysis/tools/publication_manifest.py record-figure-baseline \
  --baseline <baseline.json>

pixi run python analysis/tools/publication_manifest.py compare-figure-baseline \
  --baseline <baseline.json>
```

The baseline records the SHA-256 of every rendered artifact. The comparison checks each
PNG's dimensions and its exact or renderer-tolerant (quantized or 256 x 256 preview) pixel
fingerprint; for a PDF it checks only that one was written. A match on the preview
fingerprint covers the 256 x 256 preview, not every pixel of the full render. Use the
comparison command after structural refactors to check the rerendered figures; compare the
PDF bytes (`cmp`) when they must be identical.

For no-data validation, run:

```bash
pixi run python analysis/tools/publication_manifest.py audit-figures
```

## `_internal/_common.py` Exports

`_internal/_common.py` is the shared helper surface for figure scripts.
Internally, a few low-risk helper clusters live in smaller sibling modules
(`_paths.py`, `_sizing.py`, `_theme.py`, `_output.py`, `_annotations.py`,
`_text.py`; the marker and gene lookups come from `analysis.marker_genes`) and the main figure families live in dedicated
internal modules (`_volcano_figure.py`,
`_pathway_figure.py`, `_mediation_figure.py`, `_ppy_figure.py`). Public figure
code should usually import from `figures._internal._common` unless there is a clear
reason to depend on one of the narrower internal modules directly. The main
exception is the small set of supplements that reuse
family-specific pathway, mediation, or PPY internals.

### Path Constants

| Constant | Value | Use |
|----------|-------|-----|
| `ROOT` | repo root (`figures/..`) | Base for all paths |
| `ANALYSIS_DIR` | `ROOT / "analysis"` | Analysis pipeline outputs |
| `FIGURES_DIR` | `ROOT / "figures"` | This directory |

### Colors

**Directional (volcano/scatter/forest):**

| Constant | Hex | Meaning |
|----------|-----|---------|
| `COLOR_UP` | `#0C376D` | Higher in TZP (navy) |
| `COLOR_DOWN` | `#5A5A5A` | Higher in SEMA (gray) |
| `COLOR_NS` | `#D3D3D3` | Not significant (light gray) |

**Threshold:**

| Constant | Value |
|----------|-------|
| `FDR_THRESHOLD` | `0.05` |

**Treatment arms:**

| Constant | Value |
|----------|-------|
| `ARM_COLORS` | `{"TZP": "#0C376D", "SEMA": "#5A5A5A"}` (navy, gray; same as `COLOR_UP` / `COLOR_DOWN`) |
| `ARM_LABELS` | `{"TZP": "TZP", "SEMA": "SEMA"}` |

### Lilly Brand Palette (`c`)

The figures use the Lilly web (RGB) palette, the dict `c` in `_internal/_theme.py`
(re-exported by `_common`), so the PDFs are RGB. There is no print palette.

```python
from figures._internal._common import c
```

| Key | Hex | Typical use |
|-----|-----|-------------|
| `red` | `#D31710` | Progressive ↑ trajectory, concordant distribution (ED2) |
| `bold_blue` | `#003A6C` | Progressive ↓ trajectory |
| `bold_green` | `#00422C` | Reversal ↓↑ trajectory, Olink platform color |
| `bold_brown` | `#501009` | Reversal ↑↓ trajectory |
| `bold_grey` | `#818C94` | Discordant distributions |
| `black` | `#191919` | Threshold lines, text |
| `vibrant_azure` | `#86B3F2` | Sustained ↓ trajectory |
| `vibrant_coral` | `#FD9485` | Sustained ↑ trajectory |
| `vibrant_gold` | `#F4C003` | Transient ↑ trajectory |
| `neutral_stone` | `#D8E4EC` | Late-onset ↓ trajectory |
| `neutral_orange` | `#FFDCC6` | Late-onset ↑ trajectory |
| `neutral_cream` | `#F9F0E0` | Background accents |
| `neutral_sage` | `#BEEBE6` | Background accents |
| `neutral_rose` | `#F9EEED` | Background accents |
| `pink` | `#FFDAD4` | Highlight accents |
| `white` | `#FFFFFF` | — |

### Trajectory Constants

```python
TRAJECTORY_COLORS  # trajectory name → Lilly brand hex
```

`TRAJECTORY_COLORS` lists the classes in display order: Late-onset ↓, Progressive ↓, Sustained ↓, Transient ↓, Reversal ↓↑, Reversal ↑↓, Transient ↑, Sustained ↑, Progressive ↑, Late-onset ↑

### Gene Lookup Functions

Defined in `analysis/marker_genes.py` (shared with the tables and `analysis.trajectory_panels`)
and re-exported by `_common`. All use `analysis/outputs/uniprot_map.parquet` as the
authoritative source; the first-match lookups resolve a multi-gene marker by map row order,
Olink first.

| Function | Returns | Use case |
|----------|---------|----------|
| `load_marker_gene_pairs()` | `pl.DataFrame[marker, gene_symbol]` | Base: all marker↔gene pairs (both platforms) |
| `build_gene_lookup()` | `dict[str, str]` | Quick marker → first gene symbol |
| `build_marker_genes()` | `dict[str, list[str]]` | Marker → all gene symbols (heterodimers) |
| `load_marker_to_gene(platform)` | `dict[str, str]` | Single-platform lookup (`"olink"` or `"soma"`) |

These match and select by gene symbol. **Drawn protein labels** come from one rule,
`analysis/protein_labels.py` (`marker_labels(platform)`, the annotation's `Display_Label`): one gene is the
gene (MRFAP1L2); two genes `A/B` (IL12A/IL12B, C4A/C4B); more `FIRST +N` with N the other genes;
genes in alphabetical order, HGNC duplicate-copy symbols (C4B_2) left out; an analyte assay, and an assay whose
gene another assay shares, shows its vendor name (INS-CPEPTIDE, NT-proBNP). Every figure that
names a protein uses it (volcano labels, Fig 1 headers, trajectory and exocrine panel headers, cross-study
scatters, mediation rows). Pathway lead genes are gene-level and stay gene symbols; tables keep the full gene
list and the vendor Assay/Target names.

### Other Re-exports

| Name | From | Purpose |
|------|------|---------|
| `save_figure(fig, path, formats, **kwargs)` | `_output` | Save final figure files; honors the harness output dir/format overrides |
| `panel_figure(w, h)`, `panel_grid(fig, layout, ...)`, `grid_boxes(...)` | `_sizing` | Panel grids with margins, gaps and ratios in inches |
| `LEADER_GRAY` | `_theme` | Leader lines from repelled gene labels (`#929591`) |

### Text Placement Helpers

```python
draw_segments_left_aligned(fig, x_left, y, segments, **kwargs)
```

Place multi-colored text segments left-aligned at an exact position in figure coordinates. `segments` is a list of `(text, color)` tuples.

## Figure Sizing

Figures are plain matplotlib. Widths come from `_sizing` (`NAT_W1` 89 mm, `NAT_W2`
183 mm); panel grids give their margins, gaps and ratios in inches:

```python
fig = panel_figure(NAT_W2, 7.3)  # inches
axs = panel_grid(fig, [[1, 2], [3, 4]], left=0.55, right=0.25, top=0.45, bottom=0.55,
                 wspace=0.45, hspace=0.8)  # a number repeated over cells makes a spanning panel
```

`panel_grid` rounds each box to 1e-9 of the figure, so adjustText's gene-label placements are
reproducible, and draws 0.6 / 0.48 pt ticks; `panel_figure` keeps its dpi through
`draw_without_rendering`, so text measured after label repulsion is measured at the figure's
dpi. Fig 1 (figure fractions), Fig 5 and the trajectory grids (`GridSpec`) lay out directly.

A main figure must fit 183 x 247 mm as saved (the tight crop plus 0.02 in of padding each side):
`build_figures.py` fails the main bundle for one that does not, as it does an Extended Data figure
over 180 x 170 mm. Fig 5's horizontal gaps are set so its page is 182.0 mm wide.

## Font Size Hierarchy (5–7 pt body)

Consistent across all figures; defined as `FS_*` constants in `_internal/_theme.py` (re-exported by `_common`):

| Size (pt) | Tier | Elements |
|-----------|------|----------|
| 8.0 | NAVIGATION | Panel letters (a, b, c…) |
| 7.0 | EMPHASIS | Direction cues, banner bold names |
| 6.5 | NARRATIVE | Axis labels, Week pill, legend text, counts |
| 6.0 | BODY | Gene labels (scatter + forest y-axis) |
| 5.5 | AUXILIARY | Tick labels, banner tagline/subtitle |
| 5.0 | MICRO | Dense heatmap/forest labels where needed |

Half-point steps keep tiers visually distinct without going below the 5 pt minimum.

## Recurring Visual Patterns

### Banners

Light-gray horizontal strips above panel pairs that label platform or theme:

```python
draw_banner(fig, ax_left, ax_right, bold_text="Theme", tail_text="· subtitle")
```

Constants (`_internal/_annotations.py`): `BANNER_HEIGHT_IN = 0.20`, `BANNER_GAP_IN = 0.04` (inches).

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

Use `deterministic_adjust_text()` (`_internal/_text.py`, re-exported by `_common`) for label
repulsion: it wraps `adjustText.adjust_text` with a fixed seed and an iteration cap (500 unless
the caller sets `iter_lim` or `time_lim`), so rerenders place labels the same way.
`separate_overlapping_labels()` (same module) runs after it at the figure's final geometry: it moves the fewest
labels, the least distance, until no two texts overlap, and keeps each leader line (Fig 4's scatters).

### Spearman ρ Annotation

Upper-right corner, single line, superscript exponent:

```python
ax.text(0.97, 0.97, f"ρ = {r:.3f} · {p_str}",
        transform=ax.transAxes, fontsize=5.5, ha="right", va="top", color="#444444")
```

## Style Rules

1. **PDF only** — always `save_figure(fig, path, formats=("pdf",))` for final outputs.
2. **Solid lines only** — differentiate by weight/color/alpha, never `ls="--"` or `":"`.
3. **No dashed reference lines** — `axhline`/`axvline` use `lw=0.5` solid at `#AAAAAA`.
4. **Inter font** — registered via `init_figure_theme()`, which raises `RuntimeError` when Inter is not registered with matplotlib; install it with `pixi run setup-fonts` (SHA-256 pinned; rerun after every `pixi install`).
5. **Panel letters by hand** — place them with `place_panel_letter` or `fig.text()` for precise control.
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
| `supp_fig_pathway_ora.py` | `_pathway_figure` (pathway helpers, constants, layout logic) |
| `supp_fig_mediation_week24.py` | `_mediation_figure` (shared loaders, scatter helpers, mediation layout constants) |
| `supp_fig_mediation_weight_mediated.py` | `_mediation_figure` (forest plot helpers, banners, theme definitions) |
| `supp_fig_mediation_heterogeneous.py` | `_mediation_figure` (forest plot helpers, banners, theme definitions) |
| `supp_fig_e354q.py` | `_ppy_figure` (`draw_genotype_row`) |

This avoids duplicating complex rendering logic while keeping each script
self-contained for execution. The wrappers remain the stable entrypoints; the
internal modules are the shared implementation layer.

## Running

All render commands below require the trial data mounted through
`paths.yaml`. The no-data check is `publication_manifest.py audit-figures`.

```bash
# Main figures
pixi run python figures/fig1_trajectory.py
pixi run python figures/fig2_volcano.py
pixi run python figures/fig3_pathway.py
pixi run python figures/fig4_mediation.py
pixi run python figures/fig5_ppy.py

# Supplementals
pixi run python figures/supp_fig_proteome_pca.py
pixi run python figures/supp_fig_study_design.py
pixi run python figures/supp_fig_concordance.py
pixi run python figures/supp_fig_biomarker_correlation.py
pixi run python figures/supp_fig_tzp_induced.py
pixi run python figures/supp_fig_tzp_suppressed.py
pixi run python figures/supp_fig_sema_greater.py
pixi run python figures/supp_fig_exocrine_pancreas.py --part lead
pixi run python figures/supp_fig_exocrine_pancreas.py --part rest
pixi run python figures/supp_fig_tzp_opposing.py
pixi run python figures/supp_fig_cross_study.py
pixi run python figures/supp_fig_cross_study_pcbl.py
pixi run python figures/supp_fig_pathway_ora.py
pixi run python figures/supp_fig_mediation_week24.py
pixi run python figures/supp_fig_mediation_weight_mediated.py
pixi run python figures/supp_fig_mediation_heterogeneous.py
pixi run python figures/supp_fig_gipr_glp1r_celltype.py
pixi run python figures/supp_fig_e354q.py
```

Most scripts expose `make_figure()` or `render()` plus a `__main__` block that
writes to `figures/`. `analysis/manifest.yaml` is the central command map for
public reproduction commands.
