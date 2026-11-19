"""Render every registered figure and concatenate the Main, Extended Data and Supplementary PDFs.

Each registered script is run in this environment with the figure harness pointed at a scratch
directory (``FIGURE_OUTPUT_DIR``, PDF only); its single PDF is copied to
``figures/output/<stem>.pdf`` and the bundles are written to ``figures/main_figures.pdf``,
``figures/extended_data_figures.pdf`` and ``figures/supplementary_figures.pdf``. With
``FIGURE_OUTPUT_DIR`` set for the build itself, the bundles go to that directory and the
single-figure PDFs to its ``output/`` subdirectory.

Every bundle page is the figure's own page with a label strip stacked above it ("Figure N",
"Extended Data Fig. N", "Supplementary Fig. N"), the page sized to the figure; the bundles
carry no legends. A main figure must fit the 183 x 247 mm box and an Extended Data figure the
180 x 170 mm box as drawn: the build reports one that does not and fails, never scaling it.
A supplementary figure larger than the A4 content box (190.7 x 271.6 mm) is scaled down to it
before its strip is added, so every supplementary page fits A4.

Registry: MAIN / EXTENDED / SUPP below are the authoritative, position-ordered lists; a
figure's number in its bundle is its position there, so renumbering is a reorder and adding
a figure is one registry line. Script names do not carry a number (``fig#_`` main,
``supp_fig_<topic>`` otherwise); output stems carry bundle and slot (``fig1_``, ``ed_fig1_``,
``supp_fig1_``). A failed render is reported by name, every other entry keeps its fixed slot,
and the process exits nonzero. A bundle with any failed figure is not written, so a PDF on
disk always holds every figure of its registry. Before anything renders, the analysis outputs
the figures read are checked against ``analysis/manifest.yaml``; a missing one raises with the
stage command that writes it. When every chosen figure is placed in the default output
locations, the build writes ``analysis/outputs/provenance/figures.json``
(``figures.<which>.json`` for one registry): the commit, the ``pixi.lock`` digest and a
fingerprint of the analysis outputs the chosen figures read and of the bundles and per-figure
PDFs written (``analysis/tools/provenance.py``); ``pixi run reproduce`` writes that record
itself. A build redirected with ``FIGURE_OUTPUT_DIR`` is not recorded.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from pypdf import PageObject, PdfReader, PdfWriter

from analysis.tools.figure_baseline import _figure_environment
from analysis.tools.manifest_stages import require_stage_outputs
from analysis.tools.provenance import record_build
from figures._internal import _figure_bundle as bundle

ROOT = Path(__file__).resolve().parent.parent


FIGURES = ROOT / "figures"
OUT = FIGURES / "output"
BUNDLE_NAMES = {
    "main": "main_figures.pdf",
    "extended": "extended_data_figures.pdf",
    "supp": "supplementary_figures.pdf",
}

# Label-strip geometry (inches).
LABEL_SIDE_MARGIN_IN = 0.30
LABEL_TOP_MARGIN_IN = 0.20
LABEL_BOTTOM_MARGIN_IN = 0.18
LABEL_LEADING = 1.15

Entry = tuple[str, list[str], str]  # (script, extra args, output stem)

MAIN: list[Entry] = [
    ("fig1_trajectory.py", [], "fig1_trajectory"),
    ("fig2_volcano.py", [], "fig2_volcano"),
    ("fig3_pathway.py", [], "fig3_pathway"),
    ("fig4_mediation.py", [], "fig4_mediation"),
    ("fig5_ppy.py", [], "fig5_ppy"),
]
# Extended Data registry; position = figure number.
EXTENDED: list[Entry] = [
    ("supp_fig_study_design.py", [], "ed_fig1_study_design"),
    ("supp_fig_concordance.py", [], "ed_fig2_concordance"),
    ("supp_fig_biomarker_correlation.py", [], "ed_fig3_biomarker_correlation"),
    ("supp_fig_exocrine_pancreas.py", ["--part", "lead"], "ed_fig4_exocrine_pancreas"),
    ("supp_fig_cross_study.py", [], "ed_fig5_cross_study"),
    ("supp_fig_mediation_week24.py", [], "ed_fig6_mediation_week24"),
    ("supp_fig_gipr_glp1r_celltype.py", [], "ed_fig7_gipr_glp1r_celltype"),
]
# Supplementary registry; position = figure number. Supp 2, 3 and 5 take their proteins from
# the week-72 rule in analysis/trajectory_panels.py.
SUPP: list[Entry] = [
    ("supp_fig_proteome_pca.py", [], "supp_fig1_proteome_pca"),
    ("supp_fig_tzp_induced.py", [], "supp_fig2_tzp_induced"),
    ("supp_fig_tzp_suppressed.py", [], "supp_fig3_tzp_suppressed"),
    ("supp_fig_exocrine_pancreas.py", ["--part", "rest"], "supp_fig4_exocrine_pancreas"),
    ("supp_fig_sema_greater.py", [], "supp_fig5_sema_greater"),
    ("supp_fig_tzp_opposing.py", [], "supp_fig6_tzp_opposing"),
    ("supp_fig_cross_study_pcbl.py", [], "supp_fig7_cross_study_pcbl"),
    ("supp_fig_pathway_ora.py", [], "supp_fig8_pathway_ora"),
    ("supp_fig_mediation_weight_mediated.py", [], "supp_fig9_mediation_weight_mediated"),
    ("supp_fig_mediation_heterogeneous.py", [], "supp_fig10_mediation_heterogeneous"),
    ("supp_fig_e354q.py", [], "supp_fig11_e354q"),
]
REGISTRIES: dict[str, list[Entry]] = {"main": MAIN, "extended": EXTENDED, "supp": SUPP}


def render(script: str, args: Sequence[str], stem: str, out_dir: Path = OUT) -> Path | None:
    """Run one figure script and copy its single PDF to ``out_dir/<stem>.pdf``.

    Args:
        script: Figure script under ``figures/``.
        args: Extra arguments for the script.
        stem: Output stem.
        out_dir: Directory of the single-figure PDFs.

    Returns:
        The copied PDF, or None when the script failed or did not write exactly one PDF.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{stem}-") as td:
        scratch = Path(td)
        proc = subprocess.run(
            [sys.executable, str(FIGURES / script), *args],
            cwd=ROOT,
            env=_figure_environment(scratch, ("pdf",)),
            capture_output=True,
            text=True,
        )
        pdfs = sorted(scratch.glob("*.pdf"))
        if proc.returncode != 0 or len(pdfs) != 1:
            tail = (proc.stderr.strip().splitlines() or [proc.stdout.strip()])[-1:]
            print(f"  [FAIL] {script} {' '.join(args)}: {tail}")
            return None
        out = out_dir / f"{stem}.pdf"
        shutil.copyfile(pdfs[0], out)
    print(f"  [ok] {stem}")
    return out


def _label_strip_pdf(text: str, page_w_pt: float, out: Path) -> Path:
    """A page-wide white strip with ``text`` in bold at its top-left, stacked above a figure."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from figures._internal._theme import FS_FIGURE_LABEL, INK, PDF_METADATA, init_figure_theme

    init_figure_theme()
    page_w_in = page_w_pt / 72.0
    text_h_in = (FS_FIGURE_LABEL / 72.0) * LABEL_LEADING
    page_h_in = LABEL_TOP_MARGIN_IN + text_h_in + LABEL_BOTTOM_MARGIN_IN
    fig = plt.figure(figsize=(page_w_in, page_h_in), facecolor="white")
    fig.text(
        LABEL_SIDE_MARGIN_IN / page_w_in,
        1.0 - LABEL_TOP_MARGIN_IN / page_h_in,
        text,
        ha="left",
        va="top",
        fontsize=FS_FIGURE_LABEL,
        fontweight="bold",
        color=INK,
    )
    # The theme crops to a tight bbox, which would shrink the strip to its text.
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(out, pad_inches=0, facecolor="white", metadata=PDF_METADATA)
    plt.close(fig)
    return out


def _stack_label_above(label_pdf: Path, figure_page) -> PageObject:
    """A new page: label strip on top, figure page below, both centered."""
    return bundle.stack_pages_vertically(
        [PdfReader(str(label_pdf)).pages[0], figure_page], bottom_first=True
    )


def bundle_labelled(
    entries: Sequence[Entry],
    rendered: dict[str, Path | None],
    out_pdf: Path,
    *,
    label: str,
    title: str,
    fit_problem: Callable[[float, float], str | None] | None,
) -> list[str]:
    """Stamp each figure "<label> N" (its fixed registry slot) and concatenate the pages.

    Args:
        entries: The registry being bundled.
        rendered: Stem -> rendered PDF, None where the render failed.
        out_pdf: Bundle destination.
        label: Strip text ahead of the number, e.g. ``"Extended Data Fig."``.
        title: PDF metadata title.
        fit_problem: Checks a page's width and height in points against the bundle's journal box
            and returns why it does not fit, or None. None scales a page larger than the A4
            content box down to it instead (``_figure_bundle.fit_to_a4_content``).

    Returns:
        The scripts that failed to render or do not fit the box.
    """
    writer = PdfWriter()
    writer.add_metadata({"/Title": title})
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="figlabel-") as td:
        for n, (script, _, stem) in enumerate(entries, start=1):
            pdf = rendered[stem]
            if pdf is None:
                failures.append(script)
                continue
            pages = PdfReader(str(pdf)).pages
            if fit_problem is not None:
                problems = [fit_problem(float(pg.mediabox.width), float(pg.mediabox.height)) for pg in pages]
                problem = next((found for found in problems if found), None)
                if problem:
                    print(f"  [FAIL] {stem} does not fit the {label} box: {problem}")
                    failures.append(script)
                    continue
            if fit_problem is None:
                pages = [bundle.fit_to_a4_content(page, stem) for page in pages]
            for page in pages:
                strip = _label_strip_pdf(f"{label} {n}", float(page.mediabox.width), Path(td) / f"{stem}.pdf")
                writer.add_page(_stack_label_above(strip, page))
    _write(writer, out_pdf, failures, len(entries))
    return failures


# bundle name -> (strip label, PDF title, journal-box check; None scales down to the A4 content box)
BUNDLE_STYLES: dict[str, tuple[str, str, Callable[[float, float], str | None] | None]] = {
    "main": ("Figure", "Main Figures", bundle.main_fit_problem),
    "extended": ("Extended Data Fig.", "Extended Data Figures", bundle.extended_fit_problem),
    "supp": ("Supplementary Fig.", "Supplementary Figures", None),
}


def _write(writer: PdfWriter, out_pdf: Path, failures: Sequence[str], total: int) -> None:
    """Write the bundle only when every figure was placed; a partial bundle is never written."""
    if failures:
        print(f"[not written] {bundle_display(out_pdf)}: {len(failures)} of {total} figures failed")
        return
    with out_pdf.open("wb") as f:
        writer.write(f)
    print(f"[ok] wrote {bundle_display(out_pdf)} ({total}/{total} figures)")


def bundle_display(path: Path) -> str:
    """``path`` relative to the repository root for messages, or as given when it lies outside."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", choices=["main", "extended", "supp", "all"], default="all")
    p.add_argument("--list-figures", action="store_true", help="Print slot, stem and script, then exit.")
    return p.parse_args(argv)


def output_dirs() -> tuple[Path, Path]:
    """The bundle directory and the single-figure directory of this build.

    Returns:
        ``(figures/, figures/output/)``, or ``($FIGURE_OUTPUT_DIR, $FIGURE_OUTPUT_DIR/output)``
        when that variable is set.
    """
    override = os.environ.get("FIGURE_OUTPUT_DIR")
    if not override:
        return FIGURES, OUT
    root = Path(override).resolve()
    return root, root / "output"


def main(argv: Sequence[str] | None = None) -> int:
    """Render the chosen registries and write their bundles; nonzero if any figure failed."""
    args = _parse_args(argv)
    chosen = {name: reg for name, reg in REGISTRIES.items() if args.which in (name, "all")}
    if args.list_figures:
        for name, registry in chosen.items():
            for n, (script, extra, stem) in enumerate(registry, start=1):
                print(f"{name}\t{n}\t{stem}\tfigures/{script} {' '.join(extra)}".rstrip())
        return 0
    bundle_dir, single_dir = output_dirs()
    scripts = [f"figures/{script}" for reg in chosen.values() for script, _, _ in reg]
    require_stage_outputs("figures", scripts)
    started = time.time()
    print("Rendering:")
    rendered = {
        stem: render(script, extra, stem, single_dir) for reg in chosen.values() for script, extra, stem in reg
    }
    bundles = {name: bundle_dir / BUNDLE_NAMES[name] for name in chosen}
    failures: list[str] = []
    for name, registry in chosen.items():
        label, title, fit_problem = BUNDLE_STYLES[name]
        failures += bundle_labelled(
            registry, rendered, bundles[name], label=label, title=title, fit_problem=fit_problem
        )
    if failures:
        print(f"\nFAILED to render or place {len(failures)} figure(s): {', '.join(failures)}")
        return 1
    if bundle_dir != FIGURES:
        return 0
    written = [str(path) for path in bundles.values()] + [
        str(single_dir / f"{stem}.pdf") for reg in chosen.values() for _, _, stem in reg
    ]
    record_build(
        "figures",
        outputs=written,
        scripts=scripts,
        started=started,
        name="figures" if args.which == "all" else f"figures.{args.which}",
        arguments={"which": args.which},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
