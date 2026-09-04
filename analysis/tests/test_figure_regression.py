from __future__ import annotations

from pathlib import Path
import importlib
import pathlib
import sys

import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.tools import publication_manifest as pm
from figures._internal._common import save_figure
from paths import ensure_public_alias


def test_save_figure_respects_harness_overrides(tmp_path, monkeypatch):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    monkeypatch.setenv("FIGURE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("FIGURE_OUTPUT_FORMATS", "pdf,png")

    written = save_figure(fig, "ignored/example", formats=("pdf",))
    written_paths = [Path(path) for path in written]

    assert [path.name for path in written_paths] == ["example.pdf", "example.png"]
    assert {path.parent for path in written_paths} == {tmp_path}
    assert all(path.exists() for path in written_paths)

    plt.close(fig)


def test_figure_environment_is_deterministic(tmp_path):
    env = pm._figure_environment(tmp_path, ("pdf", "png"))
    assert env["FIGURE_OUTPUT_DIR"] == str(tmp_path)
    assert env["FIGURE_OUTPUT_FORMATS"] == "pdf,png"
    assert env["MPLBACKEND"] == "Agg"
    assert env["PYTHONHASHSEED"] == "0"


def test_internal_common_exposes_paths_sizing_theme_output_annotations_and_lookups(
    monkeypatch,
):
    monkeypatch.syspath_prepend(str(ROOT / "figures"))

    bare_common = importlib.import_module("_internal._common")
    package_common = importlib.import_module("figures._internal._common")

    for module in (bare_common, package_common):
        assert module.ROOT == ROOT
        assert module.ANALYSIS_DIR == ROOT / "analysis"
        assert module.FIGURES_DIR == ROOT / "figures"
        assert module.RESULTS_DIR == module.MMRM_ROOT
        assert module._UNIPROT_MAP_PATH == ROOT / "analysis" / "outputs" / "uniprot_map.parquet"
        assert module.NAT_W2 > module.NAT_W1
        assert module.nature_figsize("2col", 3.0) == (module.NAT_W2, 3.0)
        assert module.PALETTE_VARIANT == "web"
        assert module.ARM_COLORS["TZP"] == "#0C376D"  # manuscript navy
        assert module.BANNER_HEIGHT_IN > 0
        assert module.TRAJ_PANEL_ASPECT > 0
        assert callable(module.init_figure_theme)
        assert callable(module.save_figure)
        assert callable(module.build_cmap)
        assert callable(module.place_panel_letter)
        assert callable(module.draw_banner)
        assert callable(module.draw_segments_left_aligned)
        assert callable(module.draw_segments_right_aligned)
        assert callable(module.load_marker_gene_pairs)
        assert callable(module.build_gene_lookup)
        assert callable(module.build_marker_genes)
        assert callable(module.load_marker_to_gene)
        assert callable(module.deterministic_adjust_text)


def test_public_wrapper_modules_expose_family_entrypoints(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "figures"))

    fig1 = importlib.import_module("fig1_trajectory")
    fig2 = importlib.import_module("fig2_volcano")
    fig3 = importlib.import_module("fig3_pathway")
    fig4 = importlib.import_module("fig4_mediation")
    fig5 = importlib.import_module("fig5_ppy")
    supp9 = importlib.import_module("supp_fig9_cross_study")
    supp10 = importlib.import_module("supp_fig10_cross_study_pcbl")
    supp11 = importlib.import_module("supp_fig11_pathway_ora")
    supp12 = importlib.import_module("supp_fig12_mediation_week24")
    supp13 = importlib.import_module("supp_fig13_mediation_weight_mediated")
    supp14 = importlib.import_module("supp_fig14_mediation_heterogeneous")
    supp15 = importlib.import_module("supp_fig15_gipr_glp1r_celltype")
    supp16 = importlib.import_module("supp_fig16_reg4")
    build_main = importlib.import_module("build_main_figures")
    build_supp = importlib.import_module("build_supplementary_figures")

    assert callable(fig1.render_full)
    assert callable(fig2.make_volcano_figure)
    assert callable(fig3.make_pathway_figure)
    assert callable(fig4.make_figure)
    assert callable(fig5.render)
    assert callable(supp9.make_figure)
    assert callable(supp10.make_figure)
    assert callable(supp11.make_figure)
    assert callable(supp12.make_figure)
    assert callable(supp13.make_figure)
    assert callable(supp14.make_figure)
    assert callable(supp15.make_figure)
    assert callable(supp16.render)
    assert callable(build_main.main)
    assert len(build_main.load_main_figure_specs()) == 5
    assert callable(build_supp.main)
    assert len(build_supp.load_supplement_specs()) == 16
    assert build_supp.SUPP_PAGE_MARGIN_IN == 0.5
    assert build_supp.SUPP_BODY_FS == 8.0
    assert build_supp.SUPP_TITLE_FS == 9.0
    assert build_supp.SUPP_CAPTION_TOP_MARGIN_IN == 24.0 / 72.0
    assert build_supp.SUPP_PAGE_GAP_PT == 64.0
    assert round(build_supp._supp_content_width_pt(), 3) == round(
        build_supp.bundle.A4_WIDTH_PT - 72.0, 3
    )


def test_build_main_wrapper_composes_single_page_pdf(tmp_path):
    build_main = importlib.import_module("figures.build_main_figures")

    figure_pdf = tmp_path / "figure.pdf"
    fig, ax = plt.subplots(figsize=(3.0, 1.5))
    ax.plot([0, 1], [0, 1])
    fig.savefig(figure_pdf, format="pdf", bbox_inches=None, pad_inches=0)
    plt.close(fig)

    spec = build_main.MainFigureSpec(
        key="fig99_example",
        number=99,
    )
    label_pdf = build_main.bundle.write_text_block_pdf(
        spec.full_title,
        216.0,
        tmp_path / "label.pdf",
    )
    page = build_main.bundle.stack_pdf_pages([label_pdf, figure_pdf])

    output_pdf = tmp_path / "combined.pdf"
    writer = PdfWriter()
    writer.add_page(page)
    with output_pdf.open("wb") as handle:
        writer.write(handle)

    reader = PdfReader(str(output_pdf))
    assert len(reader.pages) == 1
    assert float(reader.pages[0].mediabox.height) > float(
        PdfReader(str(figure_pdf)).pages[0].mediabox.height
    )


def test_build_supplementary_wrapper_composes_single_page_pdf(tmp_path):
    build_supp = importlib.import_module("figures.build_supplementary_figures")

    figure_pdf = tmp_path / "figure.pdf"
    fig, ax = plt.subplots(figsize=(3.0, 1.5))
    ax.plot([0, 1], [0, 1])
    fig.savefig(figure_pdf, format="pdf", bbox_inches=None, pad_inches=0)
    plt.close(fig)

    spec = build_supp.SupplementSpec(
        key="supp_fig99_example",
        number=99,
        title="Example supplementary wrapper page.",
        legend=(
            "This is a synthetic caption used to verify that the supplementary "
            "wrapper composes one figure PDF and one caption PDF into a single page."
        ),
    )
    caption_pdf = build_supp.bundle.write_text_block_pdf(
        spec.full_title,
        build_supp._supp_content_width_pt(),
        tmp_path / "caption.pdf",
        body=spec.legend,
        title_fs=build_supp.SUPP_TITLE_FS,
        body_fs=build_supp.SUPP_BODY_FS,
        side_margin_in=build_supp.SUPP_CAPTION_SIDE_MARGIN_IN,
        top_margin_in=build_supp.SUPP_CAPTION_TOP_MARGIN_IN,
        bottom_margin_in=build_supp.SUPP_CAPTION_BOTTOM_MARGIN_IN,
        title_gap_in=build_supp.SUPP_TITLE_GAP_IN,
    )
    page = build_supp.bundle.compose_stacked_page(
        [(figure_pdf, caption_pdf)],
        side_margin_pt=build_supp._supp_page_margin_pt(),
        top_margin_pt=build_supp._supp_page_margin_pt(),
        bottom_margin_pt=build_supp._supp_page_margin_pt(),
    )

    output_pdf = tmp_path / "combined.pdf"
    writer = PdfWriter()
    writer.add_page(page)
    with output_pdf.open("wb") as handle:
        writer.write(handle)

    reader = PdfReader(str(output_pdf))
    assert len(reader.pages) == 1
    assert round(float(reader.pages[0].mediabox.width), 3) == round(
        build_supp.bundle.A4_WIDTH_PT, 3
    )
    assert round(float(reader.pages[0].mediabox.height), 3) == round(
        build_supp.bundle.A4_HEIGHT_PT, 3
    )
    text = reader.pages[0].extract_text() or ""
    assert "Supplementary Figure 99: Example supplementary wrapper page." in text


def test_build_supplementary_wrapper_composes_two_blocks_on_one_page(tmp_path):
    build_supp = importlib.import_module("figures.build_supplementary_figures")

    figure_a_pdf = tmp_path / "figure_a.pdf"
    fig, ax = plt.subplots(figsize=(3.0, 1.4))
    ax.plot([0, 1], [0, 1])
    fig.savefig(figure_a_pdf, format="pdf", bbox_inches=None, pad_inches=0)
    plt.close(fig)

    figure_b_pdf = tmp_path / "figure_b.pdf"
    fig, ax = plt.subplots(figsize=(3.0, 1.5))
    ax.plot([0, 1], [1, 0])
    fig.savefig(figure_b_pdf, format="pdf", bbox_inches=None, pad_inches=0)
    plt.close(fig)

    caption_a_pdf = build_supp.bundle.write_text_block_pdf(
        "Supplementary Figure 1: Example A",
        build_supp._supp_content_width_pt(),
        tmp_path / "caption_a.pdf",
        body="Synthetic caption A for packed supplementary layout.",
        title_fs=build_supp.SUPP_TITLE_FS,
        body_fs=build_supp.SUPP_BODY_FS,
        side_margin_in=build_supp.SUPP_CAPTION_SIDE_MARGIN_IN,
        top_margin_in=build_supp.SUPP_CAPTION_TOP_MARGIN_IN,
        bottom_margin_in=build_supp.SUPP_CAPTION_BOTTOM_MARGIN_IN,
        title_gap_in=build_supp.SUPP_TITLE_GAP_IN,
    )
    caption_b_pdf = build_supp.bundle.write_text_block_pdf(
        "Supplementary Figure 2: Example B",
        build_supp._supp_content_width_pt(),
        tmp_path / "caption_b.pdf",
        body="Synthetic caption B for packed supplementary layout.",
        title_fs=build_supp.SUPP_TITLE_FS,
        body_fs=build_supp.SUPP_BODY_FS,
        side_margin_in=build_supp.SUPP_CAPTION_SIDE_MARGIN_IN,
        top_margin_in=build_supp.SUPP_CAPTION_TOP_MARGIN_IN,
        bottom_margin_in=build_supp.SUPP_CAPTION_BOTTOM_MARGIN_IN,
        title_gap_in=build_supp.SUPP_TITLE_GAP_IN,
    )

    page = build_supp.bundle.compose_stacked_page(
        [
            (figure_a_pdf, caption_a_pdf),
            (figure_b_pdf, caption_b_pdf),
        ],
        side_margin_pt=build_supp._supp_page_margin_pt(),
        top_margin_pt=build_supp._supp_page_margin_pt(),
        bottom_margin_pt=build_supp._supp_page_margin_pt(),
    )

    output_pdf = tmp_path / "stacked.pdf"
    writer = PdfWriter()
    writer.add_page(page)
    with output_pdf.open("wb") as handle:
        writer.write(handle)

    reader = PdfReader(str(output_pdf))
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text() or ""
    assert "Supplementary Figure 1: Example A" in text
    assert "Supplementary Figure 2: Example B" in text


def test_build_supplementary_wrapper_greedily_paginates_adjacent_blocks():
    build_supp = importlib.import_module("figures.build_supplementary_figures")

    spec1 = build_supp.SupplementSpec("s1", 1, "A", "Legend")
    spec2 = build_supp.SupplementSpec("s2", 2, "B", "Legend")
    spec3 = build_supp.SupplementSpec("s3", 3, "C", "Legend")
    spec4 = build_supp.SupplementSpec("s4", 4, "D", "Legend")

    blocks = [
        build_supp.SupplementBlock(spec1, Path("a.pdf"), Path("a_cap.pdf"), 300.0),
        build_supp.SupplementBlock(spec2, Path("b.pdf"), Path("b_cap.pdf"), 300.0),
        build_supp.SupplementBlock(spec3, Path("c.pdf"), Path("c_cap.pdf"), 500.0),
        build_supp.SupplementBlock(spec4, Path("d.pdf"), Path("d_cap.pdf"), 200.0),
    ]

    pages = build_supp._paginate_blocks(blocks)
    assert [[block.spec.number for block in page] for page in pages] == [[1, 2], [3, 4]]


def test_fit_box_does_not_scale_up_narrow_figures():
    bundle = importlib.import_module("figures._internal._figure_bundle")
    scaled_width, scaled_height, scale = bundle.fit_box(
        216.0,
        108.0,
        max_width_pt=bundle.A4_WIDTH_PT,
        max_height_pt=400.0,
        allow_scale_up=False,
    )

    assert scale == 1.0
    assert scaled_width == 216.0
    assert scaled_height == 108.0


def test_fit_box_scales_down_wide_figures():
    bundle = importlib.import_module("figures._internal._figure_bundle")
    scaled_width, scaled_height, scale = bundle.fit_box(
        800.0,
        400.0,
        max_width_pt=bundle.A4_WIDTH_PT,
        max_height_pt=300.0,
        allow_scale_up=False,
    )

    assert scale < 1.0
    assert scaled_width <= bundle.A4_WIDTH_PT
    assert scaled_height <= 300.0


def test_compare_figure_records_detects_png_changes():
    expected = {
        "artifacts": {
            "pdf": {"path": "main_fig1/test.pdf", "sha256": "aaa", "size_bytes": 10},
            "png": {
                "path": "main_fig1/test.png",
                "sha256": "bbb",
                "size_bytes": 20,
                "width_px": 100,
                "height_px": 200,
                "pixel_sha256_q1": "qqq",
                "pixel_sha256_preview256": "ppp",
            },
        }
    }
    observed = {
        "pdf": {"path": "main_fig1/test.pdf", "sha256": "ccc", "size_bytes": 12},
        "png": {
            "path": "main_fig1/test.png",
            "sha256": "ddd",
            "size_bytes": 22,
            "width_px": 100,
            "height_px": 220,
            "pixel_sha256_q1": "rrr",
            "pixel_sha256_preview256": "sss",
        },
    }

    issues = pm._compare_figure_records("main:fig1", expected, observed)

    assert "main:fig1: png hash mismatch" in issues
    assert any("height_px mismatch" in issue for issue in issues)


def test_compare_figure_records_accepts_quantized_png_match():
    expected = {
        "artifacts": {
            "png": {
                "path": "main_fig1/test.png",
                "sha256": "bbb",
                "size_bytes": 20,
                "width_px": 100,
                "height_px": 200,
                "pixel_sha256_q1": "qqq",
                "pixel_sha256_preview256": "ppp",
            },
        }
    }
    observed = {
        "png": {
            "path": "main_fig1/test.png",
            "sha256": "ddd",
            "size_bytes": 22,
            "width_px": 100,
            "height_px": 200,
            "pixel_sha256_q1": "qqq",
            "pixel_sha256_preview256": "zzz",
        },
    }

    issues = pm._compare_figure_records("main:fig1", expected, observed)

    assert issues == []


def test_compare_figure_records_accepts_preview_png_match():
    expected = {
        "artifacts": {
            "png": {
                "path": "main_fig1/test.png",
                "sha256": "bbb",
                "size_bytes": 20,
                "width_px": 100,
                "height_px": 200,
                "pixel_sha256_q1": "qqq",
                "pixel_sha256_preview256": "ppp",
            },
        }
    }
    observed = {
        "png": {
            "path": "main_fig1/test.png",
            "sha256": "ddd",
            "size_bytes": 22,
            "width_px": 100,
            "height_px": 200,
            "pixel_sha256_q1": "rrr",
            "pixel_sha256_preview256": "ppp",
        },
    }

    issues = pm._compare_figure_records("main:fig1", expected, observed)

    assert issues == []


