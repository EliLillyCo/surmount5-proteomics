from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.tools import publication_manifest as pm
import paths as paths_module


def test_resolve_paths_config_path_prefers_env_override(tmp_path, monkeypatch):
    custom = tmp_path / "custom_paths.yaml"
    custom.write_text(
        "mmrm_root: /mnt/mmrm\nqc_root: /mnt/qc\npheno_root: /mnt/pheno\n",
    )

    monkeypatch.setenv(paths_module.PATHS_YAML_ENV, str(custom))

    assert paths_module.resolve_paths_config_path() == custom


def test_load_paths_config_expands_user_and_environment(tmp_path, monkeypatch):
    custom = tmp_path / "custom_paths.yaml"
    monkeypatch.setenv("SURMOUNT5_TEST_QC_ROOT", "/mnt/qc")
    custom.write_text(
        "mmrm_root: ~/mmrm\nqc_root: $SURMOUNT5_TEST_QC_ROOT\npheno_root: /mnt/pheno\n",
    )

    cfg = paths_module.load_paths_config(custom)

    assert cfg["mmrm_root"] == os.path.expanduser("~/mmrm")
    assert cfg["qc_root"] == "/mnt/qc"
    assert cfg["pheno_root"] == "/mnt/pheno"


def test_load_paths_config_requires_core_keys(tmp_path):
    custom = tmp_path / "custom_paths.yaml"
    custom.write_text("mmrm_root: /mnt/mmrm\nqc_root: /mnt/qc\n")

    with pytest.raises(KeyError, match="pheno_root"):
        paths_module.load_paths_config(custom)


def test_pytest_collection_ignores_inherited_bad_paths_env():
    bad_config = ROOT / "analysis" / "tests" / "__missing_paths__.yaml"
    assert not bad_config.exists()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "analysis/tests/test_pipeline_integrity.py",
            "-q",
        ],
        cwd=ROOT,
        env={**os.environ, paths_module.PATHS_YAML_ENV: str(bad_config)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_visit_specific_percent_change_keeps_missing_follow_up_null" in result.stdout


def test_publication_manifest_check_reports_missing_task_and_template_key(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv(pm.PATHS_ENV_VAR, raising=False)
    pixi_file = tmp_path / "pixi.toml"
    pixi_file.write_text("[tasks]\ncheck-r = 'echo ok'\n")

    paths_example = tmp_path / "paths.example.yaml"
    paths_example.write_text(
        "mmrm_root: /mnt/mmrm\nqc_root: /mnt/qc\npheno_root: /mnt/pheno\n",
    )

    legends = tmp_path / "supplementary_legends.yaml"
    legends.write_text(
        "\n".join(
            [
                "figures:",
                "  - key: supp_fig1",
                "    number: 1",
                "    title: Example",
                "    legend: Example legend.",
                "",
            ]
        )
    )

    manifest = {
        "data_inputs": {
            "qc": {"root": "paths.yaml:qa_root/example.parquet"},
        },
        "analysis_outputs": {
            "manifest": {
                "command": "uv run python analysis/tools/publication_manifest.py check"
            }
        },
        "figures": {
            "supplementary": {
                "supp_fig1": "uv run python analysis/tools/publication_manifest.py list"
            }
        },
        "validation": {
            "pixi_sync": {"commands": ["pixi run check-r", "pixi run install-r"]},
        },
    }

    monkeypatch.setattr(pm, "load_manifest", lambda path=pm.MANIFEST: manifest)
    monkeypatch.setattr(pm, "PIXI_FILE", pixi_file)
    monkeypatch.setattr(pm, "PATHS_TEMPLATE", paths_example)
    monkeypatch.setattr(pm, "LOCAL_PATHS_FILE", tmp_path / "paths.yaml")
    monkeypatch.setattr(pm, "SUPPLEMENTARY_LEGENDS", legends)

    assert pm.check_manifest() == 1

    out = capsys.readouterr().out
    assert "pixi.toml: missing tasks install-r" in out
    assert "paths.example.yaml: missing keys qa_root" in out


def test_publication_manifest_output_root_defaults_to_project_local_dir(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pm, "DEFAULT_RENDER_ROOT", tmp_path / "renders")

    with pm._output_root(None, "figure-render-") as out_dir:
        sentinel = out_dir / "sentinel.txt"
        sentinel.write_text("ok\n")
        assert out_dir.parent == tmp_path / "renders"
        assert sentinel.exists()

    assert out_dir.exists()
    assert (out_dir / "sentinel.txt").exists()


def test_publication_manifest_check_warns_for_optional_local_path_keys(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv(pm.PATHS_ENV_VAR, raising=False)
    pixi_file = tmp_path / "pixi.toml"
    pixi_file.write_text("[tasks]\ncheck-r = 'echo ok'\n")

    paths_example = tmp_path / "paths.example.yaml"
    paths_example.write_text(
        "mmrm_root: /mnt/mmrm\n"
        "qc_root: /mnt/qc\n"
        "pheno_root: /mnt/pheno\n"
        "qa_root: /mnt/qa\n",
    )

    local_paths = tmp_path / "paths.yaml"
    local_paths.write_text(
        "mmrm_root: /mnt/mmrm\nqc_root: /mnt/qc\npheno_root: /mnt/pheno\n",
    )

    legends = tmp_path / "supplementary_legends.yaml"
    legends.write_text(
        "\n".join(
            [
                "figures:",
                "  - key: supp_fig1",
                "    number: 1",
                "    title: Example",
                "    legend: Example legend.",
                "",
            ]
        )
    )

    manifest = {
        "data_inputs": {
            "qc": {"root": "paths.yaml:qa_root/example.parquet"},
        },
        "analysis_outputs": {
            "manifest": {
                "command": "uv run python analysis/tools/publication_manifest.py list"
            }
        },
        "figures": {
            "supplementary": {
                "supp_fig1": "uv run python analysis/tools/publication_manifest.py list"
            }
        },
        "validation": {
            "pixi_sync": {"commands": ["pixi run check-r"]},
        },
    }

    monkeypatch.setattr(pm, "load_manifest", lambda path=pm.MANIFEST: manifest)
    monkeypatch.setattr(pm, "PIXI_FILE", pixi_file)
    monkeypatch.setattr(pm, "PATHS_TEMPLATE", paths_example)
    monkeypatch.setattr(pm, "LOCAL_PATHS_FILE", local_paths)
    monkeypatch.setattr(pm, "SUPPLEMENTARY_LEGENDS", legends)

    assert pm.check_manifest() == 0

    out = capsys.readouterr().out
    assert "WARN paths.yaml: optional keys not configured qa_root" in out


def test_main_and_supplementary_bundle_wrappers_list_available_figures(
    monkeypatch, capsys
):
    monkeypatch.syspath_prepend(str(ROOT / "figures"))
    build_main = importlib.import_module("build_main_figures")
    build_supp = importlib.import_module("build_supplementary_figures")

    assert build_main.main(["--list-figures"]) == 0
    main_out = capsys.readouterr().out
    assert "fig1_trajectory" in main_out
    assert "Figure 1" in main_out

    assert build_supp.main(["--list-figures"]) == 0
    supp_out = capsys.readouterr().out
    assert "supp_fig1_study_design" in supp_out
    assert "Supplementary Figure 1" in supp_out


def test_table_builder_lists_tabs_without_rendering_workbook(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(ROOT / "tables"))
    build_all = importlib.import_module("build_all")

    assert build_all.main(["--list-tables"]) == 0

    out = capsys.readouterr().out
    assert "Table S1" in out
    assert "Table S16" in out
