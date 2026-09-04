from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.platform_inputs import _visit_specific_percent_change
from figures._internal import _lookups

REPO_ROOT = ROOT


def _repo_scratch(name: str) -> Path:
    scratch = REPO_ROOT / ".pytest_scratch" / f"{name}_{uuid4().hex}"
    scratch.mkdir(parents=True, exist_ok=False)
    return scratch


def test_visit_specific_percent_change_keeps_missing_follow_up_null() -> None:
    frame = pl.DataFrame(
        {
            "VISITNUM": [2, 8, 20],
            "PCTCHG_WGTV8": [None, -7.5, None],
            "PCTCHG_WGTV20": [None, None, None],
        }
    )

    out = frame.with_columns(
        _visit_specific_percent_change("PCTCHG_WGTV", "PCTCHG_WGT")
    )

    assert out.get_column("PCTCHG_WGT").to_list() == [0.0, -7.5, None]


def test_marker_gene_lookups_keep_first_match(monkeypatch: pytest.MonkeyPatch) -> None:
    synthetic = pl.DataFrame(
        {
            "gene_symbol": ["ITGAV", "ITGB3", "LEP"],
            "olink_olinkids": [["OID1"], ["OID1"], ["OID2"]],
            "soma_seqids": [["SEQ1"], ["SEQ1"], ["SEQ2"]],
            "present_olink": [True, True, True],
            "present_soma": [True, True, True],
        }
    )
    monkeypatch.setattr(_lookups.pl, "read_parquet", lambda *args, **kwargs: synthetic)

    assert _lookups.build_gene_lookup()["OID1"] == "ITGAV"
    assert _lookups.load_marker_to_gene("soma")["SEQ1"] == "ITGAV"
    with pytest.raises(ValueError):
        _lookups.load_marker_to_gene("invalid")








def test_run_mmrm_main_uses_study_name_for_output_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mmrm = types.ModuleType("analysis.mmrm")
    captured: dict[str, object] = {}

    def fake_run_mmrm_pipeline(**kwargs):
        captured.update(kwargs)

    fake_mmrm.run_mmrm_pipeline = fake_run_mmrm_pipeline
    monkeypatch.setitem(sys.modules, "analysis.mmrm", fake_mmrm)
    sys.modules.pop("analysis.pipelines.run_mmrm", None)
    run_mmrm = importlib.import_module("analysis.pipelines.run_mmrm")

    prepared = types.SimpleNamespace(
        data=pl.DataFrame(),
        annotation=pl.DataFrame(),
        response_var="NPX",
        response_var_bl="NPXBL",
        protein_id_col="OlinkID",
    )
    monkeypatch.setattr(run_mmrm, "prepare_mmrm_inputs", lambda platform: prepared)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mmrm",
            "--platform",
            "olink",
            "--study-name",
            "surmount5",
            "--output-root",
            "custom_root",
        ],
    )

    assert run_mmrm.main() == 0
    assert captured["output_dir"] == Path("custom_root") / "surmount5_olink"
    assert captured["study_name"] == "surmount5"


def test_run_mediation_main_uses_study_name_for_output_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mediation = types.ModuleType("analysis.mediation")
    captured: dict[str, object] = {}

    def fake_run_mediation_pipeline(**kwargs):
        captured.update(kwargs)

    fake_mediation.run_mediation_pipeline = fake_run_mediation_pipeline
    monkeypatch.setitem(sys.modules, "analysis.mediation", fake_mediation)
    sys.modules.pop("analysis.pipelines.run_mediation", None)
    run_mediation = importlib.import_module("analysis.pipelines.run_mediation")

    prepared = types.SimpleNamespace(
        data=pl.DataFrame(),
        annotation=pl.DataFrame(),
        response_var="NPX",
        response_var_bl="NPXBL",
        protein_id_col="OlinkID",
    )
    monkeypatch.setattr(
        run_mediation, "prepare_mediation_inputs", lambda platform: prepared
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mediation",
            "--platform",
            "soma",
            "--study-name",
            "surmount5",
            "--output-root",
            "custom_root",
            "--mediator",
            "PCTCHG_WGT",
            "--sims",
            "1000",
        ],
    )

    assert run_mediation.main() == 0
    assert captured["output_dir"] == Path("custom_root") / "surmount5_soma"
    assert captured["study_name"] == "surmount5"


def test_mediation_checkpoint_path_changes_when_configuration_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pkg = types.ModuleType("analysis.mediation")
    fake_pkg.__path__ = [str(REPO_ROOT / "analysis" / "mediation")]
    fake_core = types.ModuleType("analysis.mediation.core")
    fake_core.run_mediation_single = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "analysis.mediation", fake_pkg)
    monkeypatch.setitem(sys.modules, "analysis.mediation.core", fake_core)
    sys.modules.pop("analysis.mediation.pipeline", None)
    pipeline = importlib.import_module("analysis.mediation.pipeline")

    kwargs = {
        "mediation_dir": Path("results"),
        "visit": 8,
        "cat_covariates": ["SEX"],
        "cont_covariates": ["AGE"],
        "baseline_visitnum": 2,
        "response_var": "NPX",
        "response_var_bl": "NPXBL",
        "treat_col": "TRT01A",
        "control_value": "SEMA2.4mgorMTD",
        "treat_value": "TZP15mgorMTD",
        "sims": 1000,
    }
    weight_path = pipeline._checkpoint_path(mediator="PCTCHG_WGT", **kwargs)
    other_path = pipeline._checkpoint_path(mediator="OTHER_MEDIATOR", **kwargs)

    assert weight_path != other_path
    assert weight_path.name.startswith("_checkpoint_PCTCHG_WGT_v8_")
