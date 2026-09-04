"""Centralized path constants derived from paths.yaml.

All figure, table, and analysis scripts import from here to resolve
dxfuse-mounted data paths. After an instance restart, only paths.yaml
needs updating. Set SURMOUNT5_PATHS_YAML to point at an alternate config
file without editing the repository copy.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent
PATHS_YAML_ENV = "SURMOUNT5_PATHS_YAML"
_DEFAULT_YAML_PATH = _ROOT / "paths.yaml"
_REQUIRED_KEYS = ("mmrm_root", "qc_root", "pheno_root")


def resolve_paths_config_path(path: str | os.PathLike[str] | Path | None = None) -> Path:
    """Return the configured path YAML location."""
    if path is not None:
        return Path(path).expanduser()
    override = os.environ.get(PATHS_YAML_ENV)
    if override:
        return Path(override).expanduser()
    return _DEFAULT_YAML_PATH


def load_paths_config(path: str | os.PathLike[str] | Path | None = None) -> dict[str, str]:
    """Load and validate the path YAML."""
    yaml_path = resolve_paths_config_path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Missing path configuration: {yaml_path}. Copy {_ROOT / 'paths.example.yaml'} "
            f"to {_DEFAULT_YAML_PATH} or set {PATHS_YAML_ENV} to an alternate file."
        )
    with yaml_path.open() as f:
        raw_cfg = yaml.safe_load(f) or {}
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Path configuration must be a YAML mapping: {yaml_path}")

    missing = [key for key in _REQUIRED_KEYS if not raw_cfg.get(key)]
    if missing:
        raise KeyError(
            f"Path configuration {yaml_path} is missing required key(s): {', '.join(missing)}"
        )

    cfg: dict[str, str] = {}
    for key, value in raw_cfg.items():
        if value in (None, ""):
            cfg[str(key)] = value
            continue
        cfg[str(key)] = os.path.expandvars(os.path.expanduser(str(value)))
    return cfg


_YAML_PATH = resolve_paths_config_path()
_cfg = load_paths_config(_YAML_PATH)

MMRM_ROOT = Path(_cfg["mmrm_root"])
QC_ROOT = Path(_cfg["qc_root"])
PHENO_DIR = Path(_cfg["pheno_root"])
GENO_ROOT = Path(_cfg["geno_root"]) if _cfg.get("geno_root") else None
QA_ROOT = Path(_cfg["qa_root"]) if _cfg.get("qa_root") else None
RAW_PROTEOMICS_ROOT = (
    Path(_cfg["raw_proteomics_root"]) if _cfg.get("raw_proteomics_root") else None
)

COVAR = "covar_AGE_SEX"


def first_existing(*paths: Path) -> Path:
    if not paths:
        raise ValueError("At least one path candidate is required.")
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def ensure_public_alias(path: Path) -> Path:
    """Return *path* unchanged (no-op in the public repository).

    The internal version of this function resolved legacy directory aliases
    and created symlinks to the canonical ``surmount5_*`` layout. In the
    public release all paths already follow that convention, so no resolution
    is necessary. The function is retained as a stable call-site for downstream
    code that originally depended on the aliasing step.
    """
    return path


def _study_dir(root: Path, suffix: str) -> Path:
    return root / f"surmount5_{suffix}"


MMRM_OLINK_DIR = _study_dir(MMRM_ROOT, "olink") / COVAR
MMRM_SOMA_DIR = _study_dir(MMRM_ROOT, "soma") / COVAR

QC_OLINK_DIR = _study_dir(QC_ROOT, "olink")
QC_SOMA_DIR = _study_dir(QC_ROOT, "soma")


def mmrm_dir(platform: str) -> Path:
    """Return the MMRM covar directory for a platform ('olink' or 'soma')."""
    return _study_dir(MMRM_ROOT, platform) / COVAR


def public_mmrm_dir(platform: str) -> Path:
    """Return the public SURMOUNT-5 MMRM covariate directory for a platform."""
    return MMRM_ROOT / f"surmount5_{platform}" / COVAR
