"""Centralized path constants derived from paths.yaml.

All figure, table, and analysis scripts import from here to resolve
data paths. When the data directories move, only paths.yaml
needs updating. Set SURMOUNT5_PATHS_YAML to point at an alternate config file
without editing the repository copy.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from analysis.design import COVAR

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


def configured_value(key: str) -> str | None:
    """Return the value of a paths config key, or None when it is absent or blank.

    Args:
        key: A paths config key.

    Returns:
        The configured value, with ``~`` and environment variables expanded.
    """
    return _cfg.get(key) or None


def required_value(key: str) -> str:
    """Return the value of a paths config key that the calling stage cannot run without.

    Args:
        key: A paths config key.

    Returns:
        The configured value, with ``~`` and environment variables expanded.

    Raises:
        KeyError: If the key is absent or blank, naming the key and the config file.
    """
    value = configured_value(key)
    if value is None:
        raise KeyError(
            f"Path configuration {_YAML_PATH} does not define {key}; "
            f"see {_ROOT / 'paths.example.yaml'} for what it names"
        )
    return value


def required_path(key: str) -> Path:
    """Return the path a paths config key names, for a stage that cannot run without it.

    Args:
        key: A paths config key.

    Returns:
        The configured path.

    Raises:
        KeyError: If the key is absent or blank, naming the key and the config file.
    """
    return Path(required_value(key))


MMRM_ROOT = Path(_cfg["mmrm_root"])
QC_ROOT = Path(_cfg["qc_root"])
PHENO_DIR = Path(_cfg["pheno_root"])
GENO_ROOT = Path(_cfg["geno_root"]) if _cfg.get("geno_root") else None
QA_ROOT = Path(_cfg["qa_root"]) if _cfg.get("qa_root") else None


def _study_dir(root: Path, suffix: str) -> Path:
    """``<root>/surmount5_<suffix>``. Existence is not checked here: the first read of a
    missing directory raises, and the path constants must import with scratch roots."""
    return root / f"surmount5_{suffix}"


def mmrm_dir(platform: str) -> Path:
    """Return the MMRM covar directory for a platform ('olink' or 'soma')."""
    return _study_dir(MMRM_ROOT, platform) / COVAR
