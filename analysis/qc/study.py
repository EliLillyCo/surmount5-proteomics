"""Builds one platform run's QC from ``analysis/qc/config_<platform>.toml`` and the paths config.

SURMOUNT-5 has one trial on two platforms, so a QC "study" is a platform run:
``surmount5_olink`` (Olink Explore HT) or ``surmount5_soma`` (SomaScan 11K). Each has its
own config, chosen by platform.

``trial_config`` resolves a run's config: every ``paths:<key>`` value becomes that paths
key's value, with ``{study}`` replaced by the run (``surmount5_olink``) and ``{trial}``
by the trial (``surmount5``, for the inputs both platforms share); the corrections file
is read into ``[[corrections]]`` entries; and the study name and file stem are the ones
``analysis.qc_contract`` gives the run. ``driver_options`` turns ``[steps]`` into the
driver's options, and ``run_study`` runs the driver, writing ``<qc_root>/<study>/<study>.*``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from analysis.qc.config import TrialConfig, config_from_mapping
from analysis.qc.driver import parse_args, run
from analysis.qc_contract import (
    PLATFORM_STUDIES,
    TRIAL,
    platform_of,
    qc_stem,
    qc_study_name,
)
from analysis.qc_paths import configured_path, study_data, study_qc_root
from paths import configured_value

QC_CONFIGS: dict[str, Path] = {
    platform: Path(__file__).with_name(f"config_{platform}.toml") for platform in PLATFORM_STUDIES
}
PATHS_PREFIX = "paths:"
_PATHS_REF = re.compile(r"^paths:(\{study\}|\{trial\})_([a-z0-9_]+)(/.+)?$")
# Paths keys whose value is a name rather than a file or directory (the PGEN naming family of
# ``analysis.qc.pqtl.pgen_prefixes``, ``<prefix>.chr{chrom}``): taken as written, not made
# absolute.
NAME_KEYS = ("pgen_pattern",)
# Tables of the config the driver reads rather than the trial config.
STEPS_TABLE = "steps"
# Keys the driver takes as comma-separated lists.
LIST_STEPS = ("clinical_vars", "force_categorical", "group_col")


def config_path(study: str) -> Path:
    """The QC config of a platform run.

    Args:
        study: ``surmount5_olink`` or ``surmount5_soma``.

    Returns:
        ``analysis/qc/config_<platform>.toml``.
    """
    return QC_CONFIGS[platform_of(study)]


def _paths_key(match: re.Match, study: str) -> str:
    prefix = study if match.group(1) == "{study}" else TRIAL
    return f"{prefix}_{match.group(2)}"


def resolve_ref(value: str, study: str) -> str:
    """Returns the path a ``paths:{study}_<name>`` or ``paths:{trial}_<name>`` value names.

    Raises:
        ValueError: If the value is not such a reference.
        KeyError: If the paths config lacks the key.
    """
    m = _PATHS_REF.fullmatch(value)
    if m is None:
        raise ValueError(f"{value!r} is not a paths:{{study}}_<name> or {{trial}}_<name> reference")
    key = _paths_key(m, study)
    if m.group(2) in NAME_KEYS:
        name = configured_value(key)
        if name is None:
            raise KeyError(f"paths config has no {key} (needed by {config_path(study).name})")
        return name
    path = configured_path(key)
    if path is None:
        raise KeyError(f"paths config has no {key} (needed by {config_path(study).name})")
    return str(path) + (m.group(3) or "")


def _resolve(node, study: str):
    """Resolves every paths reference in a parsed TOML value, recursively."""
    if isinstance(node, dict):
        return {k: _resolve(v, study) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(v, study) for v in node]
    if isinstance(node, str) and node.startswith(PATHS_PREFIX):
        return resolve_ref(node, study)
    return node


def paths_keys(study: str) -> list[str]:
    """The paths config keys a run's config reads, in first-use order."""
    keys: list[str] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str) and (m := _PATHS_REF.fullmatch(node)):
            key = _paths_key(m, study)
            if key not in keys:
                keys.append(key)

    walk(tomllib.loads(config_path(study).read_text()))
    return keys


def resolved_config(study: str) -> dict:
    """Returns a run's config parsed, with paths resolved and the corrections read.

    The result is the trial config's TOML mapping plus its ``[steps]`` table, with
    ``study_name`` set.
    """
    raw = _resolve(tomllib.loads(config_path(study).read_text()), study)
    corrections_file = Path(raw.pop("corrections"))
    corrections = tomllib.loads(corrections_file.read_text())
    unknown = sorted(set(corrections) - {"corrections"})
    if unknown:
        raise ValueError(
            f"{corrections_file}: unknown top-level keys {unknown}; entries go in [[corrections]] tables"
        )
    raw["corrections"] = corrections.get("corrections", [])
    raw["study_name"] = qc_study_name(study)
    return raw


def trial_config(study: str) -> TrialConfig:
    """Builds the validated trial config of a platform run."""
    raw = resolved_config(study)
    raw.pop(STEPS_TABLE)
    return config_from_mapping(raw, config_path(study).parent)


def driver_options(study: str, output_dir: Path):
    """Returns the driver's parsed options for a run, its steps from ``[steps]``."""
    steps = resolved_config(study)[STEPS_TABLE]
    argv = [
        "--platform",
        platform_of(study),
        "--proteomics-data",
        str(study_data(study)),
        "--output-dir",
        str(output_dir),
        "--output-stem",
        qc_stem(study),
        "--study-name",
        qc_study_name(study),
    ]
    for key, value in steps.items():
        argv += [
            f"--{key.replace('_', '-')}",
            ",".join(value) if key in LIST_STEPS else str(value),
        ]
    return parse_args(argv)


def run_study(study: str, output_dir: Path | None = None) -> Path:
    """Runs the QC of a platform run into ``output_dir`` (default ``<qc_root>/<study>``).

    Returns:
        The output directory, holding the files ``analysis.qc_contract`` names.
    """
    out = Path(output_dir) if output_dir is not None else study_qc_root(study)
    run(driver_options(study, out), trial_config(study))
    return out
