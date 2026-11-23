"""Paths the QC stage reads, resolved from the paths config (``paths.yaml``).

The QC config (``analysis/qc/config_{olink,soma}.toml``) names every input as a
``paths:`` reference; ``configured_path`` resolves one key. A study here is a platform
run of the trial, ``surmount5_olink`` or ``surmount5_soma``: its keys are
``<study>_data`` (the vendor file), ``<study>_raw_key`` and ``<study>_qc_corrections``.

The QC reference inputs (the published sex-association and ProtAge tables, and the
parsed cis-pQTL catalogs) are not tracked; ``analysis.external.qc_reference`` stages
them into gitignored directories, which the optional keys ``qc_reference_dir`` and
``pqtl_catalog_dir`` override.
"""

from __future__ import annotations

from pathlib import Path

from paths import QC_ROOT, load_paths_config

ROOT = Path(__file__).resolve().parents[1]

_cfg = load_paths_config()


def configured_path(key: str) -> Path | None:
    """Return the path a paths config key names, or None when it is absent or blank.

    A relative value resolves against the repository root, not the working directory.

    Args:
        key: A paths config key.

    Returns:
        The configured path, absolute, or None.
    """
    value = _cfg.get(key)
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def required_configured_path(key: str) -> Path:
    """Like ``configured_path``, for a key the caller cannot run without.

    Args:
        key: A paths config key.

    Returns:
        The configured path, absolute.

    Raises:
        KeyError: If the key is absent or blank, naming the key.
    """
    path = configured_path(key)
    if path is None:
        raise KeyError(f"paths config does not define {key}")
    return path


_SEX_TABLE_NAME = "41467_2025_59034_MOESM3_ESM.Table_S2.tsv"
_PROTAGE_TABLE_NAME = "41591_2024_3164_MOESM3_ESM.Table_S1.tsv"
# The published sex-association (Koprulu 2025, Table S2) and ProtAge (Argentieri 2024,
# Table S1) tables.
QC_REFERENCE_DIR = configured_path("qc_reference_dir") or ROOT / "analysis" / "qc" / "data"
SEX_TABLE = QC_REFERENCE_DIR / _SEX_TABLE_NAME
PROTAGE_TABLE = QC_REFERENCE_DIR / _PROTAGE_TABLE_NAME

# The parsed cis-pQTL catalogs the identity check scores tubes with.
PQTL_CATALOG_DIR = configured_path("pqtl_catalog_dir") or (
    ROOT / "data" / "external" / "pqtl" / "parsed"
)


def study_data(study: str) -> Path:
    """The vendor proteomics file of a platform run (Olink NPX parquet or SomaScan ADAT).

    Args:
        study: ``surmount5_olink`` or ``surmount5_soma``.

    Returns:
        The configured path.

    Raises:
        KeyError: If the paths config has no ``<study>_data``, naming the key.
    """
    return required_configured_path(f"{study}_data")


def study_qc_root(study: str) -> Path:
    """QC output directory of a platform run: ``<qc_root>/<study>``.

    Args:
        study: ``surmount5_olink`` or ``surmount5_soma``.

    Returns:
        The directory.
    """
    return QC_ROOT / study
