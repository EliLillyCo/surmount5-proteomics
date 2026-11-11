"""Per-platform constants and QC defaults for Olink Explore HT and SomaScan."""

from __future__ import annotations

from dataclasses import dataclass, replace

# Olink's two measurement columns, each with its own LOD. QC reads the one
# ``[olink] value_column`` names; the long output carries both. These are the only
# places the names are spelled.
OLINK_PC_NPX = "PCNormalizedNPX"
OLINK_NPX = "NPX"
OLINK_PC_LOD = "PCNormalizedLOD"
# The NPX-scale LOD is ``NPXLOD`` in memory, because ``olink_lod`` keeps ``LOD`` for
# the one paired with the QC data column; the output files name it ``LOD``, as
# OlinkAnalyze does.
OLINK_NPX_LOD = "NPXLOD"
OLINK_NPX_LOD_OUT = "LOD"
OLINK_VALUE_COLUMNS = (OLINK_PC_NPX, OLINK_NPX)
# Value column -> its LOD column, in memory and in the output files.
OLINK_LOD_IN_MEMORY = {OLINK_PC_NPX: OLINK_PC_LOD, OLINK_NPX: OLINK_NPX_LOD}
OLINK_LOD_IN_FILES = {OLINK_PC_NPX: OLINK_PC_LOD, OLINK_NPX: OLINK_NPX_LOD_OUT}


@dataclass(frozen=True)
class IdentityDefaults:
    """Lp identity thresholds used when the trial config does not set them."""

    best_min: float
    margin_min: float


@dataclass(frozen=True)
class PqtlDefaults:
    """Per-platform pQTL keys, used when the trial config does not set them.

    These back ``TrialConfig.pqtl_best_min``, ``pqtl_weight_exponent``,
    ``pqtl_match_floor`` and ``pqtl_mismatch_floor``. ``best_min`` is 5.0 on both
    platforms and ``weight_exponent`` differs (Olink 0.0, SomaScan 0.25).
    """

    best_min: float
    weight_exponent: float
    match_floor: float
    mismatch_floor: float


@dataclass(frozen=True)
class AssayDefaults:
    """Assay-QC thresholds used when the trial config does not set them.

    ``saturation_rfu``, ``saturation_frac`` and ``sb_ratio`` are None where the check
    does not run (Olink).
    """

    intra_cv: float  # Intra-plate CV threshold (%).
    detection_rate: float  # Detection sample threshold.
    saturation_rfu: float | None
    saturation_frac: float | None
    sb_ratio: float | None


@dataclass(frozen=True)
class Platform:
    """The constants and QC defaults of one assay platform."""

    name: str  # "olink" or "soma".
    cv_formula: str  # "olink": sqrt(exp((ln2*sd)^2)-1); "standard": sd/mean.
    default_data_col: str  # Default data column for QC.
    identity_layer: str  # Column for Lp scores and the sex/age models.
    identity_defaults: IdentityDefaults
    pqtl_defaults: PqtlDefaults
    assay_defaults: AssayDefaults


OLINK = Platform(
    name="olink",
    cv_formula="olink",
    default_data_col=OLINK_PC_NPX,
    identity_layer=OLINK_PC_NPX,
    identity_defaults=IdentityDefaults(best_min=7.0, margin_min=1.0),
    pqtl_defaults=PqtlDefaults(
        best_min=5.0, weight_exponent=0.0, match_floor=5.0, mismatch_floor=3.0
    ),
    assay_defaults=AssayDefaults(
        intra_cv=25.0,
        detection_rate=0.2,
        saturation_rfu=None,
        saturation_frac=None,
        sb_ratio=None,
    ),
)
SOMA = Platform(
    name="soma",
    cv_formula="standard",
    default_data_col="log10_scaled_RFU",
    identity_layer="log10_scaled_RFU",
    identity_defaults=IdentityDefaults(best_min=7.0, margin_min=1.0),
    pqtl_defaults=PqtlDefaults(
        best_min=5.0, weight_exponent=0.25, match_floor=5.0, mismatch_floor=3.0
    ),
    # SomaLogic's thresholds: intra-CV 10, detection 0.9, 80,000 RFU in 50% of samples,
    # S/B 2.0.
    assay_defaults=AssayDefaults(
        intra_cv=10.0,
        detection_rate=0.9,
        saturation_rfu=80_000.0,
        saturation_frac=0.5,
        sb_ratio=2.0,
    ),
)

# Accepted spellings of each platform: the driver says "soma", TOML files may say
# "somascan".
PLATFORMS: dict[str, Platform] = {"olink": OLINK, "soma": SOMA, "somascan": SOMA}


def qc_platform(name: str, olink_value_column: str = OLINK_PC_NPX) -> Platform:
    """Looks up a platform with the value column QC reads.

    Olink QC reads ``olink_value_column`` everywhere the platform's value column is
    read: the default QC data column and the identity layer (the sex, age, identity
    and incretin inputs). SomaScan has one value column, so the argument must be left
    at its default there.

    Args:
        name: ``"olink"``, ``"soma"`` or ``"somascan"``.
        olink_value_column: ``"PCNormalizedNPX"`` (the default) or ``"NPX"``.

    Returns:
        The platform; ``OLINK`` itself at the default value column.

    Raises:
        ValueError: If the name is unknown, the value column is not an Olink value
            column, or a SomaScan platform is given a value column other than the
            default.
    """
    platform = get_platform(name)
    if olink_value_column not in OLINK_VALUE_COLUMNS:
        raise ValueError(
            f"[olink] value_column must be one of {', '.join(OLINK_VALUE_COLUMNS)}; "
            f"got {olink_value_column!r}"
        )
    if platform.name != "olink":
        if olink_value_column != OLINK_PC_NPX:
            raise ValueError(
                "[olink] value_column is an Olink setting; a SomaScan config leaves it unset"
            )
        return platform
    if olink_value_column == OLINK_PC_NPX:
        return OLINK
    return replace(OLINK, default_data_col=olink_value_column, identity_layer=olink_value_column)


def get_platform(name: str) -> Platform:
    """Looks up a platform by name.

    Args:
        name: ``"olink"``, ``"soma"`` or ``"somascan"``.

    Returns:
        The platform.

    Raises:
        ValueError: If the name is not a known platform.
    """
    if name not in PLATFORMS:
        raise ValueError(
            f"Unknown platform {name!r}; expected one of {', '.join(sorted(PLATFORMS))}"
        )
    return PLATFORMS[name]
