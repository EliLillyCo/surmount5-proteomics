"""Nature figure sizing constants shared across manuscript figures."""

from __future__ import annotations

_MM = 1.0 / 25.4  # millimetre -> inch

NAT_W1 = 89 * _MM  # 3.504 in — single column
NAT_W15A = 120 * _MM  # 4.724 in — 1.5 column (narrow)
NAT_W15B = 136 * _MM  # 5.354 in — 1.5 column (wide)
NAT_W2 = 183 * _MM  # 7.205 in — double column
NAT_HMAX = 247 * _MM  # 9.724 in — full page depth (hard ceiling)

_NAT_WIDTHS: dict[str, float] = {
    "1col": NAT_W1,
    "1.5col-narrow": NAT_W15A,
    "1.5col": NAT_W15B,
    "2col": NAT_W2,
}


def nature_figsize(width, height_in: float) -> tuple[float, float]:
    """Return (width_in, height_in) for a Nature figure, in inches."""
    w = _NAT_WIDTHS[width] if isinstance(width, str) else float(width)
    assert height_in <= NAT_HMAX + 1e-6, (
        f"figure height {height_in:.2f}in exceeds Nature page depth "
        f"{NAT_HMAX:.2f}in (247 mm)"
    )
    return (w, height_in)
