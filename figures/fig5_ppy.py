"""Public wrapper for Figure 5 PPY family panels."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _internal._ppy_figure as _impl  # noqa: E402

for _name, _value in vars(_impl).items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

del _name, _value


if __name__ == "__main__":
    render("PPY", "fig5_ppy", fdr_legend_text="FDR: *** <0.001")
