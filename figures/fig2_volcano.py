"""Public wrapper for Figure 2 across-treatment volcano plots."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _internal._volcano_figure as _impl  # noqa: E402

for _name, _value in vars(_impl).items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

del _name, _value


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    fig = make_volcano_figure()
    save_figure(fig, str(out_dir / "fig2_volcano"), formats=("pdf",))
    print("Wrote fig2_volcano.pdf")
