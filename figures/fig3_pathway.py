"""Public wrapper for Figure 3 pathway cameraPR panels."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _internal._pathway_figure as _impl  # noqa: E402

for _name, _value in vars(_impl).items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

del _name, _value


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    fig = make_pathway_figure()
    save_figure(fig, str(out_dir / "fig3_pathway"), formats=("pdf",))
    print("Wrote fig3_pathway.pdf")
