"""Public wrapper for Figure 4 mediation scatter and forest panels."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _internal._mediation_figure as _impl  # noqa: E402

for _name, _value in vars(_impl).items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

del _name, _value


if __name__ == "__main__":
    fig = make_figure()
    out_dir = Path(__file__).resolve().parent
    save_figure(fig, str(out_dir / "fig4_mediation"), formats=("pdf",))
    print("Wrote fig4_mediation.pdf")
