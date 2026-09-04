"""Public wrapper for Figure 1 trajectory heatmap and trajectory-line composite."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _internal._trajectory_heatmap as _impl  # noqa: E402

for _name, _value in vars(_impl).items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

del _name, _value


if __name__ == "__main__":
    init_figure_theme()
    render_full()
