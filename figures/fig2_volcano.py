"""Public wrapper for Figure 2 between-treatment volcano plots."""

from __future__ import annotations

from pathlib import Path


# The audit (analysis/tools/publication_manifest.py audit-figures) reads the implementation
# through the ``_impl`` alias; the wrapper's public API is the names bound below.
import figures._internal._volcano_figure as _impl
from figures._internal._common import save_figure

make_volcano_figure = _impl.make_volcano_figure

__all__ = ["make_volcano_figure"]


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    fig = make_volcano_figure()
    save_figure(fig, str(out_dir / "fig2_volcano"), formats=("pdf",))
    print("Wrote fig2_volcano.pdf")
