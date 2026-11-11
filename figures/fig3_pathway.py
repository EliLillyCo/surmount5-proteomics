"""Public wrapper for Figure 3: a, cameraPR pathway dot plots; b, disease-signature reversal."""

from __future__ import annotations

from pathlib import Path


# The audit (analysis/tools/publication_manifest.py audit-figures) reads the implementation
# through the ``_impl`` alias; the wrapper's public API is the names bound below.
import figures._internal._pathway_figure as _impl
from figures._internal._output import save_figure_stack

make_figure3 = _impl.make_figure3
make_pathway_figure = _impl.make_pathway_figure

__all__ = ["make_figure3", "make_pathway_figure"]


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    save_figure_stack(make_figure3(), out_dir / "fig3_pathway", formats=("pdf",))
    print("Wrote fig3_pathway.pdf")
