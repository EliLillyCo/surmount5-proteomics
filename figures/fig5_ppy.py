"""Public wrapper for Figure 5 PPY family panels."""

from __future__ import annotations



# The audit (analysis/tools/publication_manifest.py audit-figures) reads the implementation
# through the ``_impl`` alias; the wrapper's public API is the names bound below.
import figures._internal._ppy_figure as _impl

render = _impl.render

__all__ = ["render"]


if __name__ == "__main__":
    render(_impl.FIGURE_GENE, "fig5_ppy")
