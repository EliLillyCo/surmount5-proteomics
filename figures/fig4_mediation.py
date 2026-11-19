"""Public wrapper for Figure 4 mediation scatter and forest panels."""

from __future__ import annotations

from pathlib import Path


# The audit (analysis/tools/publication_manifest.py audit-figures) reads the implementation
# through the ``_impl`` alias; the wrapper's public API is the names bound below.
import figures._internal._mediation_figure as _impl
from figures._internal._common import save_figure

make_figure = _impl.make_figure

__all__ = ["make_figure"]


if __name__ == "__main__":
    fig = make_figure()
    out_dir = Path(__file__).resolve().parent
    save_figure(fig, str(out_dir / "fig4_mediation"), formats=("pdf",))
    print("Wrote fig4_mediation.pdf")
