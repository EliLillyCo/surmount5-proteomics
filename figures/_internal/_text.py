"""Deterministic text-placement helpers for manuscript figures."""

from __future__ import annotations

import numpy as np
from adjustText import adjust_text as _adjust_text


def deterministic_adjust_text(*args, seed: int = 0, iter_lim: int = 500, **kwargs):
    """Run adjustText with fixed RNG and iteration limits so rerenders stay stable."""
    if "iter_lim" not in kwargs and "time_lim" not in kwargs:
        kwargs["iter_lim"] = iter_lim
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return _adjust_text(*args, **kwargs)
    finally:
        np.random.set_state(state)
