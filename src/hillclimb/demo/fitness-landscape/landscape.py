"""The terrain for the fitness-landscape demo problem.

One deterministic 2D landscape: a dozen Gaussian peaks (one tall needle,
several broad decoys) over a cosine ripple. This module is the single source
of truth for the surface — the scorer imports it, candidate code may import
it (`problem/landscape.py`), and a 3D chart renders the same `grid()`.
"""

import numpy as np

SEED = 20260828
DOMAIN = (-5.0, 5.0)
N_PEAKS = 12
RIPPLE_AMP = 0.35
RIPPLE_FREQ = 2.2


def _params():
    rng = np.random.default_rng(SEED)
    lo, hi = DOMAIN
    centers = rng.uniform(lo + 0.5, hi - 0.5, size=(N_PEAKS, 2))
    heights = rng.uniform(2.0, 6.0, size=N_PEAKS)
    widths = rng.uniform(0.5, 1.6, size=N_PEAKS)
    heights[0] = 9.0  # the global peak is a needle among broad decoys
    widths[0] = 0.22
    return centers, heights, widths


_CENTERS, _HEIGHTS, _WIDTHS = _params()


def elevation(x, y):
    """Terrain height at (x, y); broadcasts over numpy arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d2 = (x[..., None] - _CENTERS[:, 0]) ** 2 + (y[..., None] - _CENTERS[:, 1]) ** 2
    peaks = (_HEIGHTS * np.exp(-d2 / (2.0 * _WIDTHS**2))).sum(axis=-1)
    ripple = RIPPLE_AMP * (np.cos(RIPPLE_FREQ * x) + np.sin(RIPPLE_FREQ * y))
    return peaks + ripple


def grid(n=121):
    """(xs, ys, Z) sampling of the terrain, ready for plotui.add_surface3d."""
    lo, hi = DOMAIN
    xs = np.linspace(lo, hi, n)
    ys = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(xs, ys)
    return xs, ys, elevation(X, Y)
