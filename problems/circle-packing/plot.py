"""How `hillclimb plot` draws a circle-packing solution (matplotlib): every
circle in the unit square, numbered and shaded by its radius, so where the
packing is tight and where it left room reads at a glance. Runs in the
problem's runtime venv, like the verifier.
The template lives in problems/make_circle_packing.py.
"""

import csv
from pathlib import Path

from matplotlib import colormaps
from matplotlib.patches import Circle


def plot(solution_dir, ax):
    with open(Path(solution_dir) / "submission.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    circles = [(int(r["id"]), float(r["x"]), float(r["y"]), float(r["r"])) for r in rows]
    largest = max((r for *_rest, r in circles), default=0.0) or 1.0
    shade = colormaps["viridis"]
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="0.6", linewidth=1)
    for index, x, y, r in circles:
        if r <= 0:
            continue
        ax.add_patch(Circle((x, y), r, facecolor=shade(r / largest), alpha=0.55, edgecolor="0.2", linewidth=0.8))
        ax.annotate(str(index), (x, y), ha="center", va="center", fontsize=7, color="0.1")
    ax.set_aspect("equal")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    radii = sorted(r for *_rest, r in circles)
    total = sum(radii)
    return (
        f"{len(circles)} circles · sum of radii {total:.6g}"
        + (f" · radii {radii[0]:.3g}–{radii[-1]:.3g}" if radii else "")
    )
