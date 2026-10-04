"""How `hillclimb plot` draws a heilbronn-14 solution (matplotlib): the points
in the unit square, numbered, and the smallest triangle they span — the one
the score is — shaded. A good layout ties many triangles at the minimum;
shading them all would hide the points, so one is drawn and the caption
counts the rest. Runs in the problem's runtime venv, like the verifier.
"""

import csv
from itertools import combinations
from pathlib import Path

N = 14


def area(a, b, c):
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def plot(solution_dir, ax):
    with open(Path(solution_dir) / "submission.csv", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["id"]))
    pts = [(float(row["x"]), float(row["y"])) for row in rows]
    areas = {t: area(*(pts[i] for i in t)) for t in combinations(range(len(pts)), 3)}
    smallest = min(areas.values())
    tied = [t for t, a in areas.items() if a <= smallest * (1 + 1e-6) + 1e-12]

    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="0.6", linewidth=1)
    i, j, k = tied[0]
    triangle = [pts[i], pts[j], pts[k]]
    ax.fill(*zip(*triangle), color="#eab308", alpha=0.3, linewidth=0)
    ax.plot(*zip(*triangle, triangle[0]), color="#eab308", linewidth=2, label="smallest triangle")
    ax.scatter(*zip(*pts), s=60, color="#1f9fb4", zorder=3, label="point")
    for index, (x, y) in enumerate(pts):
        ax.annotate(str(index), (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, color="0.3")
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
    ties = f" (one of {len(tied)} tied)" if len(tied) > 1 else ""
    return f"{len(pts)} points · smallest triangle {smallest:.6g}{ties}"
