#!/usr/bin/env python3
"""Stamp circle-packing instances: `problems/circle-packing-<N>/` for each N.

    uv run python problems/make_circle_packing.py 32

Every instance is the same problem as `problems/circle-packing` (n = 26,
hand-written, left untouched) — pack N disjoint circles in the unit square
so the sum of their radii is as large as possible — with only N changed:
same submission format (`id,x,y,r`) and the same tolerance semantics, so a
solution transfers between instances. The generated dirs are committed to
BOTH `problems/` and `src/hillclimb/demo/` (the wheel's bundled copy); rerun
this after editing a template here (`tests/test_circle_packing_starter.py`
checks that both committed copies match).

Best-known values are the reference lines on `hillclimb chart`. Sources:
the AlphaEvolve paper (Novikov et al. 2025, arXiv:2506.13131, Appendix B.12)
and its results notebook (google-deepmind/alphaevolve_results, Construction 2
sums to 2.937944526205518 under this verifier's tolerance); the later
2.939572 (Georgiev, Gómez-Serrano, Tao & Wagner 2025, arXiv:2511.02864) is
the value Erich Friedman's table (erich-friedman.github.io/packing/cirRsqu/)
lists as the current record ("2.939+", Berthold et al., Jan 2026).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_INSTANCES = (32,)

# n -> ((chart label, sum of radii), ...) — the LAST entry is the best known
BEST_KNOWN: dict[int, tuple[tuple[str, float], ...]] = {
    26: (
        ("AlphaEvolve 2025", 2.6358627564136983),
        ("best known (Georgiev et al. 2025)", 2.635983),
    ),
    32: (
        ("AlphaEvolve 2025", 2.937944526205518),
        ("best known (Georgiev et al. 2025)", 2.939572),
    ),
}


def grid_side(n: int) -> int:
    return math.ceil(math.sqrt(n))


def grid_circles(n: int) -> list[tuple[int, float, float, float]]:
    """N equal circles on the first N cells of a k x k grid, k = ceil(sqrt(N)),
    each inscribed in its cell (radius 1/(2k)): valid by construction —
    touching neighbours, tangent to the walls — and far from optimal (equal
    radii waste the corners and the empty cells)."""
    k = grid_side(n)
    r = 1 / (2 * k)
    return [(i, (2 * (i % k) + 1) / (2 * k), (2 * (i // k) + 1) / (2 * k), r) for i in range(n)]


def sample_submission(n: int) -> str:
    return "id,x,y,r\n" + "".join(f"{i},{x!r},{y!r},{r!r}\n" for i, x, y, r in grid_circles(n))


def problem_yaml(n: int) -> str:
    lines = [
        f"problem_id: circle-packing-{n}",
        "metric: sum-radii",
        "higher_is_better: true",
        "description: description.md",
        "# valid-by-construction floor: best/ always holds something",
        "baseline_files: {submission.csv: sample_submission.csv}",
    ]
    if n in BEST_KNOWN:
        lines.append("chart_baselines:")
        lines += [f'  "{label}": {value!r}' for label, value in BEST_KNOWN[n]]
    lines += ["time_budget_s: 900", "allow_internet_during_solution: false", ""]
    return "\n".join(lines)


def description_md(n: int) -> str:
    k = grid_side(n)
    if n in BEST_KNOWN:
        (first_label, first), (best_label, best) = BEST_KNOWN[n][0], BEST_KNOWN[n][-1]
        who = best_label.removeprefix("best known ").strip("()")
        known = (
            f"The best known value for n = {n} is about **{best:.6f}** ({who}); "
            f"{first_label} reported **{first:.6f}**."
        )
    else:
        known = f"No best-known value is recorded here for n = {n}."
    return f"""# Circle packing: {n} circles, maximize the sum of radii

Place **exactly {n} circles inside the unit square** `[0, 1] x [0, 1]` so that the
**sum of all radii is as large as possible**.

Constraints (all verified programmatically):

- every circle lies entirely inside the unit square: `r <= x <= 1 - r` and
  `r <= y <= 1 - r`
- no two circles overlap: `dist(center_i, center_j) >= r_i + r_j`
- all radii are non-negative
- exactly {n} rows

This is a hard continuous optimization problem. {known}
Equal circles on a grid are weak (the corners and the gaps between cells are
wasted). Good approaches combine constructive patterns (hexagonal / greedy
layouts, unequal radii — a few large circles surrounded by small ones), local
optimization of the sum under the non-overlap constraints (e.g. SLSQP,
projected gradient, or a physics-style relaxation that pushes overlapping
circles apart while inflating them), and restarts. `numpy` and `scipy` are
available.

## Submission format

Write `submission.csv` in the working directory with the header `id,x,y,r` and
{n} rows (`id` = 0..{n - 1}), like `sample_submission.csv` (a weak valid
baseline: {n} equal circles on a {k} x {k} grid).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the constraints (with a numerical tolerance of 1e-9) and
prints `val_score: <sum of radii>` (or a score of 0.0 with a reason if the
packing is invalid). Higher is better.

There is no train/test data; this is a pure optimization problem. Keep total
runtime well within the execution time limit.
"""


VERIFIER_SH = """#!/usr/bin/env bash
# hillclimb verifier: run the candidate, then score what it produced.
set -euo pipefail

"$HILLCLIMB_PYTHON" "$HILLCLIMB_SOLUTION"   # writes ./submission.csv

# trust boundary: only the scorer may report a score, so anything the
# solution left behind is discarded before the scorer runs
rm -f "$HILLCLIMB_RESULT"
"$HILLCLIMB_PYTHON" problem/verify.py
"""


def verify_py(n: int) -> str:
    return f'''"""Official scorer for the circle-packing-{n} problem.

Reads ./submission.csv (id,x,y,r; {n} rows), validates the packing, and writes
the score to $HILLCLIMB_RESULT (sum of radii, 0.0 if invalid) with one
`instances` entry per circle (its radius) so per-instance views can compare
candidates circle by circle. Same format and tolerance as `circle-packing`
(n = 26), so solutions transfer between the two.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N = {n}
TOL = 1e-9  # numerical slack on containment/overlap


def emit(score: float, instances: dict | None = None) -> None:
    """hillclimb's verifier contract: the score is written to the result
    file, not scraped from stdout (coding-agent-authored code shares this stream)."""
    payload = {{"split": os.environ.get("HILLCLIMB_SPLIT", "validation"), "score": score}}
    if instances:
        payload["instances"] = instances  # reserved key: per-circle breakdown of `score`
    path = Path(os.environ.get("HILLCLIMB_RESULT", "eval_result.json"))
    path.write_text(json.dumps(payload))


def fail(reason: str) -> None:
    print(f"INVALID: {{reason}}")
    emit(0.0)
    print("val_score: 0.0")
    sys.exit(0)


def main() -> None:
    try:
        df = pd.read_csv("submission.csv")
    except Exception as e:  # noqa: BLE001
        fail(f"cannot read submission.csv: {{e}}")
    for col in ("id", "x", "y", "r"):
        if col not in df.columns:
            fail(f"missing column {{col!r}}")
    if len(df) != N or sorted(df["id"].tolist()) != list(range(N)):
        fail(f"need exactly {{N}} rows with id 0..{{N - 1}}")
    df = df.sort_values("id")
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    r = df["r"].to_numpy(float)
    if not np.all(np.isfinite(x) & np.isfinite(y) & np.isfinite(r)):
        fail("non-finite values")
    if np.any(r < -TOL):
        fail("negative radius")
    r = np.clip(r, 0.0, None)
    inside = (x - r >= -TOL) & (x + r <= 1 + TOL) & (y - r >= -TOL) & (y + r <= 1 + TOL)
    if not np.all(inside):
        fail(f"circle(s) {{np.where(~inside)[0].tolist()}} outside the unit square")
    dist = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    required = r[:, None] + r[None, :]
    overlap = (dist + TOL < required) & ~np.eye(N, dtype=bool)
    if np.any(overlap):
        i, j = np.argwhere(overlap)[0]
        fail(f"circles {{i}} and {{j}} overlap (dist {{dist[i, j]:.6f}} < {{required[i, j]:.6f}})")
    # growth headroom per circle: distance to the nearest binding constraint
    # (a wall or a neighbour) — small radius + large slack = wasted space
    gaps = dist - required + np.where(np.eye(N, dtype=bool), np.inf, 0.0)
    wall = np.minimum.reduce([x - r, 1 - x - r, y - r, 1 - y - r])
    slack = np.minimum(gaps.min(axis=1), wall)
    weakest = np.argsort(r)[:6]
    print("smallest circles (id: radius / slack): " + ", ".join(f"{{i}}: {{r[i]:.4f}} / {{slack[i]:.4f}}" for i in weakest))
    total = float(r.sum())
    print(f"valid packing; sum of radii = {{total:.6f}}")
    emit(total, {{f"circle-{{i:02d}}": round(float(r[i]), 6) for i in range(N)}})
    print(f"val_score: {{total:.6f}}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 — an invalid submission is the floor, never a crash
        fail(f"unexpected error: {{e}}")
'''


def interface_py(n: int) -> str:
    return f'''"""Machine-checked output format (hillclimb spaces). Format only — the
geometry (containment, overlaps) stays with the verifier."""

from hillclimb import spaces

output = spaces.Table(
    "submission.csv",
    columns={{
        "id": spaces.Int(values=range({n}), unique=True),
        "x": spaces.Float(low=0.0, high=1.0),
        "y": spaces.Float(low=0.0, high=1.0),
        "r": spaces.Float(low=0.0),
    }},
    n_rows={n},
)

if __name__ == "__main__":
    raise SystemExit(spaces.main(output))
'''


PLOT_PY = '''"""How `hillclimb plot` draws a circle-packing solution (matplotlib): every
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
'''


def plot_py() -> str:
    return PLOT_PY


def files_for(n: int) -> dict[str, str]:
    return {
        "problem.yaml": problem_yaml(n),
        "description.md": description_md(n),
        "verifier.sh": VERIFIER_SH,
        "verify.py": verify_py(n),
        "interface.py": interface_py(n),
        "plot.py": plot_py(),
        "sample_submission.csv": sample_submission(n),
    }


def stamp(root: Path, n: int) -> Path:
    """Write circle-packing-<n>/ under `root`; returns the dir."""
    if n < 1:
        raise ValueError("a circle-packing instance needs at least one circle")
    problem_dir = root / f"circle-packing-{n}"
    problem_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files_for(n).items():
        (problem_dir / name).write_text(text)
    (problem_dir / "verifier.sh").chmod(0o755)
    return problem_dir


def main(argv: list[str]) -> None:
    instances = [int(a) for a in argv] or list(DEFAULT_INSTANCES)
    for n in instances:
        for root in (ROOT,):
            print(f"wrote {stamp(root, n)}")


if __name__ == "__main__":
    main(sys.argv[1:])
