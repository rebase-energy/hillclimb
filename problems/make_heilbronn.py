#!/usr/bin/env python3
"""Stamp the Heilbronn difficulty ladder: `problems/heilbronn-<N>/` for each N.

    uv run python problems/make_heilbronn.py 11 14 17

Every level is the same problem — place N points in the unit square so the
smallest triangle they span is as large as possible — with only N changed,
so a comparison across levels measures difficulty and nothing else. The
generated dirs are committed; rerun this after editing a template here
(`tests/test_heilbronn_ladder.py` checks the committed dirs match).

Best-known values are the reference lines on `hillclimb chart`; source:
Erich Friedman's Packing Center, https://erich-friedman.github.io/packing/heilbronn/
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# the wheel's bundled copy (`hillclimb problem get`): stamped identically
DEMO_ROOT = ROOT.parent / "src" / "hillclimb" / "demo"
DEFAULT_LEVELS = (11, 14, 17)

# n -> (best known minimum triangle area, who found it)
BEST_KNOWN: dict[int, tuple[float, str]] = {
    7: (0.08386, "Comellas & Yebra 2001"),
    8: ((math.sqrt(13) - 1) / 36, "Comellas & Yebra 2001"),
    9: ((9 * math.sqrt(65) - 55) / 320, "Comellas & Yebra 2001"),
    10: (0.04654, "Comellas & Yebra 2001"),
    11: (1 / 27, "Goldberg 1972"),
    12: (0.03260, "Comellas & Yebra 2001"),
    13: (0.02702, "Karpov 2011"),
    14: (0.02430, "Beyleveld 2006"),
    15: (0.02121, "Sudermann-Merx 2026"),
    16: (7 / 341, "Beyleveld 2006"),
    17: (0.016481, "Stead 2026"),
    18: (0.01498, "Stead 2026"),
}


def parabola_points(n: int) -> list[tuple[int, float, float]]:
    """N points on y = x², x = i/(N-1): valid by construction (a line meets
    a parabola in at most two points, so no three are collinear), inside the
    square, and far from optimal — the smallest triangle is h³, h = 1/(N-1)."""
    return [(i, round(i / (n - 1), 6), round((i / (n - 1)) ** 2, 6)) for i in range(n)]


def sample_submission(n: int) -> str:
    return "id,x,y\n" + "".join(f"{i},{x},{y}\n" for i, x, y in parabola_points(n))


def problem_yaml(n: int) -> str:
    lines = [
        f"problem_id: heilbronn-{n}",
        "metric: min-triangle-area",
        "higher_is_better: true",
        "description: description.md",
        "# valid-by-construction floor: best/ always holds something",
        "baseline_files: {submission.csv: sample_submission.csv}",
    ]
    if n in BEST_KNOWN:
        value, who = BEST_KNOWN[n]
        lines += ["chart_baselines:", f'  "best known ({who})": {value:.6f}']
    lines += ["time_budget_s: 900", "allow_internet_during_solution: false", ""]
    return "\n".join(lines)


def description_md(n: int) -> str:
    triangles = math.comb(n, 3)
    if n in BEST_KNOWN:
        value, who = BEST_KNOWN[n]
        best = f"The best known value for n = {n} is about **{value:.4f}** ({who})."
    else:
        best = f"No best-known value is recorded here for n = {n}."
    return f"""# Heilbronn triangle problem: {n} points

Place **exactly {n} points inside the unit square** `[0, 1] x [0, 1]` so that the
**smallest triangle** formed by any 3 of the points has **as large an area as
possible** (a max-min objective over all C({n},3) = {triangles} triangles).

Constraints (verified programmatically):

- all points inside the unit square (`0 <= x, y <= 1`)
- exactly {n} rows
- no 3 points may be exactly collinear (that makes the minimum area 0)

This is a classic hard continuous optimization problem. {best}
Regular grids are terrible (many nearly collinear triples). Good approaches:
random/structured starts + local optimization of the max-min objective (e.g.
maximize the smallest area with SLSQP or a soft-min surrogate), simulated
annealing on point positions, and restarts. `numpy` and `scipy` are available.

## Submission format

Write `submission.csv` in the working directory with the header `id,x,y` and {n}
rows (`id` = 0..{n - 1}), like `sample_submission.csv` (a weak valid baseline:
points on a parabola).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the layout and prints `val_score: <min triangle area>` (0.0
if invalid). Higher is better.

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
    return f'''"""Official scorer for the heilbronn-{n} problem.

Reads ./submission.csv (id,x,y; {n} rows), validates, and writes the
score to $HILLCLIMB_RESULT (minimum triangle area over all triples, 0.0 if invalid).
Generated by problems/make_heilbronn.py — edit the template there.
"""

import json
import os
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

N = {n}
TOL = 1e-9


def emit(score: float) -> None:
    """hillclimb's verifier contract: the score is written to the result
    file, not scraped from stdout (coding-agent-authored code shares this stream)."""
    path = Path(os.environ.get("HILLCLIMB_RESULT", "eval_result.json"))
    path.write_text(json.dumps({{"split": os.environ.get("HILLCLIMB_SPLIT", "validation"),
                                "score": score}}))


def fail(reason: str) -> None:
    print(f"INVALID: {{reason}}")
    print("val_score: 0.0")
    sys.exit(0)


def main() -> None:
    try:
        df = pd.read_csv("submission.csv")
    except Exception as e:  # noqa: BLE001
        fail(f"cannot read submission.csv: {{e}}")
    for col in ("id", "x", "y"):
        if col not in df.columns:
            fail(f"missing column {{col!r}}")
    if len(df) != N or sorted(df["id"].tolist()) != list(range(N)):
        fail(f"need exactly {{N}} rows with id 0..{{N - 1}}")
    pts = df.sort_values("id")[["x", "y"]].to_numpy(float)
    if not np.all(np.isfinite(pts)):
        fail("non-finite coordinates")
    if np.any(pts < -TOL) or np.any(pts > 1 + TOL):
        fail("point(s) outside the unit square")
    min_area = min(
        0.5
        * abs(
            (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1])
            - (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
        )
        for i, j, k in combinations(range(N), 3)
    )
    print(f"valid layout; min triangle area = {{min_area:.8f}}")
    emit(min_area)
    print(f"val_score: {{min_area:.8f}}")


if __name__ == "__main__":
    main()
'''


def fingerprint_py(n: int) -> str:
    return f'''"""Behavioral fingerprint for `hillclimb similarity`: the sorted areas of
all C({n},3) triangles a candidate's points span.

The flat submission.csv fingerprint would treat a relabelled or reflected
copy of the same layout as a different one; sorted triangle areas are
invariant to point order and to every isometry of the square, and they are
the objective's own geometry (the smallest entry is the score). Returns None
for anything the scorer would reject, so such candidates stay unpositioned.
Generated by problems/make_heilbronn.py — edit the template there.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

N = {n}
TOL = 1e-9


def fingerprint(candidate_dir):
    try:
        df = pd.read_csv(Path(candidate_dir) / "submission.csv")
    except Exception:  # noqa: BLE001 — unreadable output is "no fingerprint"
        return None
    if any(col not in df.columns for col in ("id", "x", "y")):
        return None
    if len(df) != N or sorted(df["id"].tolist()) != list(range(N)):
        return None
    pts = df.sort_values("id")[["x", "y"]].to_numpy(float)
    if not np.all(np.isfinite(pts)) or np.any(pts < -TOL) or np.any(pts > 1 + TOL):
        return None
    areas = [
        0.5
        * abs(
            (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1])
            - (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
        )
        for i, j, k in combinations(range(N), 3)
    ]
    return sorted(areas)
'''


PLOT_PY = '''"""How `hillclimb plot` draws a heilbronn-__N__ solution (matplotlib): the points
in the unit square, numbered, and the smallest triangle they span — the one
the score is — shaded. A good layout ties many triangles at the minimum;
shading them all would hide the points, so one is drawn and the caption
counts the rest. Runs in the problem's runtime venv, like the verifier.
Generated by problems/make_heilbronn.py — edit the template there.
"""

import csv
from itertools import combinations
from pathlib import Path

N = __N__


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
'''


def plot_py(n: int) -> str:
    return PLOT_PY.replace("__N__", str(n))


def interface_py(n: int) -> str:
    return f'''"""Machine-checked output format (hillclimb spaces). Format only — the
geometry (collinear triples, the smallest triangle) stays with the verifier.
Generated by problems/make_heilbronn.py — edit the template there.
"""

from hillclimb import spaces

output = spaces.Table(
    "submission.csv",
    columns={{
        "id": spaces.Int(values=range({n}), unique=True),
        "x": spaces.Float(low=0.0, high=1.0),
        "y": spaces.Float(low=0.0, high=1.0),
    }},
    n_rows={n},
)

if __name__ == "__main__":
    raise SystemExit(spaces.main(output))
'''


def files_for(n: int) -> dict[str, str]:
    return {
        "problem.yaml": problem_yaml(n),
        "description.md": description_md(n),
        "verifier.sh": VERIFIER_SH,
        "verify.py": verify_py(n),
        "fingerprint.py": fingerprint_py(n),
        "interface.py": interface_py(n),
        "plot.py": plot_py(n),
        "sample_submission.csv": sample_submission(n),
    }


def stamp(root: Path, n: int) -> Path:
    """Write problems/heilbronn-<n>/ under `root`; returns the dir."""
    if n < 4:
        raise ValueError("a Heilbronn instance needs at least 4 points")
    problem_dir = root / f"heilbronn-{n}"
    problem_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files_for(n).items():
        (problem_dir / name).write_text(text)
    (problem_dir / "verifier.sh").chmod(0o755)
    return problem_dir


def main(argv: list[str]) -> None:
    levels = [int(a) for a in argv] or list(DEFAULT_LEVELS)
    for n in levels:
        for root in (ROOT, DEMO_ROOT):
            print(f"wrote {stamp(root, n)}")


if __name__ == "__main__":
    main(sys.argv[1:])
