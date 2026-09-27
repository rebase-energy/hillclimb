#!/usr/bin/env python3
"""Stamp the Tammes starter instances: `problems/tammes-<N>/` for each N.

    uv run python problems/make_tammes.py 30 50

Every instance is the same problem — place N points on the unit sphere so
the smallest pairwise angular separation is as large as possible — with
only N changed. The generated dirs are committed in BOTH `problems/` and
`src/hillclimb/demo/` (the wheel's bundled copy); rerun this after editing
a template here (`tests/test_tammes_starter.py` checks both copies match).

Best-known values are the reference lines on `hillclimb chart`. The n = 30
and n = 50 values are the minimal angles of the putatively optimal codes
`pack.3.30.txt` / `pack.3.50.txt` in N. J. A. Sloane's spherical-codes
library (https://neilsloane.com/packings/, Hardin, Sloane & Smith 1994),
recomputed from the published coordinates; n = 4, 6, 12 are the regular
polyhedra proved optimal by L. Fejes Tóth (1943).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO_ROOT = ROOT.parent / "src" / "hillclimb" / "demo"
DEFAULT_INSTANCES = (30, 50)
RING_SIZE = 10  # points per latitude ring in the sample submission

# n -> (best known minimal angle in degrees, who found it)
BEST_KNOWN: dict[int, tuple[float, str]] = {
    4: (math.degrees(math.acos(-1 / 3)), "regular tetrahedron, Fejes Tóth 1943"),
    6: (90.0, "regular octahedron, Fejes Tóth 1943"),
    12: (math.degrees(math.atan(2)), "regular icosahedron, Fejes Tóth 1943"),
    30: (38.597116, "Hardin, Sloane & Smith 1994"),
    50: (29.752956, "Hardin, Sloane & Smith 1994"),
}


def ring_points(n: int) -> list[tuple[int, float, float, float]]:
    """N points on ceil(N/10) latitude rings, up to 10 per ring, rings at
    polar angles pi*(r+1)/(R+1) (never a pole) and azimuths 2*pi*j/m.
    Valid by construction, deterministic and far from optimal: the ring
    nearest a pole packs its points into a tiny circle."""
    rings = math.ceil(n / RING_SIZE)
    counts = [n // rings + (1 if r < n % rings else 0) for r in range(rings)]
    rows: list[tuple[int, float, float, float]] = []
    for r, m in enumerate(counts):
        theta = math.pi * (r + 1) / (rings + 1)
        for j in range(m):
            phi = 2 * math.pi * j / m
            rows.append((len(rows),
                         round(math.sin(theta) * math.cos(phi), 6),
                         round(math.sin(theta) * math.sin(phi), 6),
                         round(math.cos(theta), 6)))
    return rows


def sample_submission(n: int) -> str:
    return "id,x,y,z\n" + "".join(f"{i},{x:.6f},{y:.6f},{z:.6f}\n" for i, x, y, z in ring_points(n))


def problem_yaml(n: int) -> str:
    lines = [
        f"problem_id: tammes-{n}",
        "metric: min-angle-deg",
        "higher_is_better: true",
        "description: description.md",
        "# valid-by-construction floor: best/ always holds something",
        "baseline_files: {submission.csv: sample_submission.csv}",
    ]
    if n in BEST_KNOWN:
        value, who = BEST_KNOWN[n]
        lines += ["chart_baselines:", f'  "best known ({who})": {value:.6f}']
    lines += ["time_budget_s: 900", "allow_network: false", ""]
    return "\n".join(lines)


def description_md(n: int) -> str:
    pairs = math.comb(n, 2)
    if n in BEST_KNOWN:
        value, who = BEST_KNOWN[n]
        best = f"The best known value for n = {n} is **{value:.4f} degrees** ({who})."
    else:
        best = f"No best-known value is recorded here for n = {n}."
    return f"""# Tammes problem: {n} points on the sphere

Place **exactly {n} points on the unit sphere** so that the **smallest angular
separation** between any two of them is **as large as possible** (a max-min
objective over all C({n},2) = {pairs} pairs — the "spherical code" or
"repelling dictators" problem).

Constraints (verified programmatically):

- exactly {n} rows with finite coordinates
- every row is a non-zero vector; the verifier **normalizes each row to unit
  length**, so you may submit any non-zero direction

This is a classic hard continuous optimization problem. {best}
Latitude rings and lattices are weak. Good approaches: many random or
Fibonacci-spiral starts, a steep repulsion surrogate (maximize the softmin of
the pairwise angles, or minimize `sum 1/d^p` with a large `p`) followed by an
exact polish (SLSQP: maximize `t` subject to every pairwise angle `>= t`),
and restarts that keep the best. `numpy` and `scipy` are available.

## Submission format

Write `submission.csv` in the working directory with the header `id,x,y,z` and
{n} rows (`id` = 0..{n - 1}), like `sample_submission.csv` (a weak valid
baseline: points on a few latitude rings).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier normalizes the rows, computes the minimum pairwise angle in degrees
and prints `val_score: <min angle>` (0.0 if invalid). Higher is better.

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
    return f'''"""Official scorer for the tammes-{n} problem.

Reads ./submission.csv (id,x,y,z; {n} rows), normalizes each row to the unit
sphere, and writes the score to $HILLCLIMB_RESULT (minimum pairwise angle in
degrees, 0.0 if invalid).
Generated by problems/make_tammes.py — edit the template there.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N = {n}
TOL = 1e-12


def emit(score: float) -> None:
    """hillclimb's verifier contract: the score is written to the result
    file, not scraped from stdout (agent-authored code shares this stream)."""
    path = Path(os.environ.get("HILLCLIMB_RESULT", "eval_result.json"))
    path.write_text(json.dumps({{"split": os.environ.get("HILLCLIMB_SPLIT", "validation"),
                                "score": score}}))


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
    for col in ("id", "x", "y", "z"):
        if col not in df.columns:
            fail(f"missing column {{col!r}}")
    ids = pd.to_numeric(df["id"], errors="coerce").to_numpy(float)
    if len(df) != N or not np.all(np.isfinite(ids)) or sorted(ids.tolist()) != list(range(N)):
        fail(f"need exactly {{N}} rows with id 0..{{N - 1}}")
    pts = df[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)[np.argsort(ids)]
    if not np.all(np.isfinite(pts)):
        fail("non-finite or non-numeric coordinates")
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms < TOL):
        fail("zero-length vector(s): every row must be a direction")
    pts = pts / norms[:, None]
    gram = np.clip(pts @ pts.T, -1.0, 1.0)
    np.fill_diagonal(gram, -1.0)
    min_angle = float(np.degrees(np.arccos(gram.max())))
    print(f"valid code; min pairwise angle = {{min_angle:.8f}} deg")
    emit(min_angle)
    print(f"val_score: {{min_angle:.8f}}")


if __name__ == "__main__":
    main()
'''


def interface_py(n: int) -> str:
    return f'''"""Machine-checked output format (hillclimb spaces). Format only — each row
is normalized to the unit sphere and the minimum angle is computed by the verifier.
Generated by problems/make_tammes.py — edit the template there.
"""

from hillclimb import spaces

output = spaces.Table(
    "submission.csv",
    columns={{
        "id": spaces.Int(values=range({n}), unique=True),
        "x": spaces.Float(),
        "y": spaces.Float(),
        "z": spaces.Float(),
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
        "interface.py": interface_py(n),
        "sample_submission.csv": sample_submission(n),
    }


def stamp(root: Path, n: int) -> Path:
    """Write tammes-<n>/ under `root`; returns the dir."""
    if n < 2:
        raise ValueError("a Tammes instance needs at least 2 points")
    problem_dir = root / f"tammes-{n}"
    problem_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files_for(n).items():
        (problem_dir / name).write_text(text)
    (problem_dir / "verifier.sh").chmod(0o755)
    return problem_dir


def main(argv: list[str]) -> None:
    instances = [int(a) for a in argv] or list(DEFAULT_INSTANCES)
    for n in instances:
        for root in (ROOT, DEMO_ROOT):
            print(f"wrote {stamp(root, n)}")


if __name__ == "__main__":
    main(sys.argv[1:])
