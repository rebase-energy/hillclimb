#!/usr/bin/env python3
"""Stamp the Thomson starter instances: `problems/thomson-<N>/` for each N.

    uv run python problems/make_thomson.py 50 100

Every instance is the same problem — place N unit charges on the unit sphere
so their Coulomb energy sum_{i<j} 1/|p_i - p_j| is as small as possible —
with only N changed. The generated dirs are committed in BOTH `problems/`
and `src/hillclimb/demo/` (the wheel's bundled copy); rerun this after
editing a template here (`tests/test_thomson_starter.py` checks both copies
match).

Best-known values are the reference lines on `hillclimb chart`; source: the
table of smallest known energies on the Wikipedia "Thomson problem" page
(https://en.wikipedia.org/wiki/Thomson_problem, units e = 1, k_e = 1), which
collects the putative global minima of Wales et al. / the Cambridge Cluster
Database. n = 4, 6, 12 are the regular polyhedra (proved optimal).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO_ROOT = ROOT.parent / "src" / "hillclimb" / "demo"
DEFAULT_INSTANCES = (50, 100)
RING_SIZE = 10  # points per latitude ring in the sample submission

# n -> (smallest known Coulomb energy, who found it)
BEST_KNOWN: dict[int, tuple[float, str]] = {
    4: (3.674234614, "regular tetrahedron, Wikipedia Thomson table"),
    6: (9.985281374, "regular octahedron, Wikipedia Thomson table"),
    12: (49.165253058, "regular icosahedron, Wikipedia Thomson table"),
    50: (1055.182314726, "Wikipedia Thomson table"),
    100: (4448.350634331, "Wikipedia Thomson table"),
}


def ring_points(n: int) -> list[tuple[int, float, float, float]]:
    """N points on ceil(N/10) latitude rings, up to 10 per ring, rings at
    polar angles pi*(r+1)/(R+1) (never a pole) and azimuths 2*pi*j/m.
    Valid by construction, deterministic and far from optimal: the ring
    nearest a pole crowds its charges into a tiny circle."""
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


def penalty(n: int) -> float:
    """Score of an invalid submission: far above any valid layout (the
    energy of N charges scales like N^2 / 2)."""
    return float(10 * n * n)


def problem_yaml(n: int) -> str:
    lines = [
        f"problem_id: thomson-{n}",
        "metric: coulomb-energy",
        "higher_is_better: false",
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
    pairs = math.comb(n, 2)
    if n in BEST_KNOWN:
        value, who = BEST_KNOWN[n]
        best = f"The best known energy for n = {n} is **{value:.6f}** ({who})."
    else:
        best = f"No best-known value is recorded here for n = {n}."
    return f"""# Thomson problem: {n} charges on the sphere

Place **exactly {n} unit point charges on the unit sphere** so that their
**Coulomb energy** `U = sum over the C({n},2) = {pairs} pairs of 1 / |p_i - p_j|`
(units `e = 1`, `k_e = 1`) is **as small as possible**.

Constraints (verified programmatically):

- exactly {n} rows with finite coordinates
- every row is a non-zero vector; the verifier **normalizes each row to unit
  length**, so you may submit any non-zero direction
- no two charges may coincide (that makes the energy infinite)

This is a classic hard continuous optimization problem. {best}
The landscape has many local minima that differ only in the 3rd–4th decimal,
so a precise local optimizer matters as much as the start. Good approaches: a
Fibonacci-spiral or random start, then gradient descent / L-BFGS with the
gradient projected onto the tangent plane and the points renormalized after
every step, basin hopping or simulated annealing over the local minima, and
restarts that keep the best. `numpy` and `scipy` are available.

## Submission format

Write `submission.csv` in the working directory with the header `id,x,y,z` and
{n} rows (`id` = 0..{n - 1}), like `sample_submission.csv` (a weak valid
baseline: points on a few latitude rings).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier normalizes the rows, computes the Coulomb energy and prints
`val_score: <energy>` ({penalty(n):.1f} if invalid). Lower is better.

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
    return f'''"""Official scorer for the thomson-{n} problem.

Reads ./submission.csv (id,x,y,z; {n} rows), normalizes each row to the unit
sphere, and writes the score to $HILLCLIMB_RESULT (Coulomb energy
sum_{{i<j}} 1/|p_i - p_j|; {penalty(n):.1f} penalty if invalid — lower is better).
Generated by problems/make_thomson.py — edit the template there.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N = {n}
PENALTY = {penalty(n):.1f}
TOL = 1e-12


def emit(score: float) -> None:
    """hillclimb's verifier contract: the score is written to the result
    file, not scraped from stdout (coding-agent-authored code shares this stream)."""
    path = Path(os.environ.get("HILLCLIMB_RESULT", "eval_result.json"))
    path.write_text(json.dumps({{"split": os.environ.get("HILLCLIMB_SPLIT", "validation"),
                                "score": score}}))


def fail(reason: str) -> None:
    print(f"INVALID: {{reason}}")
    emit(PENALTY)
    print(f"val_score: {{PENALTY}}")
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
    i, j = np.triu_indices(N, 1)
    dist = np.linalg.norm(pts[i] - pts[j], axis=1)
    if np.any(dist < TOL):
        fail("coincident charges (infinite energy)")
    energy = float(np.sum(1.0 / dist))
    print(f"valid layout; Coulomb energy = {{energy:.9f}}")
    emit(energy)
    print(f"val_score: {{energy:.9f}}")


if __name__ == "__main__":
    main()
'''


def interface_py(n: int) -> str:
    return f'''"""Machine-checked output format (hillclimb spaces). Format only — each row
is normalized to the unit sphere and the energy is computed by the verifier.
Generated by problems/make_thomson.py — edit the template there.
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
    """Write thomson-<n>/ under `root`; returns the dir."""
    if n < 2:
        raise ValueError("a Thomson instance needs at least 2 charges")
    problem_dir = root / f"thomson-{n}"
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
