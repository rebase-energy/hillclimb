#!/usr/bin/env python3
"""Stamp the kissing-number starter problems: `problems/kissing-<d>/` for each d.

    uv run python problems/make_kissing.py 11

A kissing configuration in dimension d, in the integer formulation AlphaEvolve
used (Novikov et al. 2025, arXiv:2506.13131, Appendix B.11, Lemma 1): a set C
of N nonzero points in Z^d with

    min {||x - y|| : x != y in C}  >=  max {||x|| : x in C}.

Unit spheres centred at 2x/||x|| then kiss a unit sphere at the origin without
overlapping, so N is a lower bound on the kissing number. The score is N.
Integer coordinates make the check exact (squared norms compared as ints).

The generated dirs are committed, into both `problems/` and the wheel's bundled
copy `src/hillclimb/demo/`; rerun this after editing a template here
(`tests/test_kissing_starter.py` checks the committed dirs match).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_INSTANCES = (11,)

# maximum rows the verifier accepts, and the coordinate bound that keeps every
# squared norm / distance inside int64 (11 * (1e8)^2 * 4 << 2^63)
MAX_POINTS = 2000
MAX_COORD = 10**8

# d -> (best known kissing configuration size, who found it)
BEST_KNOWN: dict[int, tuple[int, str]] = {
    11: (593, "AlphaEvolve 2025"),  # previous record 592: Ganzhinov 2022, arXiv:2207.08266
}


def unit_points(d: int) -> list[list[int]]:
    """The 2d points ±e_i: valid by construction (norm 1, pairwise distance
    at least sqrt(2)) and far from the record."""
    pts = []
    for i in range(d):
        for sign in (1, -1):
            row = [0] * d
            row[i] = sign
            pts.append(row)
    return pts


def sample_submission(d: int) -> str:
    header = "id," + ",".join(f"c{i}" for i in range(d))
    rows = [f"{k}," + ",".join(str(c) for c in p) for k, p in enumerate(unit_points(d))]
    return "\n".join([header, *rows]) + "\n"


def problem_yaml(d: int) -> str:
    lines = [
        f"problem_id: kissing-{d}",
        "metric: points",
        "higher_is_better: true",
        "description: description.md",
        "# valid-by-construction floor: best/ always holds something",
        "baseline_files: {submission.csv: sample_submission.csv}",
    ]
    if d in BEST_KNOWN:
        value, who = BEST_KNOWN[d]
        lines += ["chart_baselines:", f'  "best known ({who})": {value}']
    lines += ["time_budget_s: 3600", "allow_internet_during_solution: false", ""]
    return "\n".join(lines)


def description_md(d: int) -> str:
    if d in BEST_KNOWN:
        value, who = BEST_KNOWN[d]
        best = (
            f"The best known configuration in dimension {d} has **{value} points** "
            f"({who}, arXiv:2506.13131 Appendix B.11; the previous record was 592, "
            f"Ganzhinov 2022)."
        )
    else:
        best = f"No best-known value is recorded here for d = {d}."
    cols = ",".join(f"c{i}" for i in range(d))
    return f"""# Kissing configuration in dimension {d}

Find **as many nonzero integer points as possible** in `Z^{d}` such that every
pairwise distance is at least the largest norm among the points:

    min over i != j of ||p_i - p_j||  >=  max over i of ||p_i||

Unit spheres centred at `2 p_i / ||p_i||` then all touch the unit sphere at the
origin without overlapping, so the number of points N is a lower bound on the
kissing number in dimension {d}. The score is N.

Constraints (verified programmatically, exactly, on squared integer norms):

- every coordinate is an integer with `|c| <= {MAX_COORD}`
- no point is the origin, no two points are equal
- `min_(i != j) ||p_i - p_j||^2 >= max_i ||p_i||^2`
- at most {MAX_POINTS} rows

{best}
The score is an integer count, so most edits are plateaus: a candidate that
keeps N is a tie, not a loss, and progress comes in unit steps. Good approaches:
start from lattice shells (the {2 * d * (d - 1)} points `±e_i ± e_j` of `D_{d}` already
satisfy the condition), scale a configuration up so there is integer room to
insert extra points, and then search for insertions/replacements that keep
the min-distance inequality, repairing the worst pair after each move. The
check is cheap (a Gram matrix), so many local moves fit in the time budget.
`numpy` and `scipy` are available.

## Submission format

Write `submission.csv` in the working directory with the header `id,{cols}`
and one row per point (`id` = 0..N-1, any N >= 1, integer coordinates), like
`sample_submission.csv` (a weak valid baseline: the {2 * d} points `±e_i`).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the configuration and prints `val_score: <N>` (0 if
invalid). Higher is better.

There is no train/test data; this is a pure construction problem. Keep total
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


def verify_py(d: int) -> str:
    return f'''"""Official scorer for the kissing-{d} problem.

Reads ./submission.csv (id,c0..c{d - 1}; N rows), validates the kissing
condition exactly on integer coordinates, and writes the score to
$HILLCLIMB_RESULT (the number of points N, 0.0 if invalid).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = {d}
MAX_POINTS = {MAX_POINTS}
MAX_COORD = {MAX_COORD}
COLS = [f"c{{i}}" for i in range(D)]


def emit(score: float) -> None:
    """hillclimb's verifier contract: the score is written to the result
    file, not scraped from stdout (coding-agent-authored code shares this stream)."""
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
    for col in ("id", *COLS):
        if col not in df.columns:
            fail(f"missing column {{col!r}}")
    n = len(df)
    if n < 1 or n > MAX_POINTS:
        fail(f"need between 1 and {{MAX_POINTS}} rows, got {{n}}")
    if sorted(df["id"].tolist()) != list(range(n)):
        fail(f"ids must be 0..{{n - 1}}")
    try:
        raw = df.sort_values("id")[COLS].to_numpy(dtype=float)
    except Exception as e:  # noqa: BLE001
        fail(f"non-numeric coordinates: {{e}}")
    if not np.all(np.isfinite(raw)):
        fail("non-finite coordinates")
    if np.any(raw != np.round(raw)):
        fail("coordinates must be integers")
    if np.any(np.abs(raw) > MAX_COORD):
        fail(f"coordinates must satisfy |c| <= {{MAX_COORD}}")
    pts = raw.astype(np.int64)
    norms = np.einsum("ij,ij->i", pts, pts)
    if np.any(norms == 0):
        fail("the origin is not allowed")
    if len(np.unique(pts, axis=0)) != n:
        fail("duplicate points")
    max_norm = int(norms.max())
    if n == 1:
        min_dist = max_norm
    else:
        gram = pts @ pts.T
        dist = norms[:, None] + norms[None, :] - 2 * gram
        np.fill_diagonal(dist, dist.max() + 1)
        min_dist = int(dist.min())
    if min_dist < max_norm:
        fail(f"kissing condition violated: min pairwise |dist|^2 {{min_dist}} < max |norm|^2 {{max_norm}}")
    print(f"valid configuration; {{n}} points, min |dist|^2 {{min_dist}} >= max |norm|^2 {{max_norm}}")
    emit(float(n))
    print(f"val_score: {{n}}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 — the verifier never crashes
        fail(f"unexpected error: {{e}}")
'''


def interface_py(d: int) -> str:
    cols = "\n".join(f'        "c{i}": spaces.Int(low=-MAX_COORD, high=MAX_COORD),' for i in range(d))
    return f'''"""Machine-checked output format (hillclimb spaces). Format only — the
distance condition, the no-origin and no-duplicate rules stay with the verifier.
"""

from hillclimb import spaces

MAX_COORD = {MAX_COORD}

output = spaces.Table(
    "submission.csv",
    columns={{
        "id": spaces.Int(low=0, unique=True),
{cols}
    }},
    n_rows=(1, {MAX_POINTS}),
)

if __name__ == "__main__":
    raise SystemExit(spaces.main(output))
'''


def files_for(d: int) -> dict[str, str]:
    return {
        "problem.yaml": problem_yaml(d),
        "description.md": description_md(d),
        "verifier.sh": VERIFIER_SH,
        "verify.py": verify_py(d),
        "interface.py": interface_py(d),
        "sample_submission.csv": sample_submission(d),
    }


def stamp(root: Path, d: int) -> Path:
    """Write kissing-<d>/ under `root`; returns the dir."""
    if d < 2:
        raise ValueError("a kissing instance needs dimension at least 2")
    problem_dir = root / f"kissing-{d}"
    problem_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files_for(d).items():
        (problem_dir / name).write_text(text)
    (problem_dir / "verifier.sh").chmod(0o755)
    return problem_dir


def main(argv: list[str]) -> None:
    instances = [int(a) for a in argv] or list(DEFAULT_INSTANCES)
    for d in instances:
        for root in (ROOT,):
            print(f"wrote {stamp(root, d)}")


if __name__ == "__main__":
    main(sys.argv[1:])
