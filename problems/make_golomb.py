#!/usr/bin/env python3
"""Stamp the Golomb-ruler starter problems: `problems/golomb-<m>/` for each m.

    uv run python problems/make_golomb.py 20 27

A Golomb ruler with m marks is a set of m distinct non-negative integers
starting at 0 whose C(m,2) pairwise differences are all distinct; its length
is the largest mark, and shorter is better. Every level is the same problem
with only m changed. The generated dirs are committed, into both `problems/`
and the wheel's bundled copy `src/hillclimb/demo/`; rerun this after editing
a template here (`tests/test_golomb_starter.py` checks the committed dirs match).

Best-known values are the optimal lengths from the table of optimal Golomb
rulers (proven optimal through m = 28), https://en.wikipedia.org/wiki/Golomb_ruler
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO_ROOT = ROOT.parent / "src" / "hillclimb" / "demo"
DEFAULT_INSTANCES = (20, 27)

# the score of an invalid submission; valid marks must stay below it
PENALTY = 10**6

# m -> (optimal length, who found it)
BEST_KNOWN: dict[int, tuple[int, str]] = {
    10: (55, "Mixon 1972"),
    11: (72, "Mixon 1972"),
    12: (85, "Robinson 1979"),
    13: (106, "Robinson 1981"),
    14: (127, "Shearer 1985"),
    15: (151, "Shearer 1985"),
    16: (177, "Shearer 1986"),
    17: (199, "Sibert 1993"),
    18: (216, "Sibert 1993"),
    19: (246, "Dollas, Rankin & McCracken 1994"),
    20: (283, "Garry, Vanderschel et al. 1997"),
    21: (333, "Garry, Vanderschel et al. 1998"),
    22: (356, "Garry, Vanderschel et al. 1999"),
    23: (372, "Garry, Vanderschel et al. 1999"),
    24: (425, "distributed.net 2004"),
    25: (480, "distributed.net 2008"),
    26: (492, "distributed.net 2009"),
    27: (553, "distributed.net 2014"),
    28: (585, "distributed.net 2022"),
}


def greedy_marks(m: int) -> list[int]:
    """The greedy ruler (the Mian–Chowla sequence shifted to start at 0):
    each new mark is the smallest integer whose differences to the existing
    marks are all new. Valid by construction and far from optimal."""
    marks = [0]
    diffs: set[int] = set()
    while len(marks) < m:
        x = marks[-1] + 1
        while True:
            new = {x - a for a in marks}
            if len(new) == len(marks) and not (new & diffs):
                break
            x += 1
        diffs |= new
        marks.append(x)
    return marks


def sample_submission(m: int) -> str:
    return "id,mark\n" + "".join(f"{i},{mark}\n" for i, mark in enumerate(greedy_marks(m)))


def problem_yaml(m: int) -> str:
    lines = [
        f"problem_id: golomb-{m}",
        "metric: length",
        "higher_is_better: false",
        "description: description.md",
        "# valid-by-construction floor: best/ always holds something",
        "baseline_files: {submission.csv: sample_submission.csv}",
    ]
    if m in BEST_KNOWN:
        value, who = BEST_KNOWN[m]
        lines += ["chart_baselines:", f'  "optimal ({who})": {value}']
    lines += ["time_budget_s: 900", "allow_internet_during_solution: false", ""]
    return "\n".join(lines)


def description_md(m: int) -> str:
    pairs = math.comb(m, 2)
    if m in BEST_KNOWN:
        value, who = BEST_KNOWN[m]
        best = (
            f"The optimal length for m = {m} is **{value}** ({who}; proven optimal, "
            f"table of optimal Golomb rulers on Wikipedia), so no ruler can score below it."
        )
    else:
        best = f"No best-known value is recorded here for m = {m}."
    return f"""# Golomb ruler with {m} marks

Find **{m} distinct non-negative integers** (the marks of a ruler), the first
of them 0, such that all C({m},2) = {pairs} **pairwise differences are distinct**,
and make the ruler **as short as possible**: the score is its length, the
largest mark.

Constraints (verified programmatically):

- exactly {m} rows, integer marks with `0 <= mark < {PENALTY}`
- mark 0 is present, no mark repeats
- no two pairs of marks have the same difference

{best}
The score is an integer, so most edits are plateaus: a candidate that keeps
the length is a tie, not a loss, and progress comes in whole units. Good
approaches: fix a target length L and search for a placement of the inner
marks (constraint propagation / backtracking over the difference table,
simulated annealing or tabu search on mark positions with the number of
repeated differences as the cost), start from affine or projective-plane
constructions (Singer, Bose–Chowla) that give near-optimal rulers directly
and then shrink, and exploit the mirror symmetry (`L - mark` is a ruler too).
`numpy` and `scipy` are available.

## Submission format

Write `submission.csv` in the working directory with the header `id,mark` and
{m} rows (`id` = 0..{m - 1}, one integer mark each, any order), like
`sample_submission.csv` (a weak valid baseline: the greedy ruler).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the ruler and prints `val_score: <length>` ({PENALTY} if
invalid). Lower is better.

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


def verify_py(m: int) -> str:
    return f'''"""Official scorer for the golomb-{m} problem.

Reads ./submission.csv (id,mark; {m} rows), validates the ruler, and writes
the score to $HILLCLIMB_RESULT (the length = largest mark, {PENALTY} if invalid).
Generated by problems/make_golomb.py — edit the template there.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

M = {m}
PENALTY = {PENALTY}


def emit(score: float) -> None:
    """hillclimb's verifier contract: the score is written to the result
    file, not scraped from stdout (coding-agent-authored code shares this stream)."""
    path = Path(os.environ.get("HILLCLIMB_RESULT", "eval_result.json"))
    path.write_text(json.dumps({{"split": os.environ.get("HILLCLIMB_SPLIT", "validation"),
                                "score": score}}))


def fail(reason: str) -> None:
    print(f"INVALID: {{reason}}")
    emit(float(PENALTY))
    print(f"val_score: {{PENALTY}}")
    sys.exit(0)


def main() -> None:
    try:
        df = pd.read_csv("submission.csv")
    except Exception as e:  # noqa: BLE001
        fail(f"cannot read submission.csv: {{e}}")
    for col in ("id", "mark"):
        if col not in df.columns:
            fail(f"missing column {{col!r}}")
    if len(df) != M or sorted(df["id"].tolist()) != list(range(M)):
        fail(f"need exactly {{M}} rows with id 0..{{M - 1}}")
    try:
        raw = df["mark"].to_numpy(dtype=float)
    except Exception as e:  # noqa: BLE001
        fail(f"non-numeric marks: {{e}}")
    if not np.all(np.isfinite(raw)):
        fail("non-finite marks")
    if np.any(raw != np.round(raw)):
        fail("marks must be integers")
    if np.any(raw < 0) or np.any(raw >= PENALTY):
        fail(f"marks must satisfy 0 <= mark < {{PENALTY}}")
    marks = np.sort(raw.astype(np.int64))
    if marks[0] != 0:
        fail("mark 0 must be present")
    if len(np.unique(marks)) != M:
        fail("duplicate marks")
    i, j = np.triu_indices(M, k=1)
    diffs = marks[j] - marks[i]
    if len(np.unique(diffs)) != len(diffs):
        fail("two pairs of marks have the same difference")
    length = int(marks[-1])
    print(f"valid Golomb ruler with {{M}} marks; length = {{length}}")
    emit(float(length))
    print(f"val_score: {{length}}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 — the verifier never crashes
        fail(f"unexpected error: {{e}}")
'''


def interface_py(m: int) -> str:
    return f'''"""Machine-checked output format (hillclimb spaces). Format only — the
distinct-differences rule and the mark at 0 stay with the verifier.
Generated by problems/make_golomb.py — edit the template there.
"""

from hillclimb import spaces

output = spaces.Table(
    "submission.csv",
    columns={{
        "id": spaces.Int(values=range({m}), unique=True),
        "mark": spaces.Int(low=0, high={PENALTY - 1}),
    }},
    n_rows={m},
)

if __name__ == "__main__":
    raise SystemExit(spaces.main(output))
'''


def files_for(m: int) -> dict[str, str]:
    return {
        "problem.yaml": problem_yaml(m),
        "description.md": description_md(m),
        "verifier.sh": VERIFIER_SH,
        "verify.py": verify_py(m),
        "interface.py": interface_py(m),
        "sample_submission.csv": sample_submission(m),
    }


def stamp(root: Path, m: int) -> Path:
    """Write golomb-<m>/ under `root`; returns the dir."""
    if m < 2:
        raise ValueError("a Golomb ruler needs at least 2 marks")
    problem_dir = root / f"golomb-{m}"
    problem_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files_for(m).items():
        (problem_dir / name).write_text(text)
    (problem_dir / "verifier.sh").chmod(0o755)
    return problem_dir


def main(argv: list[str]) -> None:
    instances = [int(a) for a in argv] or list(DEFAULT_INSTANCES)
    for m in instances:
        for root in (ROOT, DEMO_ROOT):
            print(f"wrote {stamp(root, m)}")


if __name__ == "__main__":
    main(sys.argv[1:])
