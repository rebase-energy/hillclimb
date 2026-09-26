#!/usr/bin/env python3
"""Stamp the LABS ladder: `problems/labs-<N>/` for each N.

    uv run python problems/make_labs.py 40 60

Every level is the same problem — a binary sequence of length N whose
aperiodic autocorrelation sidelobe energy is as small as possible — with only
N changed, so a comparison across levels measures difficulty and nothing
else. The generated dirs are committed to BOTH `problems/` and
`src/hillclimb/demo/` (the wheel's bundled copy); rerun this after editing a
template here (`tests/test_labs_starter.py` checks that both committed
copies match).

The reference lines on `hillclimb chart` are the PROVEN optimal energies
from Packebusch & Mertens, "Low autocorrelation binary sequences", J. Phys.
A: Math. Theor. 49 (2016) 165001 (arXiv:1512.02475), who computed every
optimal sequence for N <= 66 (their Table 1; the same numbers are OEIS
A102780, offset 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO = ROOT.parent / "src" / "hillclimb" / "demo"
DEFAULT_LEVELS = (40, 60)
PENALTY = 100000.0

OPTIMAL_SOURCE = "Packebusch & Mertens 2016"
# E(N) for N = 1..66, proven optimal (Packebusch & Mertens 2016, OEIS A102780)
_OPTIMAL_ENERGY = (
    0, 1, 1, 2, 2, 7, 3, 8, 12, 13, 5, 10, 6, 19, 15, 24, 32, 25, 29, 26, 26, 39,
    47, 36, 36, 45, 37, 50, 62, 59, 67, 64, 64, 65, 73, 82, 86, 87, 99, 108, 108,
    101, 109, 122, 118, 131, 135, 140, 136, 153, 153, 166, 170, 175, 171, 192,
    188, 197, 205, 218, 226, 235, 207, 208, 240, 257,
)
# n -> (optimal energy, who proved it)
BEST_KNOWN: dict[int, tuple[int, str]] = {
    n: (energy, OPTIMAL_SOURCE) for n, energy in enumerate(_OPTIMAL_ENERGY, start=1)
}


def growing_runs(n: int) -> list[int]:
    """Runs of +1/-1 of length 1, 2, 3, ... cut at N: deterministic, valid by
    construction, and weak — long runs give large low-lag autocorrelations
    (a few times the random-sequence expectation N(N-1)/2)."""
    spins: list[int] = []
    sign, length = 1, 1
    while len(spins) < n:
        spins += [sign] * length
        sign, length = -sign, length + 1
    return spins[:n]


def energy(spins: list[int]) -> int:
    n = len(spins)
    return sum(sum(spins[i] * spins[i + k] for i in range(n - k)) ** 2 for k in range(1, n))


def sample_submission(n: int) -> str:
    return "id,spin\n" + "".join(f"{i},{s}\n" for i, s in enumerate(growing_runs(n)))


def problem_yaml(n: int) -> str:
    lines = [
        f"problem_id: labs-{n}",
        "metric: autocorrelation-energy",
        "higher_is_better: false",
        "description: description.md",
        "# valid-by-construction floor: best/ always holds something",
        "baseline_files: {submission.csv: sample_submission.csv}",
    ]
    if n in BEST_KNOWN:
        value, who = BEST_KNOWN[n]
        lines += ["chart_baselines:", f'  "optimal ({who})": {value}']
    lines += ["time_budget_s: 900", "allow_network: false", ""]
    return "\n".join(lines)


def description_md(n: int) -> str:
    random_energy = n * (n - 1) // 2
    if n in BEST_KNOWN:
        value, who = BEST_KNOWN[n]
        best = (
            f"The optimal energy for N = {n} is **{value}** (merit factor "
            f"N^2 / (2E) = {n * n / (2 * value):.3f}), proven by exhaustive search ({who})."
        )
    else:
        best = f"No optimal or best-known energy is recorded here for N = {n}."
    return f"""# Low-autocorrelation binary sequence (LABS), length {n}

Find a binary sequence `s` of **length {n}** with entries in `{{+1, -1}}` that
**minimizes the autocorrelation sidelobe energy**

```
E(s) = sum_{{k=1}}^{{{n - 1}}} C_k(s)^2,   where   C_k(s) = sum_{{i=0}}^{{{n - 1}-k}} s_i * s_{{i+k}}
```

This is a famously rugged discrete optimization landscape (used in radar and
statistical physics as the "Bernasconi model"). A random sequence has expected
energy **{random_energy}**. {best}
Plain hill-climbing stalls quickly — good approaches use tabu search, memetic /
population methods, or self-avoiding walks over the bit-flip neighborhood,
with incremental O(N) energy updates per flip and many restarts (for odd N,
skew-symmetric sequences halve the search space and often contain the
optimum). `numpy` is available.

## Submission format

Write `submission.csv` in the working directory with the header `id,spin` and
{n} rows (`id` = 0..{n - 1}, `spin` = +1 or -1), like `sample_submission.csv` (a
weak valid baseline: runs of length 1, 2, 3, ... with alternating sign).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the sequence and prints `val_score: <energy>` (or a penalty
of {PENALTY} if invalid). **Lower is better.**

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
exec "$HILLCLIMB_PYTHON" problem/verify.py
"""


def verify_py(n: int) -> str:
    return f'''"""Official scorer for the labs-{n} problem.

Reads ./submission.csv (id,spin; {n} rows of +/-1), validates, and writes the
score to $HILLCLIMB_RESULT (autocorrelation sidelobe energy; a {PENALTY}
penalty if invalid — lower is better).
Generated by problems/make_labs.py — edit the template there.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N = {n}
PENALTY = {PENALTY}


def emit(score: float) -> None:
    """hillclimb's verifier contract: the score is written to the result
    file, not scraped from stdout (agent-authored code shares this stream)."""
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
    for col in ("id", "spin"):
        if col not in df.columns:
            fail(f"missing column {{col!r}}")
    if len(df) != N or sorted(df["id"].tolist()) != list(range(N)):
        fail(f"need exactly {{N}} rows with id 0..{{N - 1}}")
    spins = pd.to_numeric(df.sort_values("id")["spin"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(spins)) or not np.all(np.isin(spins, (-1.0, 1.0))):
        fail("spin values must be +1 or -1")
    s = spins.astype(int)
    sidelobes = [int(np.dot(s[: N - k], s[k:])) for k in range(1, N)]
    energy = sum(c * c for c in sidelobes)
    worst = sorted(range(1, N), key=lambda k: -abs(sidelobes[k - 1]))[:5]
    print("largest sidelobes (lag: C_k): " + ", ".join(f"{{k}}: {{sidelobes[k - 1]}}" for k in worst))
    print(f"valid sequence; energy = {{energy}}")
    emit(float(energy))
    print(f"val_score: {{energy}}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 — an invalid submission is the penalty, never a crash
        fail(f"unexpected error: {{e}}")
'''


def files_for(n: int) -> dict[str, str]:
    return {
        "problem.yaml": problem_yaml(n),
        "description.md": description_md(n),
        "verifier.sh": VERIFIER_SH,
        "verify.py": verify_py(n),
        "sample_submission.csv": sample_submission(n),
    }


def stamp(root: Path, n: int) -> Path:
    """Write labs-<n>/ under `root`; returns the dir."""
    if n < 2:
        raise ValueError("a LABS instance needs at least 2 spins")
    problem_dir = root / f"labs-{n}"
    problem_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files_for(n).items():
        (problem_dir / name).write_text(text)
    (problem_dir / "verifier.sh").chmod(0o755)
    return problem_dir


def main(argv: list[str]) -> None:
    levels = [int(a) for a in argv] or list(DEFAULT_LEVELS)
    for n in levels:
        for root in (ROOT, DEMO):
            print(f"wrote {stamp(root, n)}")


if __name__ == "__main__":
    main(sys.argv[1:])
