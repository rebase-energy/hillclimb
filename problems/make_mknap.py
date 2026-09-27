#!/usr/bin/env python3
"""Stamp the multidimensional-knapsack ladder: `problems/mknap-<n>-<m>/`.

    uv run python problems/make_mknap.py

Every level is the same problem — choose a subset of n items, each with a
value and m weights, of the largest total value that fits every one of the
m capacities — on a fixed instance from OR-Library's Chu & Beasley set
(P. C. Chu and J. E. Beasley, "A genetic algorithm for the multidimensional
knapsack problem", Journal of Heuristics 4 (1998) 63-86). The plain 0/1
knapsack is solved exactly by dynamic programming in milliseconds; with
several capacities at once it is NP-hard in a way that bites at these
sizes, and the literature's best feasible values are the reference lines.

The generated dirs are committed to BOTH `problems/` and `src/hillclimb/demo/`
(the wheel's bundled copy); rerun this after editing a template here
(`tests/test_mknap_starter.py` checks that both committed copies match).
The OR-Library files are fetched once into ~/.cache/hillclimb-orlib/.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO = ROOT.parent / "src" / "hillclimb" / "demo"
CACHE = Path.home() / ".cache" / "hillclimb-orlib"
ORLIB = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files"
PENALTY = 0.0   # an invalid submission scores nothing; higher is better

# level -> (OR-Library file, instance index in it, Chu & Beasley's name, best feasible value, LP bound)
# The best feasible values and LP relaxations are OR-Library's mkcbres file.
LEVELS: dict[tuple[int, int], tuple[str, int, str, int, float]] = {
    (100, 5): ("mknapcb1.txt", 0, "5.100-00", 24381, 24585.902722),
    (250, 10): ("mknapcb5.txt", 0, "10.250-00", 59187, 59489.339237),
}
BEST_KNOWN_SOURCE = "Chu & Beasley 1998"


def fetch(name: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        print(f"fetching {ORLIB}/{name} -> {path}")
        with urllib.request.urlopen(f"{ORLIB}/{name}", timeout=60) as resp:
            path.write_bytes(resp.read())
    return path.read_text()


def parse_orlib(text: str, index: int) -> tuple[list[int], list[list[int]], list[int]]:
    """The mknap1 format: K, then per problem `n m optimum`, the n profits,
    m rows of n weights, the m capacities — all whitespace-separated."""
    tokens = text.split()
    pos = 0
    count = int(tokens[pos]); pos += 1
    if not 0 <= index < count:
        raise ValueError(f"instance {index} out of range, file holds {count}")
    for k in range(count):
        n, m, _opt = int(tokens[pos]), int(tokens[pos + 1]), tokens[pos + 2]
        pos += 3
        values = [int(t) for t in tokens[pos:pos + n]]; pos += n
        weights = []
        for _ in range(m):
            weights.append([int(t) for t in tokens[pos:pos + n]]); pos += n
        capacities = [int(t) for t in tokens[pos:pos + m]]; pos += m
        if k == index:
            return values, weights, capacities
    raise AssertionError("unreachable")


def items_csv(values: list[int], weights: list[list[int]]) -> str:
    m = len(weights)
    header = "id,value," + ",".join(f"w{i + 1}" for i in range(m))
    rows = [f"{j},{values[j]}," + ",".join(str(weights[i][j]) for i in range(m)) for j in range(len(values))]
    return header + "\n" + "\n".join(rows) + "\n"


def capacities_csv(capacities: list[int]) -> str:
    return "constraint,capacity\n" + "".join(f"w{i + 1},{c}\n" for i, c in enumerate(capacities))


def sample_submission(n: int) -> str:
    """The empty knapsack: valid by construction, worth nothing."""
    return "id,take\n" + "".join(f"{i},0\n" for i in range(n))


def problem_yaml(n: int, m: int) -> str:
    _file, _index, name, best, _lp = LEVELS[(n, m)]
    return "\n".join([
        f"problem_id: mknap-{n}-{m}",
        "metric: total-value",
        "higher_is_better: true",
        "description: description.md",
        "data_dir: data",
        "# valid-by-construction floor: best/ always holds something",
        "baseline_files: {submission.csv: sample_submission.csv}",
        "chart_baselines:",
        f'  "best known ({BEST_KNOWN_SOURCE})": {best}',
        "time_budget_s: 900",
        "allow_network: false",
        "",
    ])


def description_md(n: int, m: int) -> str:
    file, index, name, best, lp = LEVELS[(n, m)]
    return f"""# Multidimensional knapsack: {n} items, {m} constraints

`data/items.csv` lists **{n} items** (`id,value,w1..w{m}`): each has an integer
value and {m} integer weights, one per constraint. `data/capacities.csv` gives
the {m} capacities (`constraint,capacity`). **Choose a subset of the items of
the largest total value** such that, for every constraint, the sum of the
chosen items' weights stays within its capacity.

This is instance **{name}** of Chu & Beasley's benchmark set (OR-Library file
`{file}`, tightness ratio 0.25: each capacity is a quarter of the total
weight on that constraint), fixed, so the score is fully reproducible. The
single-constraint knapsack falls to dynamic programming; with {m} capacities
at once the problem is NP-hard in a way that matters at this size. The LP
relaxation is **{lp:.1f}**, an upper bound no subset reaches; the best known
feasible value is **{best}** ({BEST_KNOWN_SOURCE}). A greedy by value per unit
of surrogate weight lands a few percent below it; tabu search, genetic
algorithms with a repair operator, and add/drop/swap local search with
restarts close most of the gap. `numpy` and `pandas` are available.

## Submission format

Write `submission.csv` in the working directory with the header `id,take`:
{n} rows, `id` = 0..{n - 1}, `take` = 1 for a chosen item and 0 otherwise.
`sample_submission.csv` takes nothing (a valid, worthless baseline).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier checks every constraint and prints `val_score: <total value>`, with
the slack left on each constraint so an improve step sees where the room
is (or a score of {PENALTY} if any constraint is exceeded or the file is
malformed). **Higher is better.**

There is no train/test split; this is a pure optimization problem. Keep total
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


def verify_py(n: int, m: int) -> str:
    return f'''"""Official scorer for the mknap-{n}-{m} problem.

Reads ./submission.csv (id,take; {n} rows of 0/1), checks the {m} capacity
constraints against data/items.csv and data/capacities.csv, and writes the
score to $HILLCLIMB_RESULT (the total value of the chosen items; {PENALTY} if
any constraint is exceeded or the file is malformed — higher is better).
Generated by problems/make_mknap.py — edit the template there.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N = {n}
M = {m}
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
    data = Path(__file__).resolve().parent / "data"
    items = pd.read_csv(data / "items.csv").sort_values("id")
    caps = pd.read_csv(data / "capacities.csv")
    weight_cols = [f"w{{i + 1}}" for i in range(M)]
    values = items["value"].to_numpy(float)
    weights = items[weight_cols].to_numpy(float).T          # M x N
    capacity = caps.set_index("constraint").loc[weight_cols, "capacity"].to_numpy(float)

    try:
        df = pd.read_csv("submission.csv")
    except Exception as e:  # noqa: BLE001
        fail(f"cannot read submission.csv: {{e}}")
    for col in ("id", "take"):
        if col not in df.columns:
            fail(f"missing column {{col!r}}")
    if len(df) != N or sorted(df["id"].tolist()) != list(range(N)):
        fail(f"need exactly {{N}} rows with id 0..{{N - 1}}")
    take = pd.to_numeric(df.sort_values("id")["take"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(take)) or not np.all(np.isin(take, (0.0, 1.0))):
        fail("take must be 0 or 1")

    used = weights @ take
    over = [(weight_cols[i], used[i], capacity[i]) for i in range(M) if used[i] > capacity[i]]
    if over:
        fail("over capacity: " + ", ".join(f"{{c}} {{u:.0f}}/{{cap:.0f}}" for c, u, cap in over))
    total = float(values @ take)
    for i in range(M):
        print(f"{{weight_cols[i]}}: used {{used[i]:.0f}} of {{capacity[i]:.0f}}, slack {{capacity[i] - used[i]:.0f}}")
    print(f"valid: {{int(take.sum())}} items; total value = {{total:.0f}}")
    emit(total)
    print(f"val_score: {{total:.0f}}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 — an invalid submission is the penalty, never a crash
        fail(f"unexpected error: {{e}}")
'''


def interface_py(n: int, m: int) -> str:
    return f'''"""Machine-checked output format (hillclimb spaces). Format only — the
capacities and the value are the verifier's. Generated by
problems/make_mknap.py — edit the template there.
"""

from hillclimb import spaces

output = spaces.Table(
    "submission.csv",
    columns={{
        "id": spaces.Int(values=range({n}), unique=True),
        "take": spaces.Int(values=(0, 1)),
    }},
    n_rows={n},
)

if __name__ == "__main__":
    raise SystemExit(spaces.main(output))
'''


def files_for(n: int, m: int) -> dict[str, str]:
    file, index, _name, _best, _lp = LEVELS[(n, m)]
    values, weights, capacities = parse_orlib(fetch(file), index)
    if len(values) != n or len(weights) != m:
        raise ValueError(f"{file}[{index}] is {len(values)} x {len(weights)}, not {n} x {m}")
    return {
        "problem.yaml": problem_yaml(n, m),
        "description.md": description_md(n, m),
        "verifier.sh": VERIFIER_SH,
        "verify.py": verify_py(n, m),
        "interface.py": interface_py(n, m),
        "sample_submission.csv": sample_submission(n),
        "data/items.csv": items_csv(values, weights),
        "data/capacities.csv": capacities_csv(capacities),
    }


def stamp(root: Path, n: int, m: int) -> Path:
    """Write mknap-<n>-<m>/ under `root`; returns the dir."""
    problem_dir = root / f"mknap-{n}-{m}"
    for name, text in files_for(n, m).items():
        path = problem_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (problem_dir / "verifier.sh").chmod(0o755)
    return problem_dir


def main(argv: list[str]) -> None:
    levels = [tuple(int(v) for v in a.split("-")) for a in argv] or list(LEVELS)
    for n, m in levels:
        for root in (ROOT, DEMO):
            print(f"wrote {stamp(root, n, m)}")


if __name__ == "__main__":
    main(sys.argv[1:])
