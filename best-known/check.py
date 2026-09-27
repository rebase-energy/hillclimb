#!/usr/bin/env python3
"""Re-score every file in best-known/ with its problem's own verify.py and
compare with index.json. Files for sizes the catalog has no problem for
(heilbronn-3 … heilbronn-16 apart from 11 and 14) are scored with the
same rule the Heilbronn verifiers use, since there is no verifier to run.

    uv run python best-known/check.py

Exits 1 on any mismatch or invalid file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROBLEMS = ROOT / "problems"
REL_TOL = 1e-9


def verify(problem: str, csv_text: str) -> float | None:
    """What verify.py writes for the submission, None if it calls it invalid."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "submission.csv").write_text(csv_text)
        result = Path(tmp, "result.json")
        env = {**os.environ, "HILLCLIMB_RESULT": str(result)}
        proc = subprocess.run([sys.executable, str(PROBLEMS / problem / "verify.py")], cwd=tmp, env=env,
                              capture_output=True, text=True)
        if "INVALID" in proc.stdout or not result.exists():
            return None
        return float(json.loads(result.read_text())["score"])


def heilbronn_square(csv_text: str) -> float | None:
    """The Heilbronn verifiers' rule, for sizes without a verifier: points in
    the unit square (1e-9 slack), score = the smallest triangle's area."""
    rows = [line.split(",") for line in csv_text.strip().splitlines()[1:]]
    pts = [(float(x), float(y)) for _, x, y in rows]
    if any(v < -1e-9 or v > 1 + 1e-9 for p in pts for v in p):
        return None
    return min(0.5 * abs((pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1])
                         - (pts[j][1] - pts[i][1]) * (pts[k][0] - pts[i][0]))
               for i, j, k in combinations(range(len(pts)), 3))


def main() -> None:
    index = json.loads((HERE / "index.json").read_text())
    failures = 0
    for name, entry in index.items():
        path = HERE / entry["file"]
        if not path.exists():
            print(f"FAIL {name:22s} missing {entry['file']}")
            failures += 1
            continue
        csv_text = path.read_text()
        problem = entry.get("problem")
        if problem and (PROBLEMS / problem / "verify.py").exists():
            value, how = verify(problem, csv_text), "verify.py"
        elif name.startswith("heilbronn-"):
            value, how = heilbronn_square(csv_text), "min-triangle rule"
        else:
            print(f"skip {name:22s} no verifier")
            continue
        if value is None:
            print(f"FAIL {name:22s} {how}: invalid submission")
            failures += 1
        elif abs(value - entry["value"]) > REL_TOL * max(1.0, abs(entry["value"])):
            print(f"FAIL {name:22s} {how}: {value!r} but index says {entry['value']!r}")
            failures += 1
        else:
            print(f"ok   {name:22s} {value!r:24s} {entry['status']:11s} {entry['who']}")
    print(f"\n{len(index) - failures} of {len(index)} agree" + ("" if not failures else f", {failures} FAILED"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
