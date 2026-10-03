"""Generic verifier for self-reported problems: run solution.py, collect its
score.

Standalone by construction — it runs inside the solution runtime venv, which
has no hillclimb installed. Used where the score is the solution's own claim
rather than an independent evaluation (MLE-bench: coding agents climb on their own
validation number, official grading happens once after the search).

Contract, in order of preference:

1. the solution wrote `$HILLCLIMB_RESULT` itself — kept as-is
2. otherwise the last `val_score: <float>` line of its stdout is written there

`--require <name>` fails the run when a file the search needs later (an
MLE-bench `submission.csv`, say) was not produced.

    run_solution.py <solution.py> [--require submission.csv]
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

VAL_SCORE_RE = re.compile(r"^val_score:\s*([-+0-9.eE]+)\s*$")


def _parse_score(text: str) -> float | None:
    for line in reversed(text.splitlines()):
        match = VAL_SCORE_RE.match(line.strip())
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                return None
            return None if value != value else value
    return None


def _has_score(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    score = payload.get("score") if isinstance(payload, dict) else payload
    return isinstance(score, (int, float)) and not isinstance(score, bool) and score == score


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print("run_solution.py: no solution path given", file=sys.stderr)
        return 2
    solution = Path(argv[0])
    required = [argv[i + 1] for i, token in enumerate(argv) if token == "--require"]
    result = Path(os.environ.get("HILLCLIMB_RESULT", "eval_result.json"))

    # tee: the log must still show everything the solution printed, and the
    # score line has to be readable after the fact
    proc = subprocess.Popen(
        [sys.executable, str(solution)],
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        captured.append(line)
    returncode = proc.wait()
    sys.stdout.flush()
    if returncode != 0:
        return returncode

    for name in required:
        if not Path(name).exists():
            print(f"run_solution.py: solution did not write {name}", file=sys.stderr)
            return 3

    if _has_score(result):
        return 0  # the solution reported its own result payload
    score = _parse_score("".join(captured))
    if score is None:
        print(
            "run_solution.py: no `val_score: <float>` line in the solution's output",
            file=sys.stderr,
        )
        return 4
    result.write_text(
        json.dumps(
            {
                "split": os.environ.get("HILLCLIMB_SPLIT", "validation"),
                "score": score,
                "source": "agent",
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
