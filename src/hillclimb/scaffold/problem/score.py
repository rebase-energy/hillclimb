"""Step 2: score what the solution wrote.

Reads output.json, checks every answer against the instances it builds
itself, and writes the score to $HILLCLIMB_RESULT. Exit non-zero for an
invalid answer: the candidate is then buggy, and the search repairs it.
hillclimb reads the score from that file only, never from stdout.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

from instances import make_instances


def invalid(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    instances = make_instances()
    try:
        answers = json.loads(Path("output.json").read_text())
    except (OSError, ValueError) as exc:
        invalid(f"no output.json to score: {exc}")
    if not isinstance(answers, list) or len(answers) != len(instances):
        invalid(f"output.json must hold one answer per instance ({len(instances)})")

    per_instance = {}
    for index, (numbers, answer) in enumerate(zip(instances, answers)):
        if not isinstance(answer, list) or not all(isinstance(i, int) for i in answer):
            invalid(f"instance {index}: partition must return a list of ints, got {answer!r}")
        if len(set(answer)) != len(answer):
            invalid(f"instance {index}: an index appears twice")
        if any(i < 0 or i >= len(numbers) for i in answer):
            invalid(f"instance {index}: an index is out of range")
        first = sum(numbers[i] for i in answer)
        imbalance = abs(sum(numbers) - 2 * first)
        per_instance[f"instance-{index:02d}"] = math.log10(1 + imbalance)

    score = sum(per_instance.values()) / len(per_instance)
    result = {
        "score": score,
        # optional: per-instance scores, which some climbers select on
        "instances": per_instance,
    }
    Path(os.environ.get("HILLCLIMB_RESULT", "result.json")).write_text(json.dumps(result))
    print(f"mean log10 imbalance: {score:.6f}")


if __name__ == "__main__":
    main()
