"""Step 1: run the solution.

hillclimb starts this in the solution's own sandbox. It imports `partition`
from solution.py, calls it on every instance and writes what came back to
output.json. It scores nothing: score.py does, in a sandbox of its own.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from instances import make_instances


def main() -> None:
    solution = Path(os.environ.get("HILLCLIMB_SOLUTION", "solution.py")).resolve()
    sys.path.insert(0, str(solution.parent))  # the solution may import files beside it
    import solution as candidate

    partition = getattr(candidate, "partition", None)
    if not callable(partition):
        print("solution.py must define partition(numbers)", file=sys.stderr)
        raise SystemExit(1)
    answers = []
    for index, numbers in enumerate(make_instances()):
        try:
            answer = partition(list(numbers))
        except Exception as exc:  # noqa: BLE001 - the solution's error is the run's output
            print(f"instance {index}: partition raised {exc!r}", file=sys.stderr)
            raise SystemExit(1) from exc
        # JSON carries plain ints only; score.py judges whether the answer is valid
        answers.append([int(i) for i in answer] if isinstance(answer, (list, tuple)) else repr(answer))
    Path("output.json").write_text(json.dumps(answers))


if __name__ == "__main__":
    main()
