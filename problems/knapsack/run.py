"""Runs the candidate on the knapsack instances of this split.

hillclimb starts this in the solution's own sandbox: it imports
``select_items`` from solution.py, calls it on every instance, and writes the
selections to selections.json. It scores nothing; verify.py, in a sandbox of
its own, validates and scores what it wrote.
"""

from __future__ import annotations

import json
import numbers
import os
import sys
from pathlib import Path

from instances import make_instances, seed_for


def plain(selected: object) -> object:
    """A selection as JSON can carry it (numpy integers become ints); the
    scorer judges whether it is valid."""
    if isinstance(selected, (list, tuple)):
        return [int(i) if isinstance(i, numbers.Integral) and not isinstance(i, bool) else repr(i) for i in selected]
    return repr(selected)


def main() -> None:
    split = os.environ.get("HILLCLIMB_SPLIT", "validation")
    solution = Path(os.environ.get("HILLCLIMB_SOLUTION", "solution.py")).resolve()
    sys.path.insert(0, str(solution.parent))  # an ensemble imports its candidate_N modules
    import solution as candidate

    select_items = getattr(candidate, "select_items", None)
    if not callable(select_items):
        print("solution.py must define select_items(items, capacity)", file=sys.stderr)
        raise SystemExit(1)
    selections = []
    for index, (items, capacity) in enumerate(make_instances(seed_for(split))):
        try:
            selections.append(plain(select_items(list(items), capacity)))
        except Exception as exc:  # noqa: BLE001 - candidate errors are verifier output
            print(f"instance {index}: select_items raised {exc!r}", file=sys.stderr)
            raise SystemExit(1) from exc
    Path("selections.json").write_text(json.dumps({"split": split, "selections": selections}))


if __name__ == "__main__":
    main()
