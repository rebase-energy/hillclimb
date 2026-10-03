"""Runs the candidate on the bin-packing instances of this split.

hillclimb starts this in the solution's own sandbox: it imports `pack` from
solution.py, calls it on every instance, and writes the packings to
packings.json. It scores nothing; evaluate.py, in a sandbox of its own,
validates and scores what it wrote. Stdlib only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from instances import CAPACITY, make_instances, seed_for


def main() -> None:
    split = os.environ.get("HILLCLIMB_SPLIT", "validation")
    solution = Path(os.environ.get("HILLCLIMB_SOLUTION", "solution.py")).resolve()
    sys.path.insert(0, str(solution.parent))
    import solution as candidate

    packings = []
    for items in make_instances(seed_for(split)):
        bins = candidate.pack(list(items), CAPACITY)
        try:
            json.dumps(bins)
        except (TypeError, ValueError):
            bins = repr(bins)  # the scorer rejects it, with the reason
        packings.append(bins)
    Path("packings.json").write_text(json.dumps({"split": split, "packings": packings}))


if __name__ == "__main__":
    main()
