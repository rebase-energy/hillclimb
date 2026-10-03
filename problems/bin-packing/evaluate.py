"""Official scorer for the bin-packing problem.

run.py has already called the solution's `pack` on 50 fixed instances, in the
solution's own sandbox, and written packings.json. This validates every
packing against instances it builds itself, scores the mean bin count, writes
`eval_result.json` (score + report) and prints the final `val_score:` line.
It never imports the solution. Stdlib only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from instances import CAPACITY, TOL, make_instances, seed_for, validate

WORST_K = 6


def main() -> None:
    split = "holdout" if "--holdout" in sys.argv else "validation"
    instances = make_instances(seed_for(split))
    try:
        recorded = json.loads(Path("packings.json").read_text())
    except (OSError, ValueError) as exc:
        print(f"no packings to score: {exc}", file=sys.stderr)
        sys.exit(1)
    packings = recorded.get("packings") if isinstance(recorded, dict) else None
    if recorded.get("split") != split or not isinstance(packings, list) or len(packings) != len(instances):
        print(f"packings.json must hold one packing per {split} instance", file=sys.stderr)
        sys.exit(1)
    results = []
    for index, (items, bins) in enumerate(zip(instances, packings)):
        error = validate(items, bins)
        if error is not None:
            print(f"instance {index}: INVALID packing — {error}", file=sys.stderr)
            sys.exit(1)
        lower_bound = math.ceil(sum(items) / CAPACITY - TOL)
        results.append({"instance": index, "bins": len(bins), "lower_bound": lower_bound})

    score = sum(r["bins"] for r in results) / len(results)
    worst = sorted(results, key=lambda r: r["bins"] - r["lower_bound"], reverse=True)
    report = {
        "version": 1,
        "split": split,
        "objective": "mean-bins",
        "higher_is_better": False,
        "source": "evaluator",
        "segment_label": "instance",
        "overall": {"score": score, "n_origins": len(results), "n_scored": len(results)},
        # instances furthest above their ceil(sum/capacity) floor: where a
        # better algorithm has the most to gain
        "zones": [
            {
                "zone": f"instance-{r['instance']}",
                "score": r["bins"],
                "n_scored": 1,
                "lower_bound": r["lower_bound"],
            }
            for r in worst[:WORST_K]
        ],
        "residual_bias": None,
        "report_error": None,
    }
    report = {k: v for k, v in report.items() if v is not None}
    with open("eval_result.json", "w") as fh:
        json.dump({"split": split, "score": score, "report": report}, fh, indent=2)
    print(f"mean bins over {len(results)} instances: {score:.4f} "
          f"(floor {sum(r['lower_bound'] for r in results) / len(results):.4f})")
    print(f"val_score: {score}")


if __name__ == "__main__":
    main()
