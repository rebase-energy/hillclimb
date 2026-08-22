"""Official evaluator for the bin-packing problem.

Imports `pack` from the solution in the working directory, runs it on 50
fixed instances (seed 0 = validation, seed 1 = holdout), validates every
packing, and scores the mean bin count. Writes `eval_result.json` (score +
report) and prints the final `val_score:` line — hillclimb's evaluator
contract. Stdlib only.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

N_INSTANCES = 50
N_ITEMS = 120
CAPACITY = 1.0
TOL = 1e-9
WORST_K = 6


def make_instances(seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [
        [round(rng.uniform(0.05, 0.7), 9) for _ in range(N_ITEMS)]
        for _ in range(N_INSTANCES)
    ]


def validate(items: list[float], bins: list[list[float]]) -> str | None:
    if not isinstance(bins, list) or not all(isinstance(b, list) for b in bins):
        return "pack() must return a list of bins (list[list[float]])"
    packed = sorted(value for b in bins for value in b)
    expected = sorted(items)
    if len(packed) != len(expected) or any(
        abs(a - b) > TOL for a, b in zip(packed, expected)
    ):
        return "bins do not partition the input items (missing/extra/altered values)"
    for index, b in enumerate(bins):
        if sum(b) > CAPACITY + TOL:
            return f"bin {index} overflows capacity ({sum(b):.6f} > {CAPACITY})"
        if not b:
            return f"bin {index} is empty"
    return None


def main() -> None:
    split = "holdout" if "--holdout" in sys.argv else "validation"
    sys.path.insert(0, os.getcwd())
    import solution

    instances = make_instances(seed=1 if split == "holdout" else 0)
    results = []
    for index, items in enumerate(instances):
        bins = solution.pack(list(items), CAPACITY)
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
