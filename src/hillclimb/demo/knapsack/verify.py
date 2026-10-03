"""Official scorer for the bundled 0/1-knapsack problem.

run.py has already called the candidate's ``select_items`` on every instance,
in the solution's own sandbox, and written selections.json. This validates
every selection against instances it builds itself and writes hillclimb's
result/report payload; it never imports the solution. The score is the mean
percentage of the fractional-knapsack upper bound, which makes differently
scaled instances comparable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from instances import fractional_upper_bound, make_instances, seed_for, validate_selection

WORST_K = 6


def main() -> None:
    split = "holdout" if "--holdout" in sys.argv else "validation"
    instances = make_instances(seed_for(split))
    try:
        recorded = json.loads(Path("selections.json").read_text())
    except (OSError, ValueError) as exc:
        print(f"no selections to score: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    selections = recorded.get("selections") if isinstance(recorded, dict) else None
    if recorded.get("split") != split or not isinstance(selections, list) or len(selections) != len(instances):
        print(f"selections.json must hold one selection per {split} instance", file=sys.stderr)
        raise SystemExit(1)

    results = []
    for index, ((items, capacity), selected) in enumerate(zip(instances, selections)):
        total_weight, total_value, error = validate_selection(items, capacity, selected)
        if error is not None:
            print(f"instance {index}: INVALID selection — {error}", file=sys.stderr)
            raise SystemExit(1)
        upper_bound = fractional_upper_bound(items, capacity)
        ratio = 100.0 * total_value / upper_bound
        results.append(
            {
                "instance": index,
                "score": ratio,
                "value": total_value,
                "weight": total_weight,
                "capacity": capacity,
                "upper_bound": upper_bound,
            }
        )

    score = sum(result["score"] for result in results) / len(results)
    worst = sorted(results, key=lambda result: result["score"])[:WORST_K]
    report = {
        "version": 1,
        "split": split,
        "objective": "mean-percent-of-upper-bound",
        "higher_is_better": True,
        "source": "evaluator",
        "segment_label": "instance",
        "overall": {
            "score": score,
            "n_origins": len(results),
            "n_scored": len(results),
        },
        "zones": [
            {
                "zone": f"instance-{result['instance']}",
                "score": round(result["score"], 6),
                "n_scored": 1,
                "selected_value": result["value"],
                "selected_weight": result["weight"],
                "capacity": result["capacity"],
                "fractional_upper_bound": round(result["upper_bound"], 3),
            }
            for result in worst
        ],
        "report_error": None,
    }
    payload = {"split": split, "score": score, "report": report}
    result_path = Path(os.environ.get("HILLCLIMB_RESULT", "eval_result.json"))
    result_path.write_text(json.dumps(payload, indent=2))
    print(f"{split}: mean percent of upper bound = {score:.6f}")
    print(f"val_score: {score:.12g}")


if __name__ == "__main__":
    main()
