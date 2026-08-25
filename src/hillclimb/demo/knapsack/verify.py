"""Official evaluator for the bundled 0/1-knapsack problem.

Imports ``select_items`` from the candidate, evaluates 24 deterministic
instances, validates every selection, and writes Hillclimb's result/report
payload. The score is the mean percentage of the fractional-knapsack upper
bound, which makes differently scaled instances comparable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys

N_INSTANCES = 24
N_ITEMS = 160
VALIDATION_SEED = 1729
HOLDOUT_SEED = 2718
WORST_K = 6

Item = tuple[int, int]  # (weight, value)
Instance = tuple[list[Item], int]


def _density_trap(rng: random.Random) -> Instance:
    """An instance where the fractional/density choice is locally tempting.

    Two core items exactly fill the knapsack at density 100. A smaller item at
    density 105 is considered first by greedy, but taking it blocks the core
    pair. Lower-density filler items keep the instance realistic and give
    local-repair methods useful alternatives.
    """
    capacity = rng.randint(5_000, 9_000)
    first_weight = rng.randint(int(0.48 * capacity), int(0.58 * capacity))
    second_weight = capacity - first_weight
    distractor_weight = rng.randint(int(0.10 * capacity), int(0.18 * capacity))
    items: list[Item] = [
        (first_weight, 100 * first_weight),
        (second_weight, 100 * second_weight),
        (distractor_weight, 105 * distractor_weight),
    ]
    max_filler_weight = max(60, capacity // 8)
    for _ in range(N_ITEMS - len(items)):
        weight = rng.randint(50, max_filler_weight)
        density = rng.randint(35, 80)
        value = density * weight + rng.randint(0, max(1, weight // 5))
        items.append((weight, value))
    rng.shuffle(items)
    return items, capacity


def _uncorrelated(rng: random.Random) -> Instance:
    items = [
        (rng.randint(20, 500), rng.randint(20, 1_000))
        for _ in range(N_ITEMS)
    ]
    capacity = sum(weight for weight, _ in items) // 5
    return items, capacity


def make_instances(seed: int) -> list[Instance]:
    rng = random.Random(seed)
    return [
        _density_trap(rng) if index % 3 == 0 else _uncorrelated(rng)
        for index in range(N_INSTANCES)
    ]


def fractional_upper_bound(items: list[Item], capacity: int) -> float:
    """Optimal value of the fractional relaxation (an upper bound on 0/1)."""
    remaining = capacity
    value = 0.0
    for weight, item_value in sorted(
        items, key=lambda item: item[1] / item[0], reverse=True
    ):
        taken = min(weight, remaining)
        value += item_value * taken / weight
        remaining -= taken
        if remaining == 0:
            break
    return value


def validate_selection(
    items: list[Item], capacity: int, selected: object
) -> tuple[int, int, str | None]:
    if not isinstance(selected, list):
        return 0, 0, "select_items() must return a list of indices"
    if not all(isinstance(index, int) and not isinstance(index, bool) for index in selected):
        return 0, 0, "every selected index must be an integer"
    if len(set(selected)) != len(selected):
        return 0, 0, "selected indices must be unique"
    if any(index < 0 or index >= len(items) for index in selected):
        return 0, 0, f"selected index outside 0..{len(items) - 1}"
    total_weight = sum(items[index][0] for index in selected)
    total_value = sum(items[index][1] for index in selected)
    if total_weight > capacity:
        return total_weight, total_value, (
            f"selected weight {total_weight} exceeds capacity {capacity}"
        )
    return total_weight, total_value, None


def main() -> None:
    split = "holdout" if "--holdout" in sys.argv else "validation"
    seed = HOLDOUT_SEED if split == "holdout" else VALIDATION_SEED
    sys.path.insert(0, os.getcwd())
    import solution

    select_items = getattr(solution, "select_items", None)
    if not callable(select_items):
        print("solution.py must define select_items(items, capacity)", file=sys.stderr)
        raise SystemExit(1)

    results = []
    for index, (items, capacity) in enumerate(make_instances(seed)):
        try:
            selected = select_items(list(items), capacity)
        except Exception as exc:  # noqa: BLE001 - candidate errors are verifier output
            print(f"instance {index}: select_items raised {exc!r}", file=sys.stderr)
            raise SystemExit(1) from exc
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
