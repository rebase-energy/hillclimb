"""The bundled 0/1-knapsack problem's instances, and what makes a selection valid.

Shared by the runner (run.py, in the solution's sandbox) and the scorer
(verify.py). The validation instances come from VALIDATION_SEED; the holdout
instances' seed is in holdout/seed.txt, which only holdout runs can read.
"""

from __future__ import annotations

import random
from pathlib import Path

N_INSTANCES = 24
N_ITEMS = 160
VALIDATION_SEED = 1729

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


def seed_for(split: str) -> int:
    """The instances' seed. The holdout one is a holdout input: agents and
    validation runs cannot read it."""
    if split == "holdout":
        return int((Path(__file__).resolve().parent / "holdout" / "seed.txt").read_text())
    return VALIDATION_SEED
