"""The bin-packing instances, and what makes a packing valid.

Shared by the runner (run.py, in the solution's sandbox) and the scorer
(evaluate.py). Validation instances come from seed 0; the holdout seed is in
holdout/seed.txt, which only holdout runs can read. Stdlib only.
"""

from __future__ import annotations

import random
from pathlib import Path

N_INSTANCES = 50
N_ITEMS = 120
CAPACITY = 1.0
TOL = 1e-9


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




def seed_for(split: str) -> int:
    """The instances' seed. The holdout one is a holdout input: agents and
    validation runs cannot read it."""
    if split == "holdout":
        return int((Path(__file__).resolve().parent / "holdout" / "seed.txt").read_text())
    return 0
