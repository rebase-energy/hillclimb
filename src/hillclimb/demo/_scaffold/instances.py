"""The instances, shared by run.py and score.py.

Built from a fixed seed, so every run sees the same lists and the score of
a solution never moves between runs.
"""

from __future__ import annotations

import random

SEED = 20260101
N_INSTANCES = 20
N_NUMBERS = 60
LARGEST = 10**12


def make_instances() -> list[list[int]]:
    rng = random.Random(SEED)
    return [[rng.randint(1, LARGEST) for _ in range(N_NUMBERS)] for _ in range(N_INSTANCES)]
