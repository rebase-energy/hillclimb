"""Random search over a declared parameter space: stdlib only, the default
tuner and the test double for the seam. Uniform (log-uniform when `log`)
draws snapped to `step`; categorical picks; a few redraws avoid repeating a
set already in the history."""

from __future__ import annotations

import math
import random
from typing import Sequence

from hillclimb.sdk import Observation, ParamSpace, ParamSpec, Tuner, coerce

REDRAWS = 16


def sample_param(spec: ParamSpec, rng: random.Random):
    if spec.type == "categorical":
        return rng.choice(list(spec.choices))
    low, high = float(spec.low), float(spec.high)
    if spec.log:
        value = math.exp(rng.uniform(math.log(low), math.log(high)))
    else:
        value = rng.uniform(low, high)
    if spec.step:
        value = low + round((value - low) / spec.step) * spec.step
    value = min(max(value, low), high)
    return int(round(value)) if spec.type == "int" else value


class RandomSearch(Tuner):
    name = "random"


    def ask(
        self,
        space: ParamSpace,
        history: Sequence[Observation],
        *,
        higher_is_better: bool,
        seed: int,
    ) -> dict:
        rng = random.Random(seed)
        seen = {repr(sorted(coerce(space, h.params).items())) for h in history}
        proposal = None
        for _ in range(REDRAWS):
            proposal = coerce(space, {name: sample_param(spec, rng) for name, spec in space.items()})
            if repr(sorted(proposal.items())) not in seen:
                break
        return proposal
