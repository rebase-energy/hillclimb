"""GEPA engine parameters: the validated subset of the climber's params the
MVP supports. extra="forbid" so a misspelled budget or privacy setting fails
before any model spend."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hillclimb.config import Config

SELECTION_STRATEGIES = ("pareto", "current_best")


class GEPAParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # safety ceilings on the optimizer itself; the hillclimb wall clock and
    # cost ceiling remain authoritative regardless
    max_metric_calls: int = Field(default=50, gt=0)
    reflection_minibatch_size: int | None = Field(default=None, gt=0)
    candidate_selection_strategy: str = "pareto"
    # 'instance' = Pareto frontier over the verifier's per-instance scores
    # (degenerates to one instance — the aggregate score — when the
    # verifier emits none). Upstream's 'objective' frontier needs the
    # evaluator to hand back objective_scores, which this bridge does not
    # produce, so it is rejected here rather than at the first evaluation.
    frontier_type: Literal["instance"] = "instance"
    cache_evaluation: bool = True
    use_merge: Literal[False] = False  # merge lineage is post-MVP
    failure_fitness: float = -1.0e100
    seed: int = 0

    @field_validator("candidate_selection_strategy")
    @classmethod
    def _known_strategy(cls, value: str) -> str:
        if value not in SELECTION_STRATEGIES:
            raise ValueError(
                f"candidate_selection_strategy must be one of {SELECTION_STRATEGIES}, got {value!r}"
            )
        return value

    @field_validator("failure_fitness")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("failure_fitness must be finite")
        return value


def validate_gepa_search_config(config: Config) -> GEPAParams:
    """Parse and gate the config before any candidate dir or model spend."""
    params = GEPAParams.model_validate(config.climber.params or {})
    if config.concurrency.parallel_operators > 1:
        raise ValueError(
            "the GEPA engine is serial in the MVP: set concurrency.parallel_operators=1"
        )
    return params
