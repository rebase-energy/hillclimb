"""The tuner seam: WHICH parameter set to try next for one candidate.

A `Tuner` answers `ask()` as a pure function of the candidate's trial
history (plus the parameter sets currently in flight), so replay and resume
need no persistent study state: the engine rebuilds the history from the
journal on every call. WHEN to tune, and how much, is the search policy's
decision (`Action(operator="tune")`), never the tuner's.

Implementations live in `hillclimb.modules.tuners` (`random`: stdlib, the default
and the test double; `optuna`: TPE via the optional extra). A climber's block
names one like any module: `tuner: random`, `tuner: anneal.py`, `tuner: pkg.mod:Class`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from collections.abc import Mapping
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from hillclimb.harness.candidate import Candidate
    from hillclimb.harness.params import ParamSpace


@dataclass(frozen=True)
class Observation:
    """One parameter set the tuner may learn from: `score` is the trial's
    raw journal-direction score (None = that set failed); `pending` marks a
    set in flight on the same candidate (a constant-liar hint)."""

    params: dict
    score: float | None
    pending: bool = False


class Tuner:
    """WHICH parameter set to try next for one candidate. Subclass it, set
    `name`, implement `ask`. A class with just an `ask` method still runs —
    the contract is the method."""

    name: str = ""

    def __init__(self, params: Mapping | None = None, **knobs):
        # the climber's `tuner_params` (a `seed` among them seeds every ask);
        # by keyword when composing in Python: `RandomSearch(seed=7)`
        self.params = {**dict(params or {}), **knobs}

    def ask(
        self,
        space: "ParamSpace",
        history: Sequence[Observation],
        *,
        higher_is_better: bool,
        seed: int,
    ) -> dict:
        """The next parameter set to evaluate (every declared name present,
        values inside the declared domain). A pure function of its
        arguments: no study state survives a call, so resume is free."""
        raise NotImplementedError(f"{type(self).__name__} must implement ask(space, history, ...)")


def tune_seed(base: int, candidate_id: str, n_history: int) -> int:
    """Deterministic per-ask seed: same base, candidate and history length →
    same proposal, so a resumed search re-proposes what a killed one would
    have."""
    digest = hashlib.sha256(f"hillclimb-tune:{base}:{candidate_id}:{n_history}".encode()).hexdigest()
    return int(digest[:8], 16)


def history_for(candidate: "Candidate", pending: Sequence[dict] = ()) -> list[Observation]:
    """The candidate's committed trials in index order, then the in-flight
    parameter sets."""
    observations = [
        Observation(params=dict(trial.params), score=trial.val_score)
        for trial in sorted(candidate.trials, key=lambda t: t.index)
    ]
    observations.extend(Observation(params=dict(p), score=None, pending=True) for p in pending)
    return observations
