"""hillclimb.sdk — the one import a climber needs.

A climber is the shareable bundle of exchangeable modules (a `SearchPolicy`
or `SearchLoop`, operators, memory, a tuner, similarity scores) that decides
HOW to hillclimb. Everything else — running agents, scoring, holdout, the
journal, budgets — is the harness, and a climber only ever meets it through
the names exported here. `tests/test_sdk_imports.py` enforces the other half
of that sentence: climber modules import `hillclimb.sdk` and nothing else
from hillclimb.

Exports are lazy (like the package root), so a module the sdk re-exports may
itself import from the sdk without an import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

_LAZY = {
    # what to try next
    "Action": ("hillclimb.policy", "Action"),
    "Route": ("hillclimb.policy", "Route"),
    "InflightRef": ("hillclimb.policy", "InflightRef"),
    "BudgetView": ("hillclimb.policy", "BudgetView"),
    "PolicyInput": ("hillclimb.policy", "PolicyInput"),
    "SearchPolicy": ("hillclimb.policy", "SearchPolicy"),
    "TUNE_ACTION": ("hillclimb.policy", "TUNE_ACTION"),
    # control flow over the harness (most climbers only need a SearchPolicy)
    "SearchLoop": ("hillclimb.loop", "SearchLoop"),
    "PolicyLoop": ("hillclimb.loop", "PolicyLoop"),
    "Harness": ("hillclimb.loop", "Harness"),
    "SearchInfo": ("hillclimb.loop", "SearchInfo"),
    "Ticket": ("hillclimb.loop", "Ticket"),
    "Outcome": ("hillclimb.loop", "Outcome"),
    "HarnessClosed": ("hillclimb.loop", "HarnessClosed"),
    "ClimberError": ("hillclimb.loop", "ClimberError"),
    # the records a climber reads (always holdout-blind copies)
    "Candidate": ("hillclimb.candidate", "Candidate"),
    "Trial": ("hillclimb.candidate", "Trial"),
    "Replicate": ("hillclimb.candidate", "Replicate"),
    "PolicyJournal": ("hillclimb.journal", "PolicyJournal"),
    # the accept rule, so a policy agrees with the harness on what "better" means
    "improves": ("hillclimb.evaluation", "improves"),
    "accept_band": ("hillclimb.evaluation", "accept_band"),
    # how one attempt is made
    "Operator": ("hillclimb.operators.base", "Operator"),
    "OperatorContext": ("hillclimb.operators.base", "OperatorContext"),
    "Preparation": ("hillclimb.operators.base", "Preparation"),
    "ProblemInfo": ("hillclimb.operators.base", "ProblemInfo"),
    "MemoryContext": ("hillclimb.operators.base", "MemoryContext"),
    "inspiration_filename": ("hillclimb.operators.base", "inspiration_filename"),
    "ROLES": ("hillclimb.operators.base", "ROLES"),
    # which parameter values next
    "Tuner": ("hillclimb.tuner", "Tuner"),
    "Observation": ("hillclimb.tuner", "Observation"),
    "ParamSpace": ("hillclimb.params", "ParamSpace"),
    "ParamSpec": ("hillclimb.params", "ParamSpec"),
    "coerce": ("hillclimb.params", "coerce"),
    # how alike two solutions are
    "SimilarityScore": ("hillclimb.similarity_scores.base", "SimilarityScore"),
    "SimilarityUnavailable": ("hillclimb.similarity_scores.base", "SimilarityUnavailable"),
    "Solution": ("hillclimb.similarity_scores.base", "Solution"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        value = getattr(importlib.import_module(module), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:  # eager for type checkers and editors only
    from hillclimb.candidate import Candidate, Replicate, Trial
    from hillclimb.evaluation import accept_band, improves
    from hillclimb.journal import PolicyJournal
    from hillclimb.loop import (
        ClimberError,
        Harness,
        HarnessClosed,
        Outcome,
        PolicyLoop,
        SearchInfo,
        SearchLoop,
        Ticket,
    )
    from hillclimb.operators.base import (
        ROLES,
        MemoryContext,
        Operator,
        OperatorContext,
        Preparation,
        ProblemInfo,
        inspiration_filename,
    )
    from hillclimb.params import ParamSpace, ParamSpec, coerce
    from hillclimb.policy import (
        TUNE_ACTION,
        Action,
        BudgetView,
        InflightRef,
        PolicyInput,
        Route,
        SearchPolicy,
    )
    from hillclimb.similarity_scores.base import SimilarityScore, SimilarityUnavailable, Solution
    from hillclimb.tuner import Observation, Tuner
