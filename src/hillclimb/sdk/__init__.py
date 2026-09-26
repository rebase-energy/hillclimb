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
    "Action": ("hillclimb.modules.policies.base", "Action"),
    "Route": ("hillclimb.modules.policies.base", "Route"),
    "InflightRef": ("hillclimb.modules.policies.base", "InflightRef"),
    "BudgetView": ("hillclimb.modules.policies.base", "BudgetView"),
    "PolicyInput": ("hillclimb.modules.policies.base", "PolicyInput"),
    "SearchPolicy": ("hillclimb.modules.policies.base", "SearchPolicy"),
    "TUNE_ACTION": ("hillclimb.modules.policies.base", "TUNE_ACTION"),
    "INJECT_ACTION": ("hillclimb.modules.policies.base", "INJECT_ACTION"),
    # control flow over the harness (most climbers only need a SearchPolicy)
    "SearchLoop": ("hillclimb.harness.loop", "SearchLoop"),
    "PolicyLoop": ("hillclimb.harness.loop", "PolicyLoop"),
    "Harness": ("hillclimb.harness.loop", "Harness"),
    "SearchInfo": ("hillclimb.harness.loop", "SearchInfo"),
    "Ticket": ("hillclimb.harness.loop", "Ticket"),
    "Outcome": ("hillclimb.harness.loop", "Outcome"),
    "HarnessClosed": ("hillclimb.harness.loop", "HarnessClosed"),
    "ClimberError": ("hillclimb.harness.loop", "ClimberError"),
    # the records a climber reads (always holdout-blind copies)
    "Candidate": ("hillclimb.harness.candidate", "Candidate"),
    "Trial": ("hillclimb.harness.candidate", "Trial"),
    "Replicate": ("hillclimb.harness.candidate", "Replicate"),
    "PolicyJournal": ("hillclimb.harness.journal", "PolicyJournal"),
    "source_hash": ("hillclimb.harness.candidate", "source_hash"),
    "EvalResult": ("hillclimb.harness.evaluation", "EvalResult"),
    "eval_result_for": ("hillclimb.harness.evaluation", "eval_result_for"),
    # a loop may end its search as resumable ("parked") instead of failed
    "ParkedSearch": ("hillclimb.harness.glue", "ParkedSearch"),
    # the accept rule, so a policy agrees with the harness on what "better" means
    "improves": ("hillclimb.harness.evaluation", "improves"),
    "accept_band": ("hillclimb.harness.evaluation", "accept_band"),
    # how one attempt is made
    "Operator": ("hillclimb.modules.operators.base", "Operator"),
    "OperatorContext": ("hillclimb.modules.operators.base", "OperatorContext"),
    "Preparation": ("hillclimb.modules.operators.base", "Preparation"),
    "ProblemInfo": ("hillclimb.modules.operators.base", "ProblemInfo"),
    "MemoryContext": ("hillclimb.modules.operators.base", "MemoryContext"),
    "inspiration_filename": ("hillclimb.modules.operators.base", "inspiration_filename"),
    "ROLES": ("hillclimb.modules.operators.base", "ROLES"),
    # which parameter values next
    "Tuner": ("hillclimb.modules.tuners.base", "Tuner"),
    "Observation": ("hillclimb.modules.tuners.base", "Observation"),
    "ParamSpace": ("hillclimb.harness.params", "ParamSpace"),
    "ParamSpec": ("hillclimb.harness.params", "ParamSpec"),
    "coerce": ("hillclimb.harness.params", "coerce"),
    # how alike two solutions are
    "SimilarityScore": ("hillclimb.modules.similarity.base", "SimilarityScore"),
    "SimilarityUnavailable": ("hillclimb.modules.similarity.base", "SimilarityUnavailable"),
    "Solution": ("hillclimb.modules.similarity.base", "Solution"),
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
    from hillclimb.harness.candidate import Candidate, Replicate, Trial, source_hash
    from hillclimb.harness.evaluation import EvalResult, accept_band, eval_result_for, improves
    from hillclimb.harness.journal import PolicyJournal
    from hillclimb.harness.loop import (
        ClimberError,
        Harness,
        HarnessClosed,
        Outcome,
        PolicyLoop,
        SearchInfo,
        SearchLoop,
        Ticket,
    )
    from hillclimb.modules.operators.base import (
        ROLES,
        MemoryContext,
        Operator,
        OperatorContext,
        Preparation,
        ProblemInfo,
        inspiration_filename,
    )
    from hillclimb.harness.params import ParamSpace, ParamSpec, coerce
    from hillclimb.modules.policies.base import (
        INJECT_ACTION,
        TUNE_ACTION,
        Action,
        BudgetView,
        InflightRef,
        PolicyInput,
        Route,
        SearchPolicy,
    )
    from hillclimb.modules.similarity.base import SimilarityScore, SimilarityUnavailable, Solution
    from hillclimb.harness.glue import ParkedSearch
    from hillclimb.modules.tuners.base import Observation, Tuner
