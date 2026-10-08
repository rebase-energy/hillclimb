"""hillclimb.sdk — the one import a climber needs.

A climber is the block of exchangeable modules (an `OperatorPolicy` with its
`SelectorPolicy`, or a `Loop`; operators, memory, a tuner) that decides
HOW to hillclimb. Everything else — running coding agents, scoring, holdout, the
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
    "SearchState": ("hillclimb.modules.policies.base", "SearchState"),
    "OperatorPolicy": ("hillclimb.modules.policies.base", "OperatorPolicy"),
    "Policy": ("hillclimb.modules.policies.base", "Policy"),  # the pre-0.7 name
    "TUNE_ACTION": ("hillclimb.modules.policies.base", "TUNE_ACTION"),
    "INJECT_ACTION": ("hillclimb.modules.policies.base", "INJECT_ACTION"),
    # control flow over the harness (most climbers only need an OperatorPolicy)
    "Loop": ("hillclimb.harness.loop", "Loop"),
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
    "JournalView": ("hillclimb.harness.journal", "JournalView"),
    "source_hash": ("hillclimb.harness.candidate", "source_hash"),
    "EvalResult": ("hillclimb.harness.evaluation", "EvalResult"),
    "eval_result_for": ("hillclimb.harness.evaluation", "eval_result_for"),
    # a loop may end its search as resumable ("parked") instead of failed
    "ParkedSearch": ("hillclimb.harness.glue", "ParkedSearch"),
    # the accept rule, so a policy agrees with the harness on what "better" means
    "improves": ("hillclimb.harness.evaluation", "improves"),
    "accept_band": ("hillclimb.harness.evaluation", "accept_band"),
    # which candidate to expand
    "SelectorPolicy": ("hillclimb.modules.selectors.base", "SelectorPolicy"),
    "Selector": ("hillclimb.modules.selectors.base", "Selector"),  # the pre-0.7 name
    "Selection": ("hillclimb.modules.selectors.base", "Selection"),
    "improvable": ("hillclimb.modules.selectors.base", "improvable"),
    "top_distinct": ("hillclimb.modules.selectors.base", "top_distinct"),
    # how one attempt is made
    "Operator": ("hillclimb.modules.operators.base", "Operator"),
    "OperatorContext": ("hillclimb.modules.operators.base", "OperatorContext"),
    "Attempt": ("hillclimb.modules.operators.base", "Attempt"),
    "ProblemInfo": ("hillclimb.modules.operators.base", "ProblemInfo"),
    "MemoryContext": ("hillclimb.modules.operators.base", "MemoryContext"),
    "inspiration_filename": ("hillclimb.modules.operators.base", "inspiration_filename"),
    "OPERATOR_KINDS": ("hillclimb.modules.operators.base", "OPERATOR_KINDS"),
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
    # cross-search memory: what a search knows from others and leaves for the next
    "Memory": ("hillclimb.modules.memory.base", "Memory"),
    "MemoryEnv": ("hillclimb.modules.memory.base", "MemoryEnv"),
    "Retrieved": ("hillclimb.modules.memory.base", "Retrieved"),
    # ...and the graph module that indexes one, with the model it builds
    "GraphModule": ("hillclimb.modules.memory.base", "GraphModule"),
    "KnowledgeGraph": ("hillclimb.modules.memory.base", "KnowledgeGraph"),
    "GraphNode": ("hillclimb.modules.memory.base", "GraphNode"),
    "GraphEdge": ("hillclimb.modules.memory.base", "GraphEdge"),
}

__all__ = sorted(_LAZY)

# renamed, with the version that did it; there are no aliases, the old name says where it went
_RENAMED = {
    "SearchPolicy": ("OperatorPolicy", "0.6"),
    "SearchLoop": ("Loop", "0.6"),
    "PolicyInput": ("SearchState", "0.6"),
    "PolicyJournal": ("JournalView", "0.6"),
    "Preparation": ("Attempt", "0.6"),
    "ROLES": ("OPERATOR_KINDS", "0.7"),
}


def __getattr__(name: str):
    if name in _RENAMED:
        new, version = _RENAMED[name]
        raise ImportError(f"hillclimb.sdk.{name} was renamed {new} in hillclimb {version}")
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
    from hillclimb.harness.journal import JournalView
    from hillclimb.harness.loop import (
        ClimberError,
        Harness,
        HarnessClosed,
        Outcome,
        PolicyLoop,
        SearchInfo,
        Loop,
        Ticket,
    )
    from hillclimb.modules.operators.base import (
        OPERATOR_KINDS,
        MemoryContext,
        Operator,
        OperatorContext,
        Attempt,
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
        SearchState,
        Route,
        OperatorPolicy,
        Policy,
    )
    from hillclimb.modules.similarity.base import SimilarityScore, SimilarityUnavailable, Solution
    from hillclimb.modules.selectors.base import Selection, Selector, SelectorPolicy, improvable, top_distinct
    from hillclimb.modules.memory.base import (
        GraphEdge,
        GraphModule,
        GraphNode,
        KnowledgeGraph,
        Memory,
        MemoryEnv,
        Retrieved,
    )
    from hillclimb.harness.glue import ParkedSearch
    from hillclimb.modules.tuners.base import Observation, Tuner
