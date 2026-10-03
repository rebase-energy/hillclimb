"""Operator registry: name -> class, mirroring policies and tuners.

The registry is the harness's operator vocabulary: `candidate.OPERATORS`,
`climber check`'s known operators and the route preflight all derive from it,
so registering an operator is the only step that makes it usable.
"""

from __future__ import annotations

from collections.abc import Mapping

from hillclimb.modules.operators.base import (
    CONTRACT_TOKEN,
    OPERATOR_KINDS,
    RESERVED_KINDS,
    MemoryContext,
    Operator,
    OperatorContext,
    OperatorServices,
    Attempt,
    ProblemInfo,
    inspiration_filename,
)
from hillclimb.modules import refs
from hillclimb.modules.operators.builtin import BUILTIN_OPERATORS

# the registry itself lives with every other kind's, in modules/refs.py
_OPERATORS: dict[str, type[Operator]] = refs.KINDS["operator"].registry
for _builtin in BUILTIN_OPERATORS:
    refs.register("operator", _builtin.name, _builtin)


def register_operator(cls: type[Operator]) -> type[Operator]:
    if cls.kind not in OPERATOR_KINDS:
        raise ValueError(f"operator {cls.name!r}: kind {cls.kind!r} is not one of {OPERATOR_KINDS}")
    refs.register("operator", cls.name, cls)
    return cls


def operator_names() -> tuple[str, ...]:
    """Every coding agent operator the harness can run, in registration order."""
    return tuple(_OPERATORS)


def get_operator(name: str, params: Mapping | None = None) -> Operator:
    try:
        return _OPERATORS[name](params)
    except KeyError:
        raise ValueError(f"Unknown operator {name!r}. Available: {sorted(_OPERATORS)}") from None


def operator_kind(operator: str) -> str | None:
    """The kind an operator's candidates carry (None for an unknown name —
    e.g. a record written by a climber whose operators are not loaded)."""
    if operator in RESERVED_KINDS:
        return RESERVED_KINDS[operator]
    cls = _OPERATORS.get(operator)
    return cls.kind if cls is not None else None


__all__ = [
    "CONTRACT_TOKEN",
    "OPERATOR_KINDS",
    "RESERVED_KINDS",
    "MemoryContext",
    "Operator",
    "OperatorContext",
    "OperatorServices",
    "Attempt",
    "ProblemInfo",
    "get_operator",
    "inspiration_filename",
    "operator_kind",
    "operator_names",
    "register_operator",
]
