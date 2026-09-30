"""Operator registry: name -> class, mirroring policies and tuners.

The registry is the harness's operator vocabulary: `candidate.OPERATORS`,
`policy check`'s known operators and the route preflight all derive from it,
so registering an operator is the only step that makes it usable.
"""

from __future__ import annotations

from collections.abc import Mapping

from hillclimb.modules.operators.base import (
    CONTRACT_TOKEN,
    RESERVED_ROLES,
    ROLES,
    MemoryContext,
    Operator,
    OperatorContext,
    OperatorServices,
    Attempt,
    ProblemInfo,
    inspiration_filename,
)
from hillclimb.modules.operators.builtin import BUILTIN_OPERATORS

_OPERATORS: dict[str, type[Operator]] = {cls.name: cls for cls in BUILTIN_OPERATORS}


def register_operator(cls: type[Operator]) -> type[Operator]:
    if cls.role not in ROLES:
        raise ValueError(f"operator {cls.name!r}: role {cls.role!r} is not one of {ROLES}")
    _OPERATORS[cls.name] = cls
    return cls


def operator_names() -> tuple[str, ...]:
    """Every agent operator the harness can run, in registration order."""
    return tuple(_OPERATORS)


def get_operator(name: str, params: Mapping | None = None) -> Operator:
    try:
        return _OPERATORS[name](params)
    except KeyError:
        raise ValueError(f"Unknown operator {name!r}. Available: {sorted(_OPERATORS)}") from None


def role_of(operator: str) -> str | None:
    """The role an operator's candidates carry (None for an unknown name —
    e.g. a record written by a climber whose operators are not loaded)."""
    if operator in RESERVED_ROLES:
        return RESERVED_ROLES[operator]
    cls = _OPERATORS.get(operator)
    return cls.role if cls is not None else None


__all__ = [
    "CONTRACT_TOKEN",
    "RESERVED_ROLES",
    "ROLES",
    "MemoryContext",
    "Operator",
    "OperatorContext",
    "OperatorServices",
    "Attempt",
    "ProblemInfo",
    "get_operator",
    "inspiration_filename",
    "operator_names",
    "register_operator",
    "role_of",
]
