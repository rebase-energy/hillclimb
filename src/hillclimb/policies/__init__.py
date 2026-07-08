"""Policy registry: name -> factory, mirroring backends.get_backend."""

from __future__ import annotations

from typing import Callable

from hillclimb.policies.greedy import GreedyPolicy
from hillclimb.policy import SearchPolicy


def _make_greedy(params: dict, *, complexity_start: int = 0) -> GreedyPolicy:
    return GreedyPolicy(complexity_start=complexity_start, params=params)


_POLICIES: dict[str, Callable[..., SearchPolicy]] = {
    "greedy": _make_greedy,
}


def get_policy(
    name: str, params: dict | None = None, *, complexity_start: int = 0
) -> SearchPolicy:
    if name not in _POLICIES:
        raise ValueError(f"Unknown policy: {name} (available: {', '.join(sorted(_POLICIES))})")
    return _POLICIES[name](params or {}, complexity_start=complexity_start)
