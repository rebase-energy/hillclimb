"""Tuner registry: name -> factory, mirroring policies.get_policy."""

from __future__ import annotations

from typing import Callable

from hillclimb.tuner import Tuner


def _make_random(params: dict) -> Tuner:
    from hillclimb.tuners.random_search import RandomTuner

    return RandomTuner(params)


def _make_optuna(params: dict) -> Tuner:
    from hillclimb.tuners.optuna import OptunaTuner  # optional extra

    return OptunaTuner(params)


_TUNERS: dict[str, Callable[[dict], Tuner]] = {
    "random": _make_random,
    "optuna": _make_optuna,
}


def get_tuner(name: str, params: dict | None = None) -> Tuner:
    if name not in _TUNERS:
        raise ValueError(f"Unknown tuner: {name} (available: {', '.join(sorted(_TUNERS))})")
    return _TUNERS[name](params or {})
