"""Tuners: WHICH parameter values a tunable candidate tries next.

`RandomSearch(seed=7)`; `Optuna()` is TPE (extra: optuna).

The building blocks, by name — `import hillclimb as hc; hc.tuners.RandomSearch` —
for composing a climber in Python (`hillclimb.Climber`). The classes live
under `hillclimb.modules` (and are what a block's `tuner:` names); this
module is a lazy window onto them, so importing it costs nothing until a
class is used. A climber's own files import the contracts from
`hillclimb.sdk`.
"""

from __future__ import annotations

_LAZY = {
    "Tuner": ("hillclimb.modules.tuners.base", "Tuner"),
    "RandomSearch": ("hillclimb.modules.tuners.random_search", "RandomSearch"),
    "Optuna": ("hillclimb.modules.tuners.optuna", "Optuna"),
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
