"""Operators: HOW one attempt is made — the prompt a coding agent is given.

`Draft(retrieval=False)`, `Debug()`, `Improve()`, `Ensemble()`; subclass `Operator` for your own.

The building blocks, by name — `import hillclimb as hc; hc.operators.Draft` —
for composing a climber in Python (`hillclimb.Climber`). The classes live
under `hillclimb.modules` (and are what a block's `operators:` names); this
module is a lazy window onto them, so importing it costs nothing until a
class is used. A climber's own files import the contracts from
`hillclimb.sdk`.
"""

from __future__ import annotations

_LAZY = {
    "Operator": ("hillclimb.modules.operators.base", "Operator"),
    "Attempt": ("hillclimb.modules.operators.base", "Attempt"),
    "Draft": ("hillclimb.modules.operators.builtin", "Draft"),
    "Debug": ("hillclimb.modules.operators.builtin", "Debug"),
    "Improve": ("hillclimb.modules.operators.builtin", "Improve"),
    "Ensemble": ("hillclimb.modules.operators.builtin", "Ensemble"),
    "GepaReflect": ("hillclimb.climbers.gepa.operator", "GepaReflect"),
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
