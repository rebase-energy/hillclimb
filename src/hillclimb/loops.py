"""Loops: the control flow of a search, for climbers that own it.

Most climbers are a policy on the built-in `PolicyLoop`; `GepaLoop` lets gepa drive (extra: gepa).

The building blocks, by name — `import hillclimb as hc; hc.loops.PolicyLoop` —
for composing a climber in Python (`hillclimb.Climber`). The classes live
under `hillclimb.modules` (and are what a block's `loop:` names); this
module is a lazy window onto them, so importing it costs nothing until a
class is used. A climber's own files import the contracts from
`hillclimb.sdk`.
"""

from __future__ import annotations

_LAZY = {
    "Loop": ("hillclimb.harness.loop", "Loop"),
    "PolicyLoop": ("hillclimb.harness.loop", "PolicyLoop"),
    "GepaLoop": ("hillclimb.climbers.gepa.loop", "GepaLoop"),
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
