"""Policies: WHAT to try next — the schedule of a search.

`Greedy(num_drafts=5)` is the bundled schedule; subclass `Policy` for your own.

The building blocks, by name — `import hillclimb as hc; hc.policies.Greedy` —
for composing a climber in Python (`hillclimb.Climber`). The classes live
under `hillclimb.modules` (and are what a block's `policy:` names); this
module is a lazy window onto them, so importing it costs nothing until a
class is used. A climber's own files import the contracts from
`hillclimb.sdk`.
"""

from __future__ import annotations

_LAZY = {
    "Policy": ("hillclimb.modules.policies.base", "Policy"),
    "Greedy": ("hillclimb.modules.policies.greedy", "Greedy"),
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
