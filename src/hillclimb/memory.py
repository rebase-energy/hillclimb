"""Memory: what a search knows from other searches, and leaves for the next.

`FilesMemory(max_cards=1)` is the knowledge/ directory; `NoMemory()` keeps nothing.

The building blocks, by name — `import hillclimb as hc; hc.memory.FilesMemory` —
for composing a climber in Python (`hillclimb.Climber`). The classes live
under `hillclimb.modules` (and are what a block's `memory:` names); this
module is a lazy window onto them, so importing it costs nothing until a
class is used. A climber's own files import the contracts from
`hillclimb.sdk`.
"""

from __future__ import annotations

_LAZY = {
    "Memory": ("hillclimb.modules.memory.base", "Memory"),
    "Retrieved": ("hillclimb.modules.memory.base", "Retrieved"),
    "FilesMemory": ("hillclimb.modules.memory.files", "FilesMemory"),
    "NoMemory": ("hillclimb.modules.memory.files", "NoMemory"),
    "GraphModule": ("hillclimb.modules.memory.base", "GraphModule"),
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
