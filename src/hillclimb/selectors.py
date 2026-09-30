"""Selectors: WHICH candidate a policy expands next.

`Best()` is best-first; `MapElites(num_islands=3)` a quality-diversity archive (extra: openevolve).

The building blocks, by name — `import hillclimb as hc; hc.selectors.Best` —
for composing a climber in Python (`hillclimb.Climber`). The classes live
under `hillclimb.modules` (and are what a block's `select:` names); this
module is a lazy window onto them, so importing it costs nothing until a
class is used. A climber's own files import the contracts from
`hillclimb.sdk`.
"""

from __future__ import annotations

_LAZY = {
    "Selector": ("hillclimb.modules.selectors.base", "Selector"),
    "Selection": ("hillclimb.modules.selectors.base", "Selection"),
    "Best": ("hillclimb.modules.selectors.best", "Best"),
    "MapElites": ("hillclimb.modules.selectors.map_elites", "MapElites"),
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
