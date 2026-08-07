"""hillclimb: greedy search over agent-drafted solutions.

Public API (lazy — `import hillclimb` stays dependency-light):

    hillclimb.run_search(target, budget_s=...) -> SearchOutcome
    hillclimb.Config
"""

__version__ = "0.2.0"

__all__ = ["__version__", "Config", "SearchOutcome", "run_search", "execute_search"]

_LAZY = {
    "Config": ("hillclimb.config", "Config"),
    "SearchOutcome": ("hillclimb.api", "SearchOutcome"),
    "run_search": ("hillclimb.api", "run_search"),
    "execute_search": ("hillclimb.api", "execute_search"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
