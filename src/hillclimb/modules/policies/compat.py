"""Policies that exist only so searches started before 0.6 still resume.

A search's snapshot names its policy by `module:Class`; `_moved.py` maps the
class paths those snapshots hold onto the ones here. The base they extend is
a catalog file now (`climbers/openevolve/policy.py`), so the class is built
on first use, from that file. Nothing new should name these.
"""

from __future__ import annotations

from collections.abc import Mapping


def _openevolve_policy() -> type:
    from hillclimb import catalog

    module = catalog.module("openevolve")

    class OpenEvolvePolicy(module.Greedy):
        """0.4/0.5's `openevolve` policy: greedy's schedule without ensemble or
        tune, over MAP-Elites — whose settings lived among the policy's own
        params then, so they are read from there."""

        name = "openevolve"
        strict_params = False  # MAP-Elites' settings are in here too

        def __init__(self, params: Mapping | None = None, selector=None):
            super().__init__(params, selector)
            if self._selector is None:
                mine = {key: value for key, value in self.params.items() if key in module.known_params()}  # the schedule's too
                self._selector = module.MapElites({"ensemble": False, **mine})

    return OpenEvolvePolicy


def __getattr__(name: str):
    if name == "OpenEvolvePolicy":
        cls = _openevolve_policy()
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
