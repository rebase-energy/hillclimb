"""Policies that exist only so searches started before 0.6 still resume.

A search's snapshot names its policy by `module:Class`; `_moved.py` maps the
class paths those snapshots hold onto the ones here. Nothing new should name
these: a block composes `operator_policy: greedy` with a `selector_policy:`.
"""

from __future__ import annotations

from collections.abc import Mapping

from hillclimb.modules.policies.greedy import Greedy


class OpenEvolvePolicy(Greedy):
    """0.4/0.5's `openevolve` policy: greedy's schedule without ensemble or
    tune, over MAP-Elites — whose settings lived among the policy's own
    params then, so they are read from there."""

    name = "openevolve"
    DEFAULTS = {"tune_budget": 0}
    default_selector = "map-elites"
    strict_params = False  # MAP-Elites' settings are in here too

    def __init__(self, params: Mapping | None = None, selector=None):
        super().__init__(params, selector)
        if self._selector is None:
            from hillclimb.modules.selectors import get_selector
            from hillclimb.modules.selectors.map_elites import known_params

            mine = {key: value for key, value in self.params.items() if key in known_params()}  # the schedule's too
            self._selector = get_selector("map-elites", {"ensemble": False, **mine})
