"""Search policies: WHAT to try next.

`base.py` is the contract (`Policy`, `Action`, `SearchState`), `greedy.py`
and `openevolve.py` the bundled implementations, `check.py` the conformance
check behind `hillclimb climber check`. A policy is loaded as part of a
climber (`hillclimb.climber`), by registry name (`greedy`, `openevolve`),
a `.py` file or `module:Class` (`modules/refs.py`).
"""

from __future__ import annotations

from hillclimb.modules import refs
from hillclimb.modules.policies.base import Policy
from hillclimb.modules.policies.greedy import GreedyPolicy

refs.register("policy", "greedy", GreedyPolicy)
refs.register("policy", "openevolve", "hillclimb.modules.policies.openevolve:OpenEvolvePolicy")  # optional extra

__all__ = ["GreedyPolicy", "Policy"]
