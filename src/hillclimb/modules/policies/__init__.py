"""Search policies: WHAT to try next.

`base.py` is the contract (`Policy`, `Action`, `SearchState`), `greedy.py`
and `openevolve.py` the bundled implementations, `check.py` the conformance
check behind `hillclimb climber check`. A policy is loaded as part of a
climber (`hillclimb.climber`) — there is no separate policy registry.
"""

from __future__ import annotations

from hillclimb.modules.policies.base import Policy
from hillclimb.modules.policies.greedy import GreedyPolicy

__all__ = ["GreedyPolicy", "Policy"]
