"""Search policies: WHAT to try next.

`base.py` is the contract (`Policy`, `Action`, `SearchState`), `greedy.py`
the bundled schedule, `check.py` the conformance check behind `hillclimb
climber check`, `compat.py` what pre-0.6 snapshots still name. A policy is
loaded as part of a climber (`hillclimb.climber`), by registry name
(`greedy`), a `.py` file or `module:Class` (`modules/refs.py`).
"""

from __future__ import annotations

from hillclimb.modules import refs
from hillclimb.modules.policies.base import Policy
from hillclimb.modules.policies.greedy import Greedy

refs.register("policy", "greedy", Greedy)

__all__ = ["Greedy", "Policy"]
