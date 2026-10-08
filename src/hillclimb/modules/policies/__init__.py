"""Search policies: WHAT to try next.

`base.py` is the contract (`OperatorPolicy`, `Action`, `SearchState`),
`check.py` the conformance check behind `hillclimb climber check`,
`compat.py` what pre-0.6 snapshots still name. The bundled policies live
with their selector policies, one file per climber
(`hillclimb/climbers/<name>/policy.py` — what `hillclimb climber get` copies);
the registry names here are lazy strings so importing this package never
imports a climber. A policy is loaded as part of a climber
(`hillclimb.climber`), by registry name (`greedy`), a `.py` file or
`module:Class` (`modules/refs.py`).
"""

from __future__ import annotations

from hillclimb.modules import refs
from hillclimb.modules.policies.base import OperatorPolicy

# nothing is registered: the bundled operator policies are catalog files
# (`hillclimb climber get greedy`), and a record's pre-0.9 name resolves
# through `catalog.RECORDED`

__all__ = ["OperatorPolicy"]
