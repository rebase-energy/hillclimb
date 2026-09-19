"""A search = a `Harness` plus the `SearchLoop` that drives it.

`PolicySearch` is what `build_search_strategy` hands `api.execute_search`
for every policy-driven climber: the fixed core (`hillclimb.harness`) and the
built-in `PolicyLoop` around the named `SearchPolicy`.
"""

from __future__ import annotations

from hillclimb.candidate import Candidate
from hillclimb.evaluation import TAIL_CHARS, tail  # noqa: F401 — re-exported (cli imports tail from here)
from hillclimb.harness.core import Harness, Job, OutcomeMsg  # noqa: F401 — re-exported
from hillclimb.loop import PolicyLoop, SearchLoop
from hillclimb.policy import SearchPolicy
from hillclimb.search_strategy import ParkedSearch, StopRequested  # noqa: F401 — re-exported (their historic home)


class PolicySearch:
    """The `SearchStrategy` for a policy-driven search."""

    def __init__(self, harness: Harness, loop: SearchLoop):
        self.harness = harness
        self.loop = loop

    @classmethod
    def with_policy(cls, policy: SearchPolicy, **harness_kwargs) -> PolicySearch:
        return cls(Harness(**harness_kwargs), PolicyLoop(policy))

    @property
    def journal(self):
        return self.harness.journal

    def run(self) -> Candidate | None:
        return self.harness.execute(self.loop)

    def total_cost_usd(self) -> float:
        return self.harness.total_cost_usd()
