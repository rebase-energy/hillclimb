"""Which loop drives a search, and the two ways a search ends early.

Every climber runs the same way: `Harness.execute(loop)`. A policy-driven
climber gets the built-in `PolicyLoop` around its `SearchPolicy`; a climber
that owns its control flow brings its own `SearchLoop` (`gepa`). The harness
is identical for both — budgets, the cost ceiling, stop/park, the journal and
holdout are never a loop's concern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hillclimb.config import Config
    from hillclimb.loop import SearchLoop


class ParkedSearch(Exception):
    """Raised when the backend hits a rate limit; the search can be resumed."""


class StopRequested(Exception):
    """Raised on a graceful stop (control command or SIGTERM); the search can
    be resumed."""


# climbers that bring their own SearchLoop instead of a SearchPolicy; imported
# lazily so a plain install never imports an optional library
LOOP_CLIMBERS = ("gepa",)
# a loop whose state must never meet a holdout value has the hidden split
# scored only after it returns (the mask already hides the scores; `after`
# also keeps a failed hidden split from surfacing as a verdict mid-search)
_HOLDOUT_AFTER = ("gepa",)


def is_loop_climber(name: str) -> bool:
    return name in LOOP_CLIMBERS


def holdout_timing(config: Config) -> str:
    """When the harness scores the hidden split: the user's `holdout.timing`,
    tightened to `after` for a climber that asks for it."""
    return "after" if config.search.policy in _HOLDOUT_AFTER else config.holdout.timing


def build_loop(config: Config, *, complexity_start: int = 0, log=print) -> SearchLoop:
    """The loop `config.search.policy` names (a ValueError names the unknowns)."""
    name = config.search.policy
    if name == "gepa":
        from hillclimb.integrations.gepa import build_gepa_loop

        return build_gepa_loop(config, log=log)
    from hillclimb.loop import PolicyLoop
    from hillclimb.policies import get_policy, policy_base_dir

    return PolicyLoop(
        get_policy(
            name, config.search.policy_params,
            complexity_start=complexity_start, base_dir=policy_base_dir(config),
        )
    )
