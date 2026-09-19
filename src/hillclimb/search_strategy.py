"""Which climber drives a search, and the two ways a search ends early.

Every climber runs the same way: `Harness.execute(loop)`. `search_climber`
resolves the configured reference (a bundled name, a directory, one file —
see `hillclimb.climber`); the climber builds its loop and brings its
operators and prompts. The harness is identical for all of them — budgets,
the cost ceiling, stop/park, the journal and holdout are never a climber's
concern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hillclimb.climber import Climber, OperatorSet
    from hillclimb.config import Config
    from hillclimb.loop import SearchLoop


class ParkedSearch(Exception):
    """Raised when the backend hits a rate limit; the search can be resumed."""


class StopRequested(Exception):
    """Raised on a graceful stop (control command or SIGTERM); the search can
    be resumed."""


def search_climber(config: Config) -> Climber:
    """The climber this config names (relative paths resolve from the folder
    holding the hillclimb dir). A `ClimberLoadError` names the unknowns."""
    from hillclimb.climber import load_climber
    from hillclimb.policies import policy_base_dir

    return load_climber(config.search.policy, policy_base_dir(config))


def is_loop_climber(name: str, config: Config | None = None) -> bool:
    from hillclimb.climber import load_climber
    from hillclimb.policies import policy_base_dir

    return load_climber(name, policy_base_dir(config) if config is not None else None).is_loop


def holdout_timing(config: Config) -> str:
    """When the harness scores the hidden split: the user's `holdout.timing`,
    tightened to `after` for a climber that asks for it."""
    asked = search_climber(config).manifest.holdout_timing
    return asked or config.holdout.timing


def _user_params(config: Config) -> dict:
    """What the USER set, to lay over the manifest's params: every
    `search.policy_params` key, plus the pre-manifest config blocks where
    they differ from their defaults (so a manifest's own value is not
    overwritten by a default the user never touched)."""
    from hillclimb.config import Config as ConfigModel
    from hillclimb.policies import ConfigBackedParams

    defaults = ConfigBackedParams(ConfigModel())
    blocks = ConfigBackedParams(config)
    touched = {name: blocks[name] for name in ConfigBackedParams._BLOCKS if blocks[name] != defaults[name]}
    return {**touched, **config.search.policy_params}


def build_loop(config: Config, *, complexity_start: int = 0, log=print) -> SearchLoop:
    return search_climber(config).build_loop(
        params=_user_params(config),
        complexity_start=complexity_start,
        parallelism=max(1, config.search.parallel_operators),
        log=log,
    )


def build_operators(config: Config) -> OperatorSet:
    """The climber's operators, with the pre-manifest `operators:` config
    switches laid over them where the user turned one off."""
    from hillclimb.climber import OperatorSet

    operators = search_climber(config).operator_set()
    switches = {"draft": ("retrieval", config.operators.draft_retrieval),
                "improve": ("ablation", config.operators.improve_ablation)}
    entries = dict(operators._entries)
    for name, (key, enabled) in switches.items():
        if name in entries and not enabled:
            operator_cls, params = entries[name]
            entries[name] = (operator_cls, {**params, key: False})
    return OperatorSet(entries)
