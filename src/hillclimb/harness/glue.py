"""The glue from config to climber: which climber drives a search, and the two
ways a search ends early (this module was `search_strategy.py`).

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
    from hillclimb.harness.loop import SearchLoop


class ParkedSearch(Exception):
    """Raised when the backend hits a rate limit; the search can be resumed."""


class StopRequested(Exception):
    """Raised on a graceful stop (control command or SIGTERM); the search can
    be resumed."""


def search_climber(config: Config, search_dir=None) -> Climber:
    """The climber this config names (relative paths resolve from the folder
    holding the hillclimb dir) — or, for a search that already exists, the
    snapshot it was started with. A `ClimberLoadError` names the unknowns."""
    from hillclimb.climber import load_climber, load_snapshot
    from hillclimb.modules.policies import policy_base_dir, policy_label

    if search_dir is not None:
        snapshot = load_snapshot(search_dir, name=policy_label(config.climber.ref))
        if snapshot is not None:
            return snapshot
    return load_climber(config.climber.ref, policy_base_dir(config))


def is_loop_climber(name: str, config: Config | None = None) -> bool:
    from hillclimb.climber import load_climber
    from hillclimb.modules.policies import policy_base_dir

    return load_climber(name, policy_base_dir(config) if config is not None else None).is_loop


def holdout_timing(config: Config, search_dir=None) -> str:
    """When the harness scores the hidden split: the user's `holdout.timing`,
    tightened to `after` for a climber that asks for it."""
    asked = search_climber(config, search_dir).manifest.holdout_timing
    return asked or config.holdout.timing


def build_tuner(config: Config, search_dir=None):
    """The climber's tuner, unless the user named one (`search.tuner`)."""
    from hillclimb.modules.tuners import get_tuner

    manifest = search_climber(config, search_dir).manifest
    if config.climber.tuner is not None:  # the user named one: it wins
        return get_tuner(config.climber.tuner, config.climber.tuner_params)
    return get_tuner(manifest.tuner, {**manifest.tuner_params, **config.climber.tuner_params})


def build_loop(config: Config, *, complexity_start: int = 0, log=print, search_dir=None) -> SearchLoop:
    return search_climber(config, search_dir).build_loop(
        params=config.climber.params,  # the user's overlay on the manifest's params
        complexity_start=complexity_start,
        parallelism=max(1, config.concurrency.parallel_operators),
        log=log,
    )


def build_operators(config: Config, search_dir=None) -> OperatorSet:
    """The climber's operators, with the user's per-operator params
    (`climber.operators: {draft: {retrieval: false}}`) laid over them."""
    from hillclimb.climber import OperatorSet

    entries = dict(search_climber(config, search_dir).operator_set()._entries)
    for name, overlay in config.climber.operators.items():
        if name not in entries:
            raise ValueError(
                f"climber.operators.{name}: this climber has no operator {name!r} "
                f"(it has {sorted(entries)})"
            )
        operator_cls, params = entries[name]
        entries[name] = (operator_cls, {**params, **overlay})
    return OperatorSet(entries)


def effective_memory(config: Config, search_dir=None) -> str:
    """`files` or `none`: the user's `climber.memory`, else the
    manifest's — and always `none` when learning is switched off."""
    if not config.learning.enabled:
        return "none"
    return config.climber.memory or search_climber(config, search_dir).manifest.memory
