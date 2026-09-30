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
    from hillclimb.harness.loop import Loop


class ParkedSearch(Exception):
    """Raised when the agent hits a rate limit; the search can be resumed."""


class StopRequested(Exception):
    """Raised on a graceful stop (control command or SIGTERM); the search can
    be resumed."""


def search_climber(config: Config, search_dir=None) -> Climber:
    """The climber this config names (relative paths resolve from the folder
    holding the hillclimb dir) — or, for a search that already exists, the
    snapshot it was started with. A `ClimberLoadError` names the unknowns."""
    from hillclimb.climber import climber_base_dir, climber_label, load_climber, load_snapshot

    if search_dir is not None:
        snapshot = load_snapshot(search_dir, name=climber_label(config.climber.ref))
        if snapshot is not None:
            return snapshot
    return load_climber(config.climber.ref, climber_base_dir(config))


def is_loop_climber(name: str, config: Config | None = None) -> bool:
    from hillclimb.climber import climber_base_dir, load_climber

    return load_climber(name, climber_base_dir(config) if config is not None else None).is_loop


def holdout_timing(config: Config, search_dir=None) -> str:
    """When the harness scores the hidden split: the user's `holdout.timing`,
    tightened to `after` for a climber that asks for it."""
    asked = search_climber(config, search_dir).holdout_timing
    return asked or config.holdout.timing


def build_tuner(config: Config, search_dir=None):
    """The climber's tuner, unless the user named one (`search.tuner`)."""
    from hillclimb.climber import climber_base_dir
    from hillclimb.modules.tuners import get_tuner

    climber = search_climber(config, search_dir)
    if config.climber.tuner is not None:  # the user named one: it wins
        return get_tuner(config.climber.tuner, config.climber.tuner_params, base_dir=climber_base_dir(config))
    spec = climber.spec
    return get_tuner(spec.tuner, {**spec.tuner_params, **config.climber.tuner_params}, scope=climber.scope)


def build_loop(config: Config, *, complexity_start: int = 0, log=print, search_dir=None) -> Loop:
    return search_climber(config, search_dir).build_loop(
        params=config.climber.params,  # the user's overlay on the manifest's params
        complexity_start=complexity_start,
        parallelism=max(1, config.concurrency.parallel_agents),
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


def build_graph_module(config: Config, search_dir=None, log=None):
    """The climber's graph module (`graph:` in its manifest), unless the user
    named one (`climber.graph`: a registry name, a .py file relative to the
    folder holding the hillclimb dir, or module:Class). Outside a search —
    `hillclimb knowledge …`, the graph TUI — there is no search_dir and the
    manifest is `climber.ref`'s; a climber that will not load there falls
    back to the built-in module, so reading memory never depends on it."""
    from hillclimb.climber import ClimberLoadError
    from hillclimb.modules.memory.base import DEFAULT_GRAPH
    from hillclimb.modules.memory.graphs import get_graph
    from hillclimb.climber import climber_base_dir

    if config.climber.graph is not None:  # the user named one: it wins
        return get_graph(config.climber.graph, base_dir=climber_base_dir(config))
    try:
        return search_climber(config, search_dir).graph_module()
    except ClimberLoadError as exc:
        if search_dir is not None:
            raise
        if log is not None:
            log(f"graph: {exc}; using {DEFAULT_GRAPH}")
        return get_graph(DEFAULT_GRAPH)


def effective_memory(config: Config, search_dir=None) -> str:
    """`files` or `none`: the user's `climber.memory`, else the
    manifest's — and always `none` when learning is switched off."""
    if not config.learning.enabled:
        return "none"
    return config.climber.memory or search_climber(config, search_dir).spec.memory
