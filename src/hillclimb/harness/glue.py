"""The glue from config to climber: which climber drives a search, and the two
ways a search ends early (this module was `search_strategy.py`).

Every climber runs the same way: `Harness.execute(loop)`. `search_climber`
resolves the config's `climber:` block (see `hillclimb.climber`) — or a
started search's snapshot of it; the climber builds its loop and brings its
operators, tuner and prompts. The harness is identical for all of them — budgets,
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


# snapshots already resolved in this process: (snapshot file, its mtime) -> Climber.
# One search asks for its climber many times (the loop, the operators, the
# tuner, the memory, holdout timing); the snapshot is read and hashed once.
_SNAPSHOTS: dict[tuple[str, int], "Climber"] = {}


def _snapshot(search_dir, name: str) -> "Climber | None":
    from pathlib import Path

    from hillclimb.climber import MANIFEST, SNAPSHOT_DIRNAME, load_snapshot

    root = Path(search_dir) / SNAPSHOT_DIRNAME
    if not root.is_dir():
        return None
    marker = root / MANIFEST
    if not marker.is_file():
        files = sorted(root.glob("*.py"))  # a pre-0.6 one-file snapshot
        marker = files[0] if len(files) == 1 else None
    if marker is None:
        return load_snapshot(search_dir, name=name)
    key = (str(marker), marker.stat().st_mtime_ns)
    if key not in _SNAPSHOTS:
        _SNAPSHOTS[key] = load_snapshot(search_dir, name=name)
    return _SNAPSHOTS[key]


def search_climber(config: Config, search_dir=None) -> Climber:
    """The climber the config's `climber:` block defines (relative file refs
    resolve from the hillclimb dir) — or, for a search that already exists,
    the snapshot it was started with, which is the whole truth about it. A
    `ClimberLoadError` names the unknowns."""
    from hillclimb.climber import climber_base_dir, resolve_climber

    if search_dir is not None:
        snapshot = _snapshot(search_dir, config.climber.label)
        if snapshot is not None:
            return snapshot
    return resolve_climber(config.climber, climber_base_dir(config))


def holdout_timing(config: Config, search_dir=None) -> str:
    """When the harness scores the hidden split: the user's `holdout.timing`,
    tightened to `after` for a climber whose loop asks for it."""
    asked = search_climber(config, search_dir).holdout_timing
    return asked or config.holdout.timing


def build_tuner(config: Config, search_dir=None):
    """The tuner the climber's block names, with its `tuner_params`."""
    return search_climber(config, search_dir).tuner()


def build_loop(config: Config, *, complexity_start: int = 0, log=print, search_dir=None) -> Loop:
    return search_climber(config, search_dir).build_loop(
        complexity_start=complexity_start,
        parallelism=max(1, config.concurrency.parallel_agents),
        log=log,
    )


def build_operators(config: Config, search_dir=None) -> OperatorSet:
    """The operators the climber's search may use (`operators`, with
    `operator_params` laid over them by name)."""
    return search_climber(config, search_dir).operator_set()


def build_graph_module(config: Config, search_dir=None, log=None):
    """The climber's graph module (`graph:` in its block). Outside a search —
    `hillclimb knowledge …`, the graph TUI — there is no search_dir and the
    block is the folder's; a climber that will not load there falls back to
    the built-in module, so reading memory never depends on it."""
    from hillclimb.climber import ClimberLoadError
    from hillclimb.modules.memory.base import DEFAULT_GRAPH
    from hillclimb.modules.memory.graphs import get_graph

    try:
        return search_climber(config, search_dir).graph_module()
    except ClimberLoadError as exc:
        if search_dir is not None:
            raise
        if log is not None:
            log(f"graph: {exc}; using {DEFAULT_GRAPH}")
        return get_graph(DEFAULT_GRAPH)


def effective_memory(config: Config, search_dir=None) -> str:
    """`files` or `none`: what the climber's block says — and always `none`
    when learning is switched off."""
    if not config.learning.enabled:
        return "none"
    return search_climber(config, search_dir).spec.memory
