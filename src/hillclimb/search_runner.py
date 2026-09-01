"""The engine-level runner seam: a search engine that owns its own loop.

Two integration tiers exist (docs/optimizer-host-plan.md):

- `SearchPolicy` (policy.py) — a read-only brain inside GreedySearcher's
  loop, for libraries that are archives/selectors.
- `SearchRunner` (this module) — a full engine dispatched by name before
  `get_policy()` is ever called, for optimizers that own proposal,
  selection, and iteration themselves (GEPA, future engines).

`build_search_runner` is the single construction point `api.execute_search`
calls; it receives every dependency the harness builds and returns whichever
runner the config's `search.policy` names. `run()` owns the journal for the
duration of the search (single-writer contract — see Journal).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hillclimb.backends.base import OperatorBackend
    from hillclimb.budget import BudgetManager
    from hillclimb.candidate import Candidate
    from hillclimb.config import Config
    from hillclimb.control import ControlCommand
    from hillclimb.executor import Executor, HoldoutScorer
    from hillclimb.journal import Journal
    from hillclimb.problem import ProblemSpec
    from hillclimb.routing import BackendPool, Router
    from hillclimb.slots import MachineSlots
    from hillclimb.status import StatusWriter


class ParkedSearch(Exception):
    """Raised when the backend hits a rate limit; the search can be resumed."""


class StopRequested(Exception):
    """Raised on a graceful stop (control command or SIGTERM); the search can
    be resumed."""


@runtime_checkable
class SearchRunner(Protocol):
    """What execute_search consumes: run the search to a terminal state and
    account for what it spent. `run()` may raise ParkedSearch/StopRequested;
    execute_search maps them to terminal states."""

    def run(self) -> Candidate | None: ...

    def total_cost_usd(self) -> float: ...


# Engine registry: search.policy names that dispatch to a full runner instead
# of get_policy(). Factories import their integration lazily so a greedy
# install never imports an optional engine.


def _gepa_factory(**deps) -> SearchRunner:
    from hillclimb.integrations.gepa import build_gepa_searcher

    return build_gepa_searcher(**deps)


_ENGINES: dict[str, Callable[..., SearchRunner]] = {"gepa": _gepa_factory}


def build_search_runner(
    *,
    config: Config,
    problem: ProblemSpec,
    journal: Journal,
    backend: OperatorBackend,
    executor: Executor,
    budget: BudgetManager,
    search_dir: Path,
    log=print,
    holdout_scorer: HoldoutScorer | None = None,
    status: StatusWriter | None = None,
    slots: MachineSlots | None = None,
    abort: threading.Event | None = None,
    seed_solution: Path | None = None,
    knowledge_context: str | None = None,
    reference_solution: Path | None = None,
    reference_note: str = "",
    complexity_start: int = 0,
    router: Router | None = None,
    backends: BackendPool | None = None,
    drain_commands: Callable[[], list[ControlCommand]] | None = None,
) -> SearchRunner:
    """Construct the runner `config.search.policy` names. Engine names get
    the full dependency set and never touch the policy registry; every other
    name goes through `get_policy()` (whose ValueError names the unknowns)
    into a GreedySearcher."""
    deps = dict(
        problem=problem,
        config=config,
        journal=journal,
        backend=backend,
        executor=executor,
        budget=budget,
        search_dir=search_dir,
        log=log,
        holdout_scorer=holdout_scorer,
        status=status,
        slots=slots,
        abort=abort,
        seed_solution=seed_solution,
        knowledge_context=knowledge_context,
        reference_solution=reference_solution,
        reference_note=reference_note,
        complexity_start=complexity_start,
        router=router,
        backends=backends,
        drain_commands=drain_commands,
    )
    name = config.search.policy
    if name in _ENGINES:
        return _ENGINES[name](**deps)
    from hillclimb.policies import get_policy

    # function-local on purpose: breaks the module cycle (search.py imports
    # the exceptions above at top level) and re-resolves the class per call
    # so tests can monkeypatch hillclimb.search.GreedySearcher
    from hillclimb.search import GreedySearcher

    return GreedySearcher(
        policy=get_policy(name, config.search.policy_params, complexity_start=complexity_start),
        **deps,
    )
