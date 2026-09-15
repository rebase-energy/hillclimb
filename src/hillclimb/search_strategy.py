"""The search-strategy seam: a search engine that owns its own loop.

Two integration tiers exist (docs/optimizer-host-plan.md):

- `SearchPolicy` (policy.py) — a read-only brain inside GreedySearcher's
  loop, for libraries that are archives/selectors.
- `SearchStrategy` (this module) — a full engine dispatched by name before
  `get_policy()` is ever called, for optimizers that own proposal,
  selection, and iteration themselves (GEPA, future engines).

`build_search_strategy` is the single construction point `api.execute_search`
calls; it receives every dependency the harness builds and returns whichever
strategy the config's `search.policy` names. `run()` owns the journal for the
duration of the search (single-writer contract — see Journal). Holdout is not
a strategy's concern at all: the host scores it through the `evaluator` it
hands in (evaluation.py), at the timing the engine registry declares.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hillclimb.backends.base import OperatorBackend
    from hillclimb.budget import BudgetManager
    from hillclimb.candidate import Candidate
    from hillclimb.config import Config
    from hillclimb.control import ControlCommand
    from hillclimb.evaluation import CandidateEvaluator
    from hillclimb.executor import Executor
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
class SearchStrategy(Protocol):
    """What execute_search consumes: run the search to a terminal state and
    account for what it spent. `run()` may raise ParkedSearch/StopRequested;
    execute_search maps them to terminal states."""

    def run(self) -> Candidate | None: ...

    def total_cost_usd(self) -> float: ...


# Engine registry: search.policy names that dispatch to a full strategy instead
# of get_policy(). Factories import their integration lazily so a greedy
# install never imports an optional engine. An entry may also declare how the
# HOST times holdout for that engine (evaluation.py): `inline` scores each
# candidate as it lands, `after` waits until run() has returned — for an
# optimizer whose state must never see a holdout value. A bare callable is
# accepted as an inline entry (tests stub engines that way).


@dataclass(frozen=True)
class Engine:
    factory: Callable[..., SearchStrategy]
    holdout_timing: str = "inline"


def _gepa_factory(**deps) -> SearchStrategy:
    from hillclimb.integrations.gepa import build_gepa_searcher

    return build_gepa_searcher(**deps)


_ENGINES: dict[str, Engine | Callable[..., SearchStrategy]] = {
    "gepa": Engine(_gepa_factory, holdout_timing="after"),
}


def holdout_timing(name: str) -> str:
    """When the host scores the hidden split for the strategy `name` picks:
    an engine's declared timing, `inline` for everything else."""
    entry = _ENGINES.get(name)
    return entry.holdout_timing if isinstance(entry, Engine) else "inline"


def build_search_strategy(
    *,
    config: Config,
    problem: ProblemSpec,
    journal: Journal,
    backend: OperatorBackend,
    executor: Executor,
    budget: BudgetManager,
    search_dir: Path,
    log=print,
    evaluator: CandidateEvaluator | None = None,
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
) -> SearchStrategy:
    """Construct the strategy `config.search.policy` names. Engine names get
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
        evaluator=evaluator,
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
        entry = _ENGINES[name]
        factory = entry.factory if isinstance(entry, Engine) else entry
        return factory(**deps)
    from hillclimb.policies import get_policy, policy_base_dir

    # function-local on purpose: breaks the module cycle (search.py imports
    # the exceptions above at top level) and re-resolves the class per call
    # so tests can monkeypatch hillclimb.search.GreedySearcher
    from hillclimb.search import GreedySearcher
    from hillclimb.tuners import get_tuner

    return GreedySearcher(
        policy=get_policy(
            name, config.search.policy_params,
            complexity_start=complexity_start, base_dir=policy_base_dir(config),
        ),
        tuner=get_tuner(config.search.tuner, config.search.tuner_params),
        **deps,
    )
