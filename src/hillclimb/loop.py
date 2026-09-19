"""The search-loop seam: control flow over a harness.

A `SearchLoop` decides WHEN to ask for work and how to react to results; it
does so only through the `Harness` interface below. Everything with a side
effect — candidate dirs, agents, verifier runs, the journal, `best/`, budgets,
the control queue, holdout — happens inside the harness, so a loop is small
enough to edit and cannot reach what judges it.

Most climbers never write a loop: a `SearchPolicy` (the pure "given the
state, what next?") runs on the built-in `PolicyLoop`. Write a `SearchLoop`
when the idea IS control flow — synchronized generations, islands, an
external optimizer that drives its own iteration.

Stop and park never raise into loop code. They latch the harness closed:
`open` turns False, `capacity` turns 0, in-flight work is still committed by
`wait()`, and any further `submit()`/`run()` raises `HarnessClosed`. A loop
that ignores all of that still cannot spend anything more.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from hillclimb.candidate import Candidate
    from hillclimb.policy import Action, InflightRef, PolicyInput, SearchPolicy


class HarnessClosed(Exception):
    """Work was asked of a harness that is closed (stopped, parked, out of
    budget or at its evaluation cap). Carries `Harness.closed_reason`."""


class ClimberError(Exception):
    """The climber cannot make progress (it keeps proposing actions the
    harness must refuse); the search ends as `failed`."""


@dataclass(frozen=True)
class SearchInfo:
    """What a loop may know about the search it drives."""

    problem_id: str
    description: str
    metric_name: str
    higher_is_better: bool
    parallelism: int  # how many attempts the harness runs at once


@dataclass(frozen=True)
class Ticket:
    """Receipt for one `submit()`. `rejected` names why nothing was started:
    nothing was created, journaled or spent."""

    id: str  # the candidate id, or "<candidate>:t<trial>" for a tune
    action: Action
    candidate_id: str | None = None
    trial_index: int | None = None
    rejected: str | None = None


@dataclass(frozen=True)
class Outcome:
    """One committed result, as `wait()`/`run()` hand it back. The candidate
    is the same holdout-blind copy `view().journal` holds."""

    ticket: Ticket
    # evaluated    the verifier ran: the candidate is passing, failing or buggy
    # tuned        a new trial landed on an existing candidate
    # discarded    a tune trial was dropped (its candidate changed or was pruned)
    # cut_off      killed at the budget wall — not shown to be wrong
    # agent_failed the agent call failed · no_solution it wrote no solution
    # aborted      stopped mid-attempt · parked the backend hit a limit
    # crashed      the harness's own worker failed · rejected nothing was started
    kind: str
    candidate: Candidate | None


class Harness(Protocol):
    """All a `SearchLoop` ever touches. `view`, `capacity`, `inflight` and
    `open` are pure reads; `submit`, `wait` and `run` must be called from the
    thread that runs the loop — results are committed on it."""

    info: SearchInfo
    state_dir: Path  # durable, loop-private (checkpoints, identity files)

    def view(self) -> PolicyInput: ...

    @property
    def capacity(self) -> int:
        """Free slots right now; 0 once the harness is closed."""

    @property
    def inflight(self) -> tuple[InflightRef, ...]: ...

    @property
    def open(self) -> bool:
        """False once no new work may start (stop, park, budget, cap)."""

    @property
    def closed_reason(self) -> str | None: ...

    def submit(self, action: Action) -> Ticket:
        """Start one attempt without waiting for it. The action is checked
        before anything is created; a refusal comes back as `rejected`."""

    def wait(self, timeout: float | None = None) -> list[Outcome]:
        """Block until a result is committed (or `timeout`, or nothing is in
        flight) and return what landed. An empty list is a tick: control
        commands and ceilings were checked, nothing finished."""

    def run(self, action: Action) -> Outcome:
        """`submit` + wait for that one result, on the calling thread."""

    def source(self, candidate_id: str) -> str | None:
        """A candidate's solution text (None when it has none)."""


class SearchLoop(ABC):
    """Control flow of a search. `run` returns when the loop is done; the
    harness then commits whatever is still in flight and ends the search."""

    name: str = "loop"

    @abstractmethod
    def run(self, harness: Harness) -> None: ...


class PolicyLoop(SearchLoop):
    """Keep every free slot busy with whatever the policy proposes; show it
    every result. The policy's whole state is re-derivable from the journal,
    so a resumed search catches up by replaying it through `observe`."""

    name = "policy"

    def __init__(self, policy: SearchPolicy):
        self.policy = policy
        self._caught_up: set[str] = set()

    def catch_up(self, harness: Harness) -> None:
        """Show the policy every journaled candidate it has not seen yet, in
        journal order (a resumed search; the baseline and seed of a new one)."""
        view = harness.view()
        for candidate_id, candidate in view.journal.candidates.items():
            if candidate_id not in self._caught_up:
                self._caught_up.add(candidate_id)
                self.policy.observe(view, candidate)

    def observe(self, harness: Harness, candidate: Candidate | None) -> None:
        if candidate is not None:
            self._caught_up.add(candidate.candidate_id)
            self.policy.observe(harness.view(), candidate)

    def run(self, harness: Harness) -> None:
        self.catch_up(harness)
        while True:
            while harness.capacity:
                action = self.policy.propose(harness.view())
                if action is None or harness.submit(action).rejected:
                    break  # a hold: keep the slot empty until something lands
            if not harness.inflight:
                if not harness.open:
                    return
                harness.wait(timeout=1.0)  # a tick: a holding policy must not end the search
                continue
            for outcome in harness.wait():
                self.observe(harness, outcome.candidate)
