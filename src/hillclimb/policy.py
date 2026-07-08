"""The search-policy seam: WHAT to try next is a policy decision; everything
else is the harness.

A `SearchPolicy` proposes `Action`s over a read-only `SearchView`; the
harness (`GreedySearcher`) materializes each action into a workspace, prompt,
and agent call, executes it, and journals the outcome.

Contracts every policy must honor:

- `propose()`/`observe()` run only on the scheduler thread, under the
  searcher's state lock — the same discipline as the harness's own journal
  access. Policies may read candidate workspaces from disk (greedy hashes
  solution.py to dedupe ensemble inputs) but must never write.
- Decisions must be derivable from replayed journal state: either compute
  every proposal from the `SearchView` alone, or rebuild internal caches via
  `observe()` — on construction the harness replays every existing candidate
  through `observe()` in journal order, so `hillclimb resume` works.
- Ensemble-style actions must carry their inputs in `inspiration_ids`; the
  harness copies those candidates' solutions into the new workspace.

Harness-owned, NOT policy: workspace creation, journal writes, prompt
assembly, `OperatorRequest` construction, trials, holdout gating and scoring
(`_holdout_threshold` is query-budget hygiene, not strategy), `is_best`/
selection syncing into `best/`, the control queue, and the cost ceiling.
Survival-style strategies (keep worse-but-diverse candidates alive) need none
of these: which candidate gets expanded next is already fully policy-owned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from hillclimb.candidate import Candidate
    from hillclimb.config import Config
    from hillclimb.journal import Journal


@dataclass(frozen=True)
class Route:
    """Backend/model override for one action; None fields inherit from the
    routing config, which in turn falls back to the global config scalars."""

    backend: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class Action:
    """One proposed operator invocation. Candidates are referenced by id, not
    object, so actions stay serializable and trivially journal-derivable."""

    operator: str  # draft | debug | improve | ensemble
    target_id: str | None = None  # parent candidate
    inspiration_ids: tuple[str, ...] = ()  # extra candidates as prompt/workspace context
    complexity: str | None = None  # draft complexity cue (minimal | moderate | advanced)
    route: Route | None = None  # rare per-action override; routing config is the norm
    extra_prompt_context: str = ""  # rendered as an appended prompt section
    policy_meta: dict = field(default_factory=dict)  # journaled on the candidate


@dataclass(frozen=True)
class InflightRef:
    """Snapshot of one in-flight job, as much as a policy may know about it."""

    candidate_id: str
    operator: str
    parent_id: str | None


@dataclass(frozen=True)
class BudgetView:
    """The policy sees the clock; it never owns it."""

    remaining_s: float
    total_s: int
    stop_margin_s: int


@dataclass(frozen=True)
class SearchView:
    """Read-only view of search state handed to the policy on every call.
    The journal is shared by reference (scheduler-thread only); everything
    else is a snapshot computed at call time."""

    journal: Journal
    inflight: tuple[InflightRef, ...]
    budget: BudgetView
    config: Config
    lower_is_better: bool


class SearchPolicy(Protocol):
    """What to expand next, with which operator — nothing else."""

    name: str
    params: dict  # persisted verbatim into SearchMeta for resume

    def propose(self, view: SearchView) -> Action | None:
        """Next action given current state; None = hold (keep the slot empty
        until an in-flight result lands)."""
        ...

    def observe(self, view: SearchView, candidate: Candidate) -> None:
        """Called after every journaled terminal result (and replayed for
        every existing candidate on construction). Stateless policies ignore
        it; stateful ones rebuild caches here."""
        ...
