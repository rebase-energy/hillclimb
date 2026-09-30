"""The search-policy seam: WHAT to try next is a policy decision; everything
else is the harness.

A `Policy` proposes `Action`s over a read-only `SearchState`; the
harness (`hillclimb.harness.Harness`) materializes each action into a candidate dir, prompt,
and agent call, executes it, and journals the outcome.

Contracts every policy must honor:

- `propose()`/`observe()` run only on the scheduler thread, under the
  searcher's state lock — the same discipline as the harness's own journal
  access. Policies may read candidate candidate dirs from disk (greedy hashes
  solution.py to dedupe ensemble inputs) but must never write.
- Decisions must be derivable from replayed journal state: either compute
  every proposal from the `SearchState` alone, or rebuild internal caches via
  `observe()` — on construction the harness replays every existing candidate
  through `observe()` in journal order, so `hillclimb resume` works.
- Ensemble-style actions must carry their inputs in `inspiration_ids`; the
  harness copies those candidates' solutions into the new candidate_dir.
- A policy never sees holdout: `SearchState.journal` is a `JournalView`
  (holdout-blind copies, unwritable) and `observe()` gets the candidate from
  that view. A process that may be optimized — by hand or by a meta-search —
  must not be able to select on the split that judges it.

Harness-owned, NOT policy: candidate-dir creation, journal writes, prompt
assembly, `AgentRequest` construction, trials, holdout gating and scoring
(`_holdout_threshold` is query-budget hygiene, not strategy), `is_best`/
selection syncing into `best/`, the control queue, and the cost ceiling.
Survival-style strategies (keep worse-but-diverse candidates alive) need none
of these: which candidate gets expanded next is already fully policy-owned.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from hillclimb.harness.candidate import Candidate
    from hillclimb.harness.journal import Journal


TUNE_ACTION = "tune"
# harness-native, agent-free: score `action.payload["source"]` as a candidate
# (child of `target_id` when given). How a loop evaluates a text it produced
# itself — an optimizer's merge, a seed — without an agent call.
INJECT_ACTION = "inject"


@dataclass(frozen=True)
class Route:
    """Agent/model override for one action; None fields inherit from the
    routing config, which in turn falls back to the global config scalars."""

    agent: str | None = None
    model: str | None = None
    sampling: dict[str, int | float] | None = None


@dataclass(frozen=True)
class Action:
    """One proposed operator invocation. Candidates are referenced by id, not
    object, so actions stay serializable and trivially journal-derivable."""

    # draft | debug | improve | ensemble create a candidate through an agent;
    # `tune` (TUNE_ACTION) adds one trial — a new parameter set from the
    # search's tuner — to the existing candidate `target_id`, no agent
    operator: str
    target_id: str | None = None  # parent candidate (the tuned candidate for `tune`)
    inspiration_ids: tuple[str, ...] = ()  # extra candidates as prompt/candidate-dir context
    route: Route | None = None  # rare per-action override; routing config is the norm
    extra_prompt_context: str = ""  # rendered as an appended prompt section
    # the operator's knobs for THIS attempt (JSON-able, small): journaled on
    # the candidate as `Candidate.args` — e.g. draft's {"complexity": "minimal"}
    args: Mapping = field(default_factory=dict)
    # bulk input for the operator, NEVER journaled: the source an `inject`
    # scores, the feedback `gepa-reflect` is shown
    payload: Mapping = field(default_factory=dict)
    climber_meta: dict = field(default_factory=dict)  # journaled on the candidate


@dataclass(frozen=True)
class InflightRef:
    """Snapshot of one in-flight job, as much as a policy may know about it.
    A tune job carries the tuned candidate's id, operator `tune`, no parent
    and its trial index (so a policy never over-proposes on one candidate)."""

    candidate_id: str
    operator: str
    parent_id: str | None
    trial_index: int | None = None


@dataclass(frozen=True)
class BudgetView:
    """The policy sees the clock; it never owns it."""

    remaining_s: float
    total_s: int
    stop_margin_s: int
    # what is left of the other budget dimensions; None = the user set no limit
    evaluations_remaining: int | None = None
    tokens_remaining: int | None = None
    cost_remaining_usd: float | None = None

    def remaining_str(self) -> str:
        """The clock as prompts and logs print it (`1h 05m` | `59 minutes`)."""
        from hillclimb.harness.budget import format_remaining

        return format_remaining(self.remaining_s)


@dataclass(frozen=True)
class SearchState:
    """Read-only view of search state handed to the policy on every call,
    all of it a snapshot computed at call time. The journal is always a
    `JournalView` — holdout-blind and unwritable; a plain `Journal` passed
    in is wrapped here, so no construction site can forget the mask."""

    journal: Journal
    inflight: tuple[InflightRef, ...]
    budget: BudgetView
    higher_is_better: bool
    # how much better than the best a score must be before the harness calls
    # it an improvement (the user's noise_k / min_improvement over the
    # measured replicate noise) — so a policy agrees with the harness on what
    # "better" means. A policy never sees the harness's config: its own knobs
    # arrive through its constructor's `params`.
    accept_band: float = 0.0

    def __post_init__(self) -> None:
        from hillclimb.harness.journal import JournalView

        if not isinstance(self.journal, JournalView):
            object.__setattr__(self, "journal", JournalView(self.journal))


class Policy(Protocol):
    """What to expand next, with which operator — nothing else."""

    name: str
    params: dict  # persisted verbatim into SearchMeta for resume

    def propose(self, view: SearchState) -> Action | None:
        """Next action given current state; None = hold (keep the slot empty
        until an in-flight result lands)."""
        ...

    def observe(self, view: SearchState, candidate: Candidate) -> None:
        """Called after every journaled terminal result (and replayed for
        every existing candidate on construction). Stateless policies ignore
        it; stateful ones rebuild caches here."""
        ...
