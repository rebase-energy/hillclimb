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
from typing import TYPE_CHECKING, Any

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


class Policy:
    """What to expand next, with which operator — nothing else.

    Subclass it, list your knobs in `DEFAULTS`, implement `propose`. The
    base carries what every policy ends up needing: the knobs (`param`,
    `resolved_params`), the selector that picks a parent (`self.selector`,
    `default_selector` unless the climber's block names another) and the
    journal questions a schedule asks (`debuggable_tip`,
    `prospective_branches`, `draft_complexity`, `top_distinct`).

    A class with `propose` and `observe` and no base still runs — the
    contract is the two methods — it just brings its own helpers.
    """

    name: str = ""
    # every knob and its default, merged over the class hierarchy: a
    # subclass lists only what it adds or changes
    DEFAULTS: Mapping[str, Any] = {
        "max_debug_depth": 3,  # failed fixes per failing chain
        "complexity_start": 0,  # offset of the draft-complexity cue (memory may have learned one)
    }
    # the selector a policy of this class uses when the block names none
    default_selector: str | None = "best"
    # a param the block sets must be one of `DEFAULTS` (a typo fails before
    # any spend). False for a class that takes free-form params
    strict_params: bool = True

    def __init__(self, params: Mapping | None = None, selector=None, **knobs):
        # held, never copied: what the climber resolved IS the policy's params.
        # `**knobs` is the same by keyword, for composing in Python: `Greedy(num_drafts=3)`
        from hillclimb.modules.refs import with_knobs

        known = self.defaults() if self.strict_params else None
        self.params = with_knobs(params, knobs, known, type(self).__name__)
        self._selector = selector

    # --- knobs ---

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        """Every knob and its default: `DEFAULTS` of the class and its bases."""
        merged: dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            merged.update(vars(klass).get("DEFAULTS") or {})
        return merged

    def param(self, name: str):
        """One knob: the climber's `params[name]`, else its default."""
        value = self.params.get(name, _MISSING)
        return self.defaults()[name] if value is _MISSING else value

    def resolved_params(self) -> dict:
        """Every knob the policy reads, fully resolved — the single dict that
        describes this exploration process."""
        return {name: self.param(name) for name in self.defaults()}

    @property
    def selector(self):
        """Picks the candidate to expand (`modules/selectors`)."""
        if self._selector is None and self.default_selector:
            from hillclimb.modules.selectors import get_selector

            self._selector = get_selector(self.default_selector)
        return self._selector

    # --- the contract ---

    def propose(self, state: SearchState) -> Action | None:
        """Next action given current state; None = hold (keep the slot empty
        until an in-flight result lands)."""
        raise NotImplementedError(f"{type(self).__name__} must implement propose(state)")

    def observe(self, state: SearchState, candidate: Candidate) -> None:
        """Called after every journaled terminal result (and replayed for
        every existing candidate when a search starts or resumes). A policy
        whose state is the journal ignores it; the selector is kept in sync."""
        selector = self.selector
        if selector is not None:
            selector.sync(state)

    # --- journal questions a schedule asks ---

    @staticmethod
    def improvable(candidate: Candidate) -> bool:
        """Can an attempt start from this candidate? (A declared floor —
        `baseline: 0.5` — is scored but has no code.)"""
        from hillclimb.modules.selectors.base import improvable

        return improvable(candidate)

    def debuggable_tip(self, state: SearchState) -> Candidate | None:
        """Newest failing/buggy candidate with no active child and chain depth
        under `max_debug_depth`. In serial history this is exactly the serial
        debug rule."""
        journal = state.journal
        for candidate in reversed(list(journal.candidates.values())):
            if candidate.status not in ("failing", "buggy") or candidate.pruned:
                continue
            children = journal.children(candidate.candidate_id, include_pruned=True)
            if any(c.status in ("pending", "passing", "failing", "buggy") for c in children):
                continue
            chain = journal.debug_chain(candidate.candidate_id)
            depth = sum(1 for c in chain if c.operator == "debug")
            if depth < int(self.param("max_debug_depth")):
                return candidate
        return None

    def prospective_branches(self, state: SearchState) -> int:
        """Draft branches whose subtree holds a scored OR pending candidate —
        in-flight work counts toward a number-of-drafts target."""
        journal = state.journal
        count = 0
        for draft in journal.drafts():
            frontier = [draft]
            while frontier:
                candidate = frontier.pop()
                if candidate.is_scored or candidate.status == "pending":
                    count += 1
                    break
                frontier.extend(journal.children(candidate.candidate_id))
        return count

    def draft_complexity(self, state: SearchState) -> str:
        """The complexity cue for the next draft: it escalates per draft."""
        index = len(state.journal.drafts()) + int(self.param("complexity_start"))
        return "minimal" if index == 0 else "moderate" if index == 1 else "advanced"

    def top_distinct(self, state: SearchState, k: int, skip_operator: str | None = None) -> list[Candidate]:
        """The top-k scored candidates by val score, deduped by solution
        content so near-identical ones don't fill the slots; candidates an
        operator named `skip_operator` made are left out. (`holdout.selection`
        decides what SHIPS; a policy never sees holdout.)"""
        import hashlib
        from pathlib import Path

        picked, seen_hashes = [], set()
        for candidate in state.journal.ranked_candidates(state.higher_is_better, "val"):
            if skip_operator is not None and candidate.operator == skip_operator:
                continue
            solution = Path(candidate.candidate_dir) / "solution.py"
            if not solution.exists():
                continue
            digest = hashlib.md5(solution.read_bytes()).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            picked.append(candidate)
            if len(picked) >= k:
                break
        return picked


_MISSING = object()
