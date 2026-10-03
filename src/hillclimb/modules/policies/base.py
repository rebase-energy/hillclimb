"""The policy seam — π_op: WHICH OPERATOR to apply to the node(s) the
selector chose. Everything else is the harness.

One step of a search is two decisions in a fixed order. The selector (π_sel,
`modules/selectors`) reads the history and picks the node(s) the next
attempt starts from, or none. Then the policy reads the same history and
that pick and names the operator: a root step gets a `create` operator
(draft), a failing node a `repair` one (debug), scored nodes a `refine` one
(improve, or a `tune` trial of the same code), several nodes a `combine` one
(ensemble). The loop (`harness/loop.py`) makes the two calls in that order;
a policy never picks a node, a selector never names an operator.

An `OperatorPolicy` proposes `Action`s over a read-only `SearchState`; the harness
(`hillclimb.harness.Harness`) materializes each action into a candidate dir,
prompt and coding agent call, executes it, and journals the outcome.

Contracts every policy must honor:

- `propose()`/`observe()` run only on the scheduler thread, under the
  searcher's state lock — the same discipline as the harness's own journal
  access. Policies may read candidate dirs from disk but must never write.
- Decisions must be derivable from replayed journal state: either compute
  every proposal from the `SearchState` and the `Selection` alone, or
  rebuild internal caches via `observe()` — on construction the harness
  replays every existing candidate through `observe()` in journal order, so
  `hillclimb resume` works.
- Ensemble-style actions must carry their inputs in `inspiration_ids`; the
  harness copies those candidates' solutions into the new candidate_dir.
- A policy never sees holdout: `SearchState.journal` is a `JournalView`
  (holdout-blind copies, unwritable) and `observe()` gets the candidate from
  that view. A process that may be optimized — by hand or by a meta-search —
  must not be able to select on the split that judges it.

Harness-owned, NOT policy: candidate-dir creation, journal writes, prompt
assembly, `AgentRequest` construction, trials, holdout gating and scoring,
`is_best`/selection syncing into `best/`, the control queue, and the cost
ceiling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hillclimb.harness.candidate import Candidate
    from hillclimb.harness.journal import Journal
    from hillclimb.modules.selectors.base import Selection


TUNE_ACTION = "tune"
# harness-native, coding-agent-free: score `action.payload["source"]` as a candidate
# (child of `target_id` when given). How a loop evaluates a text it produced
# itself — an optimizer's merge, a seed — without a coding agent call.
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

    # draft | debug | improve | ensemble create a candidate through a coding agent;
    # `tune` (TUNE_ACTION) adds one trial — a new parameter set from the
    # search's tuner — to the existing candidate `target_id`, no coding agent
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


_MISSING = object()


class OperatorPolicy:
    """π_op, the operator policy: which operator to apply to what the
    selector policy chose — nothing else.

    Subclass it, list your knobs in `DEFAULTS`, override `propose`. The base
    `propose` is the plain mapping — no node: draft; a failing node: debug;
    several nodes: ensemble; a scored node: improve — so an operator policy
    of your own only has to say where it differs (which `refine` operator,
    when to tune, …). The base also carries `param`, `resolved_params`, the
    selector policy the loop asks first (`self.selector`, `default_selector`
    unless the climber's block names another) and `draft_complexity`.

    A class with `propose(state, selection)` and `observe(state, candidate)`
    and no base still runs — the contract is the two methods.
    """

    name: str = ""
    # every knob and its default, merged over the class hierarchy: a
    # subclass lists only what it adds or changes
    DEFAULTS: Mapping[str, Any] = {
        "complexity_start": 0,  # offset of the draft-complexity cue (memory may have learned one)
    }
    # the selector policy an operator policy of this class uses when the block names none
    default_selector: str | None = "best"
    # a param the block sets must be one of `DEFAULTS` (a typo fails before
    # any spend). False for a class that takes free-form params
    strict_params: bool = True

    def __init__(self, params: Mapping | None = None, selector=None, **knobs):
        # held, never copied: what the climber resolved IS the policy's params.
        # `**knobs` is the same by keyword, for composing in Python: `Greedy(tune_budget=4)`
        from hillclimb.modules.refs import with_knobs

        known = self.defaults() if self.strict_params else None
        if known is not None:
            from hillclimb.modules.selectors.base import SelectorPolicy

            moved = sorted(set(knobs) & set(SelectorPolicy.defaults()) - set(known))
            if moved:
                raise TypeError(
                    f"{type(self).__name__} has no param {', '.join(map(repr, moved))}: the schedule is the "
                    f"selector policy's (π_sel), e.g. Best({moved[0]}=...) or `selector_params:` in the block"
                )
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
        """π_sel: picks the node(s) to start from (`modules/selectors`). The
        loop asks it before it asks the policy."""
        if self._selector is None and self.default_selector:
            from hillclimb.modules.selectors import get_selector

            self._selector = get_selector(self.default_selector)
        return self._selector

    # --- the contract ---

    def propose(self, state: SearchState, selection: Selection | None) -> Action | None:
        """The operator for what the selector chose, as an Action; None =
        hold (keep the slot empty until an in-flight result lands).
        `selection` is None for a root step."""
        if selection is None:
            return self.draft_action(state)
        if selection.combine:
            if state.inflight:
                return None  # drain: the inputs of a combination snapshot at launch
            return Action(
                operator="ensemble",
                target_id=selection.target_id,
                inspiration_ids=tuple(selection.inspiration_ids),
                extra_prompt_context=selection.prompt_context,
                climber_meta=dict(selection.meta),
            )
        node = state.journal.candidates[selection.target_id]
        if node.status in ("failing", "buggy"):
            return Action(operator="debug", target_id=node.candidate_id)
        return self.expand_action(state, selection)

    def observe(self, state: SearchState, candidate: Candidate) -> None:
        """Called after every journaled terminal result (and replayed for
        every existing candidate when a search starts or resumes). A policy
        whose state is the journal ignores it; the selector is kept in sync."""
        selector = self.selector
        if selector is not None:
            selector.sync(state)

    # --- the moves, for a subclass to reuse ---

    def draft_action(self, state: SearchState) -> Action:
        """A root step: a fresh draft, with the complexity cue and whatever
        the selector tags new roots with."""
        selector = self.selector
        return Action(
            operator="draft",
            args={"complexity": self.draft_complexity(state)},
            climber_meta=selector.creation_meta(state) if selector is not None else {},
        )

    def expand_action(self, state: SearchState, selection: Selection, operator: str = "improve") -> Action:
        """Build on the chosen node with a `refine` operator (`improve`)."""
        return Action(
            operator=operator,
            target_id=selection.target_id,
            inspiration_ids=tuple(selection.inspiration_ids),
            extra_prompt_context=selection.prompt_context,
            climber_meta=dict(selection.meta),
        )

    # --- journal questions ---

    @staticmethod
    def improvable(candidate: Candidate) -> bool:
        """Can an attempt start from this candidate? (A declared floor —
        `baseline: 0.5` — is scored but has no code.)"""
        from hillclimb.modules.selectors.base import improvable

        return improvable(candidate)

    def draft_complexity(self, state: SearchState) -> str:
        """The complexity cue for the next draft: it escalates per draft."""
        index = len(state.journal.drafts()) + int(self.param("complexity_start"))
        return "minimal" if index == 0 else "moderate" if index == 1 else "advanced"


# the pre-0.7 name; one release of grace
Policy = OperatorPolicy
