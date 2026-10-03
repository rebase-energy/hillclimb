"""The selector seam — π_sel: WHICH node(s) of the history the next attempt
starts from, or none.

Every step of a search is two decisions in a fixed order. First the selector
reads the history and answers "which node?": a failing tip that needs repair,
nothing (a root step: a fresh draft), one scored candidate to build on, or
several whose combination is the next candidate. Then the policy (π_op)
answers "which operator on it?". The selector never names an operator; the
policy never picks a node.

The base class carries the schedule every selector shares, and asks a
subclass only for the one choice that differs between them, `select`: which
scored candidate to build on when it is time to build. `best` selects the
best; `map-elites` samples a quality-diversity archive. Swapping the
selector swaps the search's exploration without touching that schedule.

Contract:

- Everything a selector sees is the holdout-blind `SearchState`.
- Its state is a function of the journal: `sync(state)` brings it up to date
  and is idempotent, so a resumed search rebuilds what a live one had.
- `schedule` and `select` are pure functions of the state: asked twice,
  they answer twice.
- A selector never writes: not the journal, not a candidate dir.

This module imports only the standard library, so a selector can take the
base class from `hillclimb.sdk` without pulling the harness in.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from hillclimb.harness.candidate import Candidate
    from hillclimb.modules.policies.base import SearchState


@dataclass(frozen=True)
class Selection:
    """What π_sel chose: the node the attempt starts from, and what goes
    with it. `combine` means the nodes (the target and its inspirations) are
    the INPUTS of one combined candidate rather than a parent and context."""

    target_id: str
    inspiration_ids: tuple[str, ...] = ()  # copied in beside the parent, as `inspiration_filename(i)`
    prompt_context: str = ""  # appended to the operator's prompt
    meta: Mapping = field(default_factory=dict)  # journaled on the new candidate (`climber_meta`)
    combine: bool = False

    @property
    def node_ids(self) -> tuple[str, ...]:
        return (self.target_id, *self.inspiration_ids)


class SelectorPolicy:
    """π_sel, the selector policy. Subclass, set `name`, implement `select`
    (the scored candidate to build on); override `sync` when it keeps state,
    `creation_meta` when it tags new drafts, and `schedule` only to change
    the order around `select`.

    Knobs (`selector_params`), with their defaults:
      num_drafts (3)                     root candidates before anything is built on
      debug (True)                       repair failing tips at all
      max_debug_depth (3)                failed fixes per failing chain
      ensemble (True)                    combine the top candidates in the final window
      ensemble_reserve_fraction (0.2)    final slice of the budget reserved for it
      ensemble_top_k (3)                 candidates combined
      ensemble_max_attempts (2)
    """

    name: ClassVar[str] = ""
    # every knob and its default, merged over the class hierarchy: a
    # subclass lists only what it adds or changes
    DEFAULTS: ClassVar[Mapping[str, Any]] = {
        "num_drafts": 3,
        "debug": True,
        "max_debug_depth": 3,
        "ensemble": True,
        "ensemble_reserve_fraction": 0.2,
        "ensemble_top_k": 3,
        "ensemble_max_attempts": 2,
    }

    def __init__(self, params: Mapping | None = None, **knobs):
        # held, never copied: what the climber resolved IS the selector's
        # params (a caller may hold a live mapping); knobs lie over them
        # (`Best(num_drafts=5)`)
        self.params = params if params is not None and not knobs else {**dict(params or {}), **knobs}

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        """Every knob and its default: `DEFAULTS` of the class and its bases."""
        merged: dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            merged.update(vars(klass).get("DEFAULTS") or {})
        return merged

    def param(self, name: str):
        return self.params.get(name, self.defaults()[name])

    def resolved_params(self) -> dict:
        """Every knob the selector reads, fully resolved."""
        return {name: self.param(name) for name in self.defaults()}

    def sync(self, state: SearchState) -> None:
        """Bring the selector's state up to date with the journal. Idempotent."""

    # --- the contract ---

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # `pick` was the hook's name for one day (2026-10-03); a selector
        # written against it still runs
        if "pick" in vars(cls) and "select" not in vars(cls):
            cls.select = vars(cls)["pick"]

    def schedule(self, state: SearchState, *, busy: frozenset[str] | set[str] = frozenset()) -> Selection | None:
        """Which node(s) the next attempt starts from. In order: a failing
        tip (repair comes first), the top candidates once the final window
        is open (`combine=True`), None while fewer than `num_drafts` roots
        exist (a root step), else whatever `select` chooses. `busy` holds the
        candidates an in-flight attempt is already building on."""
        self.sync(state)
        tip = self.debuggable_tip(state)
        if tip is not None:
            return Selection(tip.candidate_id)
        if self.should_combine(state):
            picks = self.combine_candidates(state)
            return Selection(
                picks[0].candidate_id, inspiration_ids=tuple(c.candidate_id for c in picks), combine=True,
            )
        if self.prospective_branches(state) < int(self.param("num_drafts")):
            return None
        return self.select(state, busy=busy)

    def select(self, state: SearchState, *, busy: frozenset[str] | set[str] = frozenset()) -> Selection | None:
        """The scored candidate to build on now, or None when there is none
        (the step is then a root step)."""
        raise NotImplementedError(f"{type(self).__name__} must implement select(state, busy=...)")

    def creation_meta(self, state: SearchState) -> dict:
        """What to journal on a draft made now (an island, a niche); {} for none."""
        return {}

    # --- the schedule's questions ---

    def debuggable_tip(self, state: SearchState) -> Candidate | None:
        """Newest failing/buggy candidate with no active child and chain depth
        under `max_debug_depth`. In serial history this is exactly the serial
        debug rule."""
        if not bool(self.param("debug")):
            return None
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

    def in_ensemble_window(self, state: SearchState) -> bool:
        # window sits ABOVE the stop margin, else margin swallows it: with a
        # 45m budget, reserve(540s) - margin(300s) left a 240s slot that one
        # improve cycle stepped over entirely
        budget = state.budget
        reserve = budget.total_s * float(self.param("ensemble_reserve_fraction"))
        return budget.remaining_s <= reserve + budget.stop_margin_s

    def should_combine(self, state: SearchState) -> bool:
        if not bool(self.param("ensemble")) or not self.in_ensemble_window(state):
            return False
        attempts = sum(1 for c in state.journal.candidates.values() if c.kind == "combine")
        if attempts >= int(self.param("ensemble_max_attempts")) or self.combine_succeeded(state):
            return False
        return len(self.combine_candidates(state)) >= 2

    def combine_succeeded(self, state: SearchState) -> bool:
        for candidate in state.journal.candidates.values():
            if candidate.status != "passing":
                continue
            root = state.journal.debug_chain(candidate.candidate_id)[0]
            if root.kind == "combine":
                return True
        return False

    def combine_candidates(self, state: SearchState) -> list[Candidate]:
        """Top-k scored candidates by val score that are not combinations
        themselves, deduped by script content so near-identical improves
        don't fill the slots."""
        return top_distinct(state, int(self.param("ensemble_top_k")), skip_kind="combine")


# the pre-0.7 name; one release of grace
Selector = SelectorPolicy


def top_distinct(state: SearchState, k: int, skip_kind: str | None = None) -> list[Candidate]:
    """The top-k scored candidates by val score, deduped by solution content
    so near-identical ones don't fill the slots; candidates of `skip_kind`
    are left out. (`holdout.selection` decides what SHIPS; a selector never
    sees holdout.)"""
    import hashlib

    picked, seen_hashes = [], set()
    for candidate in state.journal.ranked_candidates(state.higher_is_better, "val"):
        if skip_kind is not None and candidate.kind == skip_kind:
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


def improvable(candidate: Candidate) -> bool:
    """Can an attempt start from this candidate? A baseline only when it
    shipped code: a declared floor has nothing to expand."""
    if candidate.operator != "baseline":
        return True
    return (Path(candidate.candidate_dir) / "solution.py").exists()
