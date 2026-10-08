"""The selector seam — π_sel: WHICH node(s) of the history the next attempt
starts from, or none.

Every step of a search is two decisions in a fixed order. First the selector
reads the history and answers "which node?": a failing tip that needs repair,
nothing (a root step: a fresh draft), one scored candidate to build on, or
several whose combination is the next candidate. Then the policy (π_op)
answers "which operator on it?". The selector never names an operator; the
policy never picks a node.

The base class is plumbing only — the knobs (`DEFAULTS`, `param`,
`resolved_params`) and the hooks the loop calls — and decides nothing: a
selector policy writes its whole `schedule` (the order of repair, the
ensemble window, how many roots before building on one) and `select` (which
scored candidate to build on). The bundled `Best`
(`hillclimb/climbers/greedy/policy.py`) is the reference: subclass it to
change one step, or copy it (`hillclimb climber get greedy`) to own the file.
`MapElites` (`climbers/openevolve/policy.py`) samples a quality-diversity
archive instead. Swapping the selector swaps the search's exploration.

Contract:

- Everything a selector sees is the holdout-blind `SearchState`.
- Its state is a function of the journal: `sync(state)` brings it up to date
  and is idempotent, so a resumed search rebuilds what a live one had.
- `schedule` and `select` are pure functions of the state: asked twice,
  they answer twice.
- A selector never writes: not the journal, not a candidate dir.

This module imports only the standard library, so a selector can take the
base class from `hillclimb.sdk` without pulling the harness in. `top_distinct`
and `improvable` are the journal questions a selector may want (exported by
the sdk too).
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
    """π_sel, the selector policy. Subclass, set `name`, list your knobs in
    `DEFAULTS`, implement `schedule` (which node(s) the next attempt starts
    from, or None for a root step) and `select` (the scored candidate to
    build on when it is time to build); override `sync` when it keeps state
    and `creation_meta` when it tags new drafts. The base decides nothing —
    `hillclimb/climbers/greedy/policy.py:Best` is the reference to subclass
    or copy.
    """

    name: ClassVar[str] = ""
    # every knob and its default, merged over the class hierarchy: a
    # subclass lists only what it adds or changes
    DEFAULTS: ClassVar[Mapping[str, Any]] = {}

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
        # a name is the class's own: a subclass of `Best` that declares none
        # is not `best` (the loader stamps the block's label on it instead)
        if "name" not in vars(cls):
            cls.name = ""
        # `pick` was the hook's name for one day (2026-10-03); a selector
        # written against it still runs
        if "pick" in vars(cls) and "select" not in vars(cls):
            cls.select = vars(cls)["pick"]

    def schedule(self, state: SearchState, *, busy: frozenset[str] | set[str] = frozenset()) -> Selection | None:
        """Which node(s) the next attempt starts from, or None for a root
        step (a fresh draft). `busy` holds the candidates an in-flight
        attempt is already building on. The whole order of a search is
        written here — see `Best.schedule` in hillclimb/climbers/greedy/
        policy.py: a failing tip first, then the ensemble window, then roots
        up to `num_drafts`, then `select`."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement schedule(state, busy=...) — the reference is "
            "hillclimb/climbers/greedy/policy.py:Best (subclass it, or `hillclimb climber get greedy` copies it)"
        )

    def select(self, state: SearchState, *, busy: frozenset[str] | set[str] = frozenset()) -> Selection | None:
        """The scored candidate to build on now, or None when there is none
        (the step is then a root step)."""
        raise NotImplementedError(f"{type(self).__name__} must implement select(state, busy=...)")

    def creation_meta(self, state: SearchState) -> dict:
        """What to journal on a draft made now (an island, a niche); {} for none."""
        return {}


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
