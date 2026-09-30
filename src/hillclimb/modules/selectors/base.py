"""The selector seam: WHICH candidate to expand next, and with what beside it.

A policy decides the kind of move (draft, repair, tune, combine, expand); when
the move is "expand a scored candidate", its `Selector` picks the parent and
whatever rides along: inspirations copied in next to it, a paragraph of
context for the prompt, a note journaled on the new candidate. Swapping the
selector swaps the search's exploration — best-first (`best`), a
quality-diversity archive (`map-elites`) — without touching its schedule.

Contract, the same one a policy honours:

- Everything a selector sees is the holdout-blind `SearchState`.
- Its state is a function of the journal: `sync(state)` brings it up to date
  and is idempotent, so a resumed search rebuilds what a live one had. A
  policy calls it before it asks for anything.
- `select` returns None when there is nothing to expand (the policy drafts).
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
    """One pick: the parent, and what goes with it."""

    target_id: str
    inspiration_ids: tuple[str, ...] = ()  # copied in beside the parent, as `inspiration_filename(i)`
    prompt_context: str = ""  # appended to the operator's prompt
    meta: Mapping = field(default_factory=dict)  # journaled on the new candidate (`climber_meta`)


class Selector:
    """Subclass, set `name`, implement `select`; override `sync` when the
    selector keeps state, `creation_meta` when it tags new drafts."""

    name: ClassVar[str] = ""
    # every knob and its default; `params` are laid over them
    DEFAULTS: ClassVar[Mapping[str, Any]] = {}

    def __init__(self, params: Mapping | None = None):
        self.params = dict(params or {})

    def param(self, name: str):
        return self.params.get(name, self.DEFAULTS[name])

    def sync(self, state: SearchState) -> None:
        """Bring the selector's state up to date with the journal. Idempotent."""

    def select(self, state: SearchState, *, busy: frozenset[str] | set[str] = frozenset()) -> Selection | None:
        """The next candidate to expand, or None when there is none. `busy`
        holds the candidates an in-flight attempt is already expanding."""
        raise NotImplementedError(f"{type(self).__name__} must implement select(state, busy=...)")

    def creation_meta(self, state: SearchState) -> dict:
        """What to journal on a draft made now (an island, a niche); {} for none."""
        return {}


def improvable(candidate: Candidate) -> bool:
    """Can an attempt start from this candidate? A baseline only when it
    shipped code: a declared floor has nothing to expand."""
    if candidate.operator != "baseline":
        return True
    return (Path(candidate.candidate_dir) / "solution.py").exists()
