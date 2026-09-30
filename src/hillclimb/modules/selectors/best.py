"""`best`: expand the best scored candidate nobody is expanding yet.

The default selector — what makes greedy greedy. With several attempts in
flight it spreads them over the top candidates instead of piling onto one;
when every scored candidate is busy, the best gets another.
"""

from __future__ import annotations

from hillclimb.sdk import SearchState, Selection, Selector, improvable


class Best(Selector):
    """The best scored candidate that is not already being expanded."""

    name = "best"

    def select(self, state: SearchState, *, busy=frozenset()) -> Selection | None:
        direction = -1 if state.higher_is_better else 1
        ranked = sorted(
            (c for c in state.journal.scored_candidates() if improvable(c)),
            key=lambda c: direction * c.val_score,
        )
        if not ranked:
            return None
        for candidate in ranked:
            if candidate.candidate_id not in busy:
                return Selection(candidate.candidate_id)
        return Selection(ranked[0].candidate_id)
