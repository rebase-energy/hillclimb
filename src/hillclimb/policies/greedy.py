"""The default greedy policy, extracted verbatim from GreedySearcher.

Priority: DEBUG the newest buggy tip while its chain is shallow > ENSEMBLE
solo in the final budget window > DRAFT until `num_drafts` branches hold a
scored solution (complexity cue escalates per draft) > IMPROVE the best.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from hillclimb.candidate import Candidate
from hillclimb.policy import Action, SearchView


class GreedyPolicy:
    name = "greedy"

    def __init__(self, complexity_start: int = 0, params: dict | None = None):
        self.complexity_start = complexity_start  # learned draft-complexity offset
        self.params = params or {}

    # --- SearchPolicy protocol ---

    def propose(self, view: SearchView) -> Action | None:
        """Pool policy: generalizes the serial rule to in-flight state.
        Priority: debug buggy tips (one per chain) > ensemble solo in the
        final window (drain first) > drafts until num_drafts branches are
        scored-or-pending > improves on distinct top targets. None = hold."""
        tip = self.debuggable_tip(view)
        if tip is not None:
            return Action(operator="debug", target_id=tip.candidate_id)
        if self.should_ensemble(view) and not any(
            ref.operator == "ensemble" for ref in view.inflight
        ):
            if view.inflight:
                return None  # drain: ensemble inputs snapshot at launch
            return self._ensemble_action(view)
        if self.prospective_branches(view) < view.config.search.num_drafts:
            return self._draft_action(view)
        busy_targets = {
            ref.parent_id for ref in view.inflight if ref.operator == "improve"
        }
        direction = -1 if view.higher_is_better else 1
        ranked = sorted(
            (c for c in view.journal.scored_candidates() if _improvable(c)),
            key=lambda c: direction * c.val_score,
        )
        if not ranked:
            return self._draft_action(view)
        for candidate in ranked:
            if candidate.candidate_id not in busy_targets:
                return Action(operator="improve", target_id=candidate.candidate_id)
        return Action(operator="improve", target_id=ranked[0].candidate_id)

    def observe(self, view: SearchView, candidate: Candidate) -> None:
        pass  # greedy is a pure function of the journal

    def action_for(self, view: SearchView, operator: str, target_id: str | None) -> Action:
        """Fully-populated Action for an explicitly requested operator — the
        run_operator/smoke path, where the harness names the move and the
        policy fills in its decision-time details."""
        if operator == "draft":
            return self._draft_action(view)
        if operator == "ensemble":
            picks = self.ensemble_candidates(view)
            return Action(
                operator="ensemble",
                target_id=target_id or (picks[0].candidate_id if picks else None),
                inspiration_ids=tuple(c.candidate_id for c in picks),
            )
        return Action(operator=operator, target_id=target_id)

    def _draft_action(self, view: SearchView) -> Action:
        return Action(operator="draft", complexity=self.draft_complexity(view))

    def _ensemble_action(self, view: SearchView) -> Action:
        picks = self.ensemble_candidates(view)
        return Action(
            operator="ensemble",
            target_id=picks[0].candidate_id,
            inspiration_ids=tuple(c.candidate_id for c in picks),
        )

    # --- decision helpers (moved verbatim from GreedySearcher) ---

    def debuggable_tip(self, view: SearchView) -> Candidate | None:
        """Newest buggy candidate with no active child and chain depth under
        the cap. In serial history this is exactly the serial debug rule."""
        journal = view.journal
        for candidate in reversed(list(journal.candidates.values())):
            if candidate.status != "buggy" or candidate.pruned:
                continue
            children = journal.children(candidate.candidate_id, include_pruned=True)
            if any(c.status in ("pending", "ok", "buggy") for c in children):
                continue
            chain = journal.debug_chain(candidate.candidate_id)
            depth = sum(1 for c in chain if c.operator == "debug")
            if depth < view.config.search.max_debug_depth:
                return candidate
        return None

    def prospective_branches(self, view: SearchView) -> int:
        """Draft branches whose subtree holds a scored OR pending candidate —
        in-flight work counts toward the num_drafts target."""
        journal = view.journal
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

    def in_ensemble_window(self, view: SearchView) -> bool:
        # window sits ABOVE the stop margin, else margin swallows it: with a
        # 45m budget, reserve(540s) - margin(300s) left a 240s slot that one
        # improve cycle stepped over entirely
        budget = view.budget
        reserve = budget.total_s * view.config.ensemble.reserve_fraction
        return budget.remaining_s <= reserve + budget.stop_margin_s

    def should_ensemble(self, view: SearchView) -> bool:
        cfg = view.config.ensemble
        if not cfg.enabled or not self.in_ensemble_window(view):
            return False
        attempts = sum(
            1 for c in view.journal.candidates.values() if c.operator == "ensemble"
        )
        if attempts >= cfg.max_attempts or self.ensemble_succeeded(view):
            return False
        return len(self.ensemble_candidates(view)) >= 2

    def ensemble_succeeded(self, view: SearchView) -> bool:
        for candidate in view.journal.candidates.values():
            if candidate.status != "ok":
                continue
            root = view.journal.debug_chain(candidate.candidate_id)[0]
            if root.operator == "ensemble":
                return True
        return False

    def ensemble_candidates(self, view: SearchView) -> list[Candidate]:
        """Top-k scored non-ensemble candidates by the selection rule, deduped
        by script content so near-identical improves don't fill the slots."""
        ranked = view.journal.ranked_candidates(
            view.higher_is_better, view.config.holdout.selection
        )
        picked, seen_hashes = [], set()
        for candidate in ranked:
            if candidate.operator == "ensemble":
                continue
            solution = Path(candidate.candidate_dir) / "solution.py"
            if not solution.exists():
                continue
            digest = hashlib.md5(solution.read_bytes()).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            picked.append(candidate)
            if len(picked) >= view.config.ensemble.top_k:
                break
        return picked

    def draft_complexity(self, view: SearchView) -> str:
        index = len(view.journal.drafts()) + self.complexity_start
        return "minimal" if index == 0 else "moderate" if index == 1 else "advanced"


def _improvable(candidate: Candidate) -> bool:
    """A declared floor (`baseline: 0.5`) is scored but has no code to
    improve; everything else that scored is fair game."""
    if candidate.operator != "baseline":
        return True
    return (Path(candidate.candidate_dir) / "solution.py").exists()
