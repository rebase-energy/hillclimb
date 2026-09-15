"""The default greedy policy, extracted verbatim from GreedySearcher.

Priority: DEBUG the newest buggy tip while its chain is shallow > ENSEMBLE
solo in the final budget window > DRAFT until `num_drafts` branches hold a
scored solution (complexity cue escalates per draft) > TUNE a promising
candidate that declared params.json > IMPROVE the best.

The whole exploration process is ONE dict: `search.policy_params`. Every
knob below is read from it first; the strategy knobs fall back to the
config blocks they historically lived in (`search.num_drafts`,
`search.max_debug_depth`, `ensemble.*`), so an existing config is
byte-identical in behaviour and an agent editing the loop is handed a
single dict (`GreedyPolicy.resolved_params(config)` is that dict, fully
resolved). All replay-deterministic — every count is derived from the
journal and the in-flight refs, never kept.

Strategy params (fallback in brackets):
  num_drafts [search.num_drafts]              draft branches before improving
  max_debug_depth [search.max_debug_depth]    failed fixes per buggy chain
  ensemble [ensemble.enabled]                 ensemble solo in the final window
  ensemble_reserve_fraction [ensemble.reserve_fraction]
                                              final slice of budget reserved
  ensemble_top_k [ensemble.top_k]             candidates blended
  ensemble_max_attempts [ensemble.max_attempts]

Tune params:
  tune_budget (8)     extra trials per candidate beyond its defaults trial; 0 = off
  tune_gate ("band")  which ok+tunable candidates qualify: "band" = within the
                      accept band of the current best (best included), "best" =
                      the best only, "always" = every scored one
  tune_parallel (1)   tune jobs in flight per candidate (>1 engages the
                      tuner's constant liar)
  tune_burst (2)      tune trials released between agent proposals, so tuning
                      interleaves with improving instead of starving it
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from hillclimb import evaluation
from hillclimb.candidate import Candidate
from hillclimb.policy import TUNE_ACTION, Action, PolicyInput

if TYPE_CHECKING:
    from hillclimb.config import Config

TUNE_DEFAULTS = {"tune_budget": 8, "tune_gate": "band", "tune_parallel": 1, "tune_burst": 2}

# strategy knobs: policy_params first, else the config block they came from
CONFIG_FALLBACKS: dict[str, Callable[["Config"], object]] = {
    "num_drafts": lambda c: c.search.num_drafts,
    "max_debug_depth": lambda c: c.search.max_debug_depth,
    "ensemble": lambda c: c.ensemble.enabled,
    "ensemble_reserve_fraction": lambda c: c.ensemble.reserve_fraction,
    "ensemble_top_k": lambda c: c.ensemble.top_k,
    "ensemble_max_attempts": lambda c: c.ensemble.max_attempts,
}

PARAM_NAMES = (*CONFIG_FALLBACKS, *TUNE_DEFAULTS)


class GreedyPolicy:
    name = "greedy"

    def __init__(self, complexity_start: int = 0, params: dict | None = None):
        self.complexity_start = complexity_start  # learned draft-complexity offset
        self.params = params or {}

    # --- the one dict ---

    def param(self, name: str, config: "Config"):
        """One knob: `policy_params[name]`, else its config-block fallback
        (strategy knobs) or its built-in default (tune knobs)."""
        if name in self.params:
            return self.params[name]
        if name in CONFIG_FALLBACKS:
            return CONFIG_FALLBACKS[name](config)
        return TUNE_DEFAULTS[name]

    def resolved_params(self, config: "Config") -> dict:
        """Every knob the policy reads, fully resolved against `config` —
        the single dict that describes this exploration process."""
        return {name: self.param(name, config) for name in PARAM_NAMES}

    # --- SearchPolicy protocol ---

    def propose(self, view: PolicyInput) -> Action | None:
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
        if self.prospective_branches(view) < int(self.param("num_drafts", view.config)):
            return self._draft_action(view)
        tune = self.tune_target(view)
        if tune is not None:
            return Action(operator=TUNE_ACTION, target_id=tune.candidate_id)
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

    def observe(self, view: PolicyInput, candidate: Candidate) -> None:
        pass  # greedy is a pure function of the journal

    def action_for(self, view: PolicyInput, operator: str, target_id: str | None) -> Action:
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

    def tune_param(self, name: str):
        return self.params.get(name, TUNE_DEFAULTS[name])

    def tune_target(self, view: PolicyInput) -> Candidate | None:
        """The candidate to spend the next tune trial on, or None. Derived
        from the journal + in-flight refs only: `spent` on a candidate is its
        trial count beyond the defaults trial plus its in-flight tune jobs, so
        a resumed search continues exactly where a killed one stopped."""
        budget = int(self.tune_param("tune_budget"))
        if budget <= 0:
            return None
        headroom = view.budget.remaining_s - view.budget.stop_margin_s
        journal = view.journal
        best = journal.best_candidate(view.higher_is_better)
        band = evaluation.accept_band(view.config, journal)
        gate = str(self.tune_param("tune_gate"))
        parallel = int(self.tune_param("tune_parallel"))
        burst = int(self.tune_param("tune_burst"))
        direction = -1 if view.higher_is_better else 1
        ranked = sorted(
            (c for c in journal.scored_candidates() if c.tunable),
            key=lambda c: (direction * c.val_score, c.candidate_id),
        )
        agent_jobs = sum(1 for ref in view.inflight if ref.operator != TUNE_ACTION)
        for candidate in ranked:
            if not self._tune_gate_passes(candidate, best, band, gate, view.higher_is_better):
                continue
            if headroom < _trial_cost_s(candidate):
                continue  # a trial of this code could not finish before the wall
            inflight = sum(
                1 for ref in view.inflight
                if ref.operator == TUNE_ACTION and ref.candidate_id == candidate.candidate_id
            )
            spent = (len(candidate.trials) - 1) + inflight
            if spent >= budget or inflight >= parallel:
                continue
            # interleave: allow `burst` tune trials per agent proposal made
            # since this candidate landed (later ids + in-flight agent jobs)
            agents_since = sum(1 for cid in journal.candidates if cid > candidate.candidate_id)
            if spent >= burst * (1 + agents_since + agent_jobs):
                continue
            return candidate
        return None

    @staticmethod
    def _tune_gate_passes(
        candidate: Candidate, best: Candidate | None, band: float, gate: str, higher_is_better: bool
    ) -> bool:
        if gate == "always":
            return True
        if best is None or candidate.candidate_id == best.candidate_id:
            return True
        if gate == "best":
            return False
        # "band": the best does not beat this candidate by more than the band
        return not evaluation.improves(
            best.val_score, candidate.val_score, higher_is_better=higher_is_better, band=band
        )

    def _draft_action(self, view: PolicyInput) -> Action:
        return Action(operator="draft", complexity=self.draft_complexity(view))

    def _ensemble_action(self, view: PolicyInput) -> Action:
        picks = self.ensemble_candidates(view)
        return Action(
            operator="ensemble",
            target_id=picks[0].candidate_id,
            inspiration_ids=tuple(c.candidate_id for c in picks),
        )

    # --- decision helpers (moved verbatim from GreedySearcher) ---

    def debuggable_tip(self, view: PolicyInput) -> Candidate | None:
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
            if depth < int(self.param("max_debug_depth", view.config)):
                return candidate
        return None

    def prospective_branches(self, view: PolicyInput) -> int:
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

    def in_ensemble_window(self, view: PolicyInput) -> bool:
        # window sits ABOVE the stop margin, else margin swallows it: with a
        # 45m budget, reserve(540s) - margin(300s) left a 240s slot that one
        # improve cycle stepped over entirely
        budget = view.budget
        reserve = budget.total_s * float(self.param("ensemble_reserve_fraction", view.config))
        return budget.remaining_s <= reserve + budget.stop_margin_s

    def should_ensemble(self, view: PolicyInput) -> bool:
        if not bool(self.param("ensemble", view.config)) or not self.in_ensemble_window(view):
            return False
        attempts = sum(
            1 for c in view.journal.candidates.values() if c.operator == "ensemble"
        )
        max_attempts = int(self.param("ensemble_max_attempts", view.config))
        if attempts >= max_attempts or self.ensemble_succeeded(view):
            return False
        return len(self.ensemble_candidates(view)) >= 2

    def ensemble_succeeded(self, view: PolicyInput) -> bool:
        for candidate in view.journal.candidates.values():
            if candidate.status != "ok":
                continue
            root = view.journal.debug_chain(candidate.candidate_id)[0]
            if root.operator == "ensemble":
                return True
        return False

    def ensemble_candidates(self, view: PolicyInput) -> list[Candidate]:
        """Top-k scored non-ensemble candidates by the selection rule, deduped
        by script content so near-identical improves don't fill the slots."""
        ranked = view.journal.ranked_candidates(
            view.higher_is_better, view.config.holdout.selection
        )
        top_k = int(self.param("ensemble_top_k", view.config))
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
            if len(picked) >= top_k:
                break
        return picked

    def draft_complexity(self, view: PolicyInput) -> str:
        index = len(view.journal.drafts()) + self.complexity_start
        return "minimal" if index == 0 else "moderate" if index == 1 else "advanced"


MIN_TRIAL_COST_S = 30.0


def _trial_cost_s(candidate: Candidate) -> float:
    """How long a trial of this candidate takes, from its own record: the
    best trial's replicate wall-clock (parallel replicates overlap, so the
    longest one), with a floor for near-instant verifiers."""
    best = candidate.best_trial
    if best is None:
        return MIN_TRIAL_COST_S
    durations = [r.duration_s for r in best.replicates if r.duration_s is not None]
    return max([MIN_TRIAL_COST_S, *durations]) * 1.5


def _improvable(candidate: Candidate) -> bool:
    """A declared floor (`baseline: 0.5`) is scored but has no code to
    improve; everything else that scored is fair game."""
    if candidate.operator != "baseline":
        return True
    return (Path(candidate.candidate_dir) / "solution.py").exists()
