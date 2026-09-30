"""The default policy: a greedy schedule over whatever its selector picks.

Priority: DEBUG the newest failing tip while its chain is shallow > ENSEMBLE
solo in the final budget window > DRAFT until `num_drafts` branches hold a
scored solution (the complexity cue escalates per draft) > TUNE a promising
candidate that declared params.json > EXPAND the candidate the selector picks
(`improve` it; draft when there is none to pick).

The schedule is this class; WHICH candidate is expanded is the selector's
(`select:` in the climber's block — `best` by default, `map-elites` for a
quality-diversity archive). The `openevolve` preset is exactly this policy
over `map-elites`, with ensemble and tune switched off.

The whole exploration process is ONE dict: the climber's `params`. Every
knob below is read from it, else from `DEFAULTS` — a policy never sees the
harness's config, and an agent editing the process is handed a single dict
(`resolved_params()` is that dict, fully resolved). All replay-deterministic —
every count is derived from the journal and the in-flight refs, never kept.

Schedule params (default in brackets):
  num_drafts (3)                     draft branches before expanding
  debug (True)                       repair failing tips at all
  max_debug_depth (3)                failed fixes per failing chain
  complexity_start (0)               offset of the draft-complexity cue
  ensemble (True)                    ensemble solo in the final window
  ensemble_reserve_fraction (0.2)    final slice of budget reserved
  ensemble_top_k (3)                 candidates blended
  ensemble_max_attempts (2)

Tune params:
  tune_budget (8)     extra trials per candidate beyond its defaults trial; 0 = off
  tune_gate ("band")  which passing+tunable candidates qualify: "band" = within the
                      accept band of the current best (best included), "best" =
                      the best only, "always" = every scored one
  tune_parallel (1)   tune jobs in flight per candidate (>1 engages the
                      tuner's constant liar)
  tune_burst (2)      tune trials released between agent proposals, so tuning
                      interleaves with improving instead of starving it
"""

from __future__ import annotations

from hillclimb.sdk import TUNE_ACTION, Action, Candidate, Policy, SearchState, improves


class Greedy(Policy):
    """Debug failing tips, draft a few branches, then expand what the selector picks; ensemble at the end."""

    name = "greedy"
    # this policy's knobs (the base adds `max_debug_depth`, `complexity_start`)
    DEFAULTS = {
        "num_drafts": 3,
        "debug": True,
        "ensemble": True,
        "ensemble_reserve_fraction": 0.2,
        "ensemble_top_k": 3,
        "ensemble_max_attempts": 2,
        "tune_budget": 8,
        "tune_gate": "band",
        "tune_parallel": 1,
        "tune_burst": 2,
    }

    # --- the contract ---

    def propose(self, state: SearchState) -> Action | None:
        """Pool policy: generalizes the serial rule to in-flight state.
        Priority: debug failing tips (one per chain) > ensemble solo in the
        final window (drain first) > drafts until num_drafts branches are
        scored-or-pending > tune > expand the selector's pick. None = hold."""
        self.selector.sync(state)
        tip = self.debuggable_tip(state)
        if tip is not None:
            return Action(operator="debug", target_id=tip.candidate_id)
        if self.should_ensemble(state) and not any(
            ref.operator == "ensemble" for ref in state.inflight
        ):
            if state.inflight:
                return None  # drain: ensemble inputs snapshot at launch
            return self._ensemble_action(state)
        if self.prospective_branches(state) < int(self.param("num_drafts")):
            return self._draft_action(state)
        tune = self.tune_target(state)
        if tune is not None:
            return Action(operator=TUNE_ACTION, target_id=tune.candidate_id)
        busy = {ref.parent_id for ref in state.inflight if ref.operator == "improve"}
        return self._expand_action(state, busy) or self._draft_action(state)

    def action_for(self, state: SearchState, operator: str, target_id: str | None) -> Action:
        """Fully-populated Action for an explicitly requested operator — the
        run_operator/smoke path, where the harness names the move and the
        policy fills in its decision-time details."""
        if operator == "draft":
            return self._draft_action(state)
        if operator == "ensemble":
            picks = self.ensemble_candidates(state)
            return Action(
                operator="ensemble",
                target_id=target_id or (picks[0].candidate_id if picks else None),
                inspiration_ids=tuple(c.candidate_id for c in picks),
            )
        if operator == "improve" and target_id is None:
            self.selector.sync(state)
            expand = self._expand_action(state, set())
            if expand is not None:
                return expand
        return Action(operator=operator, target_id=target_id)

    # --- the moves ---

    def debuggable_tip(self, state: SearchState) -> Candidate | None:
        return super().debuggable_tip(state) if bool(self.param("debug")) else None

    def _draft_action(self, state: SearchState) -> Action:
        return Action(
            operator="draft",
            args={"complexity": self.draft_complexity(state)},
            climber_meta=self.selector.creation_meta(state),
        )

    def _expand_action(self, state: SearchState, busy: set) -> Action | None:
        """`improve` the candidate the selector picks; None when it picks none."""
        selection = self.selector.select(state, busy=busy)
        if selection is None:
            return None
        return Action(
            operator="improve",
            target_id=selection.target_id,
            inspiration_ids=tuple(selection.inspiration_ids),
            extra_prompt_context=selection.prompt_context,
            climber_meta=dict(selection.meta),
        )

    def _ensemble_action(self, state: SearchState) -> Action:
        picks = self.ensemble_candidates(state)
        return Action(
            operator="ensemble",
            target_id=picks[0].candidate_id,
            inspiration_ids=tuple(c.candidate_id for c in picks),
        )

    # --- tune ---

    def tune_param(self, name: str):
        return self.param(name)

    def tune_target(self, state: SearchState) -> Candidate | None:
        """The candidate to spend the next tune trial on, or None. Derived
        from the journal + in-flight refs only: `spent` on a candidate is its
        trial count beyond the defaults trial plus its in-flight tune jobs, so
        a resumed search continues exactly where a killed one stopped."""
        budget = int(self.tune_param("tune_budget"))
        if budget <= 0:
            return None
        headroom = state.budget.remaining_s - state.budget.stop_margin_s
        journal = state.journal
        best = journal.best_candidate(state.higher_is_better)
        band = state.accept_band
        gate = str(self.tune_param("tune_gate"))
        parallel = int(self.tune_param("tune_parallel"))
        burst = int(self.tune_param("tune_burst"))
        direction = -1 if state.higher_is_better else 1
        ranked = sorted(
            (c for c in journal.scored_candidates() if c.tunable),
            key=lambda c: (direction * c.val_score, c.candidate_id),
        )
        agent_jobs = sum(1 for ref in state.inflight if ref.operator != TUNE_ACTION)
        for candidate in ranked:
            if not self._tune_gate_passes(candidate, best, band, gate, state.higher_is_better):
                continue
            if headroom < _trial_cost_s(candidate):
                continue  # a trial of this code could not finish before the wall
            inflight = sum(
                1 for ref in state.inflight
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
        return not improves(
            best.val_score, candidate.val_score, higher_is_better=higher_is_better, band=band
        )

    # --- ensemble ---

    def in_ensemble_window(self, state: SearchState) -> bool:
        # window sits ABOVE the stop margin, else margin swallows it: with a
        # 45m budget, reserve(540s) - margin(300s) left a 240s slot that one
        # improve cycle stepped over entirely
        budget = state.budget
        reserve = budget.total_s * float(self.param("ensemble_reserve_fraction"))
        return budget.remaining_s <= reserve + budget.stop_margin_s

    def should_ensemble(self, state: SearchState) -> bool:
        if not bool(self.param("ensemble")) or not self.in_ensemble_window(state):
            return False
        attempts = sum(
            1 for c in state.journal.candidates.values() if c.operator == "ensemble"
        )
        max_attempts = int(self.param("ensemble_max_attempts"))
        if attempts >= max_attempts or self.ensemble_succeeded(state):
            return False
        return len(self.ensemble_candidates(state)) >= 2

    def ensemble_succeeded(self, state: SearchState) -> bool:
        for candidate in state.journal.candidates.values():
            if candidate.status != "passing":
                continue
            root = state.journal.debug_chain(candidate.candidate_id)[0]
            if root.operator == "ensemble":
                return True
        return False

    def ensemble_candidates(self, state: SearchState) -> list[Candidate]:
        """Top-k scored non-ensemble candidates by val score, deduped by
        script content so near-identical improves don't fill the slots."""
        return self.top_distinct(state, int(self.param("ensemble_top_k")), skip_operator="ensemble")


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
