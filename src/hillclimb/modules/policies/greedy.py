"""The bundled policy — π_op: which operator for what the selector chose.

The selector (π_sel) has already said which node the next attempt starts
from: a failing tip, nothing (a root step), the candidates to combine, or a
scored candidate to build on. This policy maps that to an operator the way
the base does — draft, debug, ensemble, improve — and adds one decision of
its own: TUNE the chosen candidate (a trial of the same code with other
parameter values, no coding agent) before improving it, while it declared
`params.json` and has tune budget left.

The whole exploration process is ONE dict per module: the climber's
`params` (this policy's knobs) and `selector_params` (the selector policy's schedule:
`num_drafts`, `debug`, `max_debug_depth`, `ensemble`, …). Every knob is read
from it, else from `DEFAULTS` — a policy never sees the harness's config, and
a coding agent editing the process is handed a single dict
(`resolved_params()` is that dict, fully resolved). All replay-deterministic —
every count is derived from the journal and the in-flight refs, never kept.

Params (default in brackets):
  complexity_start (0)  offset of the draft-complexity cue
  tune_budget (8)       extra trials per candidate beyond its defaults trial; 0 = off
  tune_gate ("band")    which chosen candidates get tuned: "band" = within the
                        accept band of the current best (best included), "best" =
                        the best only, "always" = every scored one
  tune_parallel (1)     tune jobs in flight per candidate (>1 engages the
                        tuner's constant liar)
  tune_burst (2)        tune trials released between coding agent proposals, so tuning
                        interleaves with improving instead of starving it
"""

from __future__ import annotations

from hillclimb.sdk import TUNE_ACTION, Action, Candidate, OperatorPolicy, SearchState, Selection, improves


class Greedy(OperatorPolicy):
    """Draft a root, debug a failing tip, ensemble the chosen set, and tune the chosen candidate before improving it."""

    name = "greedy"
    # this policy's knobs (the base adds `complexity_start`)
    DEFAULTS = {
        "tune_budget": 8,
        "tune_gate": "band",
        "tune_parallel": 1,
        "tune_burst": 2,
    }

    # --- the contract ---

    def propose(self, state: SearchState, selection: Selection | None) -> Action | None:
        """The base mapping, with a tune trial of a chosen scored candidate
        ahead of improving it."""
        if selection is not None and not selection.combine:
            node = state.journal.candidates[selection.target_id]
            if node.is_scored and self.tune_now(state, node):
                return Action(operator=TUNE_ACTION, target_id=node.candidate_id)
        return super().propose(state, selection)

    def action_for(self, state: SearchState, operator: str, target_id: str | None) -> Action:
        """Fully-populated Action for an explicitly requested operator — the
        run_operator path, where the harness names the move and the policy
        fills in its decision-time details."""
        selector = self.selector
        if operator == "draft":
            return self.draft_action(state)
        if operator == "ensemble":
            picks = selector.combine_candidates(state) if selector is not None else []
            return Action(
                operator="ensemble",
                target_id=target_id or (picks[0].candidate_id if picks else None),
                inspiration_ids=tuple(c.candidate_id for c in picks),
            )
        if operator == "improve" and target_id is None and selector is not None:
            selector.sync(state)
            pick = selector.pick(state)
            if pick is not None:
                return self.expand_action(state, pick)
        return Action(operator=operator, target_id=target_id)

    # --- tune ---

    def tune_param(self, name: str):
        return self.param(name)

    def tune_now(self, state: SearchState, candidate: Candidate) -> bool:
        """Spend the next tune trial on the chosen candidate? Derived from
        the journal + in-flight refs only: `spent` on a candidate is its
        trial count beyond the defaults trial plus its in-flight tune jobs, so
        a resumed search continues exactly where a killed one stopped."""
        budget = int(self.tune_param("tune_budget"))
        if budget <= 0 or not candidate.tunable:
            return False
        journal = state.journal
        best = journal.best_candidate(state.higher_is_better)
        if not self._tune_gate_passes(
            candidate, best, state.accept_band, str(self.tune_param("tune_gate")), state.higher_is_better
        ):
            return False
        headroom = state.budget.remaining_s - state.budget.stop_margin_s
        if headroom < _trial_cost_s(candidate):
            return False  # a trial of this code could not finish before the wall
        inflight = sum(
            1 for ref in state.inflight
            if ref.operator == TUNE_ACTION and ref.candidate_id == candidate.candidate_id
        )
        spent = (len(candidate.trials) - 1) + inflight
        if spent >= budget or inflight >= int(self.tune_param("tune_parallel")):
            return False
        # interleave: allow `burst` tune trials per coding agent proposal made
        # since this candidate landed (later ids + in-flight coding agent jobs)
        agent_jobs = sum(1 for ref in state.inflight if ref.operator != TUNE_ACTION)
        agents_since = sum(1 for cid in journal.candidates if cid > candidate.candidate_id)
        return spent < int(self.tune_param("tune_burst")) * (1 + agents_since + agent_jobs)

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
