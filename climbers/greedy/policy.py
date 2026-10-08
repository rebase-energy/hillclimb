"""The bundled `greedy` climber, as one file: the selector policy (π_sel,
`Best`) and the operator policy (π_op, `Greedy`) that together decide what a
search tries next. Everything that decides is written out here — there is no
hidden schedule in a base class — so `hillclimb climber get greedy` copies
this file as is, and what you read is what runs.

Every step of a search is two decisions in a fixed order. First the selector
reads the history and answers WHICH node(s) the next attempt starts from:
a failing tip (repair comes first), the top candidates once the final
ensemble window is open, nothing while fewer than `num_drafts` roots exist (a
root step: a fresh draft), else the candidate `select` picks — for `Best`,
the best scored one nobody is building on yet. Then the operator policy
answers WHICH OPERATOR to apply to it: draft, debug, ensemble, improve — and
`Greedy` adds one decision of its own: TUNE the chosen candidate (a trial of
the same code with other parameter values, no coding agent) before improving
it, while it declared `params.json` and has tune budget left.

The whole exploration process is the two `DEFAULTS` dicts below. Every knob
is read with `self.param(name)`: the climber's `selector_params` /
`params` override a default, so `--set climber.params.tune_budget=0` or
`selector_params: {num_drafts: 5}` in a block edits the process without
touching the file. Both policies are replay-deterministic — every count is
derived from the journal and the in-flight refs, never kept — so a resumed
search continues exactly where a killed one stopped.

Selector knobs (`selector_params`, default in brackets):
  num_drafts (3)                     root candidates before anything is built on
  debug (True)                       repair failing tips at all
  max_debug_depth (3)                failed fixes per failing chain
  ensemble (True)                    combine the top candidates in the final window
  ensemble_reserve_fraction (0.2)    final slice of the budget reserved for it
  ensemble_top_k (3)                 candidates combined
  ensemble_max_attempts (2)          combinations tried in the window

Operator policy knobs (`params`, default in brackets):
  complexity_start (0)  offset of the draft-complexity cue (memory may have learned one)
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

from hillclimb.sdk import (
    TUNE_ACTION,
    Action,
    Candidate,
    OperatorPolicy,
    SearchState,
    Selection,
    SelectorPolicy,
    improvable,
    improves,
    top_distinct,
)


class Best(SelectorPolicy):
    """The best scored candidate that is not already being expanded."""

    name = "best"
    DEFAULTS = {
        "num_drafts": 3,
        "debug": True,
        "max_debug_depth": 3,
        "ensemble": True,
        "ensemble_reserve_fraction": 0.2,
        "ensemble_top_k": 3,
        "ensemble_max_attempts": 2,
    }

    # --- the schedule: which node(s) the next attempt starts from ---

    def schedule(self, state: SearchState, *, busy: frozenset[str] | set[str] = frozenset()) -> Selection | None:
        """In order: a failing tip (repair comes first), the top candidates
        once the final window is open (`combine=True`), None while fewer than
        `num_drafts` roots exist (a root step), else whatever `select`
        chooses. `busy` holds the candidates an in-flight attempt is already
        building on."""
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
        """The scored candidate to build on now. With several attempts in
        flight this spreads them over the top candidates instead of piling
        onto one; when every scored candidate is busy, the best gets another."""
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


class Greedy(OperatorPolicy):
    """Draft a root, debug a failing tip, ensemble the chosen set, and tune the chosen candidate before improving it."""

    name = "greedy"
    DEFAULTS = {
        "complexity_start": 0,
        "tune_budget": 8,
        "tune_gate": "band",
        "tune_parallel": 1,
        "tune_burst": 2,
    }

    # --- the contract ---

    def propose(self, state: SearchState, selection: Selection | None) -> Action | None:
        """The operator for what the selector chose, as an Action; None =
        hold (keep the slot empty until an in-flight result lands). No node:
        draft; the nodes of a combination: ensemble, once nothing is in
        flight; a failing node: debug; a scored node: a tune trial while it
        has budget for one, else improve."""
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
        if node.is_scored and self.tune_now(state, node):
            return Action(operator=TUNE_ACTION, target_id=node.candidate_id)
        return self.expand_action(state, selection)

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
            chosen = selector.select(state)
            if chosen is not None:
                return self.expand_action(state, chosen)
        return Action(operator=operator, target_id=target_id)

    # --- the moves ---

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

    def draft_complexity(self, state: SearchState) -> str:
        """The complexity cue for the next draft: it escalates per draft."""
        index = len(state.journal.drafts()) + int(self.param("complexity_start"))
        return "minimal" if index == 0 else "moderate" if index == 1 else "advanced"

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


# --- the climber -------------------------------------------------------------
# This file IS the climber (`climber: climbers/greedy/policy.py` in hillclimb.yaml,
# `--climber climbers/greedy/policy.py`): the two policies above, the operators that
# make an attempt (each renders its template under prompts/ beside this file — a
# climber file's prompts dir), the tuner and the memory. Edit anything here or a
# template and the next run climbs with the change; a started search keeps the
# copy it snapshotted.
from hillclimb import Climber
from hillclimb.memory import FilesMemory
from hillclimb.operators import Debug, Draft, Ensemble, Improve
from hillclimb.tuners import RandomSearch

climber = Climber(
    selector_policy=Best(),  # which candidate the next attempt starts from
    operator_policy=Greedy(),  # which operator to apply to it
    operators=[Draft(), Debug(), Improve(), Ensemble()],
    tuner=RandomSearch(),  # which parameter values a tunable candidate tries
    memory=FilesMemory(),  # what a search knows from earlier ones, and leaves for the next
    name='greedy',
)
