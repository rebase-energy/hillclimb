"""`GreedySearcher`: the pre-split name for "a harness driven by a policy".

The engine now lives in two places — `hillclimb.harness.Harness` (the fixed
core) and `hillclimb.loop.PolicyLoop` (the control flow that asks a
`SearchPolicy` what to do next). This shim keeps the old constructor and the
serial `run_operator`/`decide` test API alive while callers move to
`Harness` + `harness.run(Action(...))`; it goes away with them.
"""

from __future__ import annotations

from hillclimb.candidate import Candidate
from hillclimb.evaluation import TAIL_CHARS, tail  # noqa: F401 — re-exported (cli imports tail from here)
from hillclimb.harness.core import Harness, Job, OutcomeMsg  # noqa: F401 — re-exported
from hillclimb.loop import PolicyLoop
from hillclimb.policies.greedy import GreedyPolicy
from hillclimb.policy import TUNE_ACTION, Action, SearchPolicy
from hillclimb.search_strategy import ParkedSearch, StopRequested  # noqa: F401 — re-exported (their historic home)


class GreedySearcher(Harness):
    def __init__(self, *args, complexity_start: int = 0, policy: SearchPolicy | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.complexity_start = complexity_start  # learned draft-complexity offset
        self.policy = policy or GreedyPolicy(complexity_start=complexity_start)
        self._loop = PolicyLoop(self.policy)
        self._loop.catch_up(self)  # resume contract: the policy replays the journal

    def run(self, action: Action | None = None):
        """No action: run the whole search (the historic meaning). With one:
        the harness's blocking single attempt."""
        if action is not None:
            return super().run(action)
        return self.execute(self._loop)

    # --- the serial test / smoke API ---

    def run_operator(self, operator: str, target: Candidate | None) -> Candidate:
        """One operator invocation: DRAFT/DEBUG/IMPROVE/ENSEMBLE + execute +
        journal, serially on the calling thread."""
        if operator == TUNE_ACTION:
            job = self._prepare_tune(self._action_for(operator, target))
            if job is None:
                raise ValueError(f"{target.candidate_id if target else None} cannot be tuned")
        else:
            job = self._prepare(self._action_for(operator, target))
        try:
            return self._commit(self._execute_job(job))
        finally:
            live = self.journal.candidates.get(job.candidate.candidate_id)
            self._loop.observe(self, live.holdout_blind() if live is not None else None)

    def _action_for(self, operator: str, target: Candidate | None) -> Action:
        """The action the policy would attach to (operator, target) —
        inspirations for an ensemble, a complexity cue for a draft."""
        target_id = target.candidate_id if target else None
        maker = getattr(self.policy, "action_for", None)
        if maker is not None:
            return maker(self._view(), operator, target_id)
        return Action(operator=operator, target_id=target_id)

    # --- policy introspection ---

    def decide(self) -> tuple[str, Candidate | None]:
        """What the policy would do next, as (operator, target)."""
        action = self.policy.propose(self._view())
        if action is None:
            return ("hold", None)
        target = self.journal.candidates.get(action.target_id) if action.target_id else None
        return (action.operator, target)

    def _debuggable_tip(self) -> Candidate | None:
        return self.policy.debuggable_tip(self._view())

    def _prospective_branches(self) -> int:
        return self.policy.prospective_branches(self._view())

    def _in_ensemble_window(self) -> bool:
        return self.policy.in_ensemble_window(self._view())

    def _should_ensemble(self) -> bool:
        return self.policy.should_ensemble(self._view())

    def _ensemble_succeeded(self) -> bool:
        return self.policy.ensemble_succeeded(self._view())

    def _ensemble_candidates(self) -> list[Candidate]:
        return self.policy.ensemble_candidates(self._view())

    def _draft_complexity(self) -> str:
        return self.policy.draft_complexity(self._view())
