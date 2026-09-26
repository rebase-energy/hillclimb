"""Build a bare `Harness` for tests: no policy, no loop — drive it with
`harness.run(Action(...))` or `harness.execute(loop)`."""

from __future__ import annotations

from hillclimb.harness.budget import BudgetManager
from hillclimb.harness.dirs import create_search_dir
from hillclimb.harness.core import Harness
from hillclimb.harness.journal import Journal
from tests.conftest import local_executor


def make_harness(task, config, backend, *, name: str = "test-search", **kwargs):
    """-> (harness, journal, search_dir)"""
    search_dir = create_search_dir(config.paths.runs_dir, name)
    journal = Journal(search_dir / "journal.jsonl")
    kwargs.setdefault("budget", BudgetManager(3600, stop_margin_s=1))
    harness = Harness(
        problem=task,
        config=config,
        journal=journal,
        backend=backend,
        executor=local_executor(),
        search_dir=search_dir,
        log=lambda *_: None,
        **kwargs,
    )
    return harness, journal, search_dir


# --- a policy-driven rig for tests -----------------------------------------

from hillclimb.harness.candidate import Candidate  # noqa: E402
from hillclimb.harness.loop import PolicyLoop  # noqa: E402
from hillclimb.modules.policies.greedy import GreedyPolicy  # noqa: E402
from hillclimb.modules.policies.base import TUNE_ACTION, Action, SearchPolicy  # noqa: E402


class LiveParams(dict):
    """`config.climber.params`, read at access time — tests set knobs on the
    config after building the rig, and replace the dict wholesale."""

    def __init__(self, config):
        super().__init__()
        self._config = config

    def get(self, name, default=None):
        return self._config.climber.params.get(name, default)

    def __getitem__(self, name):
        return self._config.climber.params[name]

    def __contains__(self, name):
        return name in self._config.climber.params

    def __bool__(self):
        return True  # an empty overlay is still THE params (`params or {}` must not drop the live view)


class SearchRig(Harness):
    """A harness with a policy attached, for tests that poke at both: the
    serial `run_operator(op, target)` (what the policy WOULD attach to that
    operator, run on the calling thread), `decide()`, and the greedy
    introspection helpers. Production code never sees this class — a real
    search is `Harness.execute(PolicyLoop(policy))`.

    `run()` with no action runs the whole search; with one it is
    `Harness.run(action)`."""

    def __init__(self, *args, complexity_start: int = 0, policy: SearchPolicy | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.complexity_start = complexity_start
        self.policy = policy or GreedyPolicy(
            complexity_start=complexity_start, params=LiveParams(self.config)
        )
        self._loop = PolicyLoop(self.policy)
        self._loop.catch_up(self)  # the resume contract: the policy replays the journal

    def run(self, action: Action | None = None):
        if action is not None:
            return super().run(action)
        return self.execute(self._loop)

    def run_operator(self, operator: str, target: Candidate | None) -> Candidate:
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
        target_id = target.candidate_id if target else None
        maker = getattr(self.policy, "action_for", None)
        if maker is not None:
            return maker(self._view(), operator, target_id)
        return Action(operator=operator, target_id=target_id)

    def decide(self) -> tuple[str, Candidate | None]:
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
