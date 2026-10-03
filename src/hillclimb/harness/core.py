from __future__ import annotations

import queue
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from hillclimb.agents.base import Agent, AgentRequest, AgentResult
from hillclimb.harness.baseline import write_baseline
from hillclimb.harness.budget import BudgetManager, Spend, journal_spend
from hillclimb.harness.candidate import AgentInfo, Candidate, read_solution, source_hash, utcnow
from hillclimb.config import Config
from hillclimb.harness.control import ControlCommand, apply_prune, drain_commands_dir, resync_best
from hillclimb.harness import evaluation
from hillclimb.harness.evaluation import CandidateEvaluator
from hillclimb.harness.executor import Executor
from hillclimb.harness.journal import Journal, JournalView
from hillclimb.harness.params import ParamsFile, read_candidate_space, write_inherited_params
from hillclimb.harness.loop import ClimberError, HarnessClosed, Outcome, SearchInfo, Loop, Ticket
from hillclimb.modules.policies.base import INJECT_ACTION, TUNE_ACTION, Action, BudgetView, InflightRef, SearchState
from hillclimb.climber import OperatorSet
from hillclimb.modules.operators import (
    CONTRACT_TOKEN,
    MemoryContext,
    Operator,
    OperatorContext,
    Attempt,
    ProblemInfo,
    get_operator,
    inspiration_filename,
)
from hillclimb.prompts.render import render
from hillclimb.harness.routing import AgentPool, ResolvedRoute, Router
from hillclimb.harness.sandbox import agent_policy
from hillclimb.harness.glue import ParkedSearch, StopRequested
from hillclimb.harness.slots import MachineSlots
from hillclimb.spaces import describe_params as spaces_describe, with_values
from hillclimb.harness.status import CandidateCounts, CurrentCandidate, ScoreRef, StatusWriter
from hillclimb.modules.tuners.base import Tuner, history_for, tune_seed
from hillclimb.problem import ProblemSpec
from hillclimb.harness.dirs import create_candidate_dir


def landed_line(candidate: Candidate) -> str:
    """The one log line a finished candidate gets: its status, its score (or
    why it has none) and how long it took, nested under the line that
    started it. `  new selection:` follows when it is the new best."""
    took = ""
    try:
        from datetime import datetime

        start = datetime.fromisoformat(candidate.created_at)
        end = datetime.fromisoformat(candidate.finished_at) if candidate.finished_at else None
        if end is not None:
            seconds = max(0, int((end - start).total_seconds()))
            took = f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"
    except (TypeError, ValueError):
        pass
    line = f"  {candidate.candidate_id} {candidate.status}"
    if candidate.status == "passing" and candidate.val_score is not None:
        line += f" val={candidate.val_score:.5g}"
    if took:
        line += f" ({took})"
    reason = _failure_reason(candidate)
    if reason:
        line += f" — {reason}"
    return line


def _failure_reason(candidate: Candidate) -> str | None:
    """Why a candidate has no score, in a few words. A buggy or failing one
    says what its verifier run did (the summary is the agent's own account
    of its code); an abandoned or parked one carries the engine's reason in
    its summary."""
    if candidate.status == "passing":
        return None
    if candidate.status in ("buggy", "failing"):
        trial = candidate.last_trial
        if trial is not None and trial.unit_tests is not None and not trial.unit_tests.passed:
            return "unit tests failed"
        replicate = trial.replicates[-1] if trial is not None and trial.replicates else None
        if replicate is not None:
            if replicate.timed_out:
                return f"timed out after {replicate.duration_s:.0f}s"
            if replicate.returncode == 124:
                return "ran past its time limit"  # a two-step problem's own limit (and timeout(1)'s code)
            if replicate.returncode:
                return f"verifier exited {replicate.returncode}"
            if not replicate.submission_ok:
                return "no valid output"
    first = candidate.summary.splitlines()[0][:160] if candidate.summary else ""
    return first or None


@dataclass
class Job:
    """One operator submission: everything a worker needs, nothing it must
    share. Created scheduler-side in _prepare; the candidate object is owned
    by the worker until _commit (the journal holds its own deep copy)."""

    candidate: Candidate
    request: AgentRequest | None  # None for the coding-agent-less seed candidate
    candidate_dir: Path
    ensemble_inputs: "list[Candidate] | None" = None
    agent: Agent | None = None  # routed instance; None = harness default
    # tune jobs: an extra trial on an EXISTING candidate — `candidate` is a
    # deep copy the worker may mutate, the live object is only touched in
    # _commit_tune under the state lock
    kind: str = "operator"  # operator | tune
    # `Attempt.require_change`: the parent's source hash — a coding agent that
    # hands back the same text made no attempt
    unchanged_hash: str | None = None
    trial_index: int | None = None
    params: dict | None = None
    declaration: ParamsFile | None = None

    @property
    def key(self) -> str:
        """In-flight registry key: one slot per candidate for operator jobs,
        one per (candidate, trial) for tune jobs."""
        if self.kind == "tune":
            return f"{self.candidate.candidate_id}:t{self.trial_index}"
        return self.candidate.candidate_id


@dataclass
class OutcomeMsg:
    """Terminal report from a worker; consumed by _commit on the scheduler."""

    job: Job
    kind: str  # parked | aborted | agent_failed | no_solution | unchanged | executed | tuned
    result: AgentResult | None = None
    all_ok: bool = False  # every replicate passed AND (when scored) the hidden split did too
    # the verifier ran under a timeout clamped by the search's remaining
    # budget rather than the problem's own execution limit: a trial killed
    # by that timeout was cut off by the clock, not shown to be buggy
    budget_clamped: bool = False
    error: BaseException | None = None


class _OperatorServices:
    """The harness side of an `OperatorContext`: what needs the engine's
    templates, reports or unmasked records. Candidates are named by id so the
    operator only ever holds holdout-blind copies."""

    def __init__(self, searcher: "Harness"):
        self._searcher = searcher

    def render(self, template: str, **tokens) -> str:
        # the climber's own prompts shadow the built-in operator templates;
        # the contract is rendered by the harness and never goes through here
        return render(template, _override=self._searcher.prompts_dir, **tokens)

    def live_experience(self) -> str:
        return self._searcher._live_experience()

    def failure_reason(self, candidate_id: str) -> str:
        return self._searcher._failure_reason(self._searcher.journal.get(candidate_id))

    def report_section(self, candidate_id: str) -> str:
        return self._searcher._report_section(self._searcher.journal.get(candidate_id))


class Harness:
    """The fixed core of a search: runs whatever a `Loop` submits and
    owns every state invariant — candidate dirs, coding agent calls, verifier trials,
    the journal (single writer), `best/` selection, the accept band, budgets,
    the control queue, crash recovery and holdout. A loop reaches it only
    through `view`/`capacity`/`inflight`/`open`/`submit`/`wait`/`run`/`source`
    (`hillclimb.harness.loop.Harness`); `execute(loop)` runs a search to its end.
    """

    def __init__(
        self,
        problem: ProblemSpec,
        config: Config,
        journal: Journal,
        agent: Agent,
        executor: Executor,
        budget: BudgetManager,
        search_dir: Path,
        max_candidates: int | None = None,
        log=print,
        evaluator: CandidateEvaluator | None = None,
        status: StatusWriter | None = None,
        slots: MachineSlots | None = None,
        abort: threading.Event | None = None,
        seed_solution: Path | None = None,
        memory=None,
        retrieved=None,
        router: Router | None = None,
        agents: AgentPool | None = None,
        drain_commands: Callable[[], list[ControlCommand]] | None = None,
        tuner: Tuner | None = None,
        operators: OperatorSet | None = None,
        prompts_dir: Path | None = None,
    ):
        # what this search's climber brought: the operators it may use and the
        # prompts dir that shadows built-in operator templates by name. None =
        # the four built-ins configured from the config's `operators:` block.
        self.operators = operators
        self.prompts_dir = prompts_dir
        # where queued stop/prune commands come from: the store's queue for
        # this search (the engine binds it), else the search dir's control/
        self.drain_commands = drain_commands or (lambda: drain_commands_dir(search_dir))
        self.problem = problem
        self.config = config
        self.journal = journal
        self.agent = agent
        self.executor = executor
        self.budget = budget
        self.search_dir = search_dir
        # journal-size cap (baseline and seed included): a test knob — the
        # user-facing limits are budget.max_evaluations / max_tokens
        self.max_candidates = max_candidates
        self.log = log
        # the host's evaluate service: verifier trials, and holdout — the
        # hidden split is scored in there, never here (see evaluation.py)
        self.evaluator = evaluator or CandidateEvaluator(
            executor=executor, problem=problem, config=config
        )
        self.status = status
        self.slots = slots  # machine-wide coding-agent-concurrency cap (optional)
        self.abort = abort or threading.Event()
        # a stop or a hard deadline must reach the verifier and the unit
        # tests too, not only the coding agent: whatever runs them and takes
        # the signal gets it (custom executors without `abort` are left alone)
        for runner in (self.evaluator.executor, getattr(self.evaluator, "unit_test_runner", None)):
            if runner is not None and hasattr(runner, "abort"):
                runner.abort = self.abort
        self.seed_solution = seed_solution  # incumbent model: scored as a floor candidate
        # cross-search memory (`modules/memory`): what it handed this search
        # before it started — the prior-experience prompt section, and a
        # proven solution copied into the first draft's candidate dir as
        # reference_solution.py (later drafts explore) — and the memory
        # itself, for what siblings share while the search runs
        from hillclimb.modules.memory.base import Memory, Retrieved

        self.memory = memory if memory is not None else Memory()
        self.retrieved = retrieved if retrieved is not None else Retrieved()
        if tuner is None:
            from hillclimb.modules.tuners.random_search import RandomSearch

            tuner = RandomSearch(config.climber.tuner_params)
        self.tuner = tuner  # which params a `tune` action tries; WHEN is the policy's call
        self.router = router  # None: everything routes to `agent` + config.model
        self.agents = agents
        self._consecutive_failures = 0
        # Concurrency contract: the Journal and everything below is touched
        # only by the scheduler (the thread running run()/run_operator),
        # belt-and-braces guarded by _state_lock; workers report through
        # _done_q and never see the journal.
        self._state_lock = threading.Lock()
        self._inflight: dict[str, Job] = {}
        self._done_q: "queue.Queue[OutcomeMsg]" = queue.Queue()
        self._pool: ThreadPoolExecutor | None = None  # lives for one execute()
        self._owner: int | None = None  # the loop's thread, once execute() runs
        # why no new work may start: a stop/park waiting to be raised by
        # execute(), or the hard budget deadline. Latched, never raised into
        # loop code — a loop that swallowed it could otherwise keep spending.
        self._latch: Exception | None = None
        self._finished: str | None = None
        self._rejections = 0
        # commands a control watcher drained while the loop's thread was busy
        # in a blocking job (run(), the seed); _process_control applies them
        self._deferred_commands: list[ControlCommand] = []
        self._deferred_lock = threading.Lock()
        for stale in journal.pending_candidates():
            # a pending candidate at construction time means a previous
            # orchestrator process died mid-operator (crash/kill); its work is
            # unaccounted and must not block decide() forever
            stale.status = "abandoned"
            stale.summary = stale.summary or "orchestrator died mid-operator (crash recovery)"
            stale.finished_at = utcnow()
            journal.candidate_result(stale)
            log(f"  recovered stale pending candidate {stale.candidate_id} -> abandoned")
        existing = journal.selected_candidate(problem.higher_is_better, config.holdout.selection)
        self._selection_id = existing.candidate_id if existing else None  # resume-safe
        # resume contract: the routing bandit rebuilds from the replayed
        # journal (in journal order, after stale-pending recovery); a policy
        # does the same through its loop (`PolicyLoop.catch_up`)
        for candidate in journal.candidates.values():
            self._observe_route(candidate)

    @property
    def data_dir(self) -> Path:
        return self.problem.data_dir

    def _view(self) -> SearchState:
        """Snapshot of search state for a policy call. Scheduler-thread only —
        same discipline as every other journal touch."""
        return SearchState(
            journal=self.journal,
            inflight=self.inflight,
            budget=self._budget_view(),
            accept_band=self.accept_band(),
            higher_is_better=self.problem.higher_is_better,
        )

    def _budget_view(self) -> BudgetView:
        """The clock plus what is left of every other dimension the user
        limited. Reserved in-flight evaluations are already taken off."""
        limits, spend = self.config.budget, self.spend()
        return BudgetView(
            remaining_s=self.budget.remaining(),
            total_s=self.budget.total_s,
            stop_margin_s=self.budget.stop_margin_s,
            evaluations_remaining=(
                max(0, limits.max_evaluations - spend.evaluations - len(self._inflight))
                if limits.max_evaluations else None
            ),
            tokens_remaining=max(0, limits.max_tokens - spend.tokens) if limits.max_tokens else None,
            cost_remaining_usd=(
                max(0.0, limits.max_cost_usd - spend.cost_usd) if limits.max_cost_usd else None
            ),
        )

    # --- the interface a Loop sees (hillclimb.harness.loop.Harness) ---

    @property
    def info(self) -> SearchInfo:
        return SearchInfo(
            problem_id=self.problem.problem_id,
            description=self.problem.description,
            metric_name=self.problem.metric_name,
            higher_is_better=self.problem.higher_is_better,
            parallelism=self.parallelism,
            baseline_source=self.problem.baseline_text or None,
        )

    @property
    def state_dir(self) -> Path:
        path = self.search_dir / "loop"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def parallelism(self) -> int:
        return max(1, self.config.concurrency.parallel_agents)

    def view(self) -> SearchState:
        return self._view()

    @property
    def inflight(self) -> tuple[InflightRef, ...]:
        return tuple(
            InflightRef(
                candidate_id=job.candidate.candidate_id,
                operator=TUNE_ACTION if job.kind == "tune" else job.candidate.operator,
                parent_id=None if job.kind == "tune" else job.candidate.parent_id,
                trial_index=job.trial_index,
            )
            for job in self._inflight.values()
        )

    def spend(self) -> Spend:
        """Evaluations, tokens and cost so far (the clock is `budget`)."""
        return journal_spend(self.journal)

    def _closed_on_this_thread(self) -> str | None:
        """Every reason to start no new work that can only change on the
        loop's thread (a commit, a tick) — so a loop that just read
        `capacity` cannot lose a race to it."""
        if self._latch is not None:
            return str(self._latch)
        if self._finished is not None:
            return self._finished
        if self.max_candidates is not None and len(self.journal.candidates) >= self.max_candidates:
            return "evaluation cap reached"
        limits = self.config.budget
        if limits.max_evaluations or limits.max_tokens:
            spend = self.spend()
            # work in flight has its evaluation reserved: the cap is never overshot
            if limits.max_evaluations and spend.evaluations + len(self._inflight) >= limits.max_evaluations:
                return "evaluation budget spent"
            if limits.max_tokens and spend.tokens >= limits.max_tokens:
                return "token budget spent"
        return None

    @property
    def closed_reason(self) -> str | None:
        reason = self._closed_on_this_thread()
        if reason is None and self.budget.should_stop():
            reason = "out of budget"
        return reason

    @property
    def open(self) -> bool:
        return self.closed_reason is None

    @property
    def capacity(self) -> int:
        return max(0, self.parallelism - len(self._inflight)) if self.open else 0

    def source(self, candidate_id: str) -> str | None:
        candidate = self.journal.candidates.get(candidate_id)
        return read_solution(candidate) if candidate is not None else None

    def submit(self, action: Action) -> Ticket:
        self._assert_owner()
        refusal = self._refuse_new_work()
        if refusal is not None:
            return Ticket(id="", action=action, rejected=refusal)
        if len(self._inflight) >= self.parallelism:
            return self._reject(action, "no free slot")
        job = self._prepare_job(action)
        if isinstance(job, Ticket):
            return job
        if self._pool is None:
            raise RuntimeError("submit() outside execute(): use run() for a single blocking attempt")
        self._submit(self._pool, job)
        return self._ticket(action, job)

    def wait(self, timeout: float | None = None) -> list[Outcome]:
        self._assert_owner()
        while True:
            if (
                self._inflight
                and self.config.budget.deadline == "hard"
                and self.budget.remaining() <= 0
            ):
                # the budget is a gate on starting work; `hard` makes it a
                # wall too — whatever is still running is cut off now
                self.log(
                    f"  budget deadline reached: aborting {len(self._inflight)} "
                    "in-flight operator(s) (budget.deadline=hard)"
                )
                self.abort.set()
                self._drain_aborted(reason="cut off at the budget deadline")
                self._finished = "budget deadline"
                return []
            try:
                msg = self._done_q.get(timeout=1.0 if timeout is None else timeout)
            except queue.Empty:
                self._tick()
                if timeout is not None or not self._inflight:
                    return []
                continue  # still waiting: re-check the wall, poll again
            outcome = self._commit_outcome(msg)
            self._tick()
            return [outcome]

    def run(self, action: Action) -> Outcome:
        self._assert_owner()
        refusal = self._refuse_new_work()
        if refusal is not None:
            return Outcome(ticket=Ticket(id="", action=action, rejected=refusal), kind="rejected", candidate=None)
        job = self._prepare_job(action)
        if isinstance(job, Ticket):
            return Outcome(ticket=job, kind="rejected", candidate=None)
        self._inflight[job.key] = job
        try:
            with self._watch_control():
                msg = self._execute_job(job)
        except (StopRequested, KeyboardInterrupt):
            # no worker will ever report this job: journal it now, so nothing
            # is left in flight (or pending) behind an interrupted call
            self._abandon(job)
            raise
        outcome = self._commit_outcome(msg)
        self._tick()
        return outcome

    # --- running a search ---

    def start(self) -> None:
        """The search's floor: the baseline, then the seed. Idempotent (a
        resumed search has both), so `execute` runs it first and a caller
        that drives `run()` itself calls it before its first action."""
        if not self.journal.candidates:
            baseline = write_baseline(
                self.problem,
                self.search_dir,
                evaluator=self.evaluator,
                timeout_s=self.config.budget.exec_timeout_s,
            )
            self._record_result(baseline)
            self.log(f"baseline written: {baseline.summary}")
        if self.seed_solution is not None and not any(
            c.operator == "seed" for c in self.journal.candidates.values()
        ):
            self._run_seed()  # resume-idempotent: at most one seed per search

    def execute(self, loop: Loop) -> Candidate | None:
        """Run `loop` to the end of the search. Raises ParkedSearch /
        StopRequested once everything in flight has been committed."""
        self.start()
        # one pool for every parallelism level: n=1 is a pool with one slot.
        # Recorded golden sequences pin it to the historical serial behavior.
        self._pool = ThreadPoolExecutor(max_workers=self.parallelism, thread_name_prefix="operator")
        self._owner = threading.get_ident()
        try:
            self._tick()
            if self._latch is None:
                loop.run(self)
            while self._inflight and self._finished is None:
                self.wait()  # a loop that left work in flight: commit it
        except (StopRequested, KeyboardInterrupt):
            self.abort.set()
            self._drain_aborted()
            raise
        finally:
            self._pool.shutdown(wait=True)
            self._pool, self._owner = None, None
        self.raise_latched()
        return self.journal.selected_candidate(
            self.problem.higher_is_better, self.config.holdout.selection
        )

    def raise_latched(self) -> None:
        """Raise the stop or park that closed the harness, if one did. A stop
        or park never raises into loop code (`_tick` latches it): whoever
        drives the harness asks here, once everything in flight has landed."""
        if self._latch is not None:
            if isinstance(self._latch, StopRequested):
                self.abort.set()
            raise self._latch

    def clear_strikes(self) -> None:
        """Forget refused actions. Three refusals in a row end a search whose
        POLICY keeps proposing the impossible; a person trying an action by
        hand is not a policy, and whoever drives for them says so here."""
        self._rejections = 0

    def _refuse_new_work(self) -> str | None:
        """Raise on a closed harness — except when only the CLOCK closed it.
        The budget is the one thing that moves off the loop's thread, so a
        loop that just saw `capacity` may lose that race through no fault of
        its own: that is a quiet refusal, not an error (and not a strike)."""
        reason = self._closed_on_this_thread()
        if reason is not None:
            raise HarnessClosed(reason)
        return "out of budget" if self.budget.should_stop() else None

    def request_stop(self, reason: str) -> None:
        """Close the harness from OUTSIDE the loop's call stack (a signal
        handler): latch a stop and abort what is running. Setting two
        attributes is safe at any point; the latch is what guarantees that a
        loop — or a library under it — which swallows the StopRequested the
        handler raises next still cannot start anything."""
        if self._latch is None:
            self._latch = StopRequested(reason)
        self.abort.set()

    def _assert_owner(self) -> None:
        if self._owner is not None and threading.get_ident() != self._owner:
            raise RuntimeError("the harness commits on the loop's thread: call it from there")

    def _tick(self) -> None:
        """Control commands and the cost ceiling; a stop or park closes the
        harness instead of raising into the loop."""
        if self._latch is not None:
            return
        try:
            self._process_control()
            self._check_cost_ceiling()
        except (StopRequested, ParkedSearch) as exc:
            self._close(exc)

    def _close(self, exc: Exception) -> None:
        if self._latch is None:
            self._latch = exc
            if self._inflight:
                self.log(f"  {exc}: draining in-flight operators")

    def _prepare_job(self, action: Action) -> Job | Ticket:
        """Materialize an action, or the rejected Ticket that says why not —
        in which case nothing was created, journaled or spent."""
        with self._state_lock:
            try:
                if action.operator == TUNE_ACTION:
                    job = self._prepare_tune(action)
                    reason = None if job is not None else "tune refused (see log)"
                elif action.operator == INJECT_ACTION:
                    job, reason = self._prepare_inject(action), None
                else:
                    job, reason = self._prepare(action), None
            except KeyError as exc:  # a target or inspiration id the journal does not hold
                job, reason = None, f"references {exc.args[0]} which is not in the journal"
            except ValueError as exc:
                job, reason = None, str(exc)
        if job is None:
            return self._reject(action, reason)
        self._rejections = 0
        return job

    def _reject(self, action: Action, reason: str) -> Ticket:
        self.log(f"  refused {action.operator}" + (f" -> {action.target_id}" if action.target_id else "") + f": {reason}")
        self.journal.audit_event(
            "action_rejected", operator=action.operator, target_id=action.target_id, reason=reason
        )
        self._rejections += 1
        if self._rejections >= 3 and not self._inflight:
            raise ClimberError(f"3 actions refused in a row with nothing in flight; last: {reason}")
        return Ticket(id="", action=action, rejected=reason)

    @staticmethod
    def _ticket(action: Action, job: Job) -> Ticket:
        return Ticket(
            id=job.key,
            action=action,
            candidate_id=job.candidate.candidate_id,
            trial_index=job.trial_index,
        )

    _OUTCOME_KINDS = {"executed": "evaluated", "worker_crashed": "crashed"}

    def _commit_outcome(self, msg: OutcomeMsg) -> Outcome:
        """Commit one worker report on the loop's thread and describe it to
        the loop. A park closes the harness; the candidate is journaled
        either way."""
        job = msg.job
        action = Action(
            operator=TUNE_ACTION if job.kind == "tune" else job.candidate.operator,
            target_id=job.candidate.candidate_id if job.kind == "tune" else job.candidate.parent_id,
        )
        kind = self._OUTCOME_KINDS.get(msg.kind, msg.kind)
        try:
            committed = self._commit(msg)
        except ParkedSearch as exc:
            self._close(exc)
            committed = job.candidate
        if job.kind == "tune" and committed is None:
            kind = "discarded"
        elif kind == "evaluated" and committed.status == "abandoned":
            kind = "cut_off"
        live = self.journal.candidates.get(job.candidate.candidate_id)
        blind = live.holdout_blind() if live is not None else None
        return Outcome(
            ticket=self._ticket(action, job),
            kind=kind,
            candidate=blind,
            # projected from the holdout-blind copy: a loop's scoring view can
            # never carry what the mask removed
            result=evaluation.eval_result_for(blind) if blind is not None and blind.trials else None,
        )

    def _submit(self, pool: ThreadPoolExecutor, job: Job) -> None:
        # `[m:ss left]` heads every line that starts work: the log's clock
        # gutter (the CLI aligns on it); the rest is one short clause
        if job.kind == "tune":
            values = " ".join(
                f"{name}={value:.4g}" if isinstance(value, float) else f"{name}={value}"
                for name, value in (job.params or {}).items()
            )
            self.log(
                f"[{self.budget.clock_str()} left] tune {job.candidate.candidate_id} "
                f"t{job.trial_index} {values}".rstrip()
            )
        else:
            self.log(
                f"[{self.budget.clock_str()} left] {job.candidate.operator}"
                + (f" -> {job.candidate.parent_id}" if job.candidate.parent_id else "")
                + f" ({job.candidate.candidate_id})"
            )
        self._inflight[job.key] = job

        def work() -> None:
            try:
                outcome = self._execute_job(job)
            except BaseException as exc:  # noqa: BLE001
                from hillclimb.harness.unit_tests import UnitTestInfrastructureError

                if isinstance(exc, UnitTestInfrastructureError):
                    outcome = OutcomeMsg(job=job, kind="infrastructure_failed", error=exc)
                    self._done_q.put(outcome)
                    return
                # a worker that dies without reporting would leave the
                # candidate in flight and the scheduler waiting forever
                self.log(f"  worker for {job.candidate.candidate_id} crashed: {exc!r}")
                outcome = OutcomeMsg(job=job, kind="worker_crashed")
            self._done_q.put(outcome)

        pool.submit(work)

    def _drain_aborted(self, reason: str | None = None) -> None:
        """After abort: collect whatever workers return (they die within ~1s
        poll intervals) and journal the candidates as abandoned. `reason`
        names why in the summary (a hard budget deadline); without one, a
        summary the coding agent already wrote is kept."""
        while self._inflight:
            try:
                msg = self._done_q.get(timeout=30.0)
            except queue.Empty:
                break  # workers wedged; stale-pending recovery handles them on resume
            self._abandon(msg.job, reason)

    def _abandon(self, job: Job, reason: str | None = None) -> None:
        """Journal a job that will never report as abandoned (a tune trial:
        discarded, nothing lands on its candidate)."""
        if job.kind == "tune":
            self._discard_tune(job, reason or "stopped mid-tune (abort)")
            return
        candidate = job.candidate
        candidate.status = "abandoned"
        candidate.summary = reason or candidate.summary or "stopped mid-operator (abort)"
        candidate.finished_at = utcnow()
        self._record_result(candidate)
        self._inflight.pop(job.key, None)
        if self.status is not None:
            self.status.remove_current(candidate.candidate_id)

    # --- cost accounting ---

    def total_cost_usd(self) -> float:
        return sum(
            c.agent.cost_usd or 0.0
            for c in self.journal.candidates.values()
            if c.agent is not None
        )

    def _check_cost_ceiling(self) -> None:
        """Park (resumable) when cumulative coding agent spend reaches the ceiling —
        the engine-side guarantee behind hosted credit reservations."""
        ceiling = self.config.budget.max_cost_usd
        if ceiling > 0 and self.total_cost_usd() >= ceiling:
            raise ParkedSearch(
                f"cost ceiling reached (${self.total_cost_usd():.2f} >= ${ceiling:.2f})"
            )

    # --- live cross-search sharing ---

    def _publish_live_card(self) -> None:
        """Let the memory share this search's state with concurrent ones.
        Best effort — sharing must never fail a search."""
        try:
            self.memory.publish(self.journal, budget_s=self.budget.total_s, cost_usd=self.total_cost_usd())
        except Exception as exc:  # noqa: BLE001
            self.log(f"  live card publish failed (search unaffected): {exc}")

    def _live_experience(self) -> str:
        """Prompt section with what concurrent searches found so far; "" when
        there is none. Polled fresh on every prompt build so late discoveries
        land in the very next operator."""
        try:
            return self.memory.live()
        except Exception:  # noqa: BLE001
            return ""

    # --- status reporting ---

    def _status(self, **fields) -> None:
        if self.status is None:
            return
        candidates = self.journal.candidates.values()
        fields.setdefault("cost_usd", round(self.total_cost_usd(), 6))
        spend, limits = self.spend(), self.config.budget
        fields.setdefault(
            "budget",
            self.status.status.budget.model_copy(
                update=dict(
                    evaluations=spend.evaluations,
                    max_evaluations=limits.max_evaluations,
                    tokens=spend.tokens,
                    max_tokens=limits.max_tokens,
                )
            ),
        )
        fields.setdefault(
            "candidates",
            CandidateCounts(
                total=len(self.journal.candidates),
                passing=sum(1 for c in candidates if c.status == "passing"),
                failing=sum(1 for c in candidates if c.status == "failing"),
                buggy=sum(1 for c in candidates if c.status == "buggy"),
                pruned=sum(1 for c in candidates if c.pruned),
            ),
        )
        best = self.journal.best_candidate(self.problem.higher_is_better)
        if best is not None:
            fields.setdefault(
                "best", ScoreRef(candidate_id=best.candidate_id, val_score=best.val_score)
            )
        selected = self.journal.selected_candidate(
            self.problem.higher_is_better, self.config.holdout.selection
        )
        if selected is not None:
            fields.setdefault(
                "selected",
                ScoreRef(
                    candidate_id=selected.candidate_id,
                    val_score=selected.val_score,
                    holdout_score=selected.holdout_score,
                ),
            )
        self.status.update(**fields)

    def _drain_control(self) -> list[ControlCommand]:
        with self._deferred_lock:
            commands, self._deferred_commands = self._deferred_commands, []
        return commands + self.drain_commands()

    @contextmanager
    def _watch_control(self, interval_s: float = 1.0):
        """While the loop's thread blocks in one job (`run()`, the seed), a
        watcher drains the control queue so an immediate stop still aborts
        the job within ~interval_s. It only sets `abort` (thread-safe) and
        defers every command to the next `_process_control` — the journal is
        never touched off the loop's thread."""
        done = threading.Event()

        def watch() -> None:
            while not done.wait(interval_s):
                try:
                    commands = self.drain_commands()
                except Exception:  # noqa: BLE001 - a store hiccup must not kill the job
                    continue
                if not commands:
                    continue
                with self._deferred_lock:
                    self._deferred_commands.extend(commands)
                if any(c.action == "stop" and not c.graceful for c in commands):
                    self.abort.set()

        watcher = threading.Thread(target=watch, name="control-watch", daemon=True)
        watcher.start()
        try:
            yield
        finally:
            done.set()
            watcher.join()

    def _process_control(self) -> None:
        """Apply queued user commands (control/) between operators. Prunes are
        applied before a stop so nothing is left half-processed. An immediate
        stop (the default) also aborts the operators in flight: they return
        within ~1 s and commit as abandoned; a graceful one lets them finish."""
        commands = self._drain_control()
        stop: ControlCommand | None = None
        for cmd in sorted(commands, key=lambda c: c.action != "prune"):
            if cmd.action == "stop":
                stop = cmd
            elif cmd.action == "prune" and cmd.candidate_id:
                try:
                    pruned = apply_prune(self.journal, cmd.candidate_id, cmd.reason, cmd.source)
                except ValueError as exc:
                    self.log(f"  prune {cmd.candidate_id} rejected: {exc}")
                    continue
                if not pruned:
                    continue
                self.log(f"  pruned {', '.join(pruned)} (by {cmd.source})")
                if self._selection_id in pruned:
                    self._selection_id = resync_best(
                        self.search_dir,
                        self.journal,
                        self.problem.higher_is_better,
                        self.config.holdout.selection,
                        self.problem.output_artifacts,
                    )
        if stop is not None:
            self.journal.control_event(
                "stop", reason=stop.reason, source=stop.source, graceful=stop.graceful
            )
            if not stop.graceful:
                if self._inflight:
                    self.log(f"  stop requested by {stop.source}: aborting {len(self._inflight)} in-flight operator(s)")
                self.abort.set()
            raise StopRequested(f"stop requested by {stop.source}")

    def _run_seed(self) -> Candidate:
        """Score the incumbent solution as a real candidate: the floor a
        re-search must beat. No coding agent call (the evaluator scores its hidden
        split ungated — it is the selection floor)."""
        seed = self.seed_solution.absolute()
        if not seed.exists():
            raise FileNotFoundError(f"seed solution not found: {seed}")
        self.log(f"seeding incumbent {seed.name}")
        job = self._prepare_inject(
            Action(operator=INJECT_ACTION, payload={"source": seed.read_text()}),
            operator="seed",
            summary=f"incumbent model seeded from {seed.name}",
        )
        self._inflight[job.key] = job
        with self._watch_control():
            msg = self._execute_job(job)
        return self._commit(msg)  # its landed line reports the score

    def _prepare_inject(self, action: Action, *, operator: str = INJECT_ACTION, summary: str = "") -> Job:
        """An coding-agent-free candidate from a source text the loop (or the user's
        --seed-from) already has: write it, journal `created`, score it like
        any other attempt. The text itself is never journaled — its
        `solution_sha256` is."""
        source = action.payload.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{operator} needs a non-empty payload['source']")
        target = self.journal.candidates.get(action.target_id) if action.target_id else None
        if action.target_id and target is None:
            raise KeyError(action.target_id)
        candidate_id = self.journal.next_candidate_id()
        candidate_dir = create_candidate_dir(
            self.search_dir,
            candidate_id,
            self.data_dir,
            self.problem.problem_dir,
            unit_tests_dir=(self.problem.unit_tests.root if self.problem.unit_tests else None),
        )
        (candidate_dir / "solution.py").write_text(source)
        candidate = Candidate(
            candidate_id=candidate_id,
            parent_id=target.candidate_id if target else None,
            operator=operator,
            candidate_dir=str(candidate_dir),
            summary=summary,
            args=dict(action.args),
            climber_meta=dict(action.climber_meta),
        )
        self.journal.candidate_created(candidate)
        if self.status is not None:
            self.status.add_current(
                CurrentCandidate(
                    candidate_id=candidate_id, operator=operator, phase="exec",
                    candidate_dir=str(candidate_dir),
                )
            )
        return Job(candidate=candidate, request=None, candidate_dir=candidate_dir)

    # --- candidate lifecycle: _prepare (scheduler) → _execute_job (worker)
    # --- → _commit (scheduler); run_operator is the synchronous composition

    def _prepare(self, action: Action) -> Job:
        """Scheduler-side setup: id, candidate_dir, prompt, journal `created`.
        The operator says what the attempt needs (`Attempt`); everything
        that touches disk, the journal or a coding agent happens here."""
        operator = action.operator
        target = self.journal.candidates.get(action.target_id) if action.target_id else None
        ensemble_inputs = (
            [self.journal.candidates[i] for i in action.inspiration_ids]
            if action.inspiration_ids
            else None
        )
        op, prep = self._prepare_attempt(action, target)  # pure: nothing exists yet if it raises
        candidate_id = self.journal.next_candidate_id()
        parent_solution = (
            Path(target.candidate_dir) / "solution.py"
            if target is not None and prep.copy_parent
            else None
        )
        candidate_dir = create_candidate_dir(
            self.search_dir,
            candidate_id,
            self.data_dir,
            self.problem.problem_dir,
            parent_solution,
            unit_tests_dir=(self.problem.unit_tests.root if self.problem.unit_tests else None),
        )
        for name, source in prep.files.items():
            if Path(name).name != name:
                raise ValueError(f"operator {operator!r}: extra file {name!r} must be a bare file name")
            shutil.copy(source, candidate_dir / name)
        for name, text in prep.texts.items():
            if Path(name).name != name:
                raise ValueError(f"operator {operator!r}: extra file {name!r} must be a bare file name")
            (candidate_dir / name).write_text(text)
        inherited = (
            self._inherit_params(target, candidate_dir)
            if parent_solution and prep.inherit_params
            else None
        )
        if ensemble_inputs and prep.copy_inspirations:
            for i, cand in enumerate(ensemble_inputs, 1):
                shutil.copy(Path(cand.candidate_dir) / "solution.py", candidate_dir / inspiration_filename(i))
        prompt = self._with_contract(prep.prompt, target, inherited)
        if action.extra_prompt_context:
            prompt += (
                "\n\n# Additional context from the search strategy\n\n"
                f"{action.extra_prompt_context}\n"
            )
        (candidate_dir / "prompt.md").write_text(prompt)

        # Claude Code scopes
        # sessions to the cwd, and every candidate has its own candidate_dir, so
        # --resume can't find a sibling candidate dir's session. The debug prompt
        # carries the chain's failed-fix history from the journal instead.
        # pi can fork that history into the child's cwd (see request below).
        candidate = Candidate(
            candidate_id=candidate_id,
            parent_id=target.candidate_id if target else None,
            operator=operator,
            kind=op.kind,
            args=dict(action.args),
            debug_depth=(
                sum(1 for c in self.journal.debug_chain(target.candidate_id) if c.kind == "repair") + 1
                if op.kind == "repair" and target
                else 0
            ),
            candidate_dir=str(candidate_dir),
            # inspiration ids ride along so the tree view can draw an
            # ensemble's extra in-edges (parent_id only carries the first)
            climber_meta={
                **dict(action.climber_meta),
                **({"inspiration_ids": list(action.inspiration_ids)} if action.inspiration_ids else {}),
            },
        )
        route = self._resolve_route(action)
        # known before the call starts, so a running candidate's detail view
        # can say who is writing it; the result fills in the rest
        candidate.agent = AgentInfo(
            name=route.agent, model=route.model, sampling=route.sampling
        )
        # the skills and standing instructions of the global and local
        # layers, where this coding agent reads a project's own; what it was
        # given is journaled with the candidate (a skill's track record)
        from hillclimb.harness.agent_context import render as render_context

        candidate.agent_context = render_context(self.config, route.agent, candidate_dir)
        self.journal.candidate_created(candidate)

        request = AgentRequest(
            operator=operator,
            prompt=prompt,
            candidate_dir=candidate_dir,
            timeout_s=min(
                self.config.budget.agent_timeout_s, max(60, int(self.budget.remaining()))
            ),
            model=route.model,
            sampling=route.sampling,
            kind=op.kind,
            allow_internet=self.config.allow_internet_for_agents,
            sandbox=agent_policy(self.config, self.search_dir, route.agent, self.problem),
            plugins=[
                (Path(self.config.hillclimb_dir or ".") / plugin).resolve()
                for plugin in self.config.agent_context.claude_plugins
            ],
            resume_session_id=(
                target.agent.session_id
                if prep.fork_session
                and route.agent == "pi"
                and target is not None
                and target.agent.name == "pi"
                else None
            ),
        )
        if self.status is not None:
            self.status.add_current(
                CurrentCandidate(
                    candidate_id=candidate_id,
                    operator=operator,
                    phase="agent",
                    candidate_dir=str(candidate_dir),
                )
            )
            self._status()
        return Job(
            candidate=candidate,
            request=request,
            candidate_dir=candidate_dir,
            ensemble_inputs=ensemble_inputs,
            unchanged_hash=(
                source_hash(parent_solution.read_text())
                if prep.require_change and parent_solution is not None and parent_solution.exists()
                else None
            ),
            agent=(
                self.agents.get(route.agent, route.agent_auth)
                if self.agents is not None
                else None
            ),
        )

    def _operator(self, name: str) -> Operator:
        """The operator `name` names, configured from this search's config
        (resolved per call: the config block is the live source of truth)."""
        if self.operators is not None:
            return self.operators.get(name)
        return get_operator(name, self.config.climber.operator_params.get(name))

    def _prepare_attempt(self, action: Action, target: Candidate | None) -> tuple[Operator, Attempt]:
        """Ask the operator what this attempt needs. Everything it sees is
        holdout-blind; nothing is created, journaled or spent here, so a
        refusal leaves no trace."""
        op = self._operator(action.operator)
        blind = JournalView(self.journal)

        def masked(candidate: Candidate) -> Candidate:
            return blind.candidates.get(candidate.candidate_id) or candidate.holdout_blind()

        blind_target = masked(target) if target is not None else None
        reason = op.valid_target(blind_target)
        if reason:
            raise ValueError(reason)
        ctx = OperatorContext(
            action=action,
            target=blind_target,
            inspirations=tuple(
                masked(self.journal.candidates[i]) for i in action.inspiration_ids
            ),
            journal=blind,
            problem=ProblemInfo(
                problem_id=self.problem.problem_id,
                description=self.problem.description,
                metric_name=self.problem.metric_name,
                higher_is_better=self.problem.higher_is_better,
                allow_internet_during_solution=self.problem.allow_internet_during_solution,
                data_listing=self._data_listing(),
            ),
            budget=self._view().budget,
            memory=MemoryContext(
                text=self.retrieved.text or "",
                reference=self.retrieved.reference,
                reference_note=self.retrieved.reference_note,
            ),
            services=_OperatorServices(self),
            agent_internet=self.config.allow_internet_for_agents,
        )
        return op, op.prepare(ctx)

    def _with_contract(self, body: str, target: Candidate | None, inherited: dict | None) -> str:
        """Put the problem's contract where the prompt marks its place — or
        at the end when it marks none: an operator cannot drop the contract."""
        contract = self._contract(target, inherited)
        if CONTRACT_TOKEN in body:
            return body.replace(CONTRACT_TOKEN, contract)
        return body.rstrip("\n") + "\n\n" + contract + "\n"

    def _resolve_route(self, action: Action) -> ResolvedRoute:
        if self.router is not None:
            return self.router.resolve(action.operator, action.route)
        return ResolvedRoute(
            agent=self.config.agent,
            model=self.config.model,
            agent_auth=self.config.agent_auth,
        )

    def _execute_job(self, job: Job) -> OutcomeMsg:
        """Worker-side: coding agent call + trials (the evaluator scores holdout
        inside run_trial). Lock-free — touches only the job's own
        candidate/candidate_dir, never the journal."""
        if job.kind == "tune":
            return self._execute_tune(job)
        candidate = job.candidate
        if job.request is None:  # inject / seed: there is no coding agent to call
            return self._evaluate_job(job, None)
        agent = job.agent if job.agent is not None else self.agent

        slot = None
        if self.slots is not None:
            slot = self.slots.acquire(
                abort=self.abort,
                should_stop=self.budget.should_stop,
                on_wait=lambda: self._set_phase(candidate.candidate_id, "waiting-slot"),
            )
            if slot is None and self.slots.limit > 0:
                return OutcomeMsg(job=job, kind="aborted")
            self._set_phase(candidate.candidate_id, "agent")
        try:
            result = agent.invoke(job.request)
        finally:
            if slot is not None:
                slot.release()
            self._harvest_skills(candidate)
        candidate.agent = AgentInfo(
            name=agent.name,
            model=job.request.model,
            sampling=job.request.sampling,
            model_id=result.model_id,
            session_id=result.session_id,
            cost_usd=result.cost_usd,
            num_turns=result.num_turns,
            total_tokens=result.total_tokens,
            token_usage=result.token_usage,
            quota_start=result.quota_start,
            quota_end=result.quota_end,
            agent_duration_s=result.duration_s,
            cpu_s=result.cpu_s,
            error_kind=result.error_kind,
        )

        # out_of_credits parks like a rate limit: every later call fails the
        # same way until the account is topped up, and a parked search resumes.
        if result.error_kind in ("rate_limited", "out_of_credits"):
            return OutcomeMsg(job=job, kind="parked", result=result)
        if result.error_kind == "aborted":
            return OutcomeMsg(job=job, kind="aborted", result=result)
        if not result.ok:
            # e.g. network down: solution.py may still exist as the parent's
            # copy — executing it would silently re-score the parent
            return OutcomeMsg(job=job, kind="agent_failed", result=result)

        notes = job.candidate_dir / "notes.md"
        if notes.exists():
            lines = notes.read_text().strip().splitlines()
            candidate.summary = lines[0] if lines else ""

        return self._evaluate_job(job, result)

    def _evaluate_job(self, job: Job, result: AgentResult | None) -> OutcomeMsg:
        """Worker-side: score whatever solution the attempt left behind."""
        candidate = job.candidate
        solution = job.candidate_dir / "solution.py"
        if not solution.exists():
            return OutcomeMsg(job=job, kind="no_solution", result=result)
        candidate.solution_sha256 = source_hash(solution.read_text(errors="replace"))
        if job.unchanged_hash is not None and candidate.solution_sha256 == job.unchanged_hash:
            return OutcomeMsg(job=job, kind="unchanged", result=result)

        exec_timeout = min(
            self.config.budget.exec_timeout_s, max(60, int(self.budget.remaining() - 30))
        )
        budget_clamped = exec_timeout < self.config.budget.exec_timeout_s
        self._set_phase(candidate.candidate_id, "exec")
        # the defaults trial: a declared params.json makes t0 self-describing
        # (values journaled, $HILLCLIMB_PARAMS pointing at the trial's copy);
        # a malformed one is scored on the solution's own defaults instead
        declaration, params_error = read_candidate_space(job.candidate_dir)
        candidate.tunable = declaration is not None
        candidate.params_error = params_error
        _, all_ok = self.evaluator.run_trial(
            candidate, solution, job.candidate_dir, exec_timeout,
            params=declaration.defaults if declaration else None,
            params_doc=with_values(declaration.raw, declaration.defaults) if declaration else None,
        )
        return OutcomeMsg(
            job=job,
            kind="executed",
            result=result,
            all_ok=all_ok,
            budget_clamped=budget_clamped,
        )

    def _commit(self, msg: OutcomeMsg) -> Candidate:
        """Scheduler-side: journal the outcome, update best/selection and the
        failure counter. Raises ParkedSearch on rate limit / third failure."""
        if msg.job.kind == "tune":
            return self._commit_tune(msg)
        with self._state_lock:
            candidate, result = msg.job.candidate, msg.result
            self._inflight.pop(msg.job.key, None)
            try:
                if msg.kind == "infrastructure_failed":
                    raise RuntimeError(f"unit-test infrastructure failed: {msg.error}") from msg.error
                if msg.kind == "parked":
                    candidate.status = "parked"
                    candidate.summary = f"parked: {result.error_message}"
                    candidate.finished_at = utcnow()
                    self._record_result(candidate)
                    raise ParkedSearch(result.error_message)

                if msg.kind == "worker_crashed":
                    candidate.status = "abandoned"
                    candidate.summary = "orchestrator error mid-operator (see log)"
                    candidate.finished_at = utcnow()
                    self._record_result(candidate)
                    return candidate

                if msg.kind == "aborted":
                    candidate.status = "abandoned"
                    candidate.summary = "stopped mid-operator (abort)"
                    candidate.finished_at = utcnow()
                    self._record_result(candidate)
                    return candidate

                if msg.kind == "agent_failed":
                    candidate.status = "abandoned"
                    candidate.summary = (
                        f"coding agent call failed ({result.error_kind}): {result.error_message[:150]}"
                    )
                    candidate.finished_at = utcnow()
                    self._record_result(candidate)
                    self._consecutive_failures += 1
                    self.log(f"  coding agent call failed ({self._consecutive_failures} in a row)")
                    if self._consecutive_failures >= 3:
                        raise ParkedSearch(
                            f"3 consecutive coding agent failures; last: {result.error_message[:200]}"
                        )
                    return candidate
                self._consecutive_failures = 0

                if msg.kind == "unchanged":
                    candidate.status = "abandoned"
                    candidate.summary = "agent returned the parent source unchanged"
                    candidate.finished_at = utcnow()
                    self._record_result(candidate)
                    return candidate

                if msg.kind == "no_solution":
                    candidate.status = "abandoned"
                    candidate.summary = (
                        candidate.summary
                        or f"no solution.py to score ({result.error_kind if result else 'inject'})"
                    )
                    candidate.finished_at = utcnow()
                    self._record_result(candidate)
                    return candidate

                # kind == "executed"; a failed hidden split arrives as
                # all_ok=False with the reason on the trial (evaluator's call)
                if msg.all_ok:
                    candidate.status = "passing"
                    if candidate.status == "passing":
                        previous_best = self.journal.best_candidate(self.problem.higher_is_better)
                        if previous_best is None:
                            candidate.is_best = True
                        elif self._improves(candidate.val_score, previous_best.val_score):
                            candidate.is_best = True
                        elif self._improves(
                            candidate.val_score, previous_best.val_score, band=0.0
                        ):
                            # nominally ahead, but by less than the search can
                            # measure — say so rather than silently climbing it
                            self.log(
                                f"  {candidate.candidate_id} val={candidate.val_score:.5g} beats "
                                f"{previous_best.candidate_id} "
                                f"({previous_best.val_score:.5g}) by less than the accept band "
                                f"({self.accept_band():.3g}): within noise, not promoted"
                            )
                elif self.abort.is_set() and any(
                    r.timed_out for t in candidate.trials for r in t.replicates
                ):
                    # a stop (or the hard deadline) killed the verifier: the
                    # candidate was never shown to be wrong, so it is no
                    # debug target — abandoned, like an agent call it stopped
                    candidate.status = "abandoned"
                    candidate.summary = "stopped mid-operator (abort): verifier killed" + (
                        f" — {candidate.summary}" if candidate.summary else ""
                    )
                elif msg.budget_clamped and any(
                    r.timed_out for t in candidate.trials for r in t.replicates
                ):
                    # the search's clock ran out under the verifier: the
                    # candidate was never shown to be wrong, so it is not a
                    # debug target — abandoned, like a candidate an abort
                    # stopped mid-operator
                    candidate.status = "abandoned"
                    cut = next(r for t in candidate.trials for r in t.replicates if r.timed_out)
                    candidate.summary = (
                        f"cut off at the budget wall: verifier killed after {cut.duration_s:.0f}s"
                        + (f" — {candidate.summary}" if candidate.summary else "")
                    )
                else:
                    verdict = candidate.last_trial.verdict if candidate.last_trial else None
                    candidate.status = verdict if verdict in ("failing", "buggy") else "buggy"
                candidate.finished_at = utcnow()
                if candidate.params_error:
                    self.log(
                        f"  {candidate.candidate_id} params.json rejected — scored on its "
                        f"own defaults, not tunable: {candidate.params_error}"
                    )
                self._record_result(candidate)
                if candidate.status == "passing":
                    self._sync_selection()
                self._publish_live_card()
                return candidate
            finally:
                if self.status is not None:
                    self.status.remove_current(candidate.candidate_id)
                    self._status()

    # --- tune jobs: an extra trial (parameter set) on an existing candidate ---

    def _inherit_params(self, target: Candidate | None, candidate_dir: Path) -> dict | None:
        """A child of a tunable parent starts from the parent's best-found
        values as its own defaults. Returns the inherited values (for the
        prompt) or None."""
        if target is None or not target.tunable:
            return None
        declaration, _ = read_candidate_space(Path(target.candidate_dir))
        if declaration is None:
            return None
        best = target.best_trial
        values = dict(best.params) if best is not None and best.params else declaration.defaults
        write_inherited_params(candidate_dir, declaration, values)
        return values

    def _prepare_tune(self, action: Action) -> Job | None:
        """Scheduler-side (under the lock): reserve the next trial index on
        the target, ask the tuner for its values, write the trial dir's
        params.json. None = hold: the target cannot be tuned (policy asked
        for the impossible) or the tuner failed — logged, never fatal."""
        target = self.journal.candidates.get(action.target_id) if action.target_id else None
        if target is None or target.status != "passing" or target.pruned or not target.tunable:
            self.log(f"  tune: {action.target_id} is not tunable (ignored)")
            return None
        candidate_dir = Path(target.candidate_dir)
        declaration, reason = read_candidate_space(candidate_dir)
        if declaration is None:
            # the file changed on disk since the candidate was scored
            target.tunable = False
            target.params_error = reason or "params.json vanished"
            self.journal.candidate_result(target)
            self.log(f"  tune: {target.candidate_id} params.json no longer valid: {target.params_error}")
            return None
        pending = [
            job.params for job in self._inflight.values()
            if job.kind == "tune" and job.candidate.candidate_id == target.candidate_id
        ]
        index = len(target.trials) + len(pending)
        # the tuner the search was built with holds the merged params (the
        # climber's, with the user's on top) — the seed is one of them
        seed = tune_seed(
            int((getattr(self.tuner, "params", None) or {}).get("seed", 0)), target.candidate_id, index
        )
        try:
            values = self.tuner.ask(
                declaration.space,
                history_for(target, pending),
                higher_is_better=self.problem.higher_is_better,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 — a tuner bug must not kill the search
            self.log(f"  tune: {self.tuner.name} failed on {target.candidate_id}: {exc!r}")
            return None
        self.journal.audit_event(
            "tune_started", candidate_id=target.candidate_id, trial_index=index, params=values,
            tuner=self.tuner.name,
        )
        if self.status is not None:
            self.status.add_current(
                CurrentCandidate(
                    candidate_id=target.candidate_id,
                    operator=TUNE_ACTION,
                    phase="exec",
                    candidate_dir=str(candidate_dir),
                    trial_index=index,
                )
            )
            self._status()
        return Job(
            candidate=target.model_copy(deep=True),
            request=None,
            candidate_dir=candidate_dir,
            kind="tune",
            trial_index=index,
            params=values,
            declaration=declaration,
        )

    def _execute_tune(self, job: Job) -> OutcomeMsg:
        """Worker-side: run the trial's replicates on the candidate copy (the
        evaluator scores its hidden split iff the new trial is the copy's
        best and passes the gate). No coding agent, no machine slot (those meter
        coding agent processes)."""
        copy = job.candidate
        if self.abort.is_set():
            return OutcomeMsg(job=job, kind="aborted")
        exec_timeout = min(
            self.config.budget.exec_timeout_s, max(60, int(self.budget.remaining() - 30))
        )
        _, all_ok = self.evaluator.run_trial(
            copy, job.candidate_dir / "solution.py", job.candidate_dir, exec_timeout,
            params=job.params,
            params_doc=with_values(job.declaration.raw, job.params),
            index=job.trial_index,
        )
        return OutcomeMsg(job=job, kind="tuned", all_ok=all_ok)

    def _discard_tune(self, job: Job, reason: str) -> None:
        """Nothing lands on the candidate; resume re-proposes via the policy."""
        self._inflight.pop(job.key, None)
        self.journal.audit_event(
            "tune_discarded", candidate_id=job.candidate.candidate_id,
            trial_index=job.trial_index, reason=reason,
        )
        self.log(f"  tune {job.candidate.candidate_id} t{job.trial_index} discarded: {reason}")
        if self.status is not None:
            self.status.remove_current(job.candidate.candidate_id, job.trial_index)

    def _commit_tune(self, msg: OutcomeMsg) -> Candidate | None:
        """Scheduler-side: append the trial to the LIVE candidate, restamp its
        best trial, re-hoist outputs when the new trial wins, re-evaluate
        promotion, re-journal (replay keeps the last record)."""
        job = msg.job
        with self._state_lock:
            try:
                live = self.journal.candidates.get(job.candidate.candidate_id)
                if msg.kind != "tuned" or live is None:
                    self._discard_tune(job, msg.kind)
                    return live
                if self.abort.is_set() and any(r.timed_out for r in job.candidate.trials[-1].replicates):
                    # a stop killed this trial's verifier: it says nothing
                    # about the parameters, so it is not a failed trial
                    self._discard_tune(job, "stopped mid-tune (abort)")
                    return live
                self._inflight.pop(job.key, None)
                # the trial arrives with whatever the evaluator stamped on it;
                # a parameter set whose hidden split failed still climbs on
                # val (it is just never selectable) — the candidate stays passing
                trial = job.candidate.trials[-1]
                live.trials.append(trial)
                live.trials.sort(key=lambda t: t.index)  # parallel tune commits land in any order
                live.stamp_best_trial(self.problem.higher_is_better)
                score = trial.val_score
                self.log(
                    f"  tune {live.candidate_id} t{trial.index}: val="
                    f"{score:.5g}" if score is not None else
                    f"  tune {live.candidate_id} t{trial.index}: failed"
                )
                if trial.is_best:
                    self.evaluator.hoist_trial(job.candidate_dir, trial)
                    if not live.pruned and not live.is_best:
                        previous_best = self.journal.best_candidate(self.problem.higher_is_better)
                        if previous_best is None or previous_best.candidate_id == live.candidate_id:
                            live.is_best = True
                        elif self._improves(live.val_score, previous_best.val_score):
                            live.is_best = True
                            self.log(f"  {live.candidate_id} promoted to best by t{trial.index}")
                        elif self._improves(live.val_score, previous_best.val_score, band=0.0):
                            self.log(
                                f"  {live.candidate_id} t{trial.index} val={live.val_score:.5g} beats "
                                f"{previous_best.candidate_id} ({previous_best.val_score:.5g}) by less "
                                f"than the accept band ({self.accept_band():.3g}): within noise, not promoted"
                            )
                self.journal.candidate_result(live)
                if live.status == "passing" and not live.pruned:
                    if trial.is_best and live.candidate_id == self._selection_id:
                        # the selected candidate's shipped values changed: best/
                        # must be re-materialized even though selection did not move
                        self._selection_id = None
                    self._sync_selection()
                self._publish_live_card()
                return live
            finally:
                if self.status is not None:
                    self.status.remove_current(job.candidate.candidate_id, job.trial_index)
                    self._status()

    def _record_result(self, candidate: Candidate) -> None:
        """Journal a terminal result and let the policy see it (the runtime
        half of the observe contract; construction replays history)."""
        self.journal.candidate_result(candidate)
        self._observe_route(candidate)
        if candidate.operator != "baseline":  # the baseline says so itself
            self.log(landed_line(candidate))

    def _observe_route(self, candidate: Candidate) -> None:
        """Credit the model that authored this candidate in the routing
        bandit. Same replay discipline as policy.observe: called for every
        journaled terminal result and for every candidate on construction."""
        if self.router is None:
            return
        from hillclimb.harness.bandit import candidate_reward

        parent = (
            self.journal.candidates.get(candidate.parent_id) if candidate.parent_id else None
        )
        reward = candidate_reward(
            candidate, parent, self.problem.higher_is_better, band=self.accept_band()
        )
        if reward is not None:
            self.router.observe(candidate.operator, candidate.agent.model, reward)

    def _set_phase(self, candidate_id: str, phase: str) -> None:
        if self.status is not None:
            self.status.update_current(candidate_id, phase=phase)

    def _sync_selection(self) -> None:
        """Keep best/ pointing at the currently selected candidate. Selection
        is recomputed over the whole tree because rank-blend can shift between
        existing candidates when a new one lands."""
        selected = self.journal.selected_candidate(
            self.problem.higher_is_better, self.config.holdout.selection
        )
        if selected is None or selected.candidate_id == self._selection_id:
            return
        self._selection_id = resync_best(
            self.search_dir,
            self.journal,
            self.problem.higher_is_better,
            self.config.holdout.selection,
            self.problem.output_artifacts,
        )
        scores = f"val={selected.val_score:.5g}" if selected.val_score is not None else "val=none"
        if selected.holdout_score is not None:
            scores += f" holdout={selected.holdout_score:.5g}"
        self.log(f"  new selection: {selected.candidate_id} {scores}")

    def accept_band(self) -> float:
        """Delegate to the shared helper (scheduler-thread only — it reads
        the journal's noise floor)."""
        return evaluation.accept_band(self.config, self.journal)

    def _improves(self, score: float, best: float, band: float | None = None) -> bool:
        """Strictly better by more than the accept band. Pass band=0.0 for a
        raw comparison (ranking and gating, where a near-tie should still be
        evaluated rather than dropped)."""
        return evaluation.improves(
            score,
            best,
            higher_is_better=self.problem.higher_is_better,
            band=self.accept_band() if band is None else band,
        )

    def build_prompt(
        self,
        operator: str,
        target: Candidate | None,
        complexity: str | None,
        ensemble_inputs: list[Candidate] | None = None,
        inherited: dict | None = None,
    ) -> str:
        """The full prompt for one attempt: the operator's body with the
        problem's contract filled in."""
        action = Action(
            operator=operator,
            target_id=target.candidate_id if target else None,
            inspiration_ids=tuple(c.candidate_id for c in ensemble_inputs or ()),
            args={"complexity": complexity} if complexity else {},
        )
        _op, prep = self._prepare_attempt(action, target)
        return self._with_contract(prep.prompt, target, inherited)

    def _contract(self, target: Candidate | None, inherited: dict | None = None) -> str:
        """The problem's contract as the coding agent reads it: how the solution is
        run and scored, what it may assume, what it must write. Harness-owned
        — the same for every operator."""
        holdout_clause = render(
            "holdout_clause", metric_name=self.problem.metric_name
        ).rstrip() if self.problem.holdout_cmd is not None else ""
        network_note = (
            "Internet access IS available at execution time — this problem's rules "
            "permit fetching external data; cache downloads to files in the "
            "working directory so reruns don't refetch."
            if self.problem.allow_internet_during_solution
            else "Assume no internet access at execution time."
        )
        contract_template = self.problem.contract_template
        import sys as _sys

        tools_clause = ""
        if self.config.learning.tool and self.config.learning.enabled:
            # the engine's own interpreter — the coding agent's PATH may lack uv
            tools_clause = render(
                "tools_cue", knowledge_cli=f"{_sys.executable} -m hillclimb.cli"
            ).rstrip()
        contract = render(
            contract_template,
            engine_python=_sys.executable,  # the climber contract's cheap check runs on it
            metric_name=self.problem.metric_name,
            exec_timeout_min=self.config.budget.exec_timeout_s // 60,
            runtime_pkgs=self._runtime_pkgs(),
            time_remaining=self.budget.remaining_str(),
            holdout_clause=holdout_clause,
            report_clause=self._report_clause(),
            network_note=network_note,
            verifier_clause=self._verifier_clause(),
            emflow_problem=self.problem.emflow_problem or "",
            quantile_note=self._quantile_note(),
            verifier_display=self.problem.verifier_display,
            problem_contract=self.problem.contract or "(see the problem description above)",
            interface_section=self._interface_section(),
            params_section=self._params_section(target, inherited),
            tools_clause=tools_clause,
        )
        if self.problem.unit_tests is not None:
            import shlex

            visible = [
                token.replace("{python}", "python")
                .replace("{solution}", "./solution.py")
                .replace("{tests}", "./unit_tests")
                for token in self.problem.unit_tests.command
            ]
            contract += (
                "\n\n# Frozen unit tests\n\n"
                "Your solution must pass the unit tests copied into `./unit_tests`. "
                "You may inspect and run this copy, but edits to it do not change the "
                "frozen suite used by evaluation. Run them with:\n\n"
                f"    {shlex.join(visible)}\n"
            )
        return contract

    def _failure_reason(self, candidate: Candidate) -> str:
        trial = candidate.last_trial
        replicate = candidate.last_replicate
        if trial is None or replicate is None:
            return "The script failed."
        if trial.verdict == "failing" and trial.unit_tests is not None:
            return (
                "The script ran and the verifier scored it, but the frozen unit-test "
                f"suite failed (exit code {trial.unit_tests.returncode})."
            )
        if trial.unit_tests is not None and trial.unit_tests.timed_out:
            return "The script ran, but its frozen unit-test suite timed out."
        if replicate.timed_out:
            return "The script exceeded its execution time limit and was killed."
        if replicate.returncode not in (0, None):
            return f"The script crashed (exit code {replicate.returncode})."
        problems = []
        if replicate.val_score is None:
            problems.append(
                "the verifier reported no score — it exited 0 but wrote no usable "
                "`eval_result.json` (a `{\"score\": <float>}` object, or a bare number)"
                if self.problem.report_trusted
                else "no final `val_score: <float>` line was printed"
            )
        if trial.holdout_error:
            problems.append(trial.holdout_error)
        if problems:
            return "The script ran to completion but violated the contract: " + "; ".join(problems) + "."
        return "The script failed."

    def _report_clause(self) -> str:
        """Tier-2 coding-agent-report instructions, only where the coding agent's own script
        computes the score. Where the verifier owns scoring, the executor
        takes the verifier's result file, so asking the coding agent for one would be
        a contradiction."""
        if self.problem.report_trusted:
            return ""
        return render("report_clause").rstrip()

    def _quantile_note(self) -> str:
        """Class-attribute stanza for the emflow contract's Predictor stub."""
        q = self.problem.emflow_quantiles
        if not q:
            return '# output_kind = "point" (default): predictions carry one "point" column'
        grid = [i / 100 for i in range(1, 100)]
        if [round(v, 6) for v in q] == [round(v, 6) for v in grid]:
            literal = "tuple(i / 100 for i in range(1, 100))"
        else:
            literal = "(" + ", ".join(f"{v:g}" for v in q) + ")"
        return (
            'output_kind = "quantiles"\n'
            f"    quantiles = {literal}  # {len(q)} levels — prediction columns must match exactly"
        )

    def _interface_section(self) -> str:
        """Contract section rendered from the problem's optional interface.py
        (spaces.py declaration): the machine-checked I/O contract plus the
        exact self-check command. Spelled out in full — coding agents inherit the
        orchestrator's env, not the runtime venv's, so nothing can be
        assumed importable or exported on their side."""
        if not self.problem.interface_text:
            return ""
        section = (
            "\n## Output interface (machine-checked)\n\n"
            f"{self.problem.interface_text}\n"
        )
        # the executor protocol has fakes without these attributes; the shim
        # may also have failed to materialize — then the section is text-only
        python = getattr(self.executor, "python", None)
        shim = getattr(self.executor, "pythonpath", None)
        if python and shim:
            section += (
                "\nSelf-check your output format after running your solution "
                "(cheap, no scoring):\n\n"
                f"    PYTHONPATH={shim} {python} problem/interface.py\n"
            )
        return section

    def _params_section(self, target: Candidate | None, inherited: dict | None) -> str:
        """The params.json cue (always on — declaring is optional), plus the
        parent's best-found values when a child inherits a declaration."""
        shim = getattr(self.executor, "pythonpath", None)
        shim_note = f"; for your own test runs prefix PYTHONPATH={shim}" if shim else ""
        inherited_note = ""
        if inherited and target is not None:
            declaration, _ = read_candidate_space(Path(target.candidate_dir))
            best = target.best_trial
            where = f"its best trial (t{best.index})" if best is not None else "its defaults"
            described = (
                spaces_describe(declaration.raw, inherited) if declaration is not None else str(inherited)
            )
            inherited_note = (
                f"\nThe parent declared tunable parameters; {where} found the values "
                f"below, and `./params.json` already carries them as the defaults. "
                f"Keep or revise the declaration — do not regress the values.\n\n"
                + "\n".join(f"    {line}" for line in described.splitlines())
                + "\n"
            )
        # a climber's knobs are its policy's params (the inner runs get them
        # as `climber.params`), not values a script reads through spaces
        cue = "params_cue_climber" if self.problem.solution_kind == "climber" else "params_cue"
        return render(cue, shim_note=shim_note, inherited=inherited_note).rstrip("\n") + "\n"

    def _verifier_clause(self) -> str:
        """How the coding agent's script is expected to surface its score, for the
        self-reported contract (the verifier-owned one says nothing)."""
        if self.problem.report_trusted:
            return ""
        return (
            "- prints exactly one line `val_score: <float>` "
            f"(your validation {self.problem.metric_name}) as the FINAL line of stdout"
        )

    def _report_section(self, target: Candidate) -> str:
        """Improve-prompt section: the target's validation breakdown plus,
        when its parent also has one, where the score moved. Gated by
        config.report.enabled — injection only; the data is always recorded."""
        if not self.config.report.enabled:
            return ""
        from hillclimb.harness.report import candidate_report, render_delta, render_report

        target_report = candidate_report(target)
        body = render_report(target_report, self.problem.metric_name)
        if not body:
            return ""
        section = f"# Evaluation breakdown (validation split)\n\n{body}\n"
        parent = self.journal.candidates.get(target.parent_id) if target.parent_id else None
        delta = render_delta(
            candidate_report(parent), target_report, self.problem.higher_is_better
        )
        if delta:
            section += (
                f"\n## Where this solution moved vs its parent ({parent.candidate_id})\n\n"
                f"{delta}\n"
            )
        return section

    def _harvest_skills(self, candidate: Candidate) -> None:
        """Add a skill the call created (the agent, or a plugin it ran) to
        the hillclimb dir's local layer. Best effort: a skill that cannot be
        copied never costs the candidate."""
        from hillclimb.harness.agent_context import harvest

        homes = []
        try:
            if candidate.agent is not None and candidate.agent.name == "claude-code":
                from hillclimb.agents.claude_code import claude_home

                homes.append(claude_home(self.config.agent_auth))
            created_by = {
                "run": self.search_dir.parents[1].name,
                "search": self.search_dir.name,
                "candidate": candidate.candidate_id,
                "agent": candidate.agent.name if candidate.agent else None,
                "model": candidate.agent.model if candidate.agent else None,
            }
            added = harvest(
                self.config, Path(candidate.candidate_dir), homes=tuple(homes), created_by=created_by, log=self.log
            )
            if added:  # journaled with the candidate's result
                candidate.agent_context = {**candidate.agent_context, "skills_added": added}
        except OSError as exc:
            self.log(f"  skills not harvested from {candidate.candidate_id}: {exc}")

    def _data_listing(self, limit: int = 50) -> str:
        entries = []
        # what only the scorer may read is not the agent's to know about
        private = [
            Path(p).resolve()
            for p in [*(getattr(self.problem, "private_paths", None) or ()), *(getattr(self.problem, "holdout_inputs", None) or ())]
        ]
        for path in sorted(self.problem.problem_dir.rglob("*")):
            if private and any(path.resolve() == p or p in path.resolve().parents for p in private):
                continue
            if path.is_file():
                relative = path.relative_to(self.problem.problem_dir)
                entries.append(f"- {relative} ({_human_size(path.stat().st_size)})")
            if len(entries) >= limit:
                entries.append("- ... (truncated)")
                break
        return "\n".join(entries)

    def _runtime_pkgs(self) -> str:
        from hillclimb.runtime import runtime_packages

        requirements = getattr(self.problem, "requirements_file", None)
        return ", ".join(runtime_packages(self.problem.runtime, requirements_file=requirements))


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
