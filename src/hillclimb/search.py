from __future__ import annotations

import queue
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from hillclimb.backends.base import OperatorBackend, OperatorRequest, OperatorResult
from hillclimb.baseline import write_baseline
from hillclimb.budget import BudgetManager
from hillclimb.candidate import BackendInfo, Candidate, utcnow
from hillclimb.config import Config
from hillclimb.control import ControlCommand, apply_prune, drain_commands_dir, resync_best
from hillclimb import evaluation
from hillclimb.evaluation import TAIL_CHARS, CandidateEvaluator, tail  # noqa: F401 — re-exported (cli imports tail from here)
from hillclimb.executor import Executor, HoldoutScorer
from hillclimb.journal import Journal
from hillclimb.policies.greedy import GreedyPolicy
from hillclimb.policy import Action, BudgetView, InflightRef, SearchPolicy, SearchView
from hillclimb.prompts.render import COMPLEXITY_CUES, render
from hillclimb.routing import BackendPool, ResolvedRoute, Router
from hillclimb.run import SEARCHES_DIRNAME
from hillclimb.slots import MachineSlots
from hillclimb.status import CandidateCounts, CurrentCandidate, ScoreRef, StatusWriter
from hillclimb.problem import ProblemSpec
from hillclimb.dirs import create_candidate_dir


@dataclass
class Job:
    """One operator submission: everything a worker needs, nothing it must
    share. Created scheduler-side in _prepare; the candidate object is owned
    by the worker until _commit (the journal holds its own deep copy)."""

    candidate: Candidate
    request: OperatorRequest | None  # None for the agent-less seed candidate
    candidate_dir: Path
    ensemble_inputs: "list[Candidate] | None" = None
    holdout_threshold: float | None = None  # k-th best val at prepare time; None = no gate
    backend: OperatorBackend | None = None  # routed instance; None = harness default


@dataclass
class OutcomeMsg:
    """Terminal report from a worker; consumed by _commit on the scheduler."""

    job: Job
    kind: str  # parked | aborted | agent_failed | no_solution | executed
    result: OperatorResult | None = None
    all_ok: bool = False
    holdout_score: float | None = None
    holdout_error: str | None = None
    holdout_cpu_s: float | None = None  # burned even when holdout errored
    holdout_gated: bool = False


class ParkedSearch(Exception):
    """Raised when the backend hits a rate limit; the search can be resumed."""


class StopRequested(Exception):
    """Raised on a graceful stop (control command or SIGTERM); the search can
    be resumed."""


class GreedySearcher:
    """The search harness: executes whatever `SearchPolicy` proposes while
    owning every state invariant (journal single-writer, control queue,
    budgets, holdout, best/-selection). Defaults to `GreedyPolicy`:
    (1) baseline at t=0; (2) DEBUG the newest buggy candidate while its chain
    is shallow; (3) DRAFT until `num_drafts` branches have a scored solution
    (complexity cue escalates per draft); (4) otherwise IMPROVE the best
    candidate. Stops on budget margin or max_candidates.
    """

    def __init__(
        self,
        problem: ProblemSpec,
        config: Config,
        journal: Journal,
        backend: OperatorBackend,
        executor: Executor,
        budget: BudgetManager,
        search_dir: Path,
        max_candidates: int = 50,
        log=print,
        holdout_scorer: HoldoutScorer | None = None,
        status: StatusWriter | None = None,
        slots: MachineSlots | None = None,
        abort: threading.Event | None = None,
        seed_solution: Path | None = None,
        knowledge_context: str | None = None,
        reference_solution: Path | None = None,
        reference_note: str = "",
        complexity_start: int = 0,
        policy: SearchPolicy | None = None,
        router: Router | None = None,
        backends: BackendPool | None = None,
        drain_commands: Callable[[], list[ControlCommand]] | None = None,
    ):
        # where queued stop/prune commands come from: the store's queue for
        # this search (the engine binds it), else the search dir's control/
        self.drain_commands = drain_commands or (lambda: drain_commands_dir(search_dir))
        self.problem = problem
        self.config = config
        self.journal = journal
        self.backend = backend
        self.executor = executor
        self.budget = budget
        self.search_dir = search_dir
        self.max_candidates = max_candidates
        self.log = log
        self.holdout_scorer = holdout_scorer
        self.evaluator = CandidateEvaluator(
            executor=executor, problem=problem, config=config, holdout_scorer=holdout_scorer
        )
        self.status = status
        self.slots = slots  # machine-wide agent-concurrency cap (optional)
        self.abort = abort or threading.Event()
        self.seed_solution = seed_solution  # incumbent model: scored as a floor candidate
        self.knowledge_context = knowledge_context  # prior-experience prompt section
        # skill library: a proven prior solution copied into the FIRST
        # draft's candidate dir as reference_solution.py (later drafts explore)
        self.reference_solution = reference_solution
        self.reference_note = reference_note
        self.complexity_start = complexity_start  # learned draft-complexity offset
        self.policy = policy or GreedyPolicy(complexity_start=complexity_start)
        self.router = router  # None: everything routes to `backend` + config.model
        self.backends = backends
        self._consecutive_failures = 0
        # Concurrency contract: the Journal and everything below is touched
        # only by the scheduler (the thread running run()/run_operator),
        # belt-and-braces guarded by _state_lock; workers report through
        # _done_q and never see the journal.
        self._state_lock = threading.Lock()
        self._inflight: dict[str, Job] = {}
        self._done_q: "queue.Queue[OutcomeMsg]" = queue.Queue()
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
        # resume contract: stateful policies (and the routing bandit) rebuild
        # their caches from the replayed journal (in journal order, after
        # stale-pending recovery)
        replay_view = self._view()
        for candidate in journal.candidates.values():
            self.policy.observe(replay_view, candidate)
            self._observe_route(candidate)

    @property
    def data_dir(self) -> Path:
        return self.problem.data_dir

    def _view(self) -> SearchView:
        """Snapshot of search state for a policy call. Scheduler-thread only —
        same discipline as every other journal touch."""
        return SearchView(
            journal=self.journal,
            inflight=tuple(
                InflightRef(
                    candidate_id=candidate_id,
                    operator=job.candidate.operator,
                    parent_id=job.candidate.parent_id,
                )
                for candidate_id, job in self._inflight.items()
            ),
            budget=BudgetView(
                remaining_s=self.budget.remaining(),
                total_s=self.budget.total_s,
                stop_margin_s=self.budget.stop_margin_s,
            ),
            config=self.config,
            higher_is_better=self.problem.higher_is_better,
        )

    # --- main loop ---

    def run(self) -> Candidate | None:
        if not self.journal.candidates:
            baseline = write_baseline(
                self.problem,
                self.search_dir,
                executor=self.executor,
                holdout_scorer=self.holdout_scorer,
                timeout_s=self.config.budget.exec_timeout_s,
            )
            self._record_result(baseline)
            self.log(f"baseline written: {baseline.summary}")
        if self.seed_solution is not None and not any(
            c.operator == "seed" for c in self.journal.candidates.values()
        ):
            self._run_seed()  # resume-idempotent: at most one seed per search
        # one loop for every parallelism level: n=1 is a pool with one slot.
        # Recorded golden sequences pin it to the historical serial behavior.
        return self._run_pool(max(1, self.config.search.parallel_operators))

    def _run_pool(self, n: int) -> Candidate | None:
        """Worker-pool scheduler: keep up to n operators in flight; commit
        results in completion order. Park/stop drain gracefully (in-flight
        operators finish and are committed); SIGTERM aborts them."""
        pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="operator")
        drain: Exception | None = None
        try:
            while True:
                if drain is None:
                    try:
                        self._process_control()
                        self._check_cost_ceiling()
                    except (StopRequested, ParkedSearch) as exc:
                        if self._inflight:
                            self.log(f"  {exc}: draining in-flight operators")
                            drain = exc
                        else:
                            raise
                    while (
                        drain is None
                        and len(self._inflight) < n
                        and not self.budget.should_stop()
                        and len(self.journal.candidates) < self.max_candidates
                    ):
                        job = self.decide_next()
                        if job is None:
                            break
                        self._submit(pool, job)
                if not self._inflight:
                    if drain is not None:
                        raise drain
                    if (
                        self.budget.should_stop()
                        or len(self.journal.candidates) >= self.max_candidates
                    ):
                        break
                try:
                    msg = self._done_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                try:
                    self._commit(msg)
                except ParkedSearch as exc:
                    if self._inflight:
                        self.log("  parked: draining in-flight operators")
                        drain = exc
                    else:
                        raise
        except (StopRequested, KeyboardInterrupt):
            self.abort.set()
            self._drain_aborted()
            raise
        finally:
            pool.shutdown(wait=True)
        return self.journal.selected_candidate(
            self.problem.higher_is_better, self.config.holdout.selection
        )

    def _submit(self, pool: ThreadPoolExecutor, job: Job) -> None:
        self.log(
            f"[{self.budget.remaining_str()} left] {job.candidate.operator}"
            + (f" -> {job.candidate.parent_id}" if job.candidate.parent_id else "")
            + f" ({job.candidate.candidate_id})"
        )
        self._inflight[job.candidate.candidate_id] = job

        def work() -> None:
            try:
                outcome = self._execute_job(job)
            except BaseException as exc:  # noqa: BLE001
                # a worker that dies without reporting would leave the
                # candidate in flight and the scheduler waiting forever
                self.log(f"  worker for {job.candidate.candidate_id} crashed: {exc!r}")
                outcome = OutcomeMsg(job=job, kind="worker_crashed")
            self._done_q.put(outcome)

        pool.submit(work)

    def _drain_aborted(self) -> None:
        """After abort: collect whatever workers return (they die within ~1s
        poll intervals) and journal the candidates as abandoned."""
        while self._inflight:
            try:
                msg = self._done_q.get(timeout=30.0)
            except queue.Empty:
                break  # workers wedged; stale-pending recovery handles them on resume
            candidate = msg.job.candidate
            candidate.status = "abandoned"
            candidate.summary = candidate.summary or "stopped mid-operator (abort)"
            candidate.finished_at = utcnow()
            self._record_result(candidate)
            self._inflight.pop(candidate.candidate_id, None)
            if self.status is not None:
                self.status.remove_current(candidate.candidate_id)

    def decide_next(self) -> Job | None:
        """Ask the policy for the next action and materialize it into a Job;
        None = hold (keep slots empty). Lock spans propose + prepare so the
        decision and the journal `created` event are atomic."""
        with self._state_lock:
            action = self.policy.propose(self._view())
            if action is None:
                return None
            return self._prepare(action)

    def _debuggable_tip(self) -> Candidate | None:
        return self.policy.debuggable_tip(self._view())

    def _prospective_branches(self) -> int:
        return self.policy.prospective_branches(self._view())

    # --- cost accounting ---

    def total_cost_usd(self) -> float:
        return sum(
            c.backend.cost_usd or 0.0
            for c in self.journal.candidates.values()
            if c.backend is not None
        )

    def _check_cost_ceiling(self) -> None:
        """Park (resumable) when cumulative agent spend reaches the ceiling —
        the engine-side guarantee behind hosted credit reservations."""
        ceiling = self.config.budget.max_cost_usd
        if ceiling > 0 and self.total_cost_usd() >= ceiling:
            raise ParkedSearch(
                f"cost ceiling reached (${self.total_cost_usd():.2f} >= ${ceiling:.2f})"
            )

    # --- live cross-search sharing ---

    def _live_run_dir(self) -> Path | None:
        """The run dir hosting the shared live-card folder; None when live
        sharing is off or the search dir is not in the runs/<run-id>/searches/
        layout (embedded and unit-test constructions)."""
        if not (self.config.learning.enabled and self.config.learning.live):
            return None
        if self.search_dir.parent.name != SEARCHES_DIRNAME:
            return None
        return self.search_dir.parents[1]

    def _target(self) -> str:
        """Problem target string for family grouping (same convention as the
        knowledge backfill: empty for non-emflow problems)."""
        if self.problem.runtime == "emflow":
            return f"emflow://{self.problem.emflow_problem}"
        return ""

    def _publish_live_card(self) -> None:
        """Republish this search's knowledge card into the run-scoped live dir
        so concurrent sibling searches see discoveries mid-run. Best effort —
        sharing must never fail a search."""
        run_dir = self._live_run_dir()
        if run_dir is None:
            return
        from hillclimb.knowledge import distill_card, write_live_card

        try:
            card = distill_card(
                self.journal,
                problem=self.problem,
                run_ref=f"{run_dir.name}/{self.search_dir.name}",
                target=self._target(),
                budget_s=self.budget.total_s,
                cost_usd=self.total_cost_usd(),
                selection=self.config.holdout.selection,
            )
            write_live_card(run_dir, card, self.search_dir.name)
        except Exception as exc:  # noqa: BLE001
            self.log(f"  live card publish failed (search unaffected): {exc}")

    def _live_experience(self) -> str:
        """Prompt section from sibling searches' live cards; "" when there are
        none. Polled fresh on every prompt build so late discoveries land in
        the very next operator."""
        run_dir = self._live_run_dir()
        if run_dir is None:
            return ""
        from hillclimb.knowledge import load_live_cards, problem_family, render_live_experience

        try:
            cards = load_live_cards(
                run_dir,
                exclude_search_id=self.search_dir.name,
                family=problem_family(self.problem.problem_id, self._target()),
            )
            return render_live_experience(cards, max_cards=self.config.learning.max_cards)
        except Exception:  # noqa: BLE001
            return ""

    # --- status reporting ---

    def _status(self, **fields) -> None:
        if self.status is None:
            return
        candidates = self.journal.candidates.values()
        fields.setdefault("cost_usd", round(self.total_cost_usd(), 6))
        fields.setdefault(
            "candidates",
            CandidateCounts(
                total=len(self.journal.candidates),
                ok=sum(1 for c in candidates if c.status == "ok"),
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

    def _process_control(self) -> None:
        """Apply queued user commands (control/) between operators. Prunes are
        applied before a stop so nothing is left half-processed."""
        commands = self.drain_commands()
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
                    )
        if stop is not None:
            self.journal.control_event("stop", reason=stop.reason, source=stop.source)
            raise StopRequested(f"stop requested by {stop.source}")

    def decide(self) -> tuple[str, Candidate | None]:
        """The next decision as an (operator, target) pair — the historical
        introspection surface over policy.propose(); the run loop itself
        carries the full Action."""
        action = self.policy.propose(self._view())
        if action is None:
            return "hold", None
        target = self.journal.candidates.get(action.target_id) if action.target_id else None
        return action.operator, target

    # --- ensemble stage (delegated; kept as the stable internal surface) ---

    def _in_ensemble_window(self) -> bool:
        return self.policy.in_ensemble_window(self._view())

    def _should_ensemble(self) -> bool:
        return self.policy.should_ensemble(self._view())

    def _ensemble_succeeded(self) -> bool:
        return self.policy.ensemble_succeeded(self._view())

    def _ensemble_candidates(self) -> list[Candidate]:
        return self.policy.ensemble_candidates(self._view())

    def _run_seed(self) -> Candidate:
        """Score the incumbent solution as a real candidate: the floor a
        re-search must beat. No agent call; holdout always evaluated (it is
        the selection floor, so the top-k gate does not apply)."""
        seed = self.seed_solution.absolute()
        if not seed.exists():
            raise FileNotFoundError(f"seed solution not found: {seed}")
        candidate_id = self.journal.next_candidate_id()
        candidate_dir = create_candidate_dir(
            self.search_dir,
            candidate_id,
            self.data_dir,
            self.problem.problem_dir,
            parent_solution=seed,
        )
        candidate = Candidate(
            candidate_id=candidate_id,
            operator="seed",
            candidate_dir=str(candidate_dir),
            summary=f"incumbent model seeded from {seed.name}",
        )
        self.journal.candidate_created(candidate)
        self.log(f"seeding incumbent {seed.name} as {candidate_id}")
        exec_timeout = self.config.budget.exec_timeout_s
        all_ok = self.evaluator.run_trials(candidate, candidate_dir / "solution.py", candidate_dir, exec_timeout)
        holdout_score = holdout_error = holdout_cpu = None
        if all_ok:
            holdout_score, holdout_error, holdout_cpu = self.evaluator.score_holdout(candidate_dir)
        msg = OutcomeMsg(
            job=Job(candidate=candidate, request=None, candidate_dir=candidate_dir),
            kind="executed",
            all_ok=all_ok,
            holdout_score=holdout_score,
            holdout_error=holdout_error,
            holdout_cpu_s=holdout_cpu,
        )
        committed = self._commit(msg)
        score = f"val={committed.val_score}" if committed.val_score is not None else "buggy"
        self.log(f"  seed scored: {score}")
        return committed

    # --- candidate lifecycle: _prepare (scheduler) → _execute_job (worker)
    # --- → _commit (scheduler); run_operator is the synchronous composition

    def run_operator(self, operator: str, target: Candidate | None) -> Candidate:
        job = self._prepare(self._action_for(operator, target))
        return self._commit(self._execute_job(job))

    def _action_for(self, operator: str, target: Candidate | None) -> Action:
        """Action for an explicitly named operator (serial loop, tests,
        smoke). Policies that distinguish decision-time details (draft
        complexity, ensemble inputs) fill them via action_for; others get the
        bare action."""
        target_id = target.candidate_id if target else None
        maker = getattr(self.policy, "action_for", None)
        if maker is not None:
            return maker(self._view(), operator, target_id)
        return Action(operator=operator, target_id=target_id)

    def _prepare(self, action: Action) -> Job:
        """Scheduler-side setup: id, candidate_dir, prompt, journal `created`."""
        operator = action.operator
        target = self.journal.candidates.get(action.target_id) if action.target_id else None
        candidate_id = self.journal.next_candidate_id()
        parent_solution = (
            Path(target.candidate_dir) / "solution.py"
            if target is not None and operator in ("debug", "improve")
            else None
        )
        candidate_dir = create_candidate_dir(
            self.search_dir,
            candidate_id,
            self.data_dir,
            self.problem.problem_dir,
            parent_solution,
        )
        if self._wants_reference(operator):
            shutil.copy(self.reference_solution, candidate_dir / "reference_solution.py")
        ensemble_inputs = None
        if action.inspiration_ids:
            ensemble_inputs = [self.journal.candidates[i] for i in action.inspiration_ids]
            for i, cand in enumerate(ensemble_inputs, 1):
                shutil.copy(Path(cand.candidate_dir) / "solution.py", candidate_dir / f"candidate_{i}.py")
        complexity = action.complexity
        prompt = self.build_prompt(operator, target, complexity, ensemble_inputs)
        if action.extra_prompt_context:
            prompt += (
                "\n\n# Additional context from the search strategy\n\n"
                f"{action.extra_prompt_context}\n"
            )
        (candidate_dir / "prompt.md").write_text(prompt)

        # NOTE: no session resume across candidates — Claude Code scopes
        # sessions to the cwd, and every candidate has its own candidate_dir, so
        # --resume can't find a sibling candidate dir's session. The debug prompt
        # carries the chain's failed-fix history from the journal instead.
        candidate = Candidate(
            candidate_id=candidate_id,
            parent_id=target.candidate_id if target else None,
            operator=operator,
            complexity=complexity,
            debug_depth=(
                sum(1 for c in self.journal.debug_chain(target.candidate_id) if c.operator == "debug") + 1
                if operator == "debug" and target
                else 0
            ),
            candidate_dir=str(candidate_dir),
            # inspiration ids ride along so the tree view can draw an
            # ensemble's extra in-edges (parent_id only carries the first)
            policy_meta={
                **dict(action.policy_meta),
                **({"inspiration_ids": list(action.inspiration_ids)} if action.inspiration_ids else {}),
            },
        )
        route = self._resolve_route(action)
        # known before the call starts, so a running candidate's detail view
        # can say who is writing it; the result fills in the rest
        candidate.backend = BackendInfo(name=route.backend, model=route.model)
        self.journal.candidate_created(candidate)

        request = OperatorRequest(
            operator=operator,
            prompt=prompt,
            candidate_dir=candidate_dir,
            timeout_s=min(
                self.config.budget.agent_timeout_s, max(60, int(self.budget.remaining()))
            ),
            model=route.model,
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
            holdout_threshold=evaluation.holdout_threshold(
                self.journal,
                top_k=self.config.holdout.top_k,
                higher_is_better=self.problem.higher_is_better,
            ),
            backend=(
                self.backends.get(route.backend, route.backend_auth)
                if self.backends is not None
                else None
            ),
        )

    def _wants_reference(self, operator: str) -> bool:
        """The skill-library reference goes to the FIRST draft only — later
        drafts must diverge, so seeding them all would fight exploration.
        (With parallel first drafts both may qualify; harmless.)"""
        return (
            operator == "draft"
            and self.reference_solution is not None
            and self.reference_solution.exists()
            and not self.journal.drafts()
        )

    def _resolve_route(self, action: Action) -> ResolvedRoute:
        if self.router is not None:
            return self.router.resolve(action.operator, action.route)
        return ResolvedRoute(
            backend=self.config.backend,
            model=self.config.model,
            backend_auth=self.config.backend_auth,
        )

    def _execute_job(self, job: Job) -> OutcomeMsg:
        """Worker-side: agent call + trials + holdout. Lock-free — touches
        only the job's own candidate/candidate_dir, never the journal."""
        candidate = job.candidate
        backend = job.backend if job.backend is not None else self.backend

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
            result = backend.invoke(job.request)
        finally:
            if slot is not None:
                slot.release()
        candidate.backend = BackendInfo(
            name=backend.name,
            model=job.request.model,
            model_id=result.model_id,
            session_id=result.session_id,
            cost_usd=result.cost_usd,
            num_turns=result.num_turns,
            total_tokens=result.total_tokens,
            token_usage=result.token_usage,
            quota_start=result.quota_start,
            quota_end=result.quota_end,
            agent_duration_s=result.duration_s,
            error_kind=result.error_kind,
        )

        if result.error_kind == "rate_limited":
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

        solution = job.candidate_dir / "solution.py"
        if not solution.exists():
            return OutcomeMsg(job=job, kind="no_solution", result=result)

        exec_timeout = min(
            self.config.budget.exec_timeout_s, max(60, int(self.budget.remaining() - 30))
        )
        self._set_phase(candidate.candidate_id, "exec")
        all_ok = self.evaluator.run_trials(candidate, solution, job.candidate_dir, exec_timeout)

        holdout_score = holdout_error = holdout_cpu = None
        gated = False
        if all_ok:
            if evaluation.gate_passes(
                candidate.val_score,
                job.holdout_threshold,
                higher_is_better=self.problem.higher_is_better,
            ):
                self._set_phase(candidate.candidate_id, "holdout")
                holdout_score, holdout_error, holdout_cpu = self.evaluator.score_holdout(job.candidate_dir)
            else:
                gated = True  # climbs on val; not selectable via holdout
        return OutcomeMsg(
            job=job,
            kind="executed",
            result=result,
            all_ok=all_ok,
            holdout_score=holdout_score,
            holdout_error=holdout_error,
            holdout_cpu_s=holdout_cpu,
            holdout_gated=gated,
        )

    def _commit(self, msg: OutcomeMsg) -> Candidate:
        """Scheduler-side: journal the outcome, update best/selection and the
        failure counter. Raises ParkedSearch on rate limit / third failure."""
        with self._state_lock:
            candidate, result = msg.job.candidate, msg.result
            self._inflight.pop(candidate.candidate_id, None)
            try:
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
                        f"agent call failed ({result.error_kind}): {result.error_message[:150]}"
                    )
                    candidate.finished_at = utcnow()
                    self._record_result(candidate)
                    self._consecutive_failures += 1
                    self.log(f"  agent call failed ({self._consecutive_failures} in a row)")
                    if self._consecutive_failures >= 3:
                        raise ParkedSearch(
                            f"3 consecutive agent failures; last: {result.error_message[:200]}"
                        )
                    return candidate
                self._consecutive_failures = 0

                if msg.kind == "no_solution":
                    candidate.status = "abandoned"
                    candidate.summary = (
                        candidate.summary or f"agent produced no solution.py ({result.error_kind})"
                    )
                    candidate.finished_at = utcnow()
                    self._record_result(candidate)
                    return candidate

                # kind == "executed"
                if msg.all_ok:
                    last = candidate.trials[-1]
                    # holdout CPU is spent whether or not scoring succeeded
                    last.holdout_cpu_s = msg.holdout_cpu_s
                    if msg.holdout_error is not None:
                        candidate.status = "buggy"
                        last.holdout_error = msg.holdout_error
                    else:
                        candidate.status = "ok"
                        if not msg.holdout_gated:
                            last.holdout_score = msg.holdout_score
                    if candidate.status == "ok":
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
                else:
                    candidate.status = "buggy"
                candidate.finished_at = utcnow()
                self._record_result(candidate)
                if candidate.status == "ok":
                    self._sync_selection()
                self._publish_live_card()
                return candidate
            finally:
                if self.status is not None:
                    self.status.remove_current(candidate.candidate_id)
                    self._status()

    def _record_result(self, candidate: Candidate) -> None:
        """Journal a terminal result and let the policy see it (the runtime
        half of the observe contract; construction replays history)."""
        self.journal.candidate_result(candidate)
        self.policy.observe(self._view(), candidate)
        self._observe_route(candidate)

    def _observe_route(self, candidate: Candidate) -> None:
        """Credit the model that authored this candidate in the routing
        bandit. Same replay discipline as policy.observe: called for every
        journaled terminal result and for every candidate on construction."""
        if self.router is None:
            return
        from hillclimb.bandit import candidate_reward

        parent = (
            self.journal.candidates.get(candidate.parent_id) if candidate.parent_id else None
        )
        reward = candidate_reward(
            candidate, parent, self.problem.higher_is_better, band=self.accept_band()
        )
        if reward is not None:
            self.router.observe(candidate.operator, candidate.backend.model, reward)

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
            self.search_dir, self.journal, self.problem.higher_is_better, self.config.holdout.selection
        )
        scores = f"val_score={selected.val_score}"
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

    def _draft_complexity(self) -> str:
        return self.policy.draft_complexity(self._view())

    # --- prompt assembly ---

    def build_prompt(
        self,
        operator: str,
        target: Candidate | None,
        complexity: str | None,
        ensemble_inputs: list[Candidate] | None = None,
    ) -> str:
        holdout_clause = render(
            "holdout_clause", metric_name=self.problem.metric_name
        ).rstrip() if self.problem.holdout_cmd is not None else ""
        network_note = (
            "Internet access IS available at execution time — this problem's rules "
            "permit fetching external data; cache downloads to files in the "
            "working directory so reruns don't refetch."
            if self.problem.allow_network
            else "Assume no internet access at execution time."
        )
        contract_template = self.problem.contract_template
        tools_clause = ""
        if self.config.operators.knowledge_tool and self.config.learning.enabled:
            import sys as _sys

            # the engine's own interpreter — the agent's PATH may lack uv
            tools_clause = render(
                "tools_cue", knowledge_cli=f"{_sys.executable} -m hillclimb.cli"
            ).rstrip()
        contract = render(
            contract_template,
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
            tools_clause=tools_clause,
        )
        direction = "higher is better" if self.problem.higher_is_better else "lower is better"
        if operator == "draft":
            live = self._live_experience()
            prior = self.knowledge_context or ""
            prior = "\n\n".join(part for part in (prior, live) if part)
            research_cue = (
                render("research_cue", network_note=network_note).rstrip() + "\n"
                if self.config.operators.draft_retrieval
                else ""
            )
            starter_cue = ""
            if self._wants_reference(operator):
                note = f" ({self.reference_note})" if self.reference_note else ""
                starter_cue = (
                    "# Starter reference\n\n"
                    f"A proven solution from a previous search is at "
                    f"`./reference_solution.py`{note}. Use it as a scaffold: adapt "
                    "and improve it for THIS problem — do not resubmit it unchanged.\n"
                )
            return render(
                "draft",
                description=self.problem.description,
                metric_name=self.problem.metric_name,
                direction=direction,
                data_listing=self._data_listing(),
                research_cue=research_cue,
                starter_cue=starter_cue,
                complexity_cue=COMPLEXITY_CUES[complexity or "minimal"],
                prior_experience=prior or "(no prior searches recorded)",
                prior_drafts=self._candidate_summaries(self.journal.drafts()) or "(none yet)",
                contract=contract,
            )
        if operator == "debug":
            assert target is not None
            chain = self.journal.debug_chain(target.candidate_id)
            root, attempts = chain[0], chain[1:]
            last_trial = target.last_trial
            return render(
                "debug",
                parent_summary=root.summary or "(no summary)",
                failure_reason=self._failure_reason(target),
                stderr_tail=tail(Path(target.candidate_dir) / "exec_stderr.log"),
                stdout_tail=last_trial.stdout_tail if last_trial else "",
                debug_history=self._candidate_summaries(attempts) or "(none — this is the first fix attempt)",
                contract=contract,
            )
        if operator == "ensemble":
            assert ensemble_inputs
            table = "\n".join(
                f"- `candidate_{i}.py` — validation {self.problem.metric_name}: "
                f"**{c.val_score:.5g}** ({c.candidate_id}): {c.summary or '(no summary)'}"
                for i, c in enumerate(ensemble_inputs, 1)
            )
            return render(
                "ensemble",
                description=self.problem.description,
                metric_name=self.problem.metric_name,
                direction=direction,
                candidates_table=table,
                contract=contract,
            )
        if operator == "improve":
            assert target is not None
            last_trial = target.last_trial
            live = self._live_experience()
            ablation_cue = (
                render("ablation_cue").rstrip() + "\n"
                if self.config.operators.improve_ablation
                else ""
            )
            return render(
                "improve",
                description=self.problem.description,
                metric_name=self.problem.metric_name,
                direction=direction,
                best_score=target.val_score,
                stdout_tail=last_trial.stdout_tail if last_trial else "",
                sibling_summaries=self._candidate_summaries(self.journal.children(target.candidate_id))
                or "(nothing tried from this solution yet)",
                evaluation_report=self._report_section(target),
                live_experience=(
                    f"# Discoveries from concurrent searches\n\n{live}\n" if live else ""
                ),
                prior_ablations=self._prior_ablations(target),
                ablation_cue=ablation_cue,
                contract=contract,
            )
        raise ValueError(f"Unknown operator: {operator}")

    def _failure_reason(self, candidate: Candidate) -> str:
        trial = candidate.last_trial
        if trial is None:
            return "The script failed."
        if trial.timed_out:
            return "The script exceeded its execution time limit and was killed."
        if trial.returncode not in (0, None):
            return f"The script crashed (exit code {trial.returncode})."
        problems = []
        if trial.val_score is None:
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
        """Tier-2 agent-report instructions, only where the agent's own script
        computes the score. Where the verifier owns scoring, the executor
        takes the verifier's result file, so asking the agent for one would be
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
        exact self-check command. Spelled out in full — agents inherit the
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

    def _verifier_clause(self) -> str:
        """How the agent's script is expected to surface its score, for the
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
        from hillclimb.report import candidate_report, render_delta, render_report

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

    def _prior_ablations(self, target: Candidate, max_chars: int = 3000) -> str:
        """Improve-prompt section: the newest `ablation.md` an earlier improve
        attempt wrote while analyzing this same solution, so successive
        improves of one target don't re-measure the same components. Gated
        with the cue — without the cue nothing writes ablation.md anyway."""
        if not self.config.operators.improve_ablation:
            return ""
        for child in reversed(self.journal.children(target.candidate_id, include_pruned=True)):
            path = Path(child.candidate_dir) / "ablation.md"
            if not path.exists():
                continue
            try:
                body = path.read_text(errors="replace").strip()[:max_chars]
            except OSError:
                continue
            if not body:
                continue
            return (
                f"# Prior ablation findings for this solution "
                f"(measured by {child.candidate_id})\n\n{body}\n"
            )
        return ""

    def _candidate_summaries(self, candidates: list[Candidate]) -> str:
        lines = []
        for candidate in candidates:
            score = (
                f"val_score={candidate.val_score}"
                if candidate.val_score is not None
                else candidate.status
            )
            lines.append(
                f"- {candidate.candidate_id} ({candidate.operator}, {score}): "
                f"{candidate.summary or '(no summary)'}"
            )
        return "\n".join(lines)

    def _data_listing(self, limit: int = 50) -> str:
        entries = []
        for path in sorted(self.problem.problem_dir.rglob("*")):
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
