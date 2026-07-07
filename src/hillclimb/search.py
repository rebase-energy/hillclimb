from __future__ import annotations

import hashlib
import queue
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hillclimb.backends.base import OperatorBackend, OperatorRequest, OperatorResult
from hillclimb.baseline import write_baseline
from hillclimb.budget import BudgetManager
from hillclimb.candidate import BackendInfo, Candidate, Trial, utcnow
from hillclimb.config import Config
from hillclimb.control import ControlCommand, apply_prune, read_commands, resync_best
from hillclimb.executor import Executor
from hillclimb.holdout import HoldoutInfo, HoldoutScorer
from hillclimb.journal import Journal
from hillclimb.prompts.render import COMPLEXITY_CUES, render
from hillclimb.scoring import ScoringError, score
from hillclimb.slots import MachineSlots
from hillclimb.status import CandidateCounts, CurrentCandidate, ScoreRef, StatusWriter
from hillclimb.problem import ProblemSpec
from hillclimb.workspace import create_candidate_workspace

TAIL_CHARS = 2000


@dataclass
class Job:
    """One operator submission: everything a worker needs, nothing it must
    share. Created scheduler-side in _prepare; the candidate object is owned
    by the worker until _commit (the journal holds its own deep copy)."""

    candidate: Candidate
    request: OperatorRequest | None  # None for the agent-less seed candidate
    workspace: Path
    ensemble_inputs: "list[Candidate] | None" = None
    holdout_threshold: float | None = None  # k-th best val at prepare time; None = no gate


@dataclass
class OutcomeMsg:
    """Terminal report from a worker; consumed by _commit on the scheduler."""

    job: Job
    kind: str  # parked | aborted | agent_failed | no_solution | executed
    result: OperatorResult | None = None
    all_ok: bool = False
    holdout_score: float | None = None
    holdout_error: str | None = None
    holdout_gated: bool = False


class ParkedSearch(Exception):
    """Raised when the backend hits a rate limit; the search can be resumed."""


class StopRequested(Exception):
    """Raised on a graceful stop (control command or SIGTERM); the search can
    be resumed."""


def tail(path: Path, chars: int = TAIL_CHARS) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")[-chars:]


class GreedySearcher:
    """Deterministic greedy policy over the candidate tree.

    Policy: (1) baseline at t=0; (2) DEBUG the newest buggy candidate while
    its chain is shallow; (3) DRAFT until `num_drafts` branches have a scored
    solution (complexity cue escalates per draft); (4) otherwise IMPROVE the
    best candidate. Stops on budget margin or max_candidates.
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
        holdout: HoldoutInfo | None = None,
        holdout_scorer: HoldoutScorer | None = None,
        status: StatusWriter | None = None,
        slots: MachineSlots | None = None,
        abort: threading.Event | None = None,
        seed_solution: Path | None = None,
        knowledge_context: str | None = None,
        complexity_start: int = 0,
    ):
        self.problem = problem
        self.config = config
        self.journal = journal
        self.backend = backend
        self.executor = executor
        self.budget = budget
        self.search_dir = search_dir
        self.max_candidates = max_candidates
        self.log = log
        self.holdout = holdout
        self.holdout_scorer = holdout_scorer
        self.status = status
        self.slots = slots  # machine-wide agent-concurrency cap (optional)
        self.abort = abort or threading.Event()
        self.seed_solution = seed_solution  # incumbent model: scored as a floor candidate
        self.knowledge_context = knowledge_context  # prior-experience prompt section
        self.complexity_start = complexity_start  # learned draft-complexity offset
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
        existing = journal.selected_candidate(problem.lower_is_better, config.holdout.selection)
        self._selection_id = existing.candidate_id if existing else None  # resume-safe

    @property
    def data_dir(self) -> Path:
        return self.holdout.data_view if self.holdout else self.problem.data_dir

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
            self.journal.candidate_result(baseline)
            self.log(f"baseline written: {baseline.summary}")
        if self.seed_solution is not None and not any(
            c.operator == "seed" for c in self.journal.candidates.values()
        ):
            self._run_seed()  # resume-idempotent: at most one seed per search
        if max(1, self.config.search.parallel_agents) > 1:
            return self._run_pool(self.config.search.parallel_agents)
        # serial loop kept verbatim: zero behavior change at parallel_agents=1;
        # the golden-equivalence test holds the pool path to the same sequence
        while not self.budget.should_stop() and len(self.journal.candidates) < self.max_candidates:
            self._process_control()
            self._check_cost_ceiling()
            self._status()
            operator, target = self.decide()
            self.log(
                f"[{self.budget.remaining_str()} left] {operator}"
                + (f" -> {target.candidate_id}" if target else "")
            )
            self.run_operator(operator, target)
        return self.journal.selected_candidate(
            self.problem.lower_is_better, self.config.holdout.selection
        )

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
            self.problem.lower_is_better, self.config.holdout.selection
        )

    def _submit(self, pool: ThreadPoolExecutor, job: Job) -> None:
        self.log(
            f"[{self.budget.remaining_str()} left] {job.candidate.operator}"
            + (f" -> {job.candidate.parent_id}" if job.candidate.parent_id else "")
            + f" ({job.candidate.candidate_id})"
        )
        self._inflight[job.candidate.candidate_id] = job
        pool.submit(lambda: self._done_q.put(self._execute_job(job)))

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
            self.journal.candidate_result(candidate)
            self._inflight.pop(candidate.candidate_id, None)
            if self.status is not None:
                self.status.remove_current(candidate.candidate_id)

    def decide_next(self) -> Job | None:
        """Pool policy: generalizes decide() to in-flight state. Priority:
        debug buggy tips (one per chain) > ensemble solo in the final window
        (drain first) > drafts until num_drafts branches are scored-or-pending
        > improves on distinct top targets. None = hold (keep slots empty)."""
        with self._state_lock:
            tip = self._debuggable_tip()
            if tip is not None:
                return self._prepare("debug", tip)
            if self._should_ensemble() and not any(
                j.candidate.operator == "ensemble" for j in self._inflight.values()
            ):
                if self._inflight:
                    return None  # drain: ensemble inputs snapshot at launch
                return self._prepare("ensemble", self._ensemble_candidates()[0])
            if self._prospective_branches() < self.config.search.num_drafts:
                return self._prepare("draft", None)
            busy_targets = {
                j.candidate.parent_id
                for j in self._inflight.values()
                if j.candidate.operator == "improve"
            }
            direction = 1 if self.problem.lower_is_better else -1
            ranked = sorted(
                self.journal.scored_candidates(), key=lambda c: direction * c.val_score
            )
            if not ranked:
                return self._prepare("draft", None)
            for candidate in ranked:
                if candidate.candidate_id not in busy_targets:
                    return self._prepare("improve", candidate)
            return self._prepare("improve", ranked[0])

    def _debuggable_tip(self) -> Candidate | None:
        """Newest buggy candidate with no active child and chain depth under
        the cap. In serial history this is exactly decide()'s debug rule."""
        for candidate in reversed(list(self.journal.candidates.values())):
            if candidate.status != "buggy" or candidate.pruned:
                continue
            children = self.journal.children(candidate.candidate_id, include_pruned=True)
            if any(c.status in ("pending", "ok", "buggy") for c in children):
                continue
            chain = self.journal.debug_chain(candidate.candidate_id)
            depth = sum(1 for c in chain if c.operator == "debug")
            if depth < self.config.search.max_debug_depth:
                return candidate
        return None

    def _prospective_branches(self) -> int:
        """Draft branches whose subtree holds a scored OR pending candidate —
        in-flight work counts toward the num_drafts target."""
        count = 0
        for draft in self.journal.drafts():
            frontier = [draft]
            while frontier:
                candidate = frontier.pop()
                if candidate.is_scored or candidate.status == "pending":
                    count += 1
                    break
                frontier.extend(self.journal.children(candidate.candidate_id))
        return count

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
        best = self.journal.best_candidate(self.problem.lower_is_better)
        if best is not None:
            fields.setdefault(
                "best", ScoreRef(candidate_id=best.candidate_id, val_score=best.val_score)
            )
        selected = self.journal.selected_candidate(
            self.problem.lower_is_better, self.config.holdout.selection
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
        commands = read_commands(self.search_dir)
        stop: ControlCommand | None = None
        for path, cmd in sorted(commands, key=lambda pc: pc[1].action != "prune"):
            path.unlink(missing_ok=True)
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
                        self.problem.lower_is_better,
                        self.config.holdout.selection,
                    )
        if stop is not None:
            self.journal.control_event("stop", reason=stop.reason, source=stop.source)
            raise StopRequested(f"stop requested by {stop.source}")

    def decide(self) -> tuple[str, Candidate | None]:
        candidates = list(self.journal.candidates.values())
        last = next(
            (c for c in reversed(candidates) if c.status in ("ok", "buggy") and not c.pruned), None
        )
        if last is not None and last.status == "buggy":
            chain = self.journal.debug_chain(last.candidate_id)
            depth = sum(1 for c in chain if c.operator == "debug")
            if depth < self.config.search.max_debug_depth:
                return "debug", last
        if self._should_ensemble():
            return "ensemble", self._ensemble_candidates()[0]
        if self._scored_branches() < self.config.search.num_drafts:
            return "draft", None
        best = self.journal.best_candidate(self.problem.lower_is_better)
        if best is None:
            return "draft", None
        return "improve", best

    # --- ensemble stage ---

    def _in_ensemble_window(self) -> bool:
        # window sits ABOVE the stop margin, else margin swallows it: with a
        # 45m budget, reserve(540s) - margin(300s) left a 240s slot that one
        # improve cycle stepped over entirely
        reserve = self.budget.total_s * self.config.ensemble.reserve_fraction
        return self.budget.remaining() <= reserve + self.budget.stop_margin_s

    def _should_ensemble(self) -> bool:
        cfg = self.config.ensemble
        if not cfg.enabled or not self._in_ensemble_window():
            return False
        attempts = sum(1 for c in self.journal.candidates.values() if c.operator == "ensemble")
        if attempts >= cfg.max_attempts or self._ensemble_succeeded():
            return False
        return len(self._ensemble_candidates()) >= 2

    def _ensemble_succeeded(self) -> bool:
        for candidate in self.journal.candidates.values():
            if candidate.status != "ok":
                continue
            root = self.journal.debug_chain(candidate.candidate_id)[0]
            if root.operator == "ensemble":
                return True
        return False

    def _ensemble_candidates(self) -> list[Candidate]:
        """Top-k scored non-ensemble candidates by the selection rule, deduped
        by script content so near-identical improves don't fill the slots."""
        ranked = self.journal.ranked_candidates(
            self.problem.lower_is_better, self.config.holdout.selection
        )
        picked, seen_hashes = [], set()
        for candidate in ranked:
            if candidate.operator == "ensemble":
                continue
            solution = Path(candidate.workspace) / "solution.py"
            if not solution.exists():
                continue
            digest = hashlib.md5(solution.read_bytes()).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            picked.append(candidate)
            if len(picked) >= self.config.ensemble.top_k:
                break
        return picked

    def _scored_branches(self) -> int:
        """Draft branches whose subtree contains at least one scored candidate."""
        count = 0
        for draft in self.journal.drafts():
            frontier = [draft]
            while frontier:
                candidate = frontier.pop()
                if candidate.is_scored:
                    count += 1
                    break
                frontier.extend(self.journal.children(candidate.candidate_id))
        return count

    def _run_seed(self) -> Candidate:
        """Score the incumbent solution as a real candidate: the floor a
        re-search must beat. No agent call; holdout always evaluated (it is
        the selection floor, so the top-k gate does not apply)."""
        seed = self.seed_solution.absolute()
        if not seed.exists():
            raise FileNotFoundError(f"seed solution not found: {seed}")
        candidate_id = self.journal.next_candidate_id()
        workspace = create_candidate_workspace(
            self.search_dir,
            candidate_id,
            self.data_dir,
            self.problem.problem_dir,
            parent_solution=seed,
        )
        candidate = Candidate(
            candidate_id=candidate_id,
            operator="seed",
            workspace=str(workspace),
            summary=f"incumbent model seeded from {seed.name}",
        )
        self.journal.candidate_created(candidate)
        self.log(f"seeding incumbent {seed.name} as {candidate_id}")
        exec_timeout = self.config.budget.exec_timeout_s
        all_ok = self._run_trials(candidate, workspace / "solution.py", workspace, exec_timeout)
        holdout_score = holdout_error = None
        if all_ok:
            holdout_score, holdout_error = self._score_holdout(workspace)
        msg = OutcomeMsg(
            job=Job(candidate=candidate, request=None, workspace=workspace),
            kind="executed",
            all_ok=all_ok,
            holdout_score=holdout_score,
            holdout_error=holdout_error,
        )
        committed = self._commit(msg)
        score = f"val={committed.val_score}" if committed.val_score is not None else "buggy"
        self.log(f"  seed scored: {score}")
        return committed

    # --- candidate lifecycle: _prepare (scheduler) → _execute_job (worker)
    # --- → _commit (scheduler); run_operator is the synchronous composition

    def run_operator(self, operator: str, target: Candidate | None) -> Candidate:
        job = self._prepare(operator, target)
        return self._commit(self._execute_job(job))

    def _prepare(self, operator: str, target: Candidate | None) -> Job:
        """Scheduler-side setup: id, workspace, prompt, journal `created`."""
        candidate_id = self.journal.next_candidate_id()
        parent_solution = (
            Path(target.workspace) / "solution.py"
            if target is not None and operator in ("debug", "improve")
            else None
        )
        workspace = create_candidate_workspace(
            self.search_dir,
            candidate_id,
            self.data_dir,
            self.problem.problem_dir,
            parent_solution,
        )
        ensemble_inputs = None
        if operator == "ensemble":
            ensemble_inputs = self._ensemble_candidates()
            for i, cand in enumerate(ensemble_inputs, 1):
                shutil.copy(Path(cand.workspace) / "solution.py", workspace / f"candidate_{i}.py")
        complexity = self._draft_complexity() if operator == "draft" else None
        prompt = self.build_prompt(operator, target, complexity, ensemble_inputs)
        (workspace / "prompt.md").write_text(prompt)

        # NOTE: no session resume across candidates — Claude Code scopes
        # sessions to the cwd, and every candidate has its own workspace, so
        # --resume can't find a sibling workspace's session. The debug prompt
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
            workspace=str(workspace),
        )
        self.journal.candidate_created(candidate)

        request = OperatorRequest(
            operator=operator,
            prompt=prompt,
            workspace=workspace,
            timeout_s=min(
                self.config.budget.agent_timeout_s, max(60, int(self.budget.remaining()))
            ),
            model=self.config.model,
        )
        if self.status is not None:
            self.status.add_current(
                CurrentCandidate(
                    candidate_id=candidate_id,
                    operator=operator,
                    phase="agent",
                    workspace=str(workspace),
                )
            )
            self._status()
        return Job(
            candidate=candidate,
            request=request,
            workspace=workspace,
            ensemble_inputs=ensemble_inputs,
            holdout_threshold=self._holdout_threshold(),
        )

    def _execute_job(self, job: Job) -> OutcomeMsg:
        """Worker-side: agent call + trials + holdout. Lock-free — touches
        only the job's own candidate/workspace, never the journal."""
        candidate = job.candidate

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
            result = self.backend.invoke(job.request)
        finally:
            if slot is not None:
                slot.release()
        candidate.backend = BackendInfo(
            name=self.backend.name,
            session_id=result.session_id,
            cost_usd=result.cost_usd,
            num_turns=result.num_turns,
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

        notes = job.workspace / "notes.md"
        if notes.exists():
            lines = notes.read_text().strip().splitlines()
            candidate.summary = lines[0] if lines else ""

        solution = job.workspace / "solution.py"
        if not solution.exists():
            return OutcomeMsg(job=job, kind="no_solution", result=result)

        exec_timeout = min(
            self.config.budget.exec_timeout_s, max(60, int(self.budget.remaining() - 30))
        )
        self._set_phase(candidate.candidate_id, "exec")
        all_ok = self._run_trials(candidate, solution, job.workspace, exec_timeout)

        holdout_score = holdout_error = None
        gated = False
        if all_ok:
            if self._gate_passes(candidate.val_score, job.holdout_threshold):
                self._set_phase(candidate.candidate_id, "holdout")
                holdout_score, holdout_error = self._score_holdout(job.workspace)
            else:
                gated = True  # climbs on val; not selectable via holdout
        return OutcomeMsg(
            job=job,
            kind="executed",
            result=result,
            all_ok=all_ok,
            holdout_score=holdout_score,
            holdout_error=holdout_error,
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
                    self.journal.candidate_result(candidate)
                    raise ParkedSearch(result.error_message)

                if msg.kind == "aborted":
                    candidate.status = "abandoned"
                    candidate.summary = "stopped mid-operator (abort)"
                    candidate.finished_at = utcnow()
                    self.journal.candidate_result(candidate)
                    return candidate

                if msg.kind == "agent_failed":
                    candidate.status = "abandoned"
                    candidate.summary = (
                        f"agent call failed ({result.error_kind}): {result.error_message[:150]}"
                    )
                    candidate.finished_at = utcnow()
                    self.journal.candidate_result(candidate)
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
                    self.journal.candidate_result(candidate)
                    return candidate

                # kind == "executed"
                if msg.all_ok:
                    last = candidate.trials[-1]
                    if msg.holdout_error is not None:
                        candidate.status = "buggy"
                        last.holdout_error = msg.holdout_error
                    else:
                        candidate.status = "ok"
                        if not msg.holdout_gated:
                            last.holdout_score = msg.holdout_score
                    if candidate.status == "ok":
                        previous_best = self.journal.best_candidate(self.problem.lower_is_better)
                        if previous_best is None or self._improves(
                            candidate.val_score, previous_best.val_score
                        ):
                            candidate.is_best = True
                else:
                    candidate.status = "buggy"
                candidate.finished_at = utcnow()
                self.journal.candidate_result(candidate)
                if candidate.status == "ok":
                    self._sync_selection()
                return candidate
            finally:
                if self.status is not None:
                    self.status.remove_current(candidate.candidate_id)
                    self._status()

    def _set_phase(self, candidate_id: str, phase: str) -> None:
        if self.status is not None:
            self.status.update_current(candidate_id, phase=phase)

    def _run_trials(
        self, candidate: Candidate, solution: Path, workspace: Path, exec_timeout: int
    ) -> bool:
        """Run n_trials validation evaluations (parallel when >1, each in its
        own trial dir with a distinct seed) and append the Trials in index
        order. Returns True only if every trial passed — a seed-flaky
        candidate is buggy."""
        n = max(1, self.config.search.n_trials)
        if n == 1:
            trial, ok = self._execute_one_trial(solution, workspace, exec_timeout, seed=None)
            candidate.trials.append(trial)
            return ok

        from concurrent.futures import ThreadPoolExecutor

        from hillclimb.workspace import create_trial_dir

        def run(index: int) -> tuple[Trial, bool]:
            trial_dir = create_trial_dir(workspace, index)
            return self._execute_one_trial(
                trial_dir / solution.name, trial_dir, exec_timeout, seed=index
            )

        with ThreadPoolExecutor(max_workers=n, thread_name_prefix="trial") as pool:
            results = list(pool.map(run, range(n)))
        candidate.trials.extend(trial for trial, _ in results)
        # trial-0 artifacts surface at the workspace root so best/-sync,
        # ensemble copies, and CSV holdout scoring stay untouched
        t0 = workspace / "trials" / "t0"
        for name in ("submission.csv", "holdout_predictions.csv", "eval_result.json"):
            if (t0 / name).exists():
                shutil.copy(t0 / name, workspace / name)
        return all(ok for _, ok in results)

    def _execute_one_trial(
        self, solution: Path, cwd: Path, exec_timeout: int, seed: int | None
    ) -> tuple[Trial, bool]:
        trial_started = utcnow()
        exec_result = self.executor.execute(
            solution,
            cwd,
            exec_timeout,
            verifier=self.problem.verifier,
            seed=seed,
        )
        stdout_tail = tail(Path(exec_result.stdout_path)) if exec_result.stdout_path else ""
        if not exec_result.ok and not stdout_tail.strip():
            # a silent crash is undebuggable from the journal (the only state
            # synced off remote machines) — surface stderr instead
            stderr = tail(Path(exec_result.stdout_path).with_name("exec_stderr.log"), 800)
            if stderr.strip():
                stdout_tail = f"[stderr] {stderr}"
        trial = Trial(
            seed=seed,
            returncode=exec_result.returncode,
            duration_s=exec_result.duration_s,
            timed_out=exec_result.timed_out,
            stdout_tail=stdout_tail,
            submission_ok=exec_result.submission_ok,
            val_score=exec_result.val_score if exec_result.ok else None,
            started_at=trial_started,
            finished_at=utcnow(),
        )
        return trial, exec_result.ok

    def _holdout_threshold(self) -> float | None:
        """Holdout hygiene: the k-th best val score at prepare time; a
        candidate must beat (or tie) it to earn a holdout evaluation.
        None = no gate (top_k disabled or fewer than k scored candidates).
        Snapshot semantics: slightly stale under parallelism, exact in
        serial — an acceptable heuristic for a hygiene gate."""
        top_k = self.config.holdout.top_k
        if top_k <= 0:
            return None
        scored = sorted(
            (c.val_score for c in self.journal.scored_candidates() if c.val_score is not None),
            reverse=not self.problem.lower_is_better,
        )
        if len(scored) < top_k:
            return None
        return scored[top_k - 1]

    def _gate_passes(self, val_score: float | None, threshold: float | None) -> bool:
        if threshold is None or val_score is None:
            return True
        return self._improves(val_score, threshold) or val_score == threshold

    def _sync_selection(self) -> None:
        """Keep best/ pointing at the currently selected candidate. Selection
        is recomputed over the whole tree because rank-blend can shift between
        existing candidates when a new one lands."""
        selected = self.journal.selected_candidate(
            self.problem.lower_is_better, self.config.holdout.selection
        )
        if selected is None or selected.candidate_id == self._selection_id:
            return
        self._selection_id = resync_best(
            self.search_dir, self.journal, self.problem.lower_is_better, self.config.holdout.selection
        )
        scores = f"val_score={selected.val_score}"
        if selected.holdout_score is not None:
            scores += f" holdout={selected.holdout_score:.5g}"
        self.log(f"  new selection: {selected.candidate_id} {scores}")

    def _score_holdout(self, workspace: Path) -> tuple[float | None, str | None]:
        """Score holdout predictions; (score, None) on success, (None, reason)
        on contract violation, (None, None) when holdout is disabled."""
        if self.holdout_scorer is not None:
            return self.holdout_scorer.score(workspace)
        if self.holdout is None:
            return None, None
        pred_path = workspace / "holdout_predictions.csv"
        if not pred_path.exists():
            return None, "`holdout_predictions.csv` was not written"
        try:
            predictions = pd.read_csv(pred_path)
            answers = pd.read_csv(self.holdout.answers_path)
            value = score(self.problem.metric_name, answers, predictions, self.holdout.id_col)
        except ScoringError as e:
            return None, f"holdout_predictions.csv could not be scored: {e}"
        except Exception as e:
            return None, f"holdout_predictions.csv is unreadable: {e}"
        return value, None

    def _improves(self, score: float, best: float) -> bool:
        return score < best if self.problem.lower_is_better else score > best

    def _draft_complexity(self) -> str:
        index = len(self.journal.drafts()) + self.complexity_start
        return "minimal" if index == 0 else "moderate" if index == 1 else "advanced"

    # --- prompt assembly ---

    def build_prompt(
        self,
        operator: str,
        target: Candidate | None,
        complexity: str | None,
        ensemble_inputs: list[Candidate] | None = None,
    ) -> str:
        holdout_clause = ""
        if self.holdout is not None:
            if self.holdout.strategy == "time-tail":
                if self.holdout.group_col:
                    scope = f"the most recent rows of each `{self.holdout.group_col}` block"
                elif self.holdout.time_cutoff:
                    scope = f"all rows from {self.holdout.time_cutoff} onward"
                else:
                    scope = "the most recent rows"
                split_note = (
                    f"These rows are the chronological TAIL of the training data ({scope}), "
                    "held out by the orchestrator. Treat them as a true forecast: do not "
                    "train on them, and do not use any information from the holdout period."
                )
            else:
                split_note = "These rows were held out at random from the training data."
            # class-columns problems (target column holds class names, e.g.
            # spooky's `author`): predictions must be per-class probability
            # columns in submission format, NOT the raw target column — an
            # agent following the literal column name writes hard labels the
            # log-loss scorer can't grade
            sample_cols = pd.read_csv(self.problem.sample_submission, nrows=0).columns
            if set(self.holdout.target_cols) <= set(sample_cols):
                target_cols_note = ", ".join(f"`{c}`" for c in self.holdout.target_cols)
            else:
                pred_cols = [c for c in sample_cols if c != sample_cols[0]]
                shown = ", ".join(f"`{c}`" for c in pred_cols[:6])
                if len(pred_cols) > 6:
                    shown += f", … ({len(pred_cols)} columns)"
                target_cols_note = (
                    f"the same prediction columns as submission.csv: {shown}"
                )
            holdout_clause = render(
                "holdout_clause",
                holdout_id_col=self.holdout.id_col,
                holdout_target_cols=target_cols_note,
                holdout_split_note=split_note,
            ).rstrip()
        network_note = (
            "Internet access IS available at execution time — this problem's rules "
            "permit fetching external data; cache downloads to files in the "
            "working directory so reruns don't refetch."
            if self.problem.allow_network
            else "Assume no internet access at execution time."
        )
        contract_template = "contract_emflow" if self.problem.kind == "emflow" else "contract"
        contract = render(
            contract_template,
            metric_name=self.problem.metric_name,
            exec_timeout_min=self.config.budget.exec_timeout_s // 60,
            runtime_pkgs=self._runtime_pkgs(),
            time_remaining=self.budget.remaining_str(),
            holdout_clause=holdout_clause,
            network_note=network_note,
            verifier_clause=self._verifier_clause(),
            emflow_problem=self.problem.emflow_problem or "",
            quantile_note=self._quantile_note(),
        )
        direction = "lower is better" if self.problem.lower_is_better else "higher is better"
        if operator == "draft":
            return render(
                "draft",
                description=self.problem.description,
                metric_name=self.problem.metric_name,
                direction=direction,
                data_listing=self._data_listing(),
                complexity_cue=COMPLEXITY_CUES[complexity or "minimal"],
                prior_experience=self.knowledge_context or "(no prior searches recorded)",
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
                stderr_tail=tail(Path(target.workspace) / "exec_stderr.log"),
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
            return render(
                "improve",
                description=self.problem.description,
                metric_name=self.problem.metric_name,
                direction=direction,
                best_score=target.val_score,
                stdout_tail=last_trial.stdout_tail if last_trial else "",
                sibling_summaries=self._candidate_summaries(self.journal.children(target.candidate_id))
                or "(nothing tried from this solution yet)",
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
        if not trial.submission_ok:
            problems.append("`submission.csv` was not written")
        if trial.val_score is None:
            problems.append("no final `val_score: <float>` line was printed")
        if trial.holdout_error:
            problems.append(trial.holdout_error)
        if problems:
            return "The script ran to completion but violated the contract: " + "; ".join(problems) + "."
        return "The script failed."

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

    def _verifier_clause(self) -> str:
        if self.problem.verifier is None:
            return (
                "- prints exactly one line `val_score: <float>` "
                f"(your validation {self.problem.metric_name}) as the FINAL line of stdout"
            )
        try:
            verifier_name = self.problem.verifier.relative_to(self.problem.problem_dir)
        except ValueError:
            verifier_name = self.problem.verifier.name
        return (
            f"- writes `./submission.csv`; the orchestrator then runs "
            f"`./problem/{verifier_name}` and uses the verifier's final "
            "`val_score: <float>` line as the official validation score. "
            "Do not print your own `val_score:` line."
        )

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

        kind = getattr(self.problem, "kind", "csv")
        return ", ".join(runtime_packages(kind))


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
