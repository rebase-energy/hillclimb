"""Shared candidate-evaluation helpers: trial execution, report trust,
holdout scoring, and score-comparison semantics — extracted from
the searcher so every search engine (greedy, GEPA, future runners)
interprets verifier results identically.

Concurrency contract
--------------------
`CandidateEvaluator` never writes the journal: it touches only the
executor, the `Candidate` object it is handed, and that candidate's
directory — safe to call from an operator worker or an engine thread. Its
one journal *read* (the holdout top-k gate, `holdout_threshold`) takes the
journal's lock, so it is safe from any thread. `accept_band` is the
exception: it reads journal state and must only run on the thread that
owns the journal (the scheduler, or an engine holding its state lock).

Holdout is the host's, not a strategy's
---------------------------------------
The hidden split is scored HERE, as part of evaluating a trial, never by a
search strategy: a strategy calls `run_trial` and gets back a trial that
may already carry `holdout_score`/`holdout_error`, and a trial that fails
the hidden split is reported as not-ok — the same contract failure as a
verifier that exits non-zero. WHEN holdout runs is the host's decision
(`holdout_timing`): `inline` scores each candidate's best trial as it
lands (the TUI shows holdout live; the top-k gate limits the spend),
`after` leaves it to `finalize_holdout` once the strategy has returned
(engines whose state must never see a holdout value — GEPA). A strategy
never holds the scorer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from hillclimb.candidate import Candidate, Replicate, Trial, UnitTestResult, utcnow
from hillclimb.config import Config
from hillclimb.executor import RESULT_FILE, Executor, HoldoutScorer, read_result
from hillclimb.journal import Journal
from hillclimb.problem import ProblemSpec

if TYPE_CHECKING:
    from hillclimb.status import StatusWriter
    from hillclimb.unit_tests import UnitTestRunner

HOLDOUT_TIMINGS = ("inline", "after")
# floors are always holdout-scored: they are the selection floor a re-search
# must beat, so the top-k spend gate does not apply to them
_UNGATED_OPERATORS = ("seed", "baseline")

TAIL_CHARS = 2000


def tail(path: Path, chars: int = TAIL_CHARS) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")[-chars:]


@dataclass
class CandidateEvaluator:
    """Trial/replicate execution, report reading and holdout scoring for one
    search — the host's `evaluate` service. Reads `config`/`problem` per
    call (they may be mutated by tests), writes only into the candidate dir
    it is given, never the journal (it only reads it for the holdout gate)."""

    executor: Executor
    problem: ProblemSpec
    config: Config
    unit_test_runner: UnitTestRunner | None = None
    holdout_scorer: HoldoutScorer | None = None
    # host wiring for holdout (see the module docstring); a strategy sets none of it
    holdout_timing: str = "inline"
    journal: Journal | None = None  # the top-k gate; None = no gate
    status: StatusWriter | None = None  # phase="holdout" for `watch`
    log: Callable[[str], None] = field(default=print)

    def __post_init__(self) -> None:
        if self.holdout_timing not in HOLDOUT_TIMINGS:
            raise ValueError(
                f"holdout_timing must be one of {HOLDOUT_TIMINGS}, got {self.holdout_timing!r}"
            )

    def run_trial(
        self,
        candidate: Candidate,
        solution: Path,
        candidate_dir: Path,
        exec_timeout: int,
        *,
        params: dict | None = None,
        params_doc: dict | None = None,
        index: int | None = None,
        n_replicates: int | None = None,
    ) -> tuple[Trial, bool]:
        """Run one trial — one parameter set — as `search.n_replicates`
        seeded executions, each in its own replicate dir, and append it to
        the candidate. Returns (trial, all_ok); all_ok is True only if every
        replicate passed — a seed-flaky trial is buggy.

        `params` are the trial's concrete values (journaled), `params_doc`
        the params.json document written into the trial dir (the declared
        space with a `value` per entry); both None for an undeclared
        candidate. `index` defaults to the next free slot.

        `search.replicate_mode` decides whether replicates share the machine:
        parallel for seed variance, serial when the metric is a measurement
        of the machine itself (time, memory, throughput) and concurrent runs
        would measure each other."""
        from concurrent.futures import ThreadPoolExecutor

        from hillclimb.dirs import create_replicate_dir, create_trial_dir, hoist_replicate, replicate_dir

        explicit_index = index  # tune jobs: the status entry is keyed by trial
        if index is None:
            index = len(candidate.trials)
        tdir = create_trial_dir(candidate_dir, index, params_doc)
        trial = Trial(index=index, params=dict(params or {}))
        n = max(1, n_replicates if n_replicates is not None else self.config.evaluation.n_replicates)

        def run(j: int) -> tuple[Replicate, bool]:
            rdir = create_replicate_dir(tdir, j)
            # a single replicate keeps seed=None: existing verifiers see no
            # env change; repeated replicates get distinct seeds
            return self.execute_replicate(
                rdir / solution.name, rdir, exec_timeout, seed=j if n > 1 else None
            )

        # The first verifier run proves that the solution is executable before
        # correctness failures are classified. Tests then run exactly once for
        # this code+params trial; only a passing trial earns further score
        # replicates.
        results = [run(0)]
        first_ok = results[0][1]
        if not first_ok:
            trial.verdict = "buggy"
        elif self.unit_test_runner is not None:
            if self.status is not None:
                self.status.update_current(candidate.candidate_id, explicit_index, phase="tests")
            elapsed = results[0][0].duration_s or 0.0
            remaining = exec_timeout - elapsed
            if remaining <= 0:
                trial.unit_tests = UnitTestResult(
                    timed_out=True,
                    duration_s=0.0,
                    stderr_tail="unit tests had no time remaining after the verifier run",
                )
                trial.verdict = "buggy"
            else:
                trial.unit_tests = self.unit_test_runner.run(
                    tdir / solution.name, tdir, remaining
                )
                import shutil

                for log_name in ("tests_stdout.log", "tests_stderr.log"):
                    log_path = tdir / log_name
                    if log_path.exists():
                        shutil.copy(log_path, candidate_dir / log_name)
                if trial.unit_tests.timed_out:
                    trial.verdict = "buggy"
                elif (
                    trial.unit_tests.returncode is None
                    or trial.unit_tests.returncode < 0
                ):
                    trial.verdict = "buggy"
                elif not trial.unit_tests.passed:
                    trial.verdict = "failing"
                else:
                    trial.verdict = "passing"
        else:
            trial.verdict = "passing"

        if trial.verdict == "passing" and n > 1:
            indexes = range(1, n)
            if self.config.evaluation.replicate_mode == "serial":
                results.extend(run(j) for j in indexes)
            else:
                with ThreadPoolExecutor(
                    max_workers=n - 1, thread_name_prefix="replicate"
                ) as pool:
                    results.extend(pool.map(run, indexes))
            if not all(ok for _, ok in results):
                trial.verdict = "buggy"
        trial.replicates.extend(r for r, _ in results)
        trial.finished_at = utcnow()
        candidate.trials.append(trial)
        candidate.stamp_best_trial(self.problem.higher_is_better)
        # r0 of the FIRST trial surfaces at the candidate root right away
        # (the engine re-hoists when a later trial becomes the best one)
        if index == 0:
            hoist_replicate(candidate_dir, replicate_dir(tdir, 0), self.problem.output_artifacts)
        all_ok = trial.verdict == "passing" and all(ok for _, ok in results)
        if all_ok and self._holdout_now(candidate, trial):
            # the host scores the hidden split as part of evaluation: a trial
            # that fails it is not ok, exactly like one that fails the verifier
            self._score_holdout(candidate, candidate_dir, trial, status_key=explicit_index)
            all_ok = trial.holdout_error is None
        return trial, all_ok

    # --- holdout: the host's, scored here, never by a strategy ---

    def _holdout_now(self, candidate: Candidate, trial: Trial) -> bool:
        """Score this trial's hidden split right now? Only in `inline`
        timing, only for the trial the candidate would ship (its best), and
        past the top-k spend gate — floors (seed, baseline) are never gated:
        they are the selection floor itself."""
        if self.holdout_scorer is None or self.holdout_timing != "inline":
            return False
        if not trial.is_best:
            return False  # a losing parameter set never ships
        if candidate.operator in _UNGATED_OPERATORS or self.journal is None:
            return True
        threshold = holdout_threshold(
            self.journal,
            top_k=self.config.holdout.top_k,
            higher_is_better=self.problem.higher_is_better,
        )
        return gate_passes(trial.val_score, threshold, higher_is_better=self.problem.higher_is_better)

    def _score_holdout(
        self, candidate: Candidate, candidate_dir: Path, trial: Trial, *, status_key: int | None
    ) -> None:
        """Run the scorer against `trial` and stamp the result on it (cpu is
        stamped even when scoring errored — the cost was paid)."""
        if self.status is not None:
            self.status.update_current(candidate.candidate_id, status_key, phase="holdout")
        score, error, cpu_s = self.holdout_scorer.score(candidate_dir, trial)
        trial.holdout_cpu_s = cpu_s
        if error is not None:
            trial.holdout_error = error
            self.log(f"  holdout failed for {candidate.candidate_id} t{trial.index}: {error}")
        else:
            trial.holdout_score = score

    def finalize_holdout(self, journal: Journal) -> list[Candidate]:
        """`after` timing: score the hidden split for the top-k validation
        candidates once the strategy has returned, and re-journal each
        (replay keeps the last record). Idempotent — candidates that already
        carry a holdout count toward k and are not re-scored. Returns the
        candidates it scored."""
        from hillclimb.status import CurrentCandidate

        if self.holdout_scorer is None:
            return []
        ranked = [
            c
            for c in sorted(
                journal.scored_candidates(),
                key=lambda c: (-c.val_score if self.problem.higher_is_better else c.val_score),
            )
            if c.trials
        ]
        top_k = self.config.holdout.top_k
        wanted = len(ranked) if top_k <= 0 else top_k
        scored, done = 0, []
        for candidate in ranked:
            if scored >= wanted:
                break
            if candidate.holdout_score is not None:
                scored += 1
                continue
            trial = candidate.best_trial or candidate.trials[-1]
            if self.status is not None:
                self.status.add_current(
                    CurrentCandidate(
                        candidate_id=candidate.candidate_id,
                        operator=candidate.operator,
                        phase="holdout",
                        candidate_dir=candidate.candidate_dir,
                    )
                )
            try:
                self._score_holdout(candidate, Path(candidate.candidate_dir), trial, status_key=None)
            finally:
                if self.status is not None:
                    self.status.remove_current(candidate.candidate_id)
            if trial.holdout_error is None:
                scored += 1
            journal.candidate_result(candidate)
            done.append(candidate)
        return done

    def hoist_trial(self, candidate_dir: Path, trial: Trial) -> None:
        """Re-surface a trial's r0 outputs at the candidate root — called by
        engines when a later trial becomes the candidate's best."""
        from hillclimb.dirs import hoist_replicate, replicate_dir, trial_dir

        hoist_replicate(
            candidate_dir, replicate_dir(trial_dir(candidate_dir, trial.index), 0),
            self.problem.output_artifacts,
        )

    def execute_replicate(
        self, solution: Path, cwd: Path, exec_timeout: int, seed: int | None
    ) -> tuple[Replicate, bool]:
        started = utcnow()
        exec_result = self.executor.execute(solution, cwd, exec_timeout, seed=seed)
        stdout_tail = tail(Path(exec_result.stdout_path)) if exec_result.stdout_path else ""
        if not exec_result.ok and not stdout_tail.strip():
            # a silent crash is undebuggable from the journal (the only state
            # synced off remote machines) — surface stderr instead
            stderr = tail(Path(exec_result.stdout_path).with_name("exec_stderr.log"), 800)
            if stderr.strip():
                stdout_tail = f"[stderr] {stderr}"
        replicate = Replicate(
            seed=seed,
            returncode=exec_result.returncode,
            duration_s=exec_result.duration_s,
            cpu_s=exec_result.cpu_s,
            timed_out=exec_result.timed_out,
            stdout_tail=stdout_tail,
            submission_ok=exec_result.submission_ok,
            val_score=exec_result.val_score if exec_result.ok else None,
            report=self.read_replicate_report(cwd) if exec_result.ok else None,
            metrics=exec_result.metrics if exec_result.ok else {},
            instance_scores=exec_result.instance_scores if exec_result.ok else {},
            started_at=started,
            finished_at=utcnow(),
        )
        return replicate, exec_result.ok

    def read_replicate_report(self, cwd: Path) -> dict | None:
        """Compact validation breakdown from the eval's eval_result.json —
        hillclimb's evaluator report contract. Producers: the emflow eval
        runner, a problem's verifier script (the executor discards anything
        else on verifier problems), or the agent's own solution when the
        problem has no verifier. The split check is the orchestrator half of
        the leakage contract: holdout and verify results must never reach
        prompts."""
        # the result file may legitimately be a bare number (the simplest
        # verifier form) — only the object form can carry a report
        _, payload = read_result(cwd / RESULT_FILE)
        if not isinstance(payload, dict):
            return None
        if payload.get("split") != "validation" or not isinstance(payload.get("report"), dict):
            return None
        from hillclimb.report import compact_report

        try:
            compact = compact_report(payload["report"])
        except Exception:  # noqa: BLE001 — a malformed report must never fail a trial
            return None
        # provenance is stamped from problem configuration, not file contents:
        # an agent-authored file cannot claim evaluator trust
        compact["source"] = "evaluator" if self.problem.report_trusted else "agent"
        return compact

    def score_holdout(
        self, candidate_dir: Path, trial: Trial | None = None
    ) -> tuple[float | None, str | None, float | None]:
        """Raw scorer call (the baseline writer and tests): (score, None,
        cpu_s) on success, (None, reason, cpu_s) on contract violation,
        (None, None, None) when this search has no holdout."""
        if self.holdout_scorer is None:
            return None, None, None
        return self.holdout_scorer.score(candidate_dir, trial)


# --- score-comparison semantics (pure; band always explicit — no journal) ---


def improves(score: float, best: float, *, higher_is_better: bool, band: float) -> bool:
    """Strictly better by more than the accept band. Pass band=0.0 for a
    raw comparison (ranking and gating, where a near-tie should still be
    evaluated rather than dropped)."""
    delta = (score - best) if higher_is_better else (best - score)
    return delta > band


def gate_passes(
    val_score: float | None, threshold: float | None, *, higher_is_better: bool
) -> bool:
    """Holdout hygiene gate: pass on a raw win or an exact tie against the
    k-th best val score. None threshold (gate disabled) or None score
    (unscored) always passes."""
    if threshold is None or val_score is None:
        return True
    return (
        improves(val_score, threshold, higher_is_better=higher_is_better, band=0.0)
        or val_score == threshold
    )


# --- journal-reading helpers ---
# accept_band: LOCK-HOLDER / SCHEDULER-THREAD ONLY. holdout_threshold: any
# thread (it takes the journal's lock) — the evaluator gates from workers.


def accept_band(config: Config, journal: Journal) -> float:
    """How much better a candidate must be before the engine believes it.

    `min_improvement` is the author's own floor in metric units; `noise_k`
    multiples of the measured noise floor is the search's own evidence
    about itself. Zero (the default) is the strict comparison."""
    band = config.evaluation.min_improvement
    if config.evaluation.noise_k > 0:
        floor = journal.noise_floor()
        if floor is not None:
            band = max(band, config.evaluation.noise_k * floor)
    return band


def holdout_threshold(journal: Journal, *, top_k: int, higher_is_better: bool) -> float | None:
    """Holdout hygiene: the k-th best val score right now; a candidate
    must beat (or tie) it to earn a holdout evaluation. None = no gate
    (top_k disabled or fewer than k scored candidates). Snapshot semantics:
    slightly stale under parallelism (in-flight results are not in it),
    exact in serial — an acceptable heuristic for a hygiene gate."""
    if top_k <= 0:
        return None
    with journal.lock:
        scored = sorted(
            (c.val_score for c in journal.scored_candidates() if c.val_score is not None),
            reverse=higher_is_better,
        )
    if len(scored) < top_k:
        return None
    return scored[top_k - 1]


# --- EvalResult: the rich projection engines consume ---


@dataclass(frozen=True)
class ReplicateSummary:
    """One seeded execution, flattened for engine feedback."""

    seed: int | None
    val_score: float | None
    returncode: int | None
    timed_out: bool
    duration_s: float | None
    submission_ok: bool
    stdout_tail: str


@dataclass(frozen=True)
class TrialSummary:
    """One parameter set and its replicates, flattened for engine feedback."""

    index: int
    params: dict
    val_score: float | None
    verdict: str | None = None
    test_stdout_tail: str = ""
    test_stderr_tail: str = ""
    replicates: tuple[ReplicateSummary, ...] = ()


def summarize_replicate(replicate: Replicate) -> ReplicateSummary:
    return ReplicateSummary(
        seed=replicate.seed,
        val_score=replicate.val_score,
        returncode=replicate.returncode,
        timed_out=replicate.timed_out,
        duration_s=replicate.duration_s,
        submission_ok=replicate.submission_ok,
        stdout_tail=replicate.stdout_tail,
    )


def summarize_trial(trial: Trial) -> TrialSummary:
    return TrialSummary(
        index=trial.index,
        params=dict(trial.params),
        val_score=trial.val_score,
        verdict=trial.verdict,
        test_stdout_tail=(trial.unit_tests.stdout_tail if trial.unit_tests else ""),
        test_stderr_tail=(trial.unit_tests.stderr_tail if trial.unit_tests else ""),
        replicates=tuple(summarize_replicate(r) for r in trial.replicates),
    )


@dataclass(frozen=True)
class EvalResult:
    """Read-only projection of a terminal candidate for a search engine:
    raw journal-direction score (never negated — direction transforms live
    at the engine boundary), validity, behavior-descriptor features, and
    bounded reflection feedback. Holdout values never appear here."""

    candidate_id: str
    score: float | None
    valid: bool
    instance_scores: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    trials: tuple[TrialSummary, ...] = ()
    cost_usd: float = 0.0


def eval_result_for(candidate: Candidate, *, feedback: str = "") -> EvalResult:
    """Project a committed candidate into an EvalResult. `feedback` is
    caller-supplied (engines assemble their own reflection text)."""
    valid = candidate.status == "passing"
    return EvalResult(
        candidate_id=candidate.candidate_id,
        score=candidate.val_score if valid else None,
        valid=valid,
        instance_scores=dict(candidate.instance_scores) if valid else {},
        features=dict(candidate.metrics) if valid else {},
        feedback=feedback,
        trials=tuple(summarize_trial(t) for t in candidate.trials),
        cost_usd=candidate.backend.cost_usd or 0.0,
    )
