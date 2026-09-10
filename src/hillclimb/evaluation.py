"""Shared candidate-evaluation helpers: trial execution, report trust,
holdout scoring, and score-comparison semantics — extracted from
`GreedySearcher` so every search engine (greedy, GEPA, future runners)
interprets verifier results identically.

Concurrency contract
--------------------
`CandidateEvaluator` is journal-free by construction: it touches only the
executor, the `Candidate` object it is handed, and that candidate's
directory — safe to call from an operator worker or an engine thread.
The journal-reading free functions at the bottom (`accept_band`,
`holdout_threshold`) are the exception: they read journal state and must
only run on the thread that owns the journal (the scheduler, or an engine
holding its state lock).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hillclimb.candidate import Candidate, Replicate, Trial, utcnow
from hillclimb.config import Config
from hillclimb.executor import RESULT_FILE, Executor, HoldoutScorer, read_result
from hillclimb.journal import Journal
from hillclimb.problem import ProblemSpec

TAIL_CHARS = 2000


def tail(path: Path, chars: int = TAIL_CHARS) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")[-chars:]


@dataclass
class CandidateEvaluator:
    """Trial/replicate execution and report reading for one search. Journal-free:
    reads `config`/`problem` per call (they may be mutated by tests), writes
    only into the candidate dir it is given."""

    executor: Executor
    problem: ProblemSpec
    config: Config
    holdout_scorer: HoldoutScorer | None = None

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

        if index is None:
            index = len(candidate.trials)
        tdir = create_trial_dir(candidate_dir, index, params_doc)
        trial = Trial(index=index, params=dict(params or {}))
        n = max(1, self.config.search.n_replicates)

        def run(j: int) -> tuple[Replicate, bool]:
            rdir = create_replicate_dir(tdir, j)
            # a single replicate keeps seed=None: existing verifiers see no
            # env change; repeated replicates get distinct seeds
            return self.execute_replicate(
                rdir / solution.name, rdir, exec_timeout, seed=j if n > 1 else None
            )

        if n == 1 or self.config.search.replicate_mode == "serial":
            results = [run(j) for j in range(n)]
        else:
            with ThreadPoolExecutor(max_workers=n, thread_name_prefix="replicate") as pool:
                results = list(pool.map(run, range(n)))
        trial.replicates.extend(r for r, _ in results)
        trial.finished_at = utcnow()
        candidate.trials.append(trial)
        candidate.stamp_best_trial(self.problem.higher_is_better)
        # r0 of the FIRST trial surfaces at the candidate root right away
        # (the engine re-hoists when a later trial becomes the best one)
        if index == 0:
            hoist_replicate(candidate_dir, replicate_dir(tdir, 0), self.problem.output_artifacts)
        return trial, all(ok for _, ok in results)

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
        """Score the hidden split with `trial`'s params (the candidate's
        immutable code, that trial's values); (score, None, cpu_s) on
        success, (None, reason, cpu_s) on contract violation, (None, None,
        None) when this search has no holdout."""
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


# --- journal-reading helpers: LOCK-HOLDER / SCHEDULER-THREAD ONLY ---


def accept_band(config: Config, journal: Journal) -> float:
    """How much better a candidate must be before the engine believes it.

    `min_improvement` is the author's own floor in metric units; `noise_k`
    multiples of the measured noise floor is the search's own evidence
    about itself. Zero (the default) is the strict comparison."""
    band = config.search.min_improvement
    if config.search.noise_k > 0:
        floor = journal.noise_floor()
        if floor is not None:
            band = max(band, config.search.noise_k * floor)
    return band


def holdout_threshold(journal: Journal, *, top_k: int, higher_is_better: bool) -> float | None:
    """Holdout hygiene: the k-th best val score at prepare time; a
    candidate must beat (or tie) it to earn a holdout evaluation.
    None = no gate (top_k disabled or fewer than k scored candidates).
    Snapshot semantics: slightly stale under parallelism, exact in
    serial — an acceptable heuristic for a hygiene gate."""
    if top_k <= 0:
        return None
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
    return EvalResult(
        candidate_id=candidate.candidate_id,
        score=candidate.val_score,
        valid=candidate.status == "ok",
        instance_scores=dict(candidate.instance_scores),
        features=dict(candidate.metrics),
        feedback=feedback,
        trials=tuple(summarize_trial(t) for t in candidate.trials),
        cost_usd=candidate.backend.cost_usd or 0.0,
    )
