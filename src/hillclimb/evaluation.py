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

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from hillclimb.candidate import Candidate, Trial, utcnow
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
    """Trial execution and report reading for one search. Journal-free:
    reads `config`/`problem` per call (they may be mutated by tests), writes
    only into the candidate dir it is given."""

    executor: Executor
    problem: ProblemSpec
    config: Config
    holdout_scorer: HoldoutScorer | None = None

    def run_trials(
        self, candidate: Candidate, solution: Path, candidate_dir: Path, exec_timeout: int
    ) -> bool:
        """Run n_trials validation evaluations (each in its own trial dir with
        a distinct seed) and append the Trials in index order. Returns True
        only if every trial passed — a seed-flaky candidate is buggy.

        `search.trial_mode` decides whether they share the machine: parallel
        for seed variance, serial when the metric is a measurement of the
        machine itself (time, memory, throughput) and concurrent trials would
        measure each other."""
        n = max(1, self.config.search.n_trials)
        if n == 1:
            trial, ok = self.execute_one_trial(solution, candidate_dir, exec_timeout, seed=None)
            candidate.trials.append(trial)
            return ok

        from concurrent.futures import ThreadPoolExecutor

        from hillclimb.dirs import create_trial_dir

        def run(index: int) -> tuple[Trial, bool]:
            trial_dir = create_trial_dir(candidate_dir, index)
            return self.execute_one_trial(
                trial_dir / solution.name, trial_dir, exec_timeout, seed=index
            )

        if self.config.search.trial_mode == "serial":
            results = [run(index) for index in range(n)]
        else:
            with ThreadPoolExecutor(max_workers=n, thread_name_prefix="trial") as pool:
                results = list(pool.map(run, range(n)))
        candidate.trials.extend(trial for trial, _ in results)
        # trial-0 artifacts surface at the candidate-dir root so best/-sync,
        # ensemble copies, and holdout scoring stay untouched
        t0 = candidate_dir / "trials" / "t0"
        for name in ("submission.csv", "eval_result.json"):
            if (t0 / name).exists():
                shutil.copy(t0 / name, candidate_dir / name)
        return all(ok for _, ok in results)

    def execute_one_trial(
        self, solution: Path, cwd: Path, exec_timeout: int, seed: int | None
    ) -> tuple[Trial, bool]:
        trial_started = utcnow()
        exec_result = self.executor.execute(solution, cwd, exec_timeout, seed=seed)
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
            cpu_s=exec_result.cpu_s,
            timed_out=exec_result.timed_out,
            stdout_tail=stdout_tail,
            submission_ok=exec_result.submission_ok,
            val_score=exec_result.val_score if exec_result.ok else None,
            report=self.read_trial_report(cwd) if exec_result.ok else None,
            metrics=exec_result.metrics if exec_result.ok else {},
            instance_scores=exec_result.instance_scores if exec_result.ok else {},
            started_at=trial_started,
            finished_at=utcnow(),
        )
        return trial, exec_result.ok

    def read_trial_report(self, cwd: Path) -> dict | None:
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

    def score_holdout(self, candidate_dir: Path) -> tuple[float | None, str | None, float | None]:
        """Score the hidden split; (score, None, cpu_s) on success,
        (None, reason, cpu_s) on contract violation, (None, None, None) when
        this search has no holdout."""
        if self.holdout_scorer is None:
            return None, None, None
        return self.holdout_scorer.score(candidate_dir)


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
class TrialSummary:
    """One trial, flattened for engine feedback."""

    seed: int | None
    val_score: float | None
    returncode: int | None
    timed_out: bool
    duration_s: float | None
    submission_ok: bool
    stdout_tail: str


def summarize_trial(trial: Trial) -> TrialSummary:
    return TrialSummary(
        seed=trial.seed,
        val_score=trial.val_score,
        returncode=trial.returncode,
        timed_out=trial.timed_out,
        duration_s=trial.duration_s,
        submission_ok=trial.submission_ok,
        stdout_tail=trial.stdout_tail,
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
