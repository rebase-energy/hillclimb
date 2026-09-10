from __future__ import annotations

import shutil
from pathlib import Path

from hillclimb.candidate import Candidate, Replicate, Trial, utcnow
from hillclimb.problem import ProblemSpec
from hillclimb.dirs import create_candidate_dir, create_replicate_dir, create_trial_dir, hoist_replicate


def unscored_placeholder(search_dir: Path, files: dict[str, Path] | None = None) -> Candidate:
    """c000 when the problem ships no runnable baseline: keeps the tree
    rooted (the engine requires c000) without pretending to a score. Any
    `files` maps a file name (inside the candidate dir) to a valid-by-construction source (a sample
    submission copied in as `submission.csv`) so the search always has
    something to ship."""
    candidate_dir = search_dir / "candidates" / "c000"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for name, source in (files or {}).items():
        shutil.copy(source, candidate_dir / name)
        (search_dir / "best").mkdir(parents=True, exist_ok=True)
        shutil.copy(source, search_dir / "best" / name)
    return Candidate(
        candidate_id="c000",
        operator="baseline",
        status="ok",
        candidate_dir=str(candidate_dir),
        summary="baseline: none shipped with this problem (unscored placeholder)",
        finished_at=utcnow(),
    )


def declared_floor(search_dir: Path, score: float, summary: str) -> Candidate:
    """c000 for a problem that declares its floor as a number (`baseline: 0.5`):
    a scored candidate with no code, so drafts must beat it to become best but
    nothing is ever improved or ensembled from it."""
    candidate_dir = search_dir / "candidates" / "c000"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "notes.md").write_text(summary + "\n")
    now = utcnow()
    return Candidate(
        candidate_id="c000",
        operator="baseline",
        status="ok",
        candidate_dir=str(candidate_dir),
        summary=summary,
        trials=[Trial(
            is_best=True,
            finished_at=now,
            replicates=[Replicate(returncode=0, duration_s=0.0, submission_ok=True, val_score=score, finished_at=now)],
        )],
        is_best=True,
        finished_at=now,
    )


def run_scored_baseline(
    problem: ProblemSpec,
    search_dir: Path,
    executor,
    holdout_scorer,
    timeout_s: int,
    solution_text: str,
    summary: str,
) -> Candidate:
    """c000 evaluated for real — a genuine scored floor agent drafts must
    beat. Shared by the emflow and evaluator kinds; degrades to the
    unscored-placeholder semantics when the eval fails (keeps the search
    alive)."""
    candidate_dir = create_candidate_dir(
        search_dir, "c000", problem.data_dir, problem.problem_dir
    )
    candidate = Candidate(
        candidate_id="c000",
        operator="baseline",
        status="ok",
        candidate_dir=str(candidate_dir),
        summary=summary,
    )
    solution = candidate_dir / "solution.py"
    solution.write_text(solution_text)
    (candidate_dir / "notes.md").write_text(summary + "\n")

    rdir = create_replicate_dir(create_trial_dir(candidate_dir, 0), 0)
    exec_result = executor.execute(rdir / "solution.py", rdir, timeout_s)
    replicate = Replicate(
        returncode=exec_result.returncode,
        duration_s=exec_result.duration_s,
        cpu_s=exec_result.cpu_s,
        timed_out=exec_result.timed_out,
        submission_ok=exec_result.submission_ok,
        val_score=exec_result.val_score,
        finished_at=utcnow(),
    )
    hoist_replicate(candidate_dir, rdir, problem.output_artifacts)
    (search_dir / "best").mkdir(parents=True, exist_ok=True)
    if exec_result.ok:
        trial = Trial(is_best=True, replicates=[replicate])
        if holdout_scorer is not None:
            trial.holdout_score, trial.holdout_error, trial.holdout_cpu_s = (
                holdout_scorer.score(candidate_dir, trial)
            )
        trial.finished_at = utcnow()
        candidate.trials.append(trial)
        candidate.is_best = True
        shutil.copy(solution, search_dir / "best" / "solution.py")
        # the artifacts the baseline produced (e.g. submission.csv) ship
        # alongside it, so best/ is complete from t=0
        for name in dict.fromkeys([*problem.output_artifacts, *problem.baseline_files]):
            if (candidate_dir / name).exists():
                shutil.copy(candidate_dir / name, search_dir / "best" / name)
    else:
        candidate.summary += " (baseline eval failed; unscored)"
        for name, source in problem.baseline_files.items():
            shutil.copy(source, search_dir / "best" / name)
    candidate.finished_at = utcnow()
    return candidate


def write_baseline(
    problem: ProblemSpec,
    search_dir: Path,
    executor=None,
    holdout_scorer=None,
    timeout_s: int = 1800,
) -> Candidate:
    """t=0 scored floor: the problem's baseline solution evaluated for real,
    so agent drafts must beat something honest to become best. Problems that
    ship no baseline get the unscored placeholder."""
    if problem.baseline_score is not None:
        return declared_floor(search_dir, problem.baseline_score, problem.baseline_summary)
    if problem.baseline_text is None or executor is None:
        return unscored_placeholder(search_dir, problem.baseline_files)
    return run_scored_baseline(
        problem, search_dir, executor, holdout_scorer, timeout_s,
        solution_text=problem.baseline_text,
        summary=problem.baseline_summary,
    )
