from __future__ import annotations

import shutil
from pathlib import Path

from hillclimb.candidate import Candidate, Trial, utcnow
from hillclimb.problem import ProblemSpec
from hillclimb.workspace import create_candidate_workspace


def unscored_placeholder(search_dir: Path, files: dict[str, Path] | None = None) -> Candidate:
    """c000 when the problem ships no runnable baseline: keeps the tree
    rooted (the engine requires c000) without pretending to a score. Any
    `files` maps a workspace name to a valid-by-construction source (a sample
    submission copied in as `submission.csv`) so the search always has
    something to ship."""
    workspace = search_dir / "candidates" / "c000"
    workspace.mkdir(parents=True, exist_ok=True)
    for name, source in (files or {}).items():
        shutil.copy(source, workspace / name)
        (search_dir / "best").mkdir(parents=True, exist_ok=True)
        shutil.copy(source, search_dir / "best" / name)
    return Candidate(
        candidate_id="c000",
        operator="baseline",
        status="ok",
        workspace=str(workspace),
        summary="baseline: none shipped with this problem (unscored placeholder)",
        finished_at=utcnow(),
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
    workspace = create_candidate_workspace(
        search_dir, "c000", problem.data_dir, problem.problem_dir
    )
    candidate = Candidate(
        candidate_id="c000",
        operator="baseline",
        status="ok",
        workspace=str(workspace),
        summary=summary,
    )
    solution = workspace / "solution.py"
    solution.write_text(solution_text)
    (workspace / "notes.md").write_text(summary + "\n")

    exec_result = executor.execute(solution, workspace, timeout_s)
    trial = Trial(
        returncode=exec_result.returncode,
        duration_s=exec_result.duration_s,
        timed_out=exec_result.timed_out,
        submission_ok=exec_result.submission_ok,
        val_score=exec_result.val_score,
    )
    if exec_result.ok:
        if holdout_scorer is not None:
            trial.holdout_score, trial.holdout_error = holdout_scorer.score(workspace)
        trial.finished_at = utcnow()
        candidate.trials.append(trial)
        candidate.is_best = True
        shutil.copy(solution, search_dir / "best" / "solution.py")
    else:
        candidate.summary += " (baseline eval failed; unscored)"
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
    if problem.baseline_text is None or executor is None:
        return unscored_placeholder(search_dir, problem.baseline_files)
    return run_scored_baseline(
        problem, search_dir, executor, holdout_scorer, timeout_s,
        solution_text=problem.baseline_text,
        summary=problem.baseline_summary,
    )
