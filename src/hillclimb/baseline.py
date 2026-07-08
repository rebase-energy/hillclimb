from __future__ import annotations

import shutil
from pathlib import Path

from hillclimb.candidate import Candidate, Trial, utcnow
from hillclimb.problem import ProblemSpec
from hillclimb.workspace import create_candidate_workspace


def unscored_placeholder(search_dir: Path) -> Candidate:
    """c000 when the problem ships no runnable baseline: keeps the tree
    rooted (the engine requires c000) without pretending to a score."""
    workspace = search_dir / "candidates" / "c000"
    workspace.mkdir(parents=True, exist_ok=True)
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
    """t=0 safety net / scored floor, by problem kind.

    csv: copy of sample_submission.csv — valid by construction so the search
    always has *something* gradeable in best/, never selected (no trials).

    emflow: the benchmark's reference model (get_model()) evaluated for real,
    so agent drafts must beat it to become best.

    evaluator: the problem's optional `baseline:` solution evaluated for
    real; unscored placeholder when none ships."""
    if problem.kind == "emflow":
        from hillclimb.integrations.emflow.provider import write_emflow_baseline

        return write_emflow_baseline(problem, search_dir, executor, holdout_scorer, timeout_s)

    if problem.kind == "evaluator":
        if problem.baseline_solution is None or executor is None:
            return unscored_placeholder(search_dir)
        return run_scored_baseline(
            problem, search_dir, executor, holdout_scorer, timeout_s,
            solution_text=problem.baseline_solution.read_text(),
            summary=f"baseline: {problem.baseline_solution.name}",
        )

    workspace = search_dir / "candidates" / "c000"
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy(problem.sample_submission, workspace / "submission.csv")
    shutil.copy(problem.sample_submission, search_dir / "best" / "submission.csv")
    return Candidate(
        candidate_id="c000",
        operator="baseline",
        status="ok",
        workspace=str(workspace),
        summary="baseline: copy of sample_submission.csv",
        finished_at=utcnow(),
    )
