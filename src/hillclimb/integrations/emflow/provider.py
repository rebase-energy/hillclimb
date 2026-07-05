"""emflow problem provider: resolves `emflow://<name>` targets.

The only hillclimb module that imports emflow at the top level — it is
reached exclusively through the lazy scheme dispatch in hillclimb.problem,
so the core carries no emflow dependency.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import emflow as ef

from hillclimb.candidate import Candidate, Trial, utcnow
from hillclimb.config import Config
from hillclimb.problem import ProblemSpec, ResolvedTarget, SuiteSpec

PROBLEM_CACHE_DIRNAME = "emflow-problems"


def _slug(name: str) -> str:
    return name.replace(":", "-").replace("_", "-")


def _find_baseline(name: str) -> str | None:
    """Module path of the benchmark's baseline (`<pkg>.baseline.get_model`),
    or None when the problem ships no reference model."""
    folder = name.partition(":")[0].replace("-", "_")
    for base in ("emflow.benchmarks", "emflow.examples"):
        modname = f"{base}.{folder}.baseline"
        try:
            module = importlib.import_module(modname)
        except ModuleNotFoundError:
            continue
        if hasattr(module, "get_model"):
            return modname
    return None


def materialize_problem_dir(problem, name: str, cache_root: Path) -> Path:
    """Write a small on-disk problem dir (description + API crib) so the
    engine's workspace symlinks and data listing work untouched."""
    problem_dir = cache_root / PROBLEM_CACHE_DIRNAME / _slug(name)
    problem_dir.mkdir(parents=True, exist_ok=True)

    env = problem.env("validation")
    quantiles = getattr(env, "quantiles", None)
    direction = "lower is better" if problem.objective.lower_is_better else "higher is better"
    n_val = len(problem.origins("validation"))
    n_hold = len(problem.origins("holdout"))

    lines = [
        f"# {name}",
        "",
        (problem.description or "").strip(),
        "",
        "## Evaluation",
        "",
        f"- Objective: **{problem.objective.name}** ({direction})",
        f"- Validation split: {n_val} origins (your `val_score`)",
        f"- Holdout split: {n_hold} origins (scored by the orchestrator; never touch it)",
    ]
    if quantiles:
        lines.append(
            f"- Probabilistic: predictions must carry {len(quantiles)} quantile columns "
            f"({quantiles[0]:g} … {quantiles[-1]:g})"
        )
    else:
        lines.append("- Point forecasts: predictions carry a single `point` column")
    if problem.reference_scores:
        top = problem.reference_scores[0]
        lines.append(
            f"- Official leaderboard: {len(problem.reference_scores)} teams; "
            f"winner {top.team} scored {top.score:g}"
        )
    (problem_dir / "description.md").write_text("\n".join(lines) + "\n")
    return problem_dir


def load_emflow_problem(name: str, config: Config) -> ProblemSpec:
    problem = ef.load_problem(name)  # KeyError / ProblemNotIngestedError propagate
    problem.load_dataset()  # pre-warm the HF cache so agent-time evals run offline
    cache_root = config.paths.runs_dir.parent / "cache"
    problem_dir = materialize_problem_dir(problem, name, cache_root)
    quantiles = getattr(problem.env("validation"), "quantiles", None)
    return ProblemSpec(
        kind="emflow",
        problem_id=_slug(name),
        problem_dir=problem_dir.resolve(),
        data_dir=problem_dir.resolve(),
        description=(problem_dir / "description.md").read_text(),
        metric_name=problem.objective.name,
        lower_is_better=problem.objective.lower_is_better,
        sample_submission=None,
        verifier=None,
        time_budget_s=config.budget.total_s,
        holdout_mode="evaluator",
        emflow_problem=name,
        emflow_baseline=_find_baseline(name),
        emflow_quantiles=list(quantiles) if quantiles else None,
    )


def resolve_emflow_target(name: str, config: Config) -> ResolvedTarget:
    """A bare package name with variants is a virtual suite (one search per
    variant); anything else is a single problem."""
    if ":" not in name:
        variants = [p for p in ef.list_problems() if p.startswith(f"{name}:")]
        if variants:
            return ResolvedTarget(
                kind="suite",
                suite=SuiteSpec(
                    suite_id=_slug(name),
                    suite_path=Path("."),
                    problems=[f"emflow://{v}" for v in variants],
                ),
            )
    return ResolvedTarget(kind="problem", problem=load_emflow_problem(name, config))


def write_emflow_baseline(
    problem: ProblemSpec,
    search_dir: Path,
    executor,
    holdout_scorer,
    timeout_s: int,
) -> Candidate:
    """c000 = the benchmark's reference model, evaluated for real — a genuine
    scored floor that agent drafts must beat. Degrades to an unscored
    placeholder when the problem ships no baseline or the eval fails."""
    import shutil

    workspace = search_dir / "candidates" / "c000"
    workspace.mkdir(parents=True, exist_ok=True)
    candidate = Candidate(
        candidate_id="c000",
        operator="baseline",
        status="ok",
        workspace=str(workspace),
    )

    if problem.emflow_baseline is None or executor is None:
        candidate.summary = "baseline: none shipped with this problem (unscored placeholder)"
        candidate.finished_at = utcnow()
        return candidate

    solution = workspace / "solution.py"
    solution.write_text(
        f'"""Reference baseline for {problem.emflow_problem}."""\n'
        f"from {problem.emflow_baseline} import get_model  # noqa: F401\n"
    )
    (workspace / "notes.md").write_text(
        f"baseline: {problem.emflow_baseline}.get_model()\n"
    )
    candidate.summary = f"baseline: {problem.emflow_baseline}.get_model()"

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
            holdout_score, holdout_error = holdout_scorer.score(workspace)
            trial.holdout_score = holdout_score
            trial.holdout_error = holdout_error
        trial.finished_at = utcnow()
        candidate.trials.append(trial)
        candidate.is_best = True
        shutil.copy(solution, search_dir / "best" / "solution.py")
    else:
        # keep the search alive: fall back to the unscored-placeholder semantics
        candidate.summary += " (baseline eval failed; unscored)"
    candidate.finished_at = utcnow()
    return candidate
