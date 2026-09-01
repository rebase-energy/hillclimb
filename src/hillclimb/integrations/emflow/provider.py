"""emflow problem provider: resolves `emflow://<name>` targets.

The only hillclimb module that imports emflow at the top level — it is
reached exclusively through the lazy scheme dispatch in hillclimb.problem,
so the core carries no emflow dependency.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import emflow as ef

from hillclimb.candidate import Candidate
from hillclimb.config import Config
from hillclimb.problem import ProblemSpec, ResolvedTarget, SuiteSpec

PROBLEM_CACHE_DIRNAME = "emflow-problems"


def _slug(name: str) -> str:
    return name.replace(":", "-").replace("_", "-")


def reference_baselines(problem) -> dict[str, float]:
    """The problem's published leaderboard as chart reference lines, best
    first — drawn as horizontal baselines by `hillclimb chart`."""
    refs = sorted(problem.reference_scores or [], key=lambda r: r.rank)
    return {f"#{ref.rank} {ref.team}": float(ref.score) for ref in refs}


def chart_baselines_for(name: str) -> dict[str, float]:
    """Reference lines for a registry name without materializing any data —
    cheap enough for chart-time resolution of pre-snapshot searches."""
    return reference_baselines(ef.load_problem(name))


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
    engine's candidate_dir symlinks and data listing work untouched."""
    problem_dir = cache_root / PROBLEM_CACHE_DIRNAME / _slug(name)
    problem_dir.mkdir(parents=True, exist_ok=True)

    env = problem.env("validation")
    quantiles = getattr(env, "quantiles", None)
    direction = "lower is better" if problem.objective.lower_is_better else "higher is better"  # legacy-key: emflow's own field
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
    # Materialize public data to the local build cache so agent-time evals run
    # offline and credential-free. Private holdout data deliberately stays on
    # HF: the holdout scorer fetches it live with the orchestrator's token.
    if hasattr(ef, "cache_problem_data"):
        ef.cache_problem_data(name, include_private=False)
    problem.load_dataset()  # validate loadability up front
    from hillclimb.project import machine_cache_dir

    cache_root = machine_cache_dir()  # materialize appends emflow-problems/
    problem_dir = materialize_problem_dir(problem, name, cache_root)
    quantiles = getattr(problem.env("validation"), "quantiles", None)
    baseline = _find_baseline(name)
    eval_runner = str(Path(__file__).parent / "eval_runner.py")
    verifier_cmd = [
        "{python}", eval_runner, "{solution}",
        "--problem", name,
        "--split", "validation",
        "--result-json", "{result}",
    ]
    holdout_cmd = [
        "{python}", eval_runner, "{solution}",
        "--problem", name,
        "--split", "holdout",
        "--result-json", "{result}",
    ]
    return ProblemSpec(
        problem_id=_slug(name),
        problem_dir=problem_dir.resolve(),
        data_dir=problem_dir.resolve(),
        description=(problem_dir / "description.md").read_text(),
        metric_name=problem.objective.name,
        higher_is_better=not problem.objective.lower_is_better,  # legacy-key: emflow's own field
        time_budget_s=config.budget.total_s,
        chart_baselines=reference_baselines(problem),
        verifier_cmd=verifier_cmd,
        holdout_cmd=holdout_cmd,
        # cache pre-warmed at resolve time; offline keeps agent-side evals
        # hermetic (and no ambient HF credentials exist either way)
        verifier_env={"HF_HUB_OFFLINE": "1"},
        verifier_display=f"the emflow evaluator, on the validation split of {name}",
        holdout_needs_credentials=True,  # the private holdout may be gated
        runtime="emflow",
        contract_template="contract_emflow",
        baseline_text=(
            f'"""Reference baseline for {name}."""\n'
            f"from {baseline} import get_model  # noqa: F401\n"
        ) if baseline else None,
        baseline_summary=f"baseline: {baseline}.get_model()" if baseline else "baseline",
        emflow_problem=name,
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
