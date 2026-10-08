"""Meta-problems: hillclimbing on the climber itself.

A meta-problem is an ordinary problem whose `solution.py` is a one-file
climber (`problem.yaml: solution_kind: climber`). Its verifier does not run
the file — it runs inner hillclimb searches WITH the file as their climber
and reports how far they climbed. A search on a meta-problem therefore runs
its own climber in the improver role (`SearchMeta.role`), and every part of
the harness that scores, journals, budgets or debugs a candidate applies
unchanged: a climber that fails to load is a buggy candidate and a debug
target, a verifier cut off at the budget wall is abandoned, `best/` ships
the best climber found.

This module is what such a verifier calls (`hillclimb grade`, from
`verifier.sh`, on `$HILLCLIMB_ENGINE_PYTHON`):

- `GradeSpec` (`grade.yaml` beside the meta-problem's verifier): the inner
  problems, the budget of each inner search, repeats, per problem an
  optional `floor`/`target` that override what the inner run measures,
  how a result is scored (`score:`) and combined (`aggregate:`), and the
  `outer_holdout` problems the grade is measured on for the holdout split.
- `evaluate()`: a nested hillclimb dir under the verifier's cwd (the outer
  config's coding agent, model, routing and problems; its own runs; learning
  off so every candidate is measured against the same world), one inner
  `hillclimb run` per problem × repeat, and the score.
- The default score is **gap closed**: per inner problem, the fraction of the
  distance from the inner search's floor (its baseline/seed) to the best
  known value (`chart_baselines`, or the spec's `target`) that the climber
  covered — direction-aware, 0 when it never beat the floor, above 1 when
  it beat the literature. Per problem the median over repeats, then the
  mean over problems; each instance rides the verifier's `instances` key.
  Both halves are the spec's to choose: `score:` is `gap-closed`, `solved`,
  `raw` or a function of your own, `aggregate:` is `mean`, `median`, `min`
  or a function of your own (`file.py:function` beside the spec, or
  `module:function`). They belong to the grade, never to the climber.
- The outer holdout: a search on a meta-problem climbs the grade, so the
  inner searches' results are its validation. With `holdout: true` on the
  meta-problem the engine runs the verifier once more on the holdout split,
  and the grade is then measured on `outer_holdout`: problems the search
  on the meta-problem never climbed on.
- `check_climber_source()`: the version-one permissions rule for an improver's
  candidate — the file may import `hillclimb.sdk`, `hillclimb.spaces` and
  the standard library, nothing else (the rule `tests/test_sdk_imports.py`
  holds the bundled climbers to).

The harness never imports this module; only the CLI does.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from hillclimb.config import Config, split_config
from hillclimb.project import MARKER_FILE, RUNS_CONFIG

DEFAULT_SPEC = "grade.yaml"
SPLITS = ("validation", "holdout")  # the verifier contract's `$HILLCLIMB_SPLIT`
NESTED_DIRNAME = "hillclimb"  # the inner searches' hillclimb dir, under the verifier's cwd
# what an improver's candidate may import from hillclimb
ALLOWED_HILLCLIMB_IMPORTS = (
    "hillclimb.sdk", "hillclimb.spaces", "hillclimb",
    # the prebuilt blocks, by name (`hillclimb.policies.Greedy`)
    "hillclimb.policies", "hillclimb.selectors", "hillclimb.operators", "hillclimb.tuners",
    "hillclimb.memory", "hillclimb.loops",
)
# the outer verifier's own contract, which an inner engine must never inherit
_OUTER_VERIFIER_KEYS = (
    "HILLCLIMB_PYTHON", "HILLCLIMB_SOLUTION", "HILLCLIMB_RESULT", "HILLCLIMB_SPLIT",
    "HILLCLIMB_REPLICATE_SEED", "HILLCLIMB_TRIAL_SEED", "HILLCLIMB_PARAMS",
    "HILLCLIMB_ENGINE_PYTHON",
    "PYTHONPATH",  # the outer runtime venv's `hillclimb.spaces` shim, which shadows the real package
)
_STARTUP_ALLOWANCE_S = 60  # per inner search, on top of its budget, for engine start and the last verifier
Log = Callable[[str], None]


class MetaError(RuntimeError):
    """The evaluation cannot proceed; the message names the fix."""


class InnerProblem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem: str
    floor: float | None = None  # default: the inner search's baseline/seed score
    target: float | None = None  # default: the best `chart_baselines` value in the metric's direction


class GradeSpec(BaseModel):
    """`grade.yaml`: what the inner searches are, and how their results
    become one grade."""

    model_config = ConfigDict(extra="forbid")

    problems: list[InnerProblem]
    # the OUTER HOLDOUT: problems the search on the meta-problem never climbs
    # on. The grade is measured on them for the holdout split only
    outer_holdout: list[InnerProblem] = Field(default_factory=list)
    budget: str = "5m"  # wall clock per inner search (2h / 30m / 90s)
    repeats: int = Field(default=1, ge=1)
    # the inner searches' stop margin; default: the outer's, capped at a
    # fifth of the budget so a short inner search still does work
    stop_margin_s: int | None = None
    # how one inner search's result is put on a scale every problem shares:
    # `gap-closed` | `solved` | `raw`, or `file.py:function` / `module:function`
    score: str = "gap-closed"
    # how the per-problem scores become the grade: `mean` | `median` | `min`,
    # or a function named the same way
    aggregate: str = "mean"
    # the folder of the file the spec was read from: where a `file.py` ref resolves
    _base_dir: Path | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _bare_names(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            for key in ("problems", "outer_holdout"):
                if isinstance(data.get(key), list):
                    data[key] = [{"problem": item} if isinstance(item, str) else item for item in data[key]]
        return data

    @field_validator("problems")
    @classmethod
    def _at_least_one(cls, value):
        if not value:
            raise ValueError("grade.yaml: `problems` names at least one inner problem")
        return value

    @property
    def budget_s(self) -> int:
        return parse_budget(self.budget)

    def instance_key(self, problem_id: str, repeat: int) -> str:
        return problem_id if self.repeats == 1 else f"{problem_id}/r{repeat}"

    def problems_for(self, split: str = "validation") -> list[InnerProblem]:
        """The inner problems one verifier run grades on: `problems` on the
        validation split, `outer_holdout` on the holdout split."""
        if split not in SPLITS:
            raise MetaError(f"unknown split {split!r} (one of {', '.join(SPLITS)})")
        if split == "validation":
            return self.problems
        if not self.outer_holdout:
            raise MetaError(
                "grade.yaml: the holdout split needs `outer_holdout:` — problems the search on "
                "this meta-problem never climbs on (or drop `holdout: true` from its problem.yaml)"
            )
        return self.outer_holdout

    def required_exec_s(self, split: str = "validation") -> int:
        """Wall clock the verifier needs for every inner search, serially."""
        return len(self.problems_for(split)) * self.repeats * (self.budget_s + _STARTUP_ALLOWANCE_S)

    def score_function(self) -> ScoreFn:
        return _resolve_function(self.score, SCORES, "score", self._base_dir)

    def aggregate_function(self) -> AggregateFn:
        return _resolve_function(self.aggregate, AGGREGATES, "aggregate", self._base_dir)


from hillclimb.harness.budget import parse_budget  # noqa: E402,F401 — the one parser (it lived here too)


def load_grade_spec(path: Path) -> GradeSpec:
    path = Path(path)
    if not path.is_file():
        raise MetaError(f"grade spec not found: {path}")
    try:
        spec = GradeSpec.model_validate(yaml.safe_load(path.read_text()) or {})
    except ValueError as exc:
        raise MetaError(f"{path}: {exc}") from exc
    spec._base_dir = path.resolve().parent
    return spec


# --- the score ---


def gap_closed(best: float | None, floor: float, target: float, higher_is_better: bool) -> float:
    """The fraction of the floor → target gap a search covered. Direction-
    aware; clamped at 0 (never worse than "did not move"); unbounded above
    (beating the best known value is real and counts). A degenerate gap
    (target at or behind the floor) scores 1 when the floor was reached."""
    if best is None:
        return 0.0
    sign = 1.0 if higher_is_better else -1.0
    gap = sign * (target - floor)
    gain = sign * (best - floor)
    if gap <= 0:
        return 1.0 if gain >= 0 else 0.0
    return max(0.0, gain / gap)


def solved(best: float | None, floor: float, target: float, higher_is_better: bool) -> float:
    """1 when the search reached the best known value, else 0: the grade is
    then the share of problems solved."""
    if best is None:
        return 0.0
    return 1.0 if (best >= target if higher_is_better else best <= target) else 0.0


def raw(best: float | None, floor: float, target: float, higher_is_better: bool) -> float:
    """The inner search's own score, as it is (the floor when nothing
    scored). Only for problems that share a scale and a direction."""
    return floor if best is None else best


# `score:` — one inner search's (best, floor, target, higher_is_better) on the shared scale
ScoreFn = Callable[[float | None, float, float, bool], float]
# `aggregate:` — the per-problem scores as one number
AggregateFn = Callable[[list[float]], float]
SCORES: dict[str, ScoreFn] = {"gap-closed": gap_closed, "solved": solved, "raw": raw}
AGGREGATES: dict[str, AggregateFn] = {"mean": statistics.fmean, "median": statistics.median, "min": min}


def _resolve_function(ref: str, builtins: Mapping[str, Callable], key: str, base_dir: Path | None) -> Callable:
    """A built-in by name, or the user's own: `file.py:function` (relative to
    the spec's folder) or `module:function`."""
    if ref in builtins:
        return builtins[ref]
    target, sep, name = ref.rpartition(":")
    if not sep or not target or not name:
        raise MetaError(
            f"grade.yaml `{key}: {ref}`: not one of {', '.join(builtins)}, "
            "nor `file.py:function` / `module:function`"
        )
    try:
        if target.endswith(".py"):
            path = Path(target).expanduser()
            if not path.is_absolute():
                path = (base_dir or Path.cwd()) / path
            module_spec = importlib.util.spec_from_file_location(f"_hillclimb_grade_{key}", path)
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"cannot load {path}")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(target)
        function = getattr(module, name)
    except (ImportError, AttributeError, OSError, SyntaxError) as exc:
        raise MetaError(f"grade.yaml `{key}: {ref}`: {exc}") from exc
    if not callable(function):
        raise MetaError(f"grade.yaml `{key}: {ref}`: {name} is not a function")
    return function


def aggregate(per_problem: dict[str, list[float]], combine: AggregateFn = statistics.fmean) -> float:
    """Median over a problem's repeats, then `combine` over problems (the
    mean unless the spec names another)."""
    scores = [statistics.median(values) for values in per_problem.values() if values]
    if not scores:
        return 0.0
    return float(combine(scores))


# --- version-one permissions: what an improver's file may import ---


def check_climber_source(path: Path) -> list[str]:
    """Why `path` may not be an improver's candidate, one line each, empty
    when clean: a syntax error, or an import outside `hillclimb.sdk`,
    `hillclimb.spaces` and the standard library. Pure — the file is parsed,
    never imported."""
    path = Path(path)
    if not path.is_file():
        return [f"{path}: not a file"]
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        return [f"{path.name}:{exc.lineno}: syntax error: {exc.msg}"]
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    problems: list[str] = []
    for node in ast.walk(tree):
        names: list[str]
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                problems.append(f"{path.name}:{node.lineno}: relative import — a one-file climber has no package")
                continue
            names = [node.module or ""]
            if node.module == "hillclimb":
                names = [f"hillclimb.{alias.name}" for alias in node.names]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            if root == "hillclimb":
                if name not in ALLOWED_HILLCLIMB_IMPORTS and not name.startswith("hillclimb.sdk."):
                    problems.append(
                        f"{path.name}:{node.lineno}: imports {name} — a climber reaches the harness through hillclimb.sdk only"
                    )
            elif root not in stdlib:
                problems.append(
                    f"{path.name}:{node.lineno}: imports {name} — only the standard library and hillclimb.sdk are available"
                )
    return problems


# --- running the inner searches ---


@dataclass
class InnerOutcome:
    problem_id: str
    repeat: int
    best: float | None
    floor: float
    target: float
    gap: float
    evaluations: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    search_ref: str = ""


@dataclass
class MetaResult:
    score: float
    instances: dict[str, float]
    metrics: dict[str, float] = field(default_factory=dict)
    outcomes: list[InnerOutcome] = field(default_factory=list)

    def to_result(self) -> dict:
        """The verifier's `$HILLCLIMB_RESULT` object: the score, the
        per-instance breakdown, and the spend as feature metrics."""
        return {"score": self.score, "instances": dict(self.instances), **self.metrics}


def nested_config(outer: Config, climber: Path, spec: GradeSpec, nested_dir: Path) -> dict:
    """The inner searches' hillclimb.yaml: the user's coding agent, model, routing,
    concurrency and problems; the candidate as the climber; its own runs and
    store; learning off (every candidate meets the same world)."""
    data = outer.model_dump(mode="json")
    data["climber"] = str(Path(climber).resolve())  # a one-file climber, by its file
    data["paths"]["runs_dir"] = str(nested_dir / "runs")
    data["paths"]["problems_dir"] = str(outer.paths.problems_dir)
    data["budget"]["total_s"] = spec.budget_s
    margin = spec.stop_margin_s
    if margin is None:
        margin = min(outer.budget.stop_margin_s, max(1, spec.budget_s // 5))
    data["budget"]["stop_margin_s"] = margin
    data["store"] = {"backend": "files"}
    data["learning"] = {**data.get("learning", {}), "enabled": False}
    # how the inner problems measure is theirs (their problem.yaml), and
    # whether they have a hidden split too; how the best is picked on one is
    # the candidate climber's
    data.pop("evaluation", None)
    data.pop("report", None)
    data.pop("holdout", None)
    return data


def _inner_env(nested_dir: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _OUTER_VERIFIER_KEYS}
    env["HILLCLIMB_DIR"] = str(nested_dir)
    return env


def _resolve_target(problem: InnerProblem, spec_problem, higher_is_better: bool) -> float:
    if problem.target is not None:
        return problem.target
    values = list(spec_problem.chart_baselines.values())
    if not values:
        raise MetaError(
            f"{problem.problem}: no best known value to measure against — the problem has no "
            "chart_baselines; set `target:` for it in grade.yaml"
        )
    return max(values) if higher_is_better else min(values)


def score_floor(problem, config: Config, workdir: Path, log: Log = print) -> float | None:
    """The inner problem's floor, measured the way `hillclimb verify` does:
    its declared `baseline:` score, else one verifier run of its baseline
    solution or baseline files. A search does not score a files-only floor
    itself (its c000 is an unscored placeholder), so the meta verifier
    measures it once per problem, before any inner search spends."""
    import shutil

    from hillclimb.api import build_executor
    from hillclimb.harness.dirs import create_candidate_dir

    if problem.baseline_score is not None:
        return problem.baseline_score
    source = problem.baseline_text
    if source is None and problem.baseline_files:
        source = "# the problem's declared floor: its baseline_files, scored as they are\n"
    if source is None:
        return None
    candidate_dir = create_candidate_dir(workdir / "floor", problem.problem_id, problem.data_dir, problem.problem_dir)
    script = candidate_dir / "solution.py"
    script.write_text(source)
    for dest, src in problem.baseline_files.items():
        shutil.copy2(src, candidate_dir / dest)
    result = build_executor(config, problem, log=log).execute(script, candidate_dir, config.budget.exec_timeout_s)
    return result.val_score if result.ok else None


def inner_command(
    python: str, problem_id: str, spec: GradeSpec, climber: Path, run_name: str, params: Mapping | None = None,
) -> list[str]:
    """The inner `hillclimb run`: the candidate as the climber, and the
    trial's parameter values (an improver's params.json, tuned by the outer
    harness) laid over the climber's params — the same `--set` a user types."""
    # `--no-detach`: the outer verifier waits for the inner search to finish
    # and reads its journal
    cmd = [
        python, "-m", "hillclimb.cli", "run", problem_id, "--no-detach",
        "--budget", f"{spec.budget_s}s", "--climber", str(climber), "--name", run_name,
    ]
    for name, value in (params or {}).items():
        cmd += ["--set", f"climber.params.{name}={json.dumps(value)}"]
    return cmd


def _read_inner(nested_dir: Path, problem_id: str) -> tuple[float | None, dict, str]:
    """(best val score, spend, search ref) of the newest search in the nested
    dir on `problem_id`."""
    from hillclimb.harness.budget import journal_spend
    from hillclimb.harness.journal import Journal
    from hillclimb.harness.store import open_store
    from hillclimb.problem import load_problem

    config = Config.load(path=nested_dir / MARKER_FILE)
    problem = load_problem(problem_id, config)
    with closing(open_store(config)) as store:
        records = store.searches(problem_key=problem.problem_key)
        if not records:
            return None, {}, ""
        record = max(records, key=lambda r: (r.meta.started_at, r.ref))
        journal = Journal(store.journal(record.key))
    best = journal.best_candidate(problem.higher_is_better)
    spend = journal_spend(journal)
    return (
        best.val_score if best is not None else None,
        {"evaluations": spend.evaluations, "tokens": spend.tokens, "cost_usd": spend.cost_usd},
        record.ref,
    )


def evaluate(
    spec: GradeSpec,
    climber: Path,
    outer: Config,
    workdir: Path,
    log: Log = print,
    engine_python: str | None = None,
    params: Mapping | None = None,
    split: str = "validation",
) -> MetaResult:
    """Run every inner search and score the climber. `params` are the
    trial's parameter values when the candidate declared a params.json
    (`$HILLCLIMB_PARAMS`); they reach the inner runs as `climber.params`.
    `split` picks the problems: the spec's `problems`, or its `outer_holdout`
    on the holdout split.
    Raises MetaError when an inner search fails (the outer candidate is then
    buggy, with the inner log's tail as the reason) or the spec cannot be
    measured."""
    from hillclimb.problem import load_problem

    climber = Path(climber).resolve()
    if not climber.is_file():
        raise MetaError(f"climber file not found: {climber}")
    workdir = Path(workdir).resolve()
    problems = spec.problems_for(split)
    score_of, combine = spec.score_function(), spec.aggregate_function()
    needed = spec.required_exec_s(split)
    if outer.budget.exec_timeout_s < needed:
        raise MetaError(
            f"grade.yaml needs {needed}s of inner search ({len(problems)} problem(s) × "
            f"{spec.repeats} repeat(s) × {spec.budget_s}s + start-up) but budget.exec_timeout_s "
            f"is {outer.budget.exec_timeout_s} — raise it in hillclimb.yaml or shorten the spec"
        )
    nested_dir = workdir / NESTED_DIRNAME
    nested_dir.mkdir(parents=True, exist_ok=True)
    general, runs = split_config(nested_config(outer, climber, spec, nested_dir))
    (nested_dir / MARKER_FILE).write_text(
        "# written by `hillclimb grade`: the inner searches' hillclimb dir\n"
        + yaml.safe_dump(general, sort_keys=False)
    )
    (nested_dir / "runs").mkdir(exist_ok=True)
    (nested_dir / "runs" / RUNS_CONFIG).write_text(
        "# written by `hillclimb grade`: the inner searches' run defaults\n"
        + yaml.safe_dump(runs, sort_keys=False)
    )
    inner_config = Config.load(path=nested_dir / MARKER_FILE)
    # measure before spending: every inner problem must load, have a target
    # and a floor (the spec's, else scored here once)
    targets: dict[str, tuple[float, float, bool]] = {}
    for entry in problems:
        try:
            spec_problem = load_problem(entry.problem, inner_config)
        except Exception as exc:
            raise MetaError(f"inner problem {entry.problem}: {exc}") from exc
        target = _resolve_target(entry, spec_problem, spec_problem.higher_is_better)
        floor = entry.floor if entry.floor is not None else score_floor(spec_problem, inner_config, workdir, log)
        if floor is None:
            raise MetaError(
                f"{entry.problem}: no floor to measure from — the problem ships no scorable "
                "baseline; set `floor:` for it in grade.yaml"
            )
        log(f"{entry.problem}: floor={floor} target={target}")
        targets[entry.problem] = (floor, target, spec_problem.higher_is_better)
    python = engine_python or sys.executable
    outcomes: list[InnerOutcome] = []
    per_problem: dict[str, list[float]] = {}
    for entry in problems:
        problem_id = entry.problem
        floor, target, higher_is_better = targets[problem_id]
        for repeat in range(spec.repeats):
            run_name = f"{Path(problem_id).name}-r{repeat}"
            log_path = workdir / f"inner-{run_name}.log"
            cmd = inner_command(python, problem_id, spec, climber, run_name, params)
            log(f"inner search {run_name}: {' '.join(cmd[3:])}")
            with open(log_path, "w") as out:
                proc = subprocess.run(
                    cmd, cwd=workdir, env=_inner_env(nested_dir), stdout=out, stderr=subprocess.STDOUT,
                    check=False,
                )
            if proc.returncode != 0:
                tail = log_path.read_text()[-2000:]
                raise MetaError(
                    f"inner search {run_name} exited {proc.returncode} (log: {log_path}):\n{tail}"
                )
            best, spend, ref = _read_inner(nested_dir, problem_id)
            gap = float(score_of(best, floor, target, higher_is_better))
            outcomes.append(InnerOutcome(
                problem_id=problem_id, repeat=repeat, best=best, floor=floor, target=target, gap=gap,
                evaluations=spend.get("evaluations", 0), tokens=spend.get("tokens", 0),
                cost_usd=spend.get("cost_usd", 0.0), search_ref=ref,
            ))
            per_problem.setdefault(problem_id, []).append(gap)
            log(f"inner search {run_name}: best={best} floor={floor} target={target} {spec.score}={gap:.4f}")
    instances = {spec.instance_key(o.problem_id, o.repeat): o.gap for o in outcomes}
    metrics = {
        "inner_evaluations": float(sum(o.evaluations for o in outcomes)),
        "inner_tokens": float(sum(o.tokens for o in outcomes)),
        "inner_cost_usd": float(sum(o.cost_usd for o in outcomes)),
    }
    return MetaResult(score=aggregate(per_problem, combine), instances=instances, metrics=metrics, outcomes=outcomes)
