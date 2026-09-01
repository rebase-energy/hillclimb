"""Experiments: which setup wins on a problem?

An experiment is a problem (or several) × named *arms* × N repeats. An arm
is a set of config overrides — anything `Config.apply_overrides` accepts
(`search.policy`, `learning.enabled`, `model`, `search.policy_params.*`,
…) — so "does memory help?", "greedy or openevolve?", "sonnet or opus?" are
all the same experiment with different arms. A spec is a YAML file:

    name: policy-vs-memory          # default: the file stem
    problems: [circle-packing]
    repeats: 3
    budget: 15m
    schedule: sequential            # sequential | parallel
    noise_floor: 0.02               # a number, or {problem-id: number}
    defaults: {model: sonnet}       # overrides every arm starts from
    arms:
      greedy:        {search.policy: greedy}
      greedy-nomem:  {search.policy: greedy, learning.enabled: false}
      openevolve:    {search.policy: openevolve, search.policy_params: {population_size: 50}}

`expand` turns it into jobs in a fair order — round-robin over arms within
each repeat, so shared state (the knowledge graph) is seen by every arm at
the same point in time — and the CLI runs each job as one `hillclimb run
--experiment … --arm …`. Every search is tagged in `SearchMeta` (experiment,
arm, repeat, arm_overrides), and the report groups on those tags through the
store, so a search started by hand with the same flags counts too.

Comparison is on the selected candidate's holdout score (val when holdout
was off), per (problem, arm): n, mean, median, spread, wins per repeat, and
the gap to the first arm (the control) judged against the noise floor.
Pure functions here; orchestration and printing live in the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

import yaml
from pydantic import BaseModel, Field, field_validator

from hillclimb.direction import better
from hillclimb.journal import Journal
from hillclimb.store import DataStore, FileDataStore

EXPERIMENTS_DIRNAME = "experiments"


class ExperimentSpec(BaseModel):
    name: str
    problems: list[str]
    arms: dict[str, dict] = Field(default_factory=dict)
    repeats: int = 1
    budget: str | None = None
    schedule: str = "sequential"  # sequential | parallel
    noise_floor: float | dict[str, float] | None = None
    defaults: dict = Field(default_factory=dict)
    # one executable seed solution shared by every search (--seed-from for
    # each child), so arms are compared from identical source, not merely the
    # same baseline score; relative paths resolve against the spec's dir
    seed_from: str | None = None
    spec_path: Path | None = None

    @field_validator("schedule")
    @classmethod
    def _schedule(cls, value: str) -> str:
        if value not in ("sequential", "parallel"):
            raise ValueError(f"schedule must be sequential or parallel, got {value!r}")
        return value

    @field_validator("arms")
    @classmethod
    def _arms(cls, value: dict[str, dict]) -> dict[str, dict]:
        if len(value) < 2:
            raise ValueError("an experiment needs at least two arms")
        for name, overrides in value.items():
            if not isinstance(overrides, dict):
                raise ValueError(f"arm {name!r} must map config settings to values")
        return value

    @field_validator("repeats")
    @classmethod
    def _repeats(cls, value: int) -> int:
        if value < 1:
            raise ValueError("repeats must be >= 1")
        return value

    @property
    def control(self) -> str:
        """The first arm — the one every other arm is compared against."""
        return next(iter(self.arms))

    def noise_for(self, problem_id: str) -> float | None:
        if isinstance(self.noise_floor, dict):
            return self.noise_floor.get(problem_id)
        return self.noise_floor

    def arm_overrides(self, arm: str) -> dict:
        """The overrides an arm applies: `defaults`, then the arm's own."""
        return flatten_overrides({**self.defaults, **self.arms[arm]})


def flatten_overrides(overrides: dict, prefix: str = "") -> dict:
    """Nested mappings → dotted keys (`{search: {policy: x}}` ==
    `{search.policy: x}`), except that a mapping under a key that already
    holds a dict value in Config (`policy_params`) is kept whole by the
    caller's convention: we flatten one level at a time and only recurse
    into mappings whose keys look like settings (no dots)."""
    flat: dict = {}
    for key, value in overrides.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and not name.endswith("policy_params") and not name.endswith("routing"):
            flat.update(flatten_overrides(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def resolve_experiment_path(target: str | Path, hillclimb_dir: Path | None) -> Path:
    """A spec path as given, or `<hillclimb>/experiments/<name>.yaml`."""
    path = Path(target)
    if path.is_file():
        return path.resolve()
    if hillclimb_dir is not None:
        for candidate in (
            hillclimb_dir / EXPERIMENTS_DIRNAME / f"{target}.yaml",
            hillclimb_dir / EXPERIMENTS_DIRNAME / f"{target}.yml",
            hillclimb_dir / EXPERIMENTS_DIRNAME / str(target),
        ):
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(
        f"No experiment spec {target!r} (a YAML path, or a name under hillclimb/{EXPERIMENTS_DIRNAME}/)"
    )


def load_experiment(path: Path) -> ExperimentSpec:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: an experiment spec is a mapping")
    problems = data.get("problems") or ([data["problem"]] if data.get("problem") else [])
    return ExperimentSpec(
        name=data.get("name") or path.stem,
        problems=list(problems),
        arms=data.get("arms") or {},
        repeats=data.get("repeats", 1),
        budget=data.get("budget"),
        schedule=data.get("schedule", "sequential"),
        noise_floor=data.get("noise_floor"),
        defaults=data.get("defaults") or {},
        seed_from=data.get("seed_from"),
        spec_path=path,
    )


def resolved_seed(spec: ExperimentSpec) -> Path | None:
    """The spec's shared seed as an absolute path (relative to the spec
    file's directory), validated to exist before any run is created."""
    if not spec.seed_from:
        return None
    path = Path(spec.seed_from).expanduser()
    if not path.is_absolute() and spec.spec_path is not None:
        path = spec.spec_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"experiment seed_from not found: {path}")
    return path


@dataclass(frozen=True)
class Job:
    index: int  # 1-based launch order
    problem: str
    arm: str
    repeat: int
    overrides: dict


def expand(spec: ExperimentSpec) -> list[Job]:
    """Problems × arms × repeats as jobs in launch order: repeat-major,
    arms round-robin inside, so every arm has seen the same shared state
    (knowledge, cache) when its k-th repeat starts."""
    jobs: list[Job] = []
    for repeat in range(1, spec.repeats + 1):
        for problem in spec.problems:
            for arm in spec.arms:
                jobs.append(Job(len(jobs) + 1, problem, arm, repeat, spec.arm_overrides(arm)))
    return jobs


@dataclass(frozen=True)
class ExperimentRow:
    experiment: str
    arm: str
    repeat: int
    problem_id: str
    problem_key: str
    run_id: str
    search_id: str
    state: str
    holdout: float | None
    val: float | None
    higher_is_better: bool
    started_at: str
    candidates: int = 0      # scored candidates
    tokens: int = 0          # agent tokens across the search
    minutes_to_best: float | None = None

    @property
    def ref(self) -> str:
        return f"{self.run_id}/{self.search_id}"

    @property
    def score(self) -> float | None:
        """Comparison score: holdout when available (the honest metric), else val."""
        return self.holdout if self.holdout is not None else self.val


def collect_results(
    store: DataStore | Path, *, experiment: str | None = None, problem_id: str = ""
) -> list[ExperimentRow]:
    """One row per finished, experiment-tagged search (every experiment
    unless one is named). A runs dir is shorthand for its FileDataStore."""
    from hillclimb.tree import accepted_lineage, minutes_since

    if isinstance(store, Path):
        store = FileDataStore(store)
    rows: list[ExperimentRow] = []
    for record in store.searches():
        meta = record.meta
        if not meta.experiment or not meta.arm:
            continue
        if experiment and meta.experiment != experiment:
            continue
        if problem_id and meta.problem_id != problem_id:
            continue
        if record.state == "running":
            continue
        status = store.read_status(record.key)
        selected = status.selected if status else None
        candidates = list(Journal(store.journal(record.key)).candidates.values())
        scored = [c for c in candidates if c.val_score is not None]
        accepted = accepted_lineage(candidates, meta.higher_is_better)
        best = next((c for c in candidates if accepted and c.candidate_id == accepted[-1]), None)
        rows.append(ExperimentRow(
            experiment=meta.experiment,
            arm=meta.arm,
            repeat=meta.repeat,
            problem_id=meta.problem_id,
            problem_key=meta.problem_key,
            run_id=meta.run_id,
            search_id=meta.search_id,
            state=record.state,
            holdout=selected.holdout_score if selected else None,
            val=selected.val_score if selected else None,
            higher_is_better=meta.higher_is_better,
            started_at=meta.started_at,
            candidates=len(scored),
            tokens=sum(c.backend.total_tokens or 0 for c in candidates if c.backend),
            minutes_to_best=minutes_since(best.finished_at, meta.started_at) if best else None,
        ))
    rows.sort(key=lambda r: (r.experiment, r.problem_key, r.repeat, r.started_at))
    return rows


@dataclass(frozen=True)
class ArmStats:
    arm: str
    rows: list[ExperimentRow]
    wins: int = 0  # repeats in which this arm had the best score

    @property
    def scores(self) -> list[float]:
        return [r.score for r in self.rows if r.score is not None]

    @property
    def mean(self) -> float | None:
        return mean(self.scores) if self.scores else None

    @property
    def median(self) -> float | None:
        return median(self.scores) if self.scores else None

    @property
    def spread(self) -> float | None:
        return max(self.scores) - min(self.scores) if len(self.scores) > 1 else None

    @property
    def tokens(self) -> float | None:
        return mean(r.tokens for r in self.rows) if self.rows else None

    @property
    def minutes_to_best(self) -> float | None:
        values = [r.minutes_to_best for r in self.rows if r.minutes_to_best is not None]
        return mean(values) if values else None


@dataclass(frozen=True)
class Comparison:
    """One arm against the control, across the repeats both ran."""

    arm: str
    control: str
    gap: float | None          # mean over paired repeats of (arm - control), in the metric's units
    wins: int                  # repeats where arm beat control
    losses: int
    ties: int
    within_noise: bool | None  # None when no noise floor is known


@dataclass(frozen=True)
class ExperimentSummary:
    experiment: str
    problem_id: str
    problem_key: str
    higher_is_better: bool
    arms: list[ArmStats]
    comparisons: list[Comparison]
    noise_floor: float | None
    unfinished: list[ExperimentRow] = field(default_factory=list)


def summarize(
    rows: list[ExperimentRow],
    *,
    control: str | None = None,
    noise_floor: dict[str, float | None] | None = None,
) -> list[ExperimentSummary]:
    """Group rows by (experiment, problem) and compare the arms. The control
    is the arm named, else the first arm to have started. `noise_floor` maps
    problem id → the spread below which a gap is not a result."""
    groups: dict[tuple[str, str], list[ExperimentRow]] = {}
    for row in rows:
        groups.setdefault((row.experiment, row.problem_key), []).append(row)
    summaries: list[ExperimentSummary] = []
    for (experiment, problem_key), group in sorted(groups.items()):
        higher = group[0].higher_is_better
        problem_id = group[0].problem_id
        finished = [r for r in group if r.state == "done" or r.score is not None]
        unfinished = [r for r in group if r not in finished]
        arm_order = list(dict.fromkeys(r.arm for r in sorted(group, key=lambda r: r.started_at)))
        if control in arm_order:
            arm_order.remove(control)
            arm_order.insert(0, control)
        by_arm = {arm: [r for r in finished if r.arm == arm] for arm in arm_order}
        # best-of-repeat wins: within each repeat, the best-scoring arm takes one
        wins = {arm: 0 for arm in arm_order}
        by_repeat: dict[int, list[ExperimentRow]] = {}
        for row in finished:
            if row.score is not None:
                by_repeat.setdefault(row.repeat, []).append(row)
        for contenders in by_repeat.values():
            top = contenders[0]
            for row in contenders[1:]:
                if better(row.score, top.score, higher):  # type: ignore[arg-type]
                    top = row
            if not any(
                r is not top and r.score == top.score for r in contenders
            ):
                wins[top.arm] += 1
        arms = [ArmStats(arm, by_arm[arm], wins[arm]) for arm in arm_order]
        floor = (noise_floor or {}).get(problem_id)
        comparisons = []
        if arms:
            ctrl = arms[0]
            ctrl_by_repeat = {r.repeat: r for r in ctrl.rows if r.score is not None}
            for arm in arms[1:]:
                w = l = t = 0
                diffs: list[float] = []
                for row in arm.rows:
                    partner = ctrl_by_repeat.get(row.repeat)
                    if partner is None or row.score is None:
                        continue
                    diffs.append(row.score - partner.score)  # type: ignore[operator]
                    if better(row.score, partner.score, higher):  # type: ignore[arg-type]
                        w += 1
                    elif better(partner.score, row.score, higher):  # type: ignore[arg-type]
                        l += 1
                    else:
                        t += 1
                # paired by repeat where possible (same shared state, same
                # point in time); unpaired means only when nothing pairs
                if diffs:
                    gap: float | None = mean(diffs)
                elif arm.mean is not None and ctrl.mean is not None:
                    gap = arm.mean - ctrl.mean
                else:
                    gap = None
                within = abs(gap) < floor if gap is not None and floor is not None else None
                comparisons.append(Comparison(arm.arm, ctrl.arm, gap, w, l, t, within))
        summaries.append(ExperimentSummary(
            experiment=experiment, problem_id=problem_id, problem_key=problem_key,
            higher_is_better=higher, arms=arms, comparisons=comparisons,
            noise_floor=floor, unfinished=unfinished,
        ))
    return summaries


def _fmt(value: float | None, digits: int = 5) -> str:
    return f"{value:.{digits}g}" if value is not None else "-"


def _fmt_tokens(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def render_report(summaries: list[ExperimentSummary]) -> str:
    """Markdown-ish text: one table per (experiment, problem) — the arms with
    their n/mean/median/spread/wins/cost — then every arm against the
    control with the gap judged against the noise floor."""
    if not summaries:
        return "no finished experiment searches found (run `hillclimb experiment run <spec>` first)"
    lines: list[str] = []
    for summary in summaries:
        direction = "higher is better" if summary.higher_is_better else "lower is better"
        lines.append(f"## {summary.experiment} · {summary.problem_key} ({direction})")
        lines.append("")
        lines.append("| arm | n | mean | median | spread | wins | min→best | tokens |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for arm in summary.arms:
            tag = " (control)" if arm is summary.arms[0] and len(summary.arms) > 1 else ""
            lines.append(
                f"| {arm.arm}{tag} | {len(arm.scores)} | {_fmt(arm.mean)} | {_fmt(arm.median)} | "
                f"{_fmt(arm.spread, 3)} | {arm.wins} | {_fmt(arm.minutes_to_best, 3)} | "
                f"{_fmt_tokens(arm.tokens)} |"
            )
        lines.append("")
        for cmp in summary.comparisons:
            if cmp.gap is None:
                verdict = "no scores to compare"
            else:
                sign = "+" if cmp.gap >= 0 else ""
                improves = (cmp.gap > 0) == summary.higher_is_better and cmp.gap != 0
                verdict = f"{sign}{cmp.gap:.4g} vs {cmp.control}"
                if cmp.within_noise:
                    verdict += f" — within noise ({summary.noise_floor:g}), not a result"
                elif cmp.within_noise is False:
                    verdict += " — " + ("better" if improves else "worse") + f" beyond noise ({summary.noise_floor:g})"
                else:
                    verdict += " — " + ("better" if improves else "worse") + " (no noise floor known)"
                verdict += f"; wins {cmp.wins}, loses {cmp.losses}, ties {cmp.ties}"
            lines.append(f"{cmp.arm}: {verdict}")
        if summary.unfinished:
            refs = ", ".join(f"{r.ref} ({r.arm}, {r.state})" for r in summary.unfinished)
            lines.append(f"not finished: {refs}")
        lines.append("")
    return "\n".join(lines).rstrip()
