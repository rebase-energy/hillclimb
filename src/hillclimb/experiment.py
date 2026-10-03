"""Studies: which setup wins on a problem?

A study is a problem (or several) × named *experiments* × N repeats. An
experiment is a set of config overrides — anything `Config.apply_overrides`
accepts (`climber`, `learning.enabled`, `model`, `climber.params.*`, …) —
so "does memory help?", "greedy or openevolve?", "sonnet or opus?" are
all the same study with different experiments. A spec is a YAML file
(`hillclimb experiment run` takes it; `arms:` is the old name of
`experiments:` and still loads):

    name: policy-vs-memory          # default: the file stem
    problems: [circle-packing]
    repeats: 3
    budget: 15m
    schedule: sequential            # sequential | parallel
    max_concurrent: 8               # parallel only: searches alive at once
    noise_floor: 0.02               # a number, or {problem-id: number}
    defaults: {model: sonnet}       # overrides every experiment starts from
    experiments:
      greedy:        {climber: greedy}
      greedy-nomem:  {climber: greedy, learning.enabled: false}
      openevolve:    {climber: openevolve, climber.params: {population_size: 50}}
      mine:                         # a whole `climber:` block, as a run spec takes it
        climber: {operator_policy: mine.py, tuner: optuna}

An experiment's `climber` names or defines its climber — a preset, one .py
file, or the block — and replaces the block whole; `climber.<field>` edits
the block it ends up with, whichever key came first.

`expand` turns it into jobs in a fair order — round-robin over experiments
within each repeat, so shared state (the knowledge graph) is seen by every
experiment at the same point in time — and the CLI runs each job as one
`hillclimb run --study … --experiment …`. Every search is tagged in
`SearchMeta` (study, experiment, repeat, experiment_overrides), and the
report groups on those tags through the store, so a search started by hand
with the same flags counts too.

Comparison is on the selected candidate's holdout score (val when holdout
was off), per (problem, experiment): n, mean, median, spread, wins per repeat, and
the gap to the first experiment (the control) judged against the noise floor.
Pure functions here; orchestration and printing live in the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

import yaml
from pydantic import BaseModel, Field, field_validator

from hillclimb.harness.direction import better
from hillclimb.harness.journal import Journal
from hillclimb.harness.store import DataStore, FileDataStore

EXPERIMENTS_DIRNAME = "experiments"


class StudySpec(BaseModel):
    name: str
    problems: list[str]
    experiments: dict[str, dict] = Field(default_factory=dict)
    repeats: int = 1
    budget: str | None = None
    schedule: str = "sequential"  # sequential | parallel
    # parallel only: at most this many searches alive at once — the launcher
    # waits on its children before starting the next. Without it every job
    # starts at once and, past the machine's coding agent slots, burns its wall
    # clock in `waiting-slot`. None = unbounded (the old behaviour)
    max_concurrent: int | None = None
    noise_floor: float | dict[str, float] | None = None
    defaults: dict = Field(default_factory=dict)
    # one executable seed solution shared by every search (--seed-from for
    # each child), so experiments are compared from identical source, not merely
    # the same baseline score; relative paths resolve against the spec's dir
    seed_from: str | None = None
    spec_path: Path | None = None

    @field_validator("schedule")
    @classmethod
    def _schedule(cls, value: str) -> str:
        if value not in ("sequential", "parallel"):
            raise ValueError(f"schedule must be sequential or parallel, got {value!r}")
        return value

    @field_validator("experiments")
    @classmethod
    def _experiments(cls, value: dict[str, dict]) -> dict[str, dict]:
        if len(value) < 2:
            raise ValueError("a study needs at least two experiments")
        for name, overrides in value.items():
            if not isinstance(overrides, dict):
                raise ValueError(f"experiment {name!r} must map config settings to values")
        return value

    @field_validator("max_concurrent")
    @classmethod
    def _max_concurrent(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("max_concurrent must be >= 1")
        return value

    @field_validator("repeats")
    @classmethod
    def _repeats(cls, value: int) -> int:
        if value < 1:
            raise ValueError("repeats must be >= 1")
        return value

    @property
    def control(self) -> str:
        """The first experiment — the one every other experiment is compared against."""
        return next(iter(self.experiments))

    def noise_for(self, problem_id: str) -> float | None:
        if isinstance(self.noise_floor, dict):
            return self.noise_floor.get(problem_id)
        return self.noise_floor

    def experiment_overrides(self, experiment: str) -> dict:
        """The overrides an experiment applies: `defaults`, then the
        experiment's own — with whatever names the climber first, since
        naming one replaces the block the `climber.<field>` overrides edit."""
        flat = flatten_overrides({**self.defaults, **self.experiments[experiment]})
        return {
            **{key: value for key, value in flat.items() if key in NAMES_THE_CLIMBER},
            **{key: value for key, value in flat.items() if key not in NAMES_THE_CLIMBER},
        }


# override keys that name (and so replace) the whole climber; the last two
# are the 0.5 and 0.3 spellings, which still load
NAMES_THE_CLIMBER = ("climber", "climber.ref", "search.policy")


def flatten_overrides(overrides: dict, prefix: str = "") -> dict:
    """Nested mappings → dotted keys (`{climber: {params: {k: 1}}}` would
    be `{climber.params.k: 1}`), except the mappings that are ONE value: a
    `climber:` block (it defines the climber; its fields are not overrides
    to merge), `routing`, and the 0.3 `policy_params`."""
    flat: dict = {}
    for key, value in overrides.items():
        name = f"{prefix}{key}"
        whole = name == "climber" or name.endswith("policy_params") or name.endswith("routing")
        if isinstance(value, dict) and not whole:
            flat.update(flatten_overrides(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def resolve_study_path(target: str | Path, hillclimb_dir: Path | None) -> Path:
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
        f"No study spec {target!r} (a YAML path, or a name under {EXPERIMENTS_DIRNAME}/)"
    )


def load_study(path: Path) -> StudySpec:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a study spec is a mapping")
    problems = data.get("problems") or ([data["problem"]] if data.get("problem") else [])
    return StudySpec(
        name=data.get("name") or path.stem,
        problems=list(problems),
        experiments=data.get("experiments") or data.get("arms") or {},
        repeats=data.get("repeats", 1),
        budget=data.get("budget"),
        schedule=data.get("schedule", "sequential"),
        max_concurrent=data.get("max_concurrent"),
        noise_floor=data.get("noise_floor"),
        defaults=data.get("defaults") or {},
        seed_from=data.get("seed_from"),
        spec_path=path,
    )


def resolved_seed(spec: StudySpec) -> Path | None:
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
    experiment: str
    repeat: int
    overrides: dict


def expand(spec: StudySpec, first_repeat: int = 1) -> list[Job]:
    """Problems × experiments × repeats as jobs in launch order: repeat-major,
    experiments round-robin inside, so every experiment has seen the same shared state
    (knowledge, cache) when its k-th repeat starts. `first_repeat` numbers
    the repeats from K — how a finished run gains repeats K.. later."""
    if first_repeat < 1:
        raise ValueError("first_repeat must be >= 1")
    jobs: list[Job] = []
    for repeat in range(first_repeat, first_repeat + spec.repeats):
        for problem in spec.problems:
            for experiment in spec.experiments:
                jobs.append(Job(len(jobs) + 1, problem, experiment, repeat, spec.experiment_overrides(experiment)))
    return jobs


@dataclass(frozen=True)
class StudyRow:
    study: str
    experiment: str
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
    tokens: int = 0          # coding agent tokens across the search
    minutes_to_best: float | None = None

    @property
    def ref(self) -> str:
        return f"{self.run_id}/{self.search_id}"

    @property
    def score(self) -> float | None:
        """Comparison score: holdout when available (the honest metric), else val."""
        return self.holdout if self.holdout is not None else self.val


def collect_results(
    store: DataStore | Path, *, study: str | None = None, problem_id: str = ""
) -> list[StudyRow]:
    """One row per finished, study-tagged search (every study unless one
    is named). A runs dir is shorthand for its FileDataStore."""
    from hillclimb.tui.tree import accepted_lineage, minutes_since

    if isinstance(store, Path):
        store = FileDataStore(store)
    rows: list[StudyRow] = []
    for record in store.searches():
        meta = record.meta
        if not meta.study or not meta.experiment:
            continue
        if study and meta.study != study:
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
        rows.append(StudyRow(
            study=meta.study,
            experiment=meta.experiment,
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
            tokens=sum(c.agent.total_tokens or 0 for c in candidates if c.agent),
            minutes_to_best=minutes_since(best.finished_at, meta.started_at) if best else None,
        ))
    rows.sort(key=lambda r: (r.study, r.problem_key, r.repeat, r.started_at))
    return rows


@dataclass(frozen=True)
class ExperimentStats:
    experiment: str
    rows: list[StudyRow]
    wins: int = 0  # repeats in which this experiment had the best score

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
    """One experiment against the control, across the repeats both ran."""

    experiment: str
    control: str
    gap: float | None          # mean over paired repeats of (experiment - control), in the metric's units
    wins: int                  # repeats where experiment beat control
    losses: int
    ties: int
    within_noise: bool | None  # None when no noise floor is known


@dataclass(frozen=True)
class StudySummary:
    study: str
    problem_id: str
    problem_key: str
    higher_is_better: bool
    experiments: list[ExperimentStats]
    comparisons: list[Comparison]
    noise_floor: float | None
    unfinished: list[StudyRow] = field(default_factory=list)


def summarize(
    rows: list[StudyRow],
    *,
    control: str | None = None,
    noise_floor: dict[str, float | None] | None = None,
) -> list[StudySummary]:
    """Group rows by (study, problem) and compare the experiments. The
    control is the experiment named, else the first experiment to have
    started. `noise_floor` maps problem id → the spread below which a gap is not a result."""
    groups: dict[tuple[str, str], list[StudyRow]] = {}
    for row in rows:
        groups.setdefault((row.study, row.problem_key), []).append(row)
    summaries: list[StudySummary] = []
    for (study, problem_key), group in sorted(groups.items()):
        higher = group[0].higher_is_better
        problem_id = group[0].problem_id
        finished = [r for r in group if r.state == "done" or r.score is not None]
        unfinished = [r for r in group if r not in finished]
        order = list(dict.fromkeys(r.experiment for r in sorted(group, key=lambda r: r.started_at)))
        if control in order:
            order.remove(control)
            order.insert(0, control)
        by_experiment = {name: [r for r in finished if r.experiment == name] for name in order}
        # best-of-repeat wins: within each repeat, the best-scoring experiment takes one
        wins = {name: 0 for name in order}
        by_repeat: dict[int, list[StudyRow]] = {}
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
                wins[top.experiment] += 1
        experiments = [ExperimentStats(name, by_experiment[name], wins[name]) for name in order]
        floor = (noise_floor or {}).get(problem_id)
        comparisons = []
        if experiments:
            ctrl = experiments[0]
            ctrl_by_repeat = {r.repeat: r for r in ctrl.rows if r.score is not None}
            for stats in experiments[1:]:
                w = l = t = 0
                diffs: list[float] = []
                for row in stats.rows:
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
                elif stats.mean is not None and ctrl.mean is not None:
                    gap = stats.mean - ctrl.mean
                else:
                    gap = None
                within = abs(gap) < floor if gap is not None and floor is not None else None
                comparisons.append(Comparison(stats.experiment, ctrl.experiment, gap, w, l, t, within))
        summaries.append(StudySummary(
            study=study, problem_id=problem_id, problem_key=problem_key,
            higher_is_better=higher, experiments=experiments, comparisons=comparisons,
            noise_floor=floor, unfinished=unfinished,
        ))
    return summaries


def summary_to_dict(summary: StudySummary) -> dict:
    """The summary as plain JSON-able data (`experiment report --json`):
    per experiment the scores and aggregates, per comparison the paired gap and
    its verdict — enough for a meta-verifier to read a score off without
    parsing the table. `verdict` is one of `better`, `worse`, `tie`,
    `within-noise`, `unknown` (no scores or no noise floor)."""
    experiments = []
    for stats in summary.experiments:
        experiments.append({
            "experiment": stats.experiment,
            "control": stats is summary.experiments[0],
            "n": len(stats.scores),
            "scores": list(stats.scores),
            "mean": stats.mean,
            "median": stats.median,
            "spread": stats.spread,
            "wins": stats.wins,
            "minutes_to_best": stats.minutes_to_best,
            "tokens": stats.tokens,
            "searches": [r.ref for r in stats.rows],
        })
    comparisons = []
    for cmp in summary.comparisons:
        if cmp.gap is None:
            verdict = "unknown"
        elif cmp.within_noise:
            verdict = "within-noise"
        elif cmp.gap == 0:
            verdict = "tie"
        elif cmp.within_noise is None:
            verdict = "unknown"
        else:
            verdict = "better" if (cmp.gap > 0) == summary.higher_is_better else "worse"
        comparisons.append({
            "experiment": cmp.experiment,
            "control": cmp.control,
            "gap": cmp.gap,
            "wins": cmp.wins,
            "losses": cmp.losses,
            "ties": cmp.ties,
            "within_noise": cmp.within_noise,
            "verdict": verdict,
        })
    return {
        "study": summary.study,
        "problem_id": summary.problem_id,
        "problem_key": summary.problem_key,
        "higher_is_better": summary.higher_is_better,
        "noise_floor": summary.noise_floor,
        "experiments": experiments,
        "comparisons": comparisons,
        "unfinished": [{"search": r.ref, "experiment": r.experiment, "state": r.state} for r in summary.unfinished],
    }


def summaries_to_dict(summaries: list[StudySummary]) -> list[dict]:
    return [summary_to_dict(s) for s in summaries]


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


def render_report(summaries: list[StudySummary]) -> str:
    """Markdown-ish text: one table per (study, problem) — the experiments
    with their n/mean/median/spread/wins/cost — then every experiment against
    the control with the gap judged against the noise floor."""
    if not summaries:
        return "no finished study searches found (run `hillclimb experiment run <spec>` first)"
    lines: list[str] = []
    for summary in summaries:
        direction = "higher is better" if summary.higher_is_better else "lower is better"
        lines.append(f"## {summary.study} · {summary.problem_key} ({direction})")
        lines.append("")
        lines.append("| experiment | n | mean | median | spread | wins | min→best | tokens |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for stats in summary.experiments:
            tag = " (control)" if stats is summary.experiments[0] and len(summary.experiments) > 1 else ""
            lines.append(
                f"| {stats.experiment}{tag} | {len(stats.scores)} | {_fmt(stats.mean)} | {_fmt(stats.median)} | "
                f"{_fmt(stats.spread, 3)} | {stats.wins} | {_fmt(stats.minutes_to_best, 3)} | "
                f"{_fmt_tokens(stats.tokens)} |"
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
                elif cmp.gap == 0:
                    verdict += " — tie"  # the same score is not "worse" (the --json verdict says so too)
                elif cmp.within_noise is False:
                    verdict += " — " + ("better" if improves else "worse") + f" beyond noise ({summary.noise_floor:g})"
                else:
                    verdict += " — " + ("better" if improves else "worse") + " (no noise floor known)"
                verdict += f"; wins {cmp.wins}, loses {cmp.losses}, ties {cmp.ties}"
            lines.append(f"{cmp.experiment}: {verdict}")
        if summary.unfinished:
            refs = ", ".join(f"{r.ref} ({r.experiment}, {r.state})" for r in summary.unfinished)
            lines.append(f"not finished: {refs}")
        lines.append("")
    return "\n".join(lines).rstrip()
