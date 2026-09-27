"""`hillclimb experiment run|report`: arms × repeats, and the paired report."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import typer

from hillclimb.api import (
    child_launch_context,
    create_run,
    new_run_id,
    spawn_search_proc,
    spec_entry,
    write_run_spec,
)
from hillclimb.cli import common
from hillclimb.cli._app import HillclimbGroup, app
from hillclimb.config import Config
from hillclimb.harness.run import RunMeta, load_run_meta
from hillclimb.harness.store import open_store
from hillclimb.problem import load_problem, resolve_target

experiment_app = typer.Typer(
    cls=HillclimbGroup, help="Compare setups: named arms of config overrides × repeats on a problem"
)


app.add_typer(experiment_app, name="experiment")


@experiment_app.command("run")
def experiment_run(
    spec: str = typer.Argument(..., help="Spec YAML path, or a name under hillclimb/experiments/"),
    budget: str = typer.Option(None, "--budget", help="Per-search budget, e.g. 10m (overrides the spec)"),
    repeats: int = typer.Option(None, "--repeats", help="Override the spec's repeat count"),
    parallel: bool = typer.Option(
        None, "--parallel/--sequential",
        help="Launch every search detached at once, or one after another (default: the spec's schedule)",
    ),
    max_concurrent: int = typer.Option(
        None, "--max-concurrent", min=1,
        help="Parallel, but at most N searches alive at once: launch detached in job order, "
        "wait on the children before starting the next (overrides the spec's max_concurrent)",
    ),
    run_id: str = typer.Option(
        None, "--run-id",
        help="Append the searches to this existing experiment run instead of creating a new one",
    ),
    first_repeat: int = typer.Option(
        1, "--first-repeat", min=1,
        help="Number the repeats from K (with --run-id: add repeats K.. to a finished run)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the jobs, start nothing"),
):
    """Run an experiment: every arm × every problem × N repeats.

    Sequential (the default) runs jobs in a fair order — repeat by repeat,
    arms round-robin inside — so shared state such as cross-search memory
    is seen by every arm at the same point; use it whenever an arm touches
    shared state. Parallel launches all jobs detached at once (machine
    slots still cap concurrency) — fine for stateless comparisons such as
    policy or model. Bounded parallel (--max-concurrent N, or the spec's
    max_concurrent) keeps at most N alive so no search spends its wall
    clock waiting for a machine slot; the launcher stays up until the last
    child exits, then prints the report — run it under nohup or in tmux.
    Unlike sequential it runs the whole matrix rather than stopping at the
    first failure; its exit code is 1 if any child failed, else 2 if any
    parked, else 0. Real agent runs — the repeat count is your cost dial.
    """
    from hillclimb.experiment import (
        expand,
        load_experiment,
        resolve_experiment_path,
        resolved_seed,
    )

    config = common.load_config()
    try:
        spec_path = resolve_experiment_path(spec, config.hillclimb_dir)
        experiment = load_experiment(spec_path)
        seed_path = resolved_seed(experiment)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if repeats is not None:
        experiment = experiment.model_copy(update={"repeats": repeats})
    for problem_target in experiment.problems:
        if resolve_target(problem_target, config).kind == "suite":
            raise typer.BadParameter(f"experiments take problems, not suites ({problem_target!r})")
    jobs = expand(experiment, first_repeat)
    if parallel is False and max_concurrent is not None:
        raise typer.BadParameter("--sequential and --max-concurrent contradict each other")
    schedule = "parallel" if parallel else "sequential" if parallel is False else experiment.schedule
    limit = max_concurrent if max_concurrent is not None else experiment.max_concurrent
    if max_concurrent is not None:
        schedule = "parallel"  # the flag implies detached launches
    if schedule != "parallel":
        limit = None
    child_budget = budget or experiment.budget
    repeats_text = (
        f"{experiment.repeats} repeat(s)" if first_repeat == 1
        else f"repeats {first_repeat}..{first_repeat + experiment.repeats - 1}"
    )
    common.say(
        f"[head]Experiment {common._m(experiment.name)}:[/] {len(experiment.arms)} arms × "
        f"{len(experiment.problems)} problem(s) × {repeats_text} = {len(jobs)} searches, {schedule}"
        + (f" [note](at most {limit} at once)[/]" if limit else "")
    )
    if seed_path is not None:
        import hashlib

        digest = hashlib.sha256(seed_path.read_bytes()).hexdigest()[:12]
        common.say(f"  shared seed: [path]{common._m(seed_path)}[/] [note](sha256 {digest})[/]")
    for job in jobs:
        settings = ", ".join(f"{k}={v}" for k, v in job.overrides.items()) or "(defaults)"
        common.say(
            f"  {job.index:2d}. [path]{common._m(job.problem)}[/] · {common._m(job.arm)} · r{job.repeat}"
            f"  [note]{common._m(settings)}[/]"
        )
    if dry_run:
        return
    if run_id:
        run_dir = config.paths.runs_dir / run_id
        existing = load_run_meta(run_dir)
        if existing is None or existing.kind != "experiment":
            raise typer.BadParameter(f"--run-id {run_id!r} is not an existing experiment run")
        run_name = existing.name
        common.say(f"[head]Appending to run[/] [path]{common._m(run_id)}[/]")
    else:
        run_name = experiment.name
        run_id = new_run_id(run_name)
        run_dir = create_run(
            config,
            RunMeta(
                run_id=run_id, name=run_name, kind="experiment", target=spec,
                spec=common._spec_provenance(config, spec_path),
                problem_ids=list(dict.fromkeys(load_problem(p, config).problem_id for p in experiment.problems)),
            ),
        )
        # the run's own recipe: one entry per search, named by arm and
        # repeat, its overrides as `set` pairs — reruns the same searches
        # as a plain suite (the experiment tagging is run.yaml's)
        write_run_spec(run_dir, [
            spec_entry(
                job.problem, name=f"{job.arm}-r{job.repeat}", budget=child_budget, seed_from=seed_path,
                set=[f"{key}={_set_value(value)}" for key, value in job.overrides.items()],
            )
            for job in jobs
        ], source=common._spec_provenance(config, spec_path))
    launched = []
    alive: dict[int, tuple[subprocess.Popen, str]] = {}  # bounded parallel: pid -> (child, slug)
    exits: dict[str, int] = {}  # bounded parallel: slug -> non-zero exit code
    for job in jobs:
        argv = [
            job.problem, "--run-id", run_id, "--run-name", run_name,
            "--experiment", experiment.name, "--arm", job.arm, "--repeat", str(job.repeat),
        ]
        if seed_path is not None:
            argv += ["--seed-from", str(seed_path)]
        if child_budget:
            argv += ["--budget", child_budget]
        for key, value in job.overrides.items():
            argv += ["--set", f"{key}={_set_value(value)}"]
        slug = f"{Path(job.problem).name}-{job.arm}-r{job.repeat}"
        if schedule == "parallel":
            if limit:
                _reap_until_below(alive, limit, exits, _say_reaped)
            proc, log_path = spawn_search_proc(config, run_dir, job.index, slug, argv)
            launched.append((slug, proc.pid, log_path))
            if limit:
                alive[proc.pid] = (proc, slug)
                common.say(
                    f"[head]=== {job.index}/{len(jobs)}: {common._m(slug)} started[/] "
                    f"[note](pid {proc.pid}, log {common._m(log_path)})[/]"
                )
            continue
        common.say(f"[head]=== {job.index}/{len(jobs)}: {common._m(slug)} ===[/]")
        cwd, env = child_launch_context(config)
        result = subprocess.run([sys.executable, "-m", "hillclimb.cli", "run", *argv], cwd=cwd, env=env)
        if result.returncode != 0:
            hint = " (parked — resume it, then `experiment report`)" if result.returncode == 2 else ""
            common.fail(
                f"{common._m(slug)} exited {result.returncode}{common._m(hint)}; stopping the experiment"
            )
            raise typer.Exit(result.returncode)
    if schedule == "parallel" and not limit:
        common.say(
            f"[head]Run {common._m(run_id)}:[/] launched {len(launched)} searches "
            "[note](`hillclimb experiment report` when done)[/]"
        )
        for slug, pid, log_path in launched:
            common.say(f"  pid={pid} [path]{common._m(slug)}[/]  log=[path]{common._m(log_path)}[/]")
        return
    if limit:
        _reap_until_below(alive, 1, exits, _say_reaped)  # drain: every child has exited
        common.say(
            f"[head]Run {common._m(run_id)}:[/] {len(launched)} searches finished, "
            f"{len(exits)} with a non-zero exit"
        )
        for slug, code in exits.items():
            hint = "parked — resume it" if code == 2 else f"exit {code}"
            common.say(
                f"  [path]{common._m(slug)}[/]: {common._m(hint)}; log under [path]{common._m(run_dir / 'logs')}[/]"
            )
    common.say()
    _experiment_report_impl(config, experiment.name, "", spec_path=spec_path)
    if exits:
        raise typer.Exit(1 if any(code != 2 for code in exits.values()) else 2)


# How often the bounded experiment launcher polls its children for exits.
_REAP_POLL_S = 5.0


def _say_reaped(slug: str, code: int) -> None:
    """One reaped child, in the CLI's voice: the exit code coloured by verdict."""
    verdict = "ok" if code == 0 else "bad"
    common.say(f"    finished: [path]{common._m(slug)}[/] [{verdict}](exit {code})[/]")


def _reap_until_below(
    alive: dict[int, tuple[subprocess.Popen, str]], limit: int, exits: dict[str, int], log
) -> None:
    """Block until fewer than `limit` of the detached children in `alive`
    (pid -> (child, slug)) are still running, reaping each one that exits and
    recording non-zero exit codes in `exits`; `log(slug, code)` announces
    each one. `limit=1` drains them all.
    Polls the Popen objects themselves: a dropped Popen gets reaped behind
    our back by the next subprocess call, and its exit code with it."""
    while len(alive) >= limit:
        for pid, (proc, slug) in list(alive.items()):
            code = proc.poll()
            if code is None:
                continue
            alive.pop(pid)
            log(slug, code)
            if code != 0:
                exits[slug] = code
        if len(alive) >= limit:
            time.sleep(_REAP_POLL_S)


def _set_value(value) -> str:
    """An override value as `--set` text: scalars verbatim, mappings/lists as
    JSON (which `--set` parses back as YAML)."""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@experiment_app.command("report")
def experiment_report(
    experiment: str = typer.Argument(None, help="Experiment name or spec (default: every experiment)"),
    problem: str = typer.Option("", "--problem", help="Filter to one problem id"),
    control: str = typer.Option(None, "--control", help="Arm to compare against (default: the first)"),
    noise_floor: float = typer.Option(
        None, "--noise-floor", help="Gap below which arms are not different (default: the spec's)"
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Machine-readable: the summaries as JSON (a meta-verifier reads the gaps)"
    ),
):
    """Compare the arms of an experiment.

    On the selected candidate's holdout score (val when holdout was off):
    per arm n/mean/median/spread, best-of-repeat wins, time to best and
    tokens; then every arm against the control, with the gap judged against
    the noise floor (`hillclimb verify <problem> --repeat 5` measures it).
    """
    from hillclimb.experiment import resolve_experiment_path

    config = common.load_config()
    spec_path = None
    if experiment:
        try:
            spec_path = resolve_experiment_path(experiment, config.hillclimb_dir)
        except FileNotFoundError:
            spec_path = None  # a name with no spec on disk: report by tag alone
    _experiment_report_impl(
        config, experiment, problem, spec_path=spec_path, control=control,
        noise_floor=noise_floor, as_json=as_json,
    )


def _experiment_report_impl(
    config: Config,
    experiment: str | None,
    problem_id: str,
    *,
    spec_path: Path | None = None,
    control: str | None = None,
    noise_floor: float | None = None,
    as_json: bool = False,
) -> None:
    from hillclimb.experiment import (
        collect_results,
        load_experiment,
        render_report,
        summaries_to_dict,
        summarize,
    )

    floors: dict[str, float | None] = {}
    name = experiment
    if spec_path is not None:
        spec = load_experiment(spec_path)
        name = spec.name
        control = control or spec.control
        for target in spec.problems:
            try:
                pid = load_problem(target, config).problem_id
            except Exception:  # noqa: BLE001 — a problem that no longer loads still has rows
                pid = Path(target).name
            floors[pid] = spec.noise_for(pid)
    with closing(open_store(config)) as store:
        rows = collect_results(store, experiment=name, problem_id=problem_id)
    if noise_floor is not None:
        floors = {row.problem_id: noise_floor for row in rows}
    summaries = summarize(rows, control=control, noise_floor=floors)
    if as_json:
        typer.echo(json.dumps(summaries_to_dict(summaries), indent=2))
    else:
        typer.echo(render_report(summaries))
