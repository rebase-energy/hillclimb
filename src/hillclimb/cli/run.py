"""`hillclimb run`, `resume`, `smoke`: starting searches and fleets."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from hillclimb.api import (
    FleetEngine,
    build_evaluator,
    child_launch_context,
    climber_argv,
    create_problem_run,
    create_run,
    create_search,
    execute_search,
    mixed_fleet,
    new_run_id,
    resume_spent_seconds,
    run_fleet,
    spawn_search_proc,
    spec_entry,
    write_run_spec,
)
from hillclimb.agents import get_agent
from hillclimb.cli import common
from hillclimb.cli._app import app
from hillclimb.cli.common import _m, fail, next_steps, say, warn
from hillclimb.config import Config, RouteConfig
from hillclimb.terms import ENGINE
from hillclimb.harness.budget import BudgetManager
from hillclimb.harness.core import Harness
from hillclimb.harness.journal import Journal
from hillclimb.harness.oscompat import new_group_kwargs, runnable
from hillclimb.harness.run import RunMeta, search_ref
from hillclimb.harness.store import SearchRecord, key_for, open_store
from hillclimb.modules.policies.base import Action
from hillclimb.problem import (
    ProblemSpec,
    load_problem,
    resolve_target,
    suite_problem_targets,
)

# The child-engine launcher lives in api.py (hosted runs use it too); these
# names stay for the call sites in this module.
_child_launch_context = child_launch_context


_spawn_search_proc = spawn_search_proc


_create_problem_run = create_problem_run


def _spawn_search(config: Config, run_dir: Path, index: int, slug: str, run_argv: list[str]) -> tuple[int, Path]:
    """`api.spawn_search_proc` for launchers that only keep the pid."""
    proc, log_path = _spawn_search_proc(config, run_dir, index, slug, run_argv)
    return proc.pid, log_path


def _read_knowledge_context(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.read_text() or None
    except OSError as exc:
        raise typer.BadParameter(f"--knowledge-context-file: {exc}") from exc


def _execute(
    config: Config,
    problem: ProblemSpec,
    search_dir: Path,
    budget: BudgetManager,
    seed_from: Path | None = None,
    knowledge_context: str | None = None,
) -> None:
    """CLI shell over api.execute_search: messages + exit codes."""
    outcome = execute_search(
        config, problem, search_dir, budget, log=common.engine_log, seed_from=seed_from,
        knowledge_context=knowledge_context,
    )
    ref = outcome.ref
    if outcome.state == "parked":
        say(f"\n[warn]Parked:[/] {_m(outcome.error)}")
        say(f"Resume later with: [cmd]hillclimb resume {_m(ref)}[/]")
        raise typer.Exit(2)
    if outcome.state == "stopped":
        say("\n[head]Stopped.[/]")
        say(f"Resume with: [cmd]hillclimb resume {_m(ref)}[/]")
        raise typer.Exit(2)
    store = open_store(config)
    try:
        common.warn_if_none_passed(Journal(store.journal(key_for(search_dir))), ref, config.agent)
    finally:
        store.close()
    selected = outcome.selected
    best = search_dir / "best" / "solution.py"
    if not best.is_file():
        # a declared numeric baseline (`baseline: 0`) is a score with no
        # solution behind it: when it stays the best, there is nothing to ship
        floor = f" ({problem.baseline_score:g})" if problem.baseline_score is not None else ""
        say(
            f"\n[head]Done.[/] [warn]No candidate beat the baseline{_m(floor)}, so best/ holds no solution.[/] "
            f"[note]({_m(problem.metric_name)}, {'higher' if problem.higher_is_better else 'lower'} is better)[/]"
        )
        say(f"See what was tried: [cmd]hillclimb status {_m(ref)}[/]")
        return
    if selected is not None:
        scores = f"val_score={selected.val_score}"
        if selected.holdout_score is not None:
            scores += f", holdout={selected.holdout_score:.5g}"
        say(
            f"\n[head]Done.[/] Selected candidate [head]{_m(selected.candidate_id)}[/]: {_m(scores)} "
            f"[note]({_m(problem.metric_name)}, {'higher' if problem.higher_is_better else 'lower'} is better)[/]"
        )
    else:
        say("\n[head]Done.[/] No scored solution; best/ holds the t=0 baseline.")
    # the solution is always the artifact; a submission file only exists
    # where the problem's verifier asks for one
    artifact = "solution.py"
    say(f"Best artifact: [path]{_m(search_dir / 'best' / artifact)}[/]")
    say(f"Inspect with:  [cmd]hillclimb status {_m(ref)}[/]")


def _run_problem(
    target: str,
    config: Config,
    budget: str | None,
    run_id: str | None = None,
    run_name: str | None = None,
    seed_from: Path | None = None,
    study: str | None = None,
    experiment: str | None = None,
    repeat: int = 0,
    experiment_overrides: dict | None = None,
    knowledge_context: str | None = None,
) -> None:
    if experiment_overrides:
        try:
            config.apply_overrides(experiment_overrides)
        except (KeyError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    problem = load_problem(target, config)
    total_s = common.resolve_run_budget(budget, config)
    if run_id is None:
        run_name = run_name or problem.problem_id
        run_dir = _create_problem_run(config, run_name, target, problem.problem_id)
        run_id = run_dir.name
        # the run's own recipe, next to its record (spec.yaml)
        write_run_spec(run_dir, [spec_entry(
            target, budget=budget or total_s, agent=config.agent, model=config.model,
            climber=config.climber_block(), parallel_agents=config.concurrency.parallel_agents,
            n_replicates=config.evaluation.n_replicates, seed_from=seed_from,
            set=[f"{key}={value}" for key, value in (experiment_overrides or {}).items()],
        )])
    else:
        # suite child: the parent already wrote run.yaml and spec.yaml
        run_name = run_name or run_id
        run_dir = config.paths.runs_dir / run_id
    search_dir = create_search(
        config, problem, run_dir, run_id, total_s, seed_from=seed_from,
        study=study, experiment=experiment, repeat=repeat, experiment_overrides=experiment_overrides,
    )
    tag = f", study={study}/{experiment}" + (f" r{repeat}" if repeat else "") if study else ""
    say(
        f"[head]Search {_m(search_ref(search_dir))}[/] [note](run={_m(run_name)}, problem={_m(problem.problem_id)}, "
        f"agent={_m(config.agent)}, model={_m(config.model)}, budget={total_s}s{_m(tag)})[/]"
    )
    _execute(
        config, problem, search_dir,
        BudgetManager(total_s, config.budget.stop_margin_s),
        seed_from=seed_from,
        knowledge_context=knowledge_context,
    )


def _spec_climber(config: Config, named) -> dict:
    """What a run's spec records as a search's climber: the full block. A
    spec entry's is one already (anchored at the spec); a name given on the
    command line resolves from the hillclimb dir; none is the folder's."""
    from hillclimb.climber import ClimberLoadError, as_spec

    if named is None:
        return config.climber_block()
    if isinstance(named, dict):
        return named
    try:
        return as_spec(named, config.hillclimb_dir).anchored(config.hillclimb_dir).block()
    except ClimberLoadError as exc:
        raise typer.BadParameter(str(exc), param_hint="--climber") from exc


def _run_suite(
    target: str,
    config: Config,
    budget: str | None,
    agent: str | None,
    model: str | None,
    holdout: bool,
    name: str | None,
    climber: str | None = None,
    parallel_agents: int | None = None,
    n_replicates: int | None = None,
    seed_from: Path | None = None,
    learning: bool = True,
    set_: list[str] | None = None,
) -> None:
    resolved = resolve_target(target, config)
    if resolved.kind != "suite" or resolved.suite is None:
        raise typer.BadParameter(f"{target!r} is not a suite")
    suite = resolved.suite
    run_name = name or suite.suite_id
    run_id = new_run_id(run_name)
    # one budget for the entries that name none, settled (or asked for, once)
    # before the run is created
    default_budget = None
    if not budget and any(not entry.budget for entry in suite.problems):
        default_budget = f"{common.resolve_run_budget(None, config)}s"
    problem_targets = suite_problem_targets(suite, config)
    # a problem may appear more than once (two models on one problem, say):
    # each entry is its own search, and search ids get a -2/-3 suffix
    problem_ids = list(dict.fromkeys(
        load_problem(problem_target, config).problem_id for problem_target in problem_targets
    ))
    run_dir = create_run(
        config,
        RunMeta(
            run_id=run_id,
            name=run_name,
            kind="suite",
            target=target,
            spec=common._spec_provenance(config, suite.suite_path),
            problem_ids=problem_ids,
        ),
    )
    # Snapshot every suite before launching the first child search. A child
    # only reuses these run-owned bundles; it never observes later live edits.
    from hillclimb.harness.unit_tests import freeze_for_run

    for problem_target in problem_targets:
        freeze_for_run(load_problem(problem_target, config), run_dir)
    launched = []
    entries: list[dict] = []
    for index, (entry, problem_target) in enumerate(zip(suite.problems, problem_targets), 1):
        slug = Path(problem_target).name or f"problem-{index}"
        cmd = [problem_target, "--run-id", run_id, "--run-name", run_name]
        # CLI flags override the spec entry's committed values
        child_budget = budget or entry.budget or default_budget
        child_agent = agent or entry.agent
        child_model = model or entry.model
        child_climber = climber or entry.climber
        child_parallel = parallel_agents if parallel_agents is not None else entry.parallel_agents
        child_replicates = n_replicates if n_replicates is not None else entry.n_replicates
        child_seed = seed_from or entry.seed_from
        child_set = [*entry.set, *(set_ or [])]  # the CLI's pairs apply last, so they win
        seed_path: Path | None = None
        if child_budget:
            cmd += ["--budget", child_budget]
        if child_agent:
            cmd += ["--agent", child_agent]
        if child_model:
            cmd += ["--model", child_model]
        # a name goes as `--climber`, a block as the first `--set` (the
        # pairs after it may edit its fields)
        cmd += climber_argv(child_climber)
        if child_parallel is not None:
            cmd += ["--parallel-agents", str(child_parallel)]
        if child_replicates is not None:
            cmd += ["--n-replicates", str(child_replicates)]
        if child_seed:
            # spec-relative paths resolve against the spec's own directory
            seed_path = Path(child_seed)
            if not seed_path.is_absolute():
                seed_path = (suite.suite_path.parent / seed_path).resolve()
            cmd += ["--seed-from", str(seed_path)]
        if not holdout:
            cmd.append("--no-holdout")
        if not learning:
            cmd.append("--no-learning")
        for pair in child_set:
            cmd += ["--set", pair]
        entries.append(spec_entry(
            problem_target, name=entry.name, budget=child_budget, agent=child_agent,
            model=child_model, climber=_spec_climber(config, child_climber), parallel_agents=child_parallel,
            n_replicates=child_replicates, seed_from=seed_path, set=child_set,
        ))
        pid, log_path = _spawn_search(config, run_dir, index, slug, cmd)
        launched.append((problem_target, pid, log_path))
    # the run's own recipe: the entries as resolved, next to run.yaml
    write_run_spec(run_dir, entries, source=common._spec_provenance(config, suite.suite_path))
    say(f"[head]Run {_m(run_id)}[/]: launched {len(launched)} searches")
    for problem_target, pid, log_path in launched:
        say(f"  pid={pid} [path]{_m(problem_target)}[/]  log=[path]{_m(log_path)}[/]")


def _warn_oversubscribed(config, overrides: dict, searches: int) -> None:
    """Say so when the run can keep more cores busy than the machine has:
    every search evaluates up to `parallel_agents` candidates at once, each
    with its replicates side by side, each run on `solution_cpus` cores. A
    solution that searches until a deadline then finds less the busier the
    machine is, and its score measures the contention."""
    import os

    cpus = int(overrides.get("concurrency.solution_cpus", config.concurrency.solution_cpus))
    agents = int(overrides.get("concurrency.parallel_agents", config.concurrency.parallel_agents))
    replicates = int(overrides.get("evaluation.n_replicates", config.evaluation.n_replicates))
    at_once = int(overrides.get("concurrency.parallel_replicates", config.concurrency.parallel_replicates))
    replicates = max(1, replicates) if at_once == 0 else min(max(1, replicates), at_once)
    runs = max(1, searches) * max(1, agents) * replicates
    cores = os.cpu_count() or 1
    if runs * cpus > cores:
        warn(
            f"up to {runs} solution runs at once ({searches} searches x {agents} agents x "
            f"{replicates} replicates at once) at {cpus} CPU core{'s' if cpus != 1 else ''} each want "
            f"{runs * cpus} cores; this machine has {cores}. Scores of solutions that search "
            "until a deadline will depend on the load: lower --solution-cpus or the parallelism"
        )


@app.command()
def run(
    target: str,
    budget: str = typer.Option(None, help="Wall-clock budget, e.g. 2h / 30m"),
    agent: str = typer.Option(
        None, "--agent", "--backend",
        help="The coding agent that runs the operators: claude-code | codex | pi | dummy | toy (the last two need no LLM; --backend is the old spelling)",
    ),
    model: str = typer.Option(None, help="Model the coding agent runs, e.g. sonnet / opus"),
    climber: list[str] = typer.Option(
        None, "--climber",
        help=(
            "The climber: a .py file, a climber folder, or package.module:Class (hillclimb "
            "climber list shows the catalog's). It "
            "replaces the `climber:` block of hillclimb.yaml (or of the run spec); --set "
            "climber.params.k=v edits it. Repeat it "
            "(--climber climbers/greedy --climber climbers/gepa) for a mixed fleet: one search per climber "
            "on the problem, under one run, each tagged as an experiment"
        ),
    ),
    holdout: bool = typer.Option(True, "--holdout/--no-holdout", help="Hidden selection holdout"),
    learning: bool = typer.Option(
        True, "--learning/--no-learning",
        help="Cross-search memory (cards/claims injection + distillation); off = memory-blind experiment",
    ),
    name: str = typer.Option(None, "--name", help="Run name shown in the TUI"),
    parallel_agents: int = typer.Option(
        None, "--parallel-agents", "--parallel-operators",
        help="Concurrent coding agents per search, one candidate each (--parallel-operators is the old spelling)",
    ),
    parallel_searches: int = typer.Option(
        1, "--parallel-searches", min=1,
        help="Independent searches on the problem at once, each in the background",
    ),
    solution_cpus: int = typer.Option(
        None, "--solution-cpus", min=1,
        help=(
            "CPU cores each run of a solution may use, given to it as $HILLCLIMB_CPUS "
            "(default 1; --set concurrency.solution_cpus=N is the same)"
        ),
    ),
    detach: bool = typer.Option(
        True, "--detach/--no-detach",
        help="Run in the background (the default; `hillclimb watch` follows it) or in this terminal (Ctrl-C stops it)",
    ),
    n_replicates: int = typer.Option(
        None, "--n-replicates", "--n-trials",
        help="Seeded runs per trial (the median is the trial's score; --n-trials is the old spelling)",
    ),
    parallel_replicates: int = typer.Option(
        None, "--parallel-replicates", min=0,
        help=(
            "How many of a trial's replicates run at once: 0 = all (default), 1 = one after another, "
            "required when the metric measures the machine (time, throughput, memory)"
        ),
    ),
    seed_from: Path = typer.Option(
        None, "--seed-from", help="Incumbent solution.py scored as the floor candidate"
    ),
    set_: list[str] = typer.Option(
        None, "--set", help="Any config setting, dotted: --set climber.params.num_drafts=5 --set learning.enabled=false",
    ),
    experiment_set: list[str] = typer.Option(
        None, "--experiment-set", "--arm-set",
        help=(
            "A setting for one experiment of a mixed fleet, EXPERIMENT:KEY=VALUE: "
            "--experiment-set gepa:concurrency.parallel_agents=1 (applied after --set; "
            "--arm-set is the old spelling)"
        ),
    ),
    study: str = typer.Option(
        None, "--study",
        help="Tag the search as one experiment of a study (with --experiment); names a mixed fleet's study",
    ),
    experiment: str = typer.Option(None, "--experiment", help="The experiment name (with --study)"),
    repeat: int = typer.Option(0, "--repeat", hidden=True),
    run_id: str = typer.Option(None, "--run-id", hidden=True),
    run_name: str = typer.Option(None, "--run-name", hidden=True),
    knowledge_context_file: Path = typer.Option(
        None, "--knowledge-context-file", hidden=True,
        help="Pre-built knowledge context (markdown) injected into every operator prompt",
    ),
):
    """Start a run on a problem or a run-spec YAML.

    The search runs in the background and the terminal comes straight
    back: `hillclimb watch` follows it, `hillclimb stop --all` ends it.
    `--no-detach` keeps it in this terminal instead (Ctrl-C stops it).
    """
    config = common.load_config(agent=agent, model=model)
    if not holdout:
        config.holdout.enabled = False
    if not learning:
        config.learning.enabled = False
    climbers = list(climber or [])
    mixed = len(climbers) > 1
    experiment_overrides = common._parse_experiment_set(experiment_set or [])
    if experiment_overrides and not mixed:
        raise typer.BadParameter("--experiment-set needs a mixed fleet (two or more --climber)")
    single_climber = None if mixed else (climbers[0] if climbers else None)
    if single_climber is not None:
        from hillclimb.climber import ClimberLoadError, as_spec

        try:
            config.climber = as_spec(single_climber, config.hillclimb_dir)  # naming a climber replaces the folder's block
        except ClimberLoadError as exc:
            raise typer.BadParameter(str(exc), param_hint="--climber") from exc
    if parallel_agents is not None:
        config.concurrency.parallel_agents = parallel_agents
    if n_replicates is not None:
        config.evaluation.n_replicates = n_replicates
    if parallel_replicates is not None:
        # a --set, so it reaches every engine the run starts and its spec.yaml
        set_ = [*(set_ or []), f"concurrency.parallel_replicates={parallel_replicates}"]
    if solution_cpus is not None:
        # a --set, so it reaches every engine the run starts and its spec.yaml
        set_ = [*(set_ or []), f"concurrency.solution_cpus={solution_cpus}"]
    overrides = common._parse_set(set_ or [])
    _warn_oversubscribed(config, overrides, parallel_searches)
    common.require_sandbox(config, overrides)
    common.ensure_agents_ready(config, agent, model)
    if mixed:
        if experiment or run_id:
            raise typer.BadParameter("a mixed fleet names its experiments itself; --experiment/--run-id do not apply")
    elif (study is None) != (experiment is None):
        raise typer.BadParameter("--study and --experiment go together")
    resolved = resolve_target(target, config)
    if resolved.kind != "suite" and not mixed and config.climber is None:
        # the engine ships no climber: say so here, in the terminal, not in a detached child's log
        from hillclimb.modules.refs import NO_CLIMBER_HINT

        fail(f"error: {_m(NO_CLIMBER_HINT)}")
        raise typer.Exit(1)
    if resolved.kind != "suite" and run_id is None:
        # the budget is settled here too, before anything is written: the
        # flag, else the folder's run defaults, else a yes to the default.
        # Every child is then handed it explicitly. (A suite asks once, for
        # the entries that name none; a suite's child has its --budget.)
        budget = budget or f"{common.resolve_run_budget(overrides.get('budget.total_s'), config)}s"
    if resolved.kind == "suite":
        if mixed:
            raise typer.BadParameter("a spec takes one --climber; mixed fleets run on a single problem")
        _run_suite(
            target, config, budget, agent, model, holdout, name,
            climber=single_climber, parallel_agents=parallel_agents, n_replicates=n_replicates,
            seed_from=seed_from, learning=learning, set_=set_,
        )
        return
    if mixed:
        try:
            engines = mixed_fleet(climbers, repeats=parallel_searches, experiment_overrides=experiment_overrides)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        _run_problem_fleet(
            target, config, budget, parallel_searches, name,
            agent=agent, model=model, climber=None, parallel_agents=parallel_agents,
            n_replicates=n_replicates, holdout=holdout, learning=learning, set_=set_ or [],
            seed_from=seed_from, knowledge_context_file=knowledge_context_file,
            engines=engines, study=study,
        )
        return
    if parallel_searches > 1 and (study or run_id):
        raise typer.BadParameter("--parallel-searches does not combine with --study/--run-id")
    # The default is a detached engine — the terminal comes straight back with
    # the run id and `hillclimb watch` to follow it. A suite's or study's
    # child (`--run-id`), a study's experiment, and `--no-detach` run here.
    detached = detach and run_id is None and study is None
    if parallel_searches > 1 or detached:
        _run_problem_fleet(
            target, config, budget, parallel_searches, name,
            agent=agent, model=model, climber=single_climber, parallel_agents=parallel_agents,
            n_replicates=n_replicates, holdout=holdout, learning=learning, set_=set_ or [],
            seed_from=seed_from, knowledge_context_file=knowledge_context_file,
        )
        return
    _run_problem(
        target,
        config,
        budget,
        run_id=run_id,
        run_name=run_name or name,
        seed_from=seed_from,
        study=study,
        experiment=experiment,
        repeat=repeat,
        experiment_overrides=overrides,
        knowledge_context=_read_knowledge_context(knowledge_context_file),
    )


def _run_problem_fleet(
    target: str,
    config: Config,
    budget: str | None,
    parallel_searches: int,
    name: str | None,
    *,
    agent: str | None,
    model: str | None,
    climber: str | None,
    parallel_agents: int | None,
    n_replicates: int | None,
    holdout: bool,
    learning: bool,
    set_: list[str],
    seed_from: Path | None = None,
    knowledge_context_file: Path | None = None,
    engines: list[FleetEngine] | None = None,
    study: str | None = None,
) -> Path:
    """CLI shell over api.run_fleet: N independent searches on one problem
    (or one per `engines` entry — a mixed fleet), each its own detached
    engine under one run. Returns the run dir."""
    fleet = run_fleet(
        target,
        config=config,
        parallel_searches=parallel_searches,
        run_name=name,
        budget=budget,
        agent=agent, model=model, climber=climber,
        parallel_agents=parallel_agents, n_replicates=n_replicates,
        holdout=holdout, learning=learning, seed_from=seed_from,
        knowledge_context_file=knowledge_context_file, overrides=set_,
        engines=engines, study=study,
        log=common.engine_log,
    )
    agents = parallel_agents if parallel_agents is not None else config.concurrency.parallel_agents
    failed = fleet.startup_failures()
    if failed:
        # an engine that died while starting is not running in the background
        for log_path, code in failed:
            fail(f"The {ENGINE} stopped while starting (exit {code}). The end of [path]{_m(log_path)}[/]:")
            try:
                lines = log_path.read_text(errors="replace").strip().splitlines()
            except OSError:
                lines = []
            errors = [line for line in lines if line.strip()][-12:]
            for line in errors:
                say(f"  [note]{_m(line)}[/]", err=True)
        if len(failed) < len(fleet.procs):
            warn(f"{len(fleet.procs) - len(failed)} other {ENGINE.form(len(fleet.procs) - len(failed))} of run {fleet.run_id} still running: "
                 "hillclimb stop --all ends them")
        raise typer.Exit(1)
    if engines:
        experiments = ", ".join(dict.fromkeys(engine.experiment for engine in engines))
        say(
            f"[head]Run {_m(fleet.run_id)}[/]: {len(engines)} searches ({_m(experiments)}) x {agents} coding agents "
            f"running in the background"
        )
    else:
        searches = "1 search" if parallel_searches == 1 else f"{parallel_searches} searches"
        say(
            f"[head]Run {_m(fleet.run_id)}[/]: {searches} x {agents} coding agent{'' if agents == 1 else 's'} "
            f"running in the background"
        )
    say(f"{ENGINE.cap} logs in [path]{_m(fleet.run_dir / 'logs')}[/]")
    steps = [
        ("hillclimb watch", "follow the search: every coding agent, what it is doing, its candidate's score"),
        ("hillclimb chart", "best score so far against time"),
        ("hillclimb stop --all", "end the run; the best solution of every search stays in runs/"),
    ]
    if engines:
        steps.insert(2, (f"hillclimb experiment report {study or fleet.run_id}", "compare the experiments"))
    next_steps(steps)
    return fleet.run_dir


RESUMABLE_STATES = ("parked", "stopped", "crashed")


def _live_engine_pid(store, record: SearchRecord) -> int | None:
    """The pid of an engine still working on this search, or None. A
    `running` state is proof enough; otherwise the pid the engine last
    recorded counts while that process is still a hillclimb process (a
    foreground engine, or one still draining a stop, may already read as
    stopped or crashed). A second engine on a live search duplicates its
    candidates and abandons work the first one still has in flight."""
    import os

    status = store.read_status(record.key)
    if status is None or not status.pid:
        return None
    if record.state == "running":
        return status.pid
    if status.pid == os.getpid():
        return None
    try:
        from hillclimb.harness.orphans import process_table

        proc = process_table().get(status.pid)
    except Exception:  # noqa: BLE001 - no process listing: trust the state
        return None
    return status.pid if proc is not None and "hillclimb" in proc.command else None


def _launch_entry(config: Config, record: SearchRecord) -> dict | None:
    """This search's entry in its run's `spec.yaml`: the settings it was
    launched with. A run of several problems (a suite) or several climbers
    (a mixed fleet) has one entry each; the one whose problem resolves to
    this search's, and whose `set` carries this search's experiment
    overrides, is it."""
    import yaml

    from hillclimb.api import RUN_SPEC_FILE

    try:
        spec = yaml.safe_load((record.search_dir.parents[1] / RUN_SPEC_FILE).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None
    entries = [entry for entry in spec.get("problems") or [] if isinstance(entry, dict)]
    if len(entries) > 1:
        def same_problem(entry: dict) -> bool:
            try:
                return load_problem(str(entry.get("target")), config).problem_id == record.meta.problem_id
            except Exception:  # noqa: BLE001 - an entry that no longer resolves is not this one
                return False

        entries = [entry for entry in entries if same_problem(entry)]
        wanted = {f"{key}={value}" for key, value in record.meta.experiment_overrides.items()}
        if len(entries) > 1 and wanted:
            entries = [entry for entry in entries if wanted <= set(map(str, entry.get("set") or []))] or entries
    return entries[0] if entries else None


def _restore_launch_settings(config: Config, record: SearchRecord) -> None:
    """Put back what the search was launched with beyond the folder's
    hillclimb.yaml: its `--set` settings (spend caps among them), its agents
    per search and replicates per trial, and its study experiment's
    overrides. Without them a resumed search runs uncapped."""
    from hillclimb.config import parse_set_overrides

    entry = _launch_entry(config, record)
    if entry is not None:
        if entry.get("parallel_agents") is not None:
            config.concurrency.parallel_agents = int(entry["parallel_agents"])
        if entry.get("n_replicates") is not None:
            config.evaluation.n_replicates = int(entry["n_replicates"])
        config.apply_overrides(parse_set_overrides([str(pair) for pair in entry.get("set") or []]))
    if record.meta.experiment_overrides:
        config.apply_overrides(dict(record.meta.experiment_overrides))


def _stop_what_the_dead_engine_left(record: SearchRecord) -> None:
    """A search whose engine died outright (SIGKILL, a crash) may still have
    its coding agents and verifiers running, and billing: stop them before
    the new engine starts its own."""
    from hillclimb.harness.orphans import stop_orphaned_children

    stopped = stop_orphaned_children(record.search_dir)
    if stopped:
        names = ", ".join(sorted({entry.get("program") or "?" for entry in stopped}))
        warn(f"stopped {len(stopped)} process(es) left running after {record.ref} died ({names})")


def _refuse_unless_resumable(store, record: SearchRecord) -> None:
    """Exit with a reason when resuming this search would do harm or nothing:
    an engine is still on it, or it already finished."""
    pid = _live_engine_pid(store, record)
    if pid is not None:
        fail(f"{record.ref} is still running (pid {pid}); a second {ENGINE} would duplicate its work")
        next_steps([
            (f"hillclimb stop {record.ref}", "stop it, then resume"),
            ("hillclimb watch", "follow it instead"),
        ])
        raise typer.Exit(1)
    if record.state == "done":
        fail(f"{record.ref} finished (done); there is nothing to resume")
        raise typer.Exit(1)


def _spawn_resume(config: Config, record: SearchRecord) -> tuple[int, Path]:
    """Start a detached `hillclimb resume <ref>` engine for this search,
    logging to <run>/logs/resume-<search-id>.log. Returns (pid, log path)."""
    run_dir = record.search_dir.parents[1]
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"resume-{record.meta.search_id}.log"
    cwd, env = _child_launch_context(config)
    # --no-detach: the engine IS the detached process (resume detaches by default)
    cmd = [sys.executable, "-m", "hillclimb.cli", "resume", record.ref, "--no-detach"]
    with log_path.open("a") as out:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
            **new_group_kwargs(detached=True),
        )
    return proc.pid, log_path


@app.command()
def resume(
    search: str = typer.Argument("latest"),
    all_: Annotated[
        bool, typer.Option("--all", help="Resume every parked/stopped/crashed search, each detached")
    ] = False,
    detach: Annotated[
        bool,
        typer.Option(
            "--detach/--no-detach",
            help="Resume in the background (the default, like `run`); "
            "--no-detach keeps it in this terminal",
        ),
    ] = True,
):
    """Resume a parked or interrupted search.

    SEARCH is `<run-id>/<search-id>`, `<run-id>`, or `latest`. `--all` resumes
    everything resumable (the counterpart of `hillclimb stop --all`), each as
    its own background process — the pause/resume flow for changing code or env
    under a live project. The search continues with the settings it was
    launched with (its `--set` caps, agents and replicates). A search whose
    process is still running, or one that is done, is not resumed.
    """
    config = common.load_config()
    common.require_sandbox(config)
    if all_:
        store = open_store(config)
        targets = [
            r for r in store.searches()
            if r.state in RESUMABLE_STATES and _live_engine_pid(store, r) is None
        ]
        if not targets:
            say("[head]No parked, stopped, or crashed searches to resume.[/]")
            raise typer.Exit(1)
        for agent, model in dict.fromkeys((r.meta.agent, r.meta.model) for r in targets):
            common.ensure_agents_ready(config, agent, model)
        for record in targets:
            pid, log_path = _spawn_resume(config, record)
            say(f"[head]Resuming {_m(record.ref)}[/] ({_m(record.state)}) detached: pid {pid}, log [path]{_m(log_path)}[/]")
        return
    store, record = common.open_search(config, search)
    _refuse_unless_resumable(store, record)
    common.ensure_agents_ready(config, record.meta.agent, record.meta.model)
    _stop_what_the_dead_engine_left(record)
    if detach:
        pid, log_path = _spawn_resume(config, record)
        say(f"[head]Resuming {_m(record.ref)}[/] ({_m(record.state)}) in the background: pid {pid}")
        say(f"{ENGINE.cap} log in [path]{_m(log_path)}[/]")
        next_steps([
            ("hillclimb watch", "follow the search: every coding agent, what it is doing, its candidate's score"),
            (f"hillclimb stop {record.ref}", "park it again (resumable)"),
        ])
        return
    meta, search_dir = record.meta, record.search_dir
    config = common.load_config(agent=meta.agent, model=meta.model)
    _restore_launch_settings(config, record)
    config.holdout.enabled = meta.holdout_enabled
    if not meta.learning_enabled:
        config.learning.enabled = False  # started without learning: it stays out of the knowledge
    # the search resumes as the climber it started as: its snapshot is the
    # whole truth (policy, params, operators, tuner, memory), whatever the
    # live config says by now
    from hillclimb.climber import ClimberLoadError, as_spec, climber_base_dir, climber_label, load_snapshot, resolve_climber

    try:
        snapshot = load_snapshot(search_dir, name=climber_label(meta.climber))
    except ClimberLoadError as exc:
        raise typer.BadParameter(f"this search's climber snapshot does not load: {exc}") from exc
    if snapshot is not None:
        config.climber = snapshot.spec
    else:
        # a search from before snapshots: all there is is what the record says
        try:
            block = dict(meta.climber_spec)
            if not any(key in block for key in ("operator_policy", "policy", "loop")):
                # the record names it without saying what it is: a preset of before 0.9, a file
                block = {**as_spec(meta.climber_ref or meta.climber, legacy=True).block(), **block}
            config.climber = as_spec(block, legacy=True)
            config._climber_legacy_names = True  # a record's names: the engine resolves them the same way
            resolve_climber(config.climber, climber_base_dir(config), legacy_names=True).brain  # noqa: B018
        except ClimberLoadError as exc:
            raise typer.BadParameter(
                f"climber {meta.climber} is gone and this search has no snapshot of it: {exc}"
            ) from exc
        warn("note: this search predates climber snapshots; resuming from the live climber")
    if snapshot is not None and meta.climber_ref is None and meta.climber_sha256 is not None and meta.climber_spec:
        # say so when the climber it was launched from has changed since
        # (a pre-0.6 record carries a hash of another kind: nothing to compare)
        try:
            now = resolve_climber(meta.climber_spec).sha256
        except ClimberLoadError:
            now = None  # its files are gone: the snapshot is what runs
        if now is not None and now != meta.climber_sha256:
            warn(
                f"note: climber {_m(meta.climber)} changed since the search started "
                f"({_m(meta.climber_sha256[:12])} -> {_m(now[:12])}); resuming the version it started with"
            )
    config.routing = {op: RouteConfig(**route) for op, route in meta.routing.items()}
    problem = load_problem(meta.problem, config)
    from hillclimb.harness.unit_tests import restore_frozen

    problem = restore_frozen(
        problem,
        search_dir.parents[1],
        bundle_path=getattr(meta, "unit_tests_bundle", None),
        command=getattr(meta, "unit_tests_command", []),
        sha256=getattr(meta, "unit_tests_sha256", None),
    )
    journal = Journal(store.journal(record.key))
    spent = resume_spent_seconds(store.read_status(record.key), journal)
    say(
        f"[head]Resuming {_m(search_ref(search_dir))}[/]: {len(journal.candidates)} candidates, ~{int(spent)}s spent"
    )
    _execute(
        config,
        problem,
        search_dir,
        BudgetManager(meta.budget_s, config.budget.stop_margin_s, spent_s=spent),
    )


def _agent_preflight(agent: str) -> None:
    """Fail fast, with the fix, on the two tools the engine shells out to."""
    import shutil

    missing = []
    if shutil.which("uv") is None:
        missing.append("uv is not on PATH (it builds the solution venv): pip install uv")
    if agent == "claude-code" and shutil.which("claude") is None:
        missing.append(
            "claude (Claude Code CLI) is not on PATH — the coding agents run through it:\n"
            "    npm install -g @anthropic-ai/claude-code && claude login"
        )
    if agent == "codex" and shutil.which("codex") is None:
        missing.append(
            "codex (Codex CLI) is not on PATH — install it and run `codex login`"
        )
    if agent == "pi" and shutil.which("pi") is None:
        missing.append("pi (pi coding-agent CLI) is not on PATH — install pi before running this agent")
    if missing:
        for line in missing:
            fail(f"error: {_m(line)}")
        say("[cmd]hillclimb connect[/] checks every coding agent's credential.", err=True)
        raise typer.Exit(1)


@app.command()
def smoke(
    target: str = typer.Argument("circle-packing"),
    model: str = typer.Option(None),
    agent: str = typer.Option(
        None, "--agent", "--backend",
        help="The coding agent that runs the operators: claude-code | codex | pi | dummy | toy (the last two need no LLM; --backend is the old spelling)",
    ),
):
    """One real DRAFT call through the selected coding agent, end to end.

    Executes the result and reports — verifies auth, JSON field names, and
    the filesystem contract.
    """
    config = common.load_config(agent=agent, model=model)
    from hillclimb.catalog import PROBLEM_IDS, install_problem

    if target in PROBLEM_IDS and not (config.paths.problems_dir / target / "problem.yaml").exists():
        # a check of the agent, not of the folder: fetch the catalog problem it uses
        problem_dir, _ = install_problem(config.paths.problems_dir, target)
        say(f"fetched {_m(target)} for the check: [path]{_m(problem_dir)}[/]")
    problem = load_problem(target, config)
    _agent_preflight(config.agent)
    version_cmd = {
        "claude-code": ["claude", "-v"],
        "codex": ["codex", "--version"],
        "pi": ["pi", "--version"],
    }.get(config.agent)
    if version_cmd:
        version = subprocess.run(
            runnable(version_cmd), capture_output=True, text=True
        ).stdout.strip()
        say(f"{_m(config.agent)} version: [head]{_m(version)}[/]")
    run_id = f"smoke-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = create_run(
        config,
        RunMeta(
            run_id=run_id,
            name=run_id,
            kind="problem",
            target=target,
            problem_ids=[problem.problem_id],
        ),
    )
    search_dir = create_search(config, problem, run_dir, run_id, total_s=1800)
    from hillclimb.harness.routing import AgentPool, Router

    agent_instance = get_agent(
        config.agent, auth=config.agent_auth, pi_models_file=config.pi.models_file
    )
    agents = AgentPool(pi_models_file=config.pi.models_file)
    agents.seed(config.agent, config.agent_auth, agent_instance)
    journal = Journal(open_store(config).journal(key_for(search_dir)))
    evaluator = build_evaluator(config, problem, search_dir, journal, log=common.engine_log)
    harness = Harness(
        problem=problem,
        config=config,
        journal=journal,
        agent=agent_instance,
        router=Router(config),
        agents=agents,
        executor=evaluator.executor,
        budget=BudgetManager(1800, stop_margin_s=0),
        search_dir=search_dir,
        log=common.engine_log,
        evaluator=evaluator,
    )
    say(
        f"[head]Running one {_m(config.agent)} DRAFT in the foreground;[/] "
        "[note]this can take several minutes.[/]"
    )
    say("To follow it live, open another terminal and run: [cmd]hillclimb watch[/]")
    outcome = harness.run(Action(operator="draft"))
    if outcome.candidate is None:
        fail(f"the harness refused the draft: {_m(outcome.ticket.rejected)}")
        raise typer.Exit(1)
    # the journal's own record: the smoke report shows holdout, which a
    # loop-facing Outcome never carries
    candidate = journal.get(outcome.candidate.candidate_id)
    trial = candidate.last_trial
    say(f"\n[head]candidate:[/]   [path]{_m(candidate.candidate_id)}[/] status={_m(candidate.status)}")
    say(f"[head]val_score:[/]   {_m(candidate.val_score)}")
    say(f"[head]holdout:[/]     {_m(candidate.holdout_score)} [note](error: {_m(trial.holdout_error if trial else '-')})[/]")
    say(f"[head]session_id:[/]  {_m(candidate.agent.session_id)}")
    say(f"[head]cost_usd:[/]    {_m(candidate.agent.cost_usd)}")
    say(f"[head]num_turns:[/]   {_m(candidate.agent.num_turns)}")
    say(f"[head]error_kind:[/]  {_m(candidate.agent.error_kind)}")
    say(f"[head]raw output:[/]  [path]{_m(Path(candidate.candidate_dir) / 'agent_raw.json')}[/]")
    if candidate.agent.session_id is None and candidate.agent.error_kind is None:
        warn("WARNING: session_id not parsed — check agent_raw.json for actual field names")
