"""`hillclimb run`, `resume`, `demo`, `smoke`: starting searches and fleets."""

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
    create_problem_run,
    create_run,
    create_search,
    execute_search,
    mixed_fleet,
    new_run_id,
    resume_spent_seconds,
    run_fleet,
    spawn_search_proc,
)
from hillclimb.backends import get_backend
from hillclimb.cli import common
from hillclimb.cli._app import app, print_banner
from hillclimb.cli.common import _m, next_steps, say
from hillclimb.config import Config, RouteConfig
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
        config, problem, search_dir, budget, log=typer.echo, seed_from=seed_from,
        knowledge_context=knowledge_context,
    )
    ref = outcome.ref
    if outcome.state == "parked":
        typer.echo(f"\nParked: {outcome.error}")
        typer.echo(f"Resume later with: hillclimb resume {ref}")
        raise typer.Exit(2)
    if outcome.state == "stopped":
        typer.echo("\nStopped.")
        typer.echo(f"Resume with: hillclimb resume {ref}")
        raise typer.Exit(2)
    selected = outcome.selected
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
    experiment: str | None = None,
    arm: str | None = None,
    repeat: int = 0,
    arm_overrides: dict | None = None,
    knowledge_context: str | None = None,
) -> None:
    if arm_overrides:
        try:
            config.apply_overrides(arm_overrides)
        except (KeyError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    problem = load_problem(target, config)
    if run_id is None:
        run_name = run_name or problem.problem_id
        run_dir = _create_problem_run(config, run_name, target, problem.problem_id)
        run_id = run_dir.name
    else:
        # suite child: the parent already wrote run.yaml
        run_name = run_name or run_id
        run_dir = config.paths.runs_dir / run_id
    total_s = common.parse_budget(budget) if budget else problem.time_budget_s
    search_dir = create_search(
        config, problem, run_dir, run_id, total_s, seed_from=seed_from,
        experiment=experiment, arm=arm, repeat=repeat, arm_overrides=arm_overrides,
    )
    tag = f", experiment={experiment}/{arm}" + (f" r{repeat}" if repeat else "") if experiment else ""
    typer.echo(
        f"Search {search_ref(search_dir)} (run={run_name}, problem={problem.problem_id}, "
        f"backend={config.backend}, model={config.model}, budget={total_s}s{tag})"
    )
    _execute(
        config, problem, search_dir,
        BudgetManager(total_s, config.budget.stop_margin_s),
        seed_from=seed_from,
        knowledge_context=knowledge_context,
    )


def _run_suite(
    target: str,
    config: Config,
    budget: str | None,
    backend: str | None,
    model: str | None,
    holdout: bool,
    name: str | None,
    climber: str | None = None,
    parallel_operators: int | None = None,
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
    for index, (entry, problem_target) in enumerate(zip(suite.problems, problem_targets), 1):
        slug = Path(problem_target).name or f"problem-{index}"
        cmd = [problem_target, "--run-id", run_id, "--run-name", run_name]
        # CLI flags override the spec entry's committed values
        child_budget = budget or entry.budget
        child_backend = backend or entry.backend
        child_model = model or entry.model
        child_parallel = parallel_operators if parallel_operators is not None else entry.parallel_operators
        child_replicates = n_replicates if n_replicates is not None else entry.n_replicates
        child_seed = seed_from or entry.seed_from
        if child_budget:
            cmd += ["--budget", child_budget]
        if child_backend:
            cmd += ["--backend", child_backend]
        if child_model:
            cmd += ["--model", child_model]
        if climber:
            cmd += ["--climber", climber]
        if child_parallel is not None:
            cmd += ["--parallel-operators", str(child_parallel)]
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
        for pair in set_ or []:
            cmd += ["--set", pair]
        pid, log_path = _spawn_search(config, run_dir, index, slug, cmd)
        launched.append((problem_target, pid, log_path))
    typer.echo(f"Run {run_id}: launched {len(launched)} searches")
    for problem_target, pid, log_path in launched:
        typer.echo(f"  pid={pid} {problem_target}  log={log_path}")


@app.command()
def run(
    target: str,
    budget: str = typer.Option(None, help="Wall-clock budget, e.g. 2h / 30m"),
    backend: str = typer.Option(None, help="Operator backend: claude-code | codex | pi | dummy"),
    model: str = typer.Option(None, help="Model for operator calls, e.g. sonnet / opus"),
    climber: list[str] = typer.Option(
        None, "--climber",
        help=(
            "The climber (default: greedy): a bundled name, a directory holding climber.yaml, "
            "or one .py file; params via config climber.params. Repeat it "
            "(--climber greedy --climber gepa) for a mixed fleet: one search per climber "
            "on the problem, under one run, each tagged as an arm"
        ),
    ),
    policy: list[str] = typer.Option(None, "--policy", hidden=True),
    holdout: bool = typer.Option(True, "--holdout/--no-holdout", help="Hidden selection holdout"),
    learning: bool = typer.Option(
        True, "--learning/--no-learning",
        help="Cross-search memory (cards/claims injection + distillation); off = memory-blind arm",
    ),
    name: str = typer.Option(None, "--name", help="Run name shown in the TUI"),
    parallel_operators: int = typer.Option(
        None, "--parallel-operators", help="Concurrent operators per search (worker pool)"
    ),
    parallel_searches: int = typer.Option(
        1, "--parallel-searches", min=1,
        help="Independent searches on the problem at once (>1 runs them detached, in the background)",
    ),
    n_replicates: int = typer.Option(
        None, "--n-replicates", "--n-trials",
        help="Seeded runs per trial (the median is the trial's score; --n-trials is the old spelling)",
    ),
    seed_from: Path = typer.Option(
        None, "--seed-from", help="Incumbent solution.py scored as the floor candidate"
    ),
    set_: list[str] = typer.Option(
        None, "--set", help="Any config setting, dotted: --set climber.ref=openevolve --set learning.enabled=false",
    ),
    arm_set: list[str] = typer.Option(
        None, "--arm-set",
        help=(
            "A setting for one arm of a mixed fleet, ARM:KEY=VALUE: "
            "--arm-set gepa:concurrency.parallel_operators=1 (applied after --set)"
        ),
    ),
    experiment: str = typer.Option(
        None, "--experiment",
        help="Tag the search as one arm of an experiment (with --arm); names a mixed fleet's experiment",
    ),
    arm: str = typer.Option(None, "--arm", help="The arm name (with --experiment)"),
    repeat: int = typer.Option(0, "--repeat", hidden=True),
    run_id: str = typer.Option(None, "--run-id", hidden=True),
    run_name: str = typer.Option(None, "--run-name", hidden=True),
    knowledge_context_file: Path = typer.Option(
        None, "--knowledge-context-file", hidden=True,
        help="Pre-built knowledge context (markdown) injected into every operator prompt",
    ),
):
    """Start a run on a problem or a run-spec YAML."""
    config = common.load_config(backend=backend, model=model)
    if not holdout:
        config.holdout.enabled = False
    if not learning:
        config.learning.enabled = False
    if policy:
        typer.echo("note: `--policy` is now `--climber` (same values)", err=True)
    climbers = [*(climber or []), *(policy or [])]
    mixed = len(climbers) > 1
    arm_overrides = common._parse_arm_set(arm_set or [])
    if arm_overrides and not mixed:
        raise typer.BadParameter("--arm-set needs a mixed fleet (two or more --climber)")
    single_climber = None if mixed else (climbers[0] if climbers else None)
    if single_climber is not None:
        config.climber.ref = single_climber
    if parallel_operators is not None:
        config.concurrency.parallel_operators = parallel_operators
    if n_replicates is not None:
        config.evaluation.n_replicates = n_replicates
    overrides = common._parse_set(set_ or [])
    if mixed:
        if arm or run_id:
            raise typer.BadParameter("a mixed fleet names its arms itself; --arm/--run-id do not apply")
    elif (experiment is None) != (arm is None):
        raise typer.BadParameter("--experiment and --arm go together")
    resolved = resolve_target(target, config)
    if resolved.kind == "suite":
        if mixed:
            raise typer.BadParameter("a spec takes one --climber; mixed fleets run on a single problem")
        _run_suite(
            target, config, budget, backend, model, holdout, name,
            climber=single_climber, parallel_operators=parallel_operators, n_replicates=n_replicates,
            seed_from=seed_from, learning=learning, set_=set_,
        )
        return
    if mixed:
        try:
            engines = mixed_fleet(climbers, repeats=parallel_searches, arm_overrides=arm_overrides)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        _run_problem_fleet(
            target, config, budget, parallel_searches, name,
            backend=backend, model=model, climber=None, parallel_operators=parallel_operators,
            n_replicates=n_replicates, holdout=holdout, learning=learning, set_=set_ or [],
            seed_from=seed_from, knowledge_context_file=knowledge_context_file,
            engines=engines, experiment=experiment,
        )
        return
    if parallel_searches > 1:
        if experiment or run_id:
            raise typer.BadParameter("--parallel-searches does not combine with --experiment/--run-id")
        _run_problem_fleet(
            target, config, budget, parallel_searches, name,
            backend=backend, model=model, climber=single_climber, parallel_operators=parallel_operators,
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
        experiment=experiment,
        arm=arm,
        repeat=repeat,
        arm_overrides=overrides,
        knowledge_context=_read_knowledge_context(knowledge_context_file),
    )


def _run_problem_fleet(
    target: str,
    config: Config,
    budget: str | None,
    parallel_searches: int,
    name: str | None,
    *,
    backend: str | None,
    model: str | None,
    climber: str | None,
    parallel_operators: int | None,
    n_replicates: int | None,
    holdout: bool,
    learning: bool,
    set_: list[str],
    seed_from: Path | None = None,
    knowledge_context_file: Path | None = None,
    engines: list[FleetEngine] | None = None,
    experiment: str | None = None,
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
        backend=backend, model=model, climber=climber,
        parallel_operators=parallel_operators, n_replicates=n_replicates,
        holdout=holdout, learning=learning, seed_from=seed_from,
        knowledge_context_file=knowledge_context_file, overrides=set_,
        engines=engines, experiment=experiment,
        log=typer.echo,
    )
    operators = parallel_operators if parallel_operators is not None else config.concurrency.parallel_operators
    if engines:
        arms = ", ".join(dict.fromkeys(engine.arm for engine in engines))
        say(
            f"[head]Run {_m(fleet.run_id)}[/]: {len(engines)} searches ({_m(arms)}) x {operators} operators "
            f"running in the background"
        )
    else:
        say(
            f"[head]Run {_m(fleet.run_id)}[/]: {parallel_searches} searches x {operators} operators "
            f"running in the background"
        )
    say(f"Engine logs in [path]{_m(fleet.run_dir / 'logs')}[/]")
    steps = [
        ("hillclimb watch", "every agent, what it is doing, its candidate's score"),
        ("hillclimb chart", "best score so far against time"),
        ("hillclimb stop --all", "end the run; the best solution of every search stays in runs/"),
    ]
    if engines:
        steps.insert(2, (f"hillclimb experiment report {experiment or fleet.run_id}", "compare the arms"))
    next_steps(steps)
    return fleet.run_dir


RESUMABLE_STATES = ("parked", "stopped", "crashed")


def _spawn_resume(config: Config, record: SearchRecord) -> tuple[int, Path]:
    """Start a detached `hillclimb resume <ref>` engine for this search,
    logging to <run>/logs/resume-<search-id>.log. Returns (pid, log path)."""
    run_dir = record.search_dir.parents[1]
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"resume-{record.meta.search_id}.log"
    cwd, env = _child_launch_context(config)
    cmd = [sys.executable, "-m", "hillclimb.cli", "resume", record.ref]
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
        typer.Option("--detach", help="Resume in a detached background engine instead of the foreground"),
    ] = False,
):
    """Resume a parked or interrupted search.

    SEARCH is `<run-id>/<search-id>`, `<run-id>`, or `latest`. `--all` resumes
    everything resumable (the counterpart of `hillclimb stop --all`), each as
    its own detached engine — the pause/resume flow for changing code or env
    under a live project.
    """
    config = common.load_config()
    if all_:
        store = open_store(config)
        targets = [r for r in store.searches() if r.state in RESUMABLE_STATES]
        if not targets:
            typer.echo("No parked, stopped, or crashed searches to resume.")
            raise typer.Exit(1)
        for record in targets:
            pid, log_path = _spawn_resume(config, record)
            typer.echo(f"Resuming {record.ref} ({record.state}) detached: pid {pid}, log {log_path}")
        return
    store, record = common.open_search(config, search)
    if detach:
        pid, log_path = _spawn_resume(config, record)
        typer.echo(f"Resuming {record.ref} ({record.state}) detached: pid {pid}, log {log_path}")
        return
    meta, search_dir = record.meta, record.search_dir
    config = common.load_config(backend=meta.backend, model=meta.model)
    config.holdout.enabled = meta.holdout_enabled
    # the search resumes under the policy/routing it started with, not
    # whatever the live config currently says
    config.climber.ref = meta.climber
    config.climber.params = meta.climber_params
    if meta.climber_sha256 is not None:
        from hillclimb.climber import ClimberLoadError, load_climber, load_snapshot
        from hillclimb.modules.policies import policy_base_dir

        # the search resumes from the snapshot in its own folder, so an
        # edited (or deleted) live climber changes nothing — but say so
        snapshot = load_snapshot(search_dir)
        try:
            now = load_climber(meta.climber, policy_base_dir(config)).sha256
        except ClimberLoadError as exc:
            if snapshot is None:  # nothing left to resume WITH
                raise typer.BadParameter(
                    f"climber {meta.climber} is gone and this search has no snapshot of it: {exc}"
                ) from exc
            now = None
        if snapshot is None:
            typer.echo("note: this search predates climber snapshots; resuming from the live climber", err=True)
        if now is not None and now != meta.climber_sha256:
            typer.echo(
                f"note: climber {meta.climber} changed since the search started "
                f"({meta.climber_sha256[:12]} -> {now[:12]}); resuming the version it started with",
                err=True,
            )
    if meta.tuner is not None:
        config.climber.tuner = meta.tuner
    config.climber.tuner_params = meta.tuner_params
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
    typer.echo(
        f"Resuming {search_ref(search_dir)}: {len(journal.candidates)} candidates, ~{int(spent)}s spent"
    )
    _execute(
        config,
        problem,
        search_dir,
        BudgetManager(meta.budget_s, config.budget.stop_margin_s, spent_s=spent),
    )


def _demo_preflight(backend: str) -> None:
    """Fail fast, with the fix, on the two tools the engine shells out to."""
    import shutil

    missing = []
    if shutil.which("uv") is None:
        missing.append("uv is not on PATH (it builds the solution venv): pip install uv")
    if backend == "claude-code" and shutil.which("claude") is None:
        missing.append(
            "claude (Claude Code CLI) is not on PATH — the agents run through it:\n"
            "    npm install -g @anthropic-ai/claude-code && claude login"
        )
    if backend == "codex" and shutil.which("codex") is None:
        missing.append(
            "codex (Codex CLI) is not on PATH — install it and run `codex login`"
        )
    if backend == "pi" and shutil.which("pi") is None:
        missing.append("pi (pi coding-agent CLI) is not on PATH — install pi before running this backend")
    if missing:
        for line in missing:
            typer.echo(f"error: {line}", err=True)
        typer.echo("`hillclimb connect` checks every backend's credential.", err=True)
        raise typer.Exit(1)


def _print_demo_intro(folder: Path, parallel_searches: int, parallel_operators: int, budget: str) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console(highlight=False)
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    for command, what in DEMO_COMMANDS:
        table.add_row(command, what)
    body = Table.grid(padding=(0, 0))
    body.add_row(
        f"Circle packing: 26 circles in the unit square, maximize the sum of radii.\n"
        f"{parallel_searches} searches of {budget} are climbing in parallel in [cyan]{folder}[/],\n"
        f"each running {parallel_operators} operators at a time, "
        f"starting from a one-circle baseline (sum of radii 0.5).\n"
    )
    body.add_row("They run in the background — watch them from this terminal:\n")
    body.add_row(table)
    body.add_row("\n[bold cyan]hillclimb stop --all[/] ends the demo; the best solutions stay in runs/.")
    console.print(Panel(body, title="hillclimb demo", border_style="cyan", expand=False))
    console.print()


DEMO_COMMANDS = (
    ("hillclimb watch candidates", "one search's candidates: agents drafting, debugging, improving"),
    ("hillclimb watch", "all the searches side by side"),
    ("hillclimb chart", "the hillclimb curve: best score vs candidates, live"),
    ("hillclimb tree", "one search's exploration tree: expanded vs discontinued lineages"),
    ("hillclimb graph", "the knowledge graph growing as searches finish"),
)


@app.command()
def demo(
    budget: str = typer.Option("10m", help="Wall-clock budget per search, e.g. 10m"),
    parallel_searches: int = typer.Option(3, "--parallel-searches", min=1, help="Searches to run at once"),
    parallel_operators: int = typer.Option(
        3, "--parallel-operators", min=1, help="Concurrent operators (one candidate each) per search"
    ),
    model: str = typer.Option(None, help="Model for operator calls, e.g. sonnet / opus"),
    backend: str = typer.Option(None, help="Operator backend: claude-code | codex | pi | dummy"),
):
    """Try hillclimb in one command: agents climb the circle-packing problem.

    Creates a hillclimb/ dir here if there is none, installs the bundled
    problem, starts several searches in parallel in the background, and
    prints the commands that show them live — run those right here.
    """
    from hillclimb.demo import DEMO_PROBLEM_ID, install_demo_problem
    from hillclimb.project import find_hillclimb_dir

    print_banner()
    if find_hillclimb_dir() is None:
        folder = common.scaffold_hillclimb_dir(Path.cwd())
        typer.echo(f"Created hillclimb dir at {folder}")
    config = common.load_config(backend=backend, model=model)
    problem_dir, created = install_demo_problem(config.paths.problems_dir)
    if created:
        typer.echo(f"Installed the {DEMO_PROBLEM_ID} problem at {problem_dir}")
    _demo_preflight(config.backend)
    run_dir = _run_problem_fleet(
        DEMO_PROBLEM_ID, config, budget, parallel_searches, "demo",
        backend=backend, model=model, climber=None, parallel_operators=parallel_operators,
        n_replicates=None, holdout=True, learning=True, set_=[],
    )
    _print_demo_intro(config.hillclimb_dir, parallel_searches, parallel_operators, budget)
    typer.echo(f"Engine logs in {run_dir / 'logs'}")


@app.command()
def smoke(
    target: str = typer.Argument("circle-packing"),
    model: str = typer.Option(None),
    backend: str = typer.Option(None, help="Operator backend: claude-code | codex | pi | dummy"),
):
    """One real DRAFT call through the selected backend, end to end.

    Executes the result and reports — verifies auth, JSON field names, and
    the filesystem contract.
    """
    config = common.load_config(backend=backend, model=model)
    problem = load_problem(target, config)
    _demo_preflight(config.backend)
    version_cmd = {
        "claude-code": ["claude", "-v"],
        "codex": ["codex", "--version"],
        "pi": ["pi", "--version"],
    }.get(config.backend)
    if version_cmd:
        version = subprocess.run(
            runnable(version_cmd), capture_output=True, text=True
        ).stdout.strip()
        typer.echo(f"{config.backend} version: {version}")
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
    from hillclimb.harness.routing import BackendPool, Router

    backend_instance = get_backend(
        config.backend, auth=config.backend_auth, pi_models_file=config.pi.models_file
    )
    backends = BackendPool(pi_models_file=config.pi.models_file)
    backends.seed(config.backend, config.backend_auth, backend_instance)
    journal = Journal(open_store(config).journal(key_for(search_dir)))
    evaluator = build_evaluator(config, problem, search_dir, journal, log=typer.echo)
    harness = Harness(
        problem=problem,
        config=config,
        journal=journal,
        backend=backend_instance,
        router=Router(config),
        backends=backends,
        executor=evaluator.executor,
        budget=BudgetManager(1800, stop_margin_s=0),
        search_dir=search_dir,
        log=typer.echo,
        evaluator=evaluator,
    )
    typer.echo(
        f"Running one {config.backend} DRAFT in the foreground; "
        "this can take several minutes."
    )
    typer.echo("To follow it live, open another terminal and run: hillclimb watch")
    outcome = harness.run(Action(operator="draft"))
    if outcome.candidate is None:
        typer.echo(f"the harness refused the draft: {outcome.ticket.rejected}")
        raise typer.Exit(1)
    # the journal's own record: the smoke report shows holdout, which a
    # loop-facing Outcome never carries
    candidate = journal.get(outcome.candidate.candidate_id)
    trial = candidate.last_trial
    typer.echo(f"\ncandidate:   {candidate.candidate_id} status={candidate.status}")
    typer.echo(f"val_score:   {candidate.val_score}")
    typer.echo(f"holdout:     {candidate.holdout_score} (error: {trial.holdout_error if trial else '-'})")
    typer.echo(f"session_id:  {candidate.backend.session_id}")
    typer.echo(f"cost_usd:    {candidate.backend.cost_usd}")
    typer.echo(f"num_turns:   {candidate.backend.num_turns}")
    typer.echo(f"error_kind:  {candidate.backend.error_kind}")
    typer.echo(f"raw output:  {Path(candidate.candidate_dir) / 'agent_raw.json'}")
    if candidate.backend.session_id is None and candidate.backend.error_kind is None:
        typer.echo("WARNING: session_id not parsed — check agent_raw.json for actual field names")
