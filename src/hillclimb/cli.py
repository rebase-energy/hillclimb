from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import typer

from hillclimb.api import (
    create_search,
    ensure_runtime_venv,
    execute_search,
    new_run_id,
    search_ref,
    spent_seconds,
)
from hillclimb.backends import get_backend
from hillclimb.budget import BudgetManager
from hillclimb.config import Config
from hillclimb.control import request_prune, request_stop
from hillclimb.executor import LocalExecutor
from hillclimb.journal import Journal
from hillclimb.problem import (
    ProblemSpec,
    load_problem,
    resolve_target,
    suite_problem_targets,
)
from hillclimb.run import (
    SEARCHES_DIRNAME,
    RunMeta,
    iter_search_dirs,
    latest_search_dir,
    load_run_meta,
    load_search_meta,
    write_run_meta,
)
from hillclimb.search import GreedySearcher
from hillclimb.status import effective_state, read_status
from hillclimb.workspace import create_run_dir

app = typer.Typer(help="rebase-hillclimb: auto-hillclimbing for verifier-defined problems")


def parse_budget(value: str) -> int:
    match = re.fullmatch(r"(\d+)\s*([hms]?)", value.strip())
    if not match:
        raise typer.BadParameter(f"Cannot parse budget {value!r} (use e.g. 2h, 30m, 3600s)")
    amount, unit = int(match.group(1)), match.group(2)
    return amount * {"h": 3600, "m": 60, "s": 1, "": 1}[unit]


def resolve_search_dir(config: Config, ref: str | None) -> Path:
    """Resolve a search reference:

    - `latest` (or empty) — the most recently active search across v2 runs
    - `<run-id>/<search-id>` — exact address
    - `<run-id>` — the run's only search; error listing choices if several
    """
    runs_dir = config.paths.runs_dir
    if not ref or ref == "latest":
        latest = latest_search_dir(runs_dir)
        if latest is None:
            raise typer.BadParameter(f"No searches found in {runs_dir}")
        return latest
    if "/" in ref:
        run_id, _, search_id = ref.partition("/")
        search_dir = runs_dir / run_id / SEARCHES_DIRNAME / search_id
        if load_search_meta(search_dir) is None:
            raise typer.BadParameter(f"No search at {search_dir}")
        return search_dir
    run_dir = runs_dir / ref
    if load_run_meta(run_dir) is None:
        raise typer.BadParameter(f"No run named {ref!r} in {runs_dir}")
    searches = iter_search_dirs(run_dir)
    if not searches:
        raise typer.BadParameter(f"Run {ref} has no searches")
    if len(searches) > 1:
        choices = "\n".join(f"  {ref}/{s.name}" for s in searches)
        raise typer.BadParameter(f"Run {ref} has {len(searches)} searches; pick one:\n{choices}")
    return searches[0]


def _execute(config: Config, problem: ProblemSpec, search_dir: Path, budget: BudgetManager) -> None:
    """CLI shell over api.execute_search: messages + exit codes."""
    outcome = execute_search(config, problem, search_dir, budget, log=typer.echo)
    ref = outcome.ref
    if outcome.state == "parked":
        typer.echo(f"\nRate limited: {outcome.error}")
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
        typer.echo(
            f"\nDone. Selected candidate {selected.candidate_id}: {scores} "
            f"({problem.metric_name}, {'lower' if problem.lower_is_better else 'higher'} is better)"
        )
    else:
        typer.echo("\nDone. No scored solution; best/ holds the baseline submission.")
    artifact = "solution.py" if problem.kind == "emflow" else "submission.csv"
    typer.echo(f"Best artifact: {search_dir / 'best' / artifact}")
    typer.echo(f"Inspect with: hillclimb status {ref}")


def _run_problem(
    target: str,
    config: Config,
    budget: str | None,
    run_id: str | None = None,
    run_name: str | None = None,
) -> None:
    problem = load_problem(target, config)
    if run_id is None:
        run_name = run_name or problem.problem_id
        run_id = new_run_id(run_name)
        run_dir = create_run_dir(config.paths.runs_dir, run_id)
        write_run_meta(
            run_dir,
            RunMeta(
                run_id=run_id,
                name=run_name,
                kind="problem",
                target=target,
                problem_ids=[problem.problem_id],
            ),
        )
    else:
        # suite child: the parent already wrote run.yaml
        run_name = run_name or run_id
        run_dir = config.paths.runs_dir / run_id
    total_s = parse_budget(budget) if budget else problem.time_budget_s
    search_dir = create_search(config, problem, run_dir, run_id, total_s)
    typer.echo(
        f"Search {search_ref(search_dir)} (run={run_name}, problem={problem.problem_id}, "
        f"backend={config.backend}, model={config.model}, budget={total_s}s)"
    )
    _execute(config, problem, search_dir, BudgetManager(total_s, config.budget.stop_margin_s))


def _run_suite(
    target: str,
    config: Config,
    budget: str | None,
    backend: str | None,
    model: str | None,
    holdout: bool,
    name: str | None,
) -> None:
    resolved = resolve_target(target, config)
    if resolved.kind != "suite" or resolved.suite is None:
        raise typer.BadParameter(f"{target!r} is not a suite")
    suite = resolved.suite
    run_name = name or suite.suite_id
    run_id = new_run_id(run_name)
    problem_targets = suite_problem_targets(suite, config)
    problem_ids = [load_problem(problem_target, config).problem_id for problem_target in problem_targets]
    duplicates = {p for p in problem_ids if problem_ids.count(p) > 1}
    if duplicates:
        # search ids are problem ids, unique within a run
        raise typer.BadParameter(
            f"Suite {target!r} lists duplicate problem ids: {', '.join(sorted(duplicates))}"
        )
    run_dir = create_run_dir(config.paths.runs_dir, run_id)
    write_run_meta(
        run_dir,
        RunMeta(
            run_id=run_id,
            name=run_name,
            kind="suite",
            target=target,
            problem_ids=problem_ids,
        ),
    )
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    launched = []
    for index, problem_target in enumerate(problem_targets, 1):
        slug = Path(problem_target).name or f"problem-{index}"
        log_path = log_dir / f"{index:02d}-{slug}.log"
        cmd = [
            sys.executable,
            "-m",
            "hillclimb.cli",
            "run",
            problem_target,
            "--run-id",
            run_id,
            "--run-name",
            run_name,
        ]
        if budget:
            cmd += ["--budget", budget]
        if backend:
            cmd += ["--backend", backend]
        if model:
            cmd += ["--model", model]
        if not holdout:
            cmd.append("--no-holdout")
        out = log_path.open("w")
        proc = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        out.close()
        launched.append((problem_target, proc.pid, log_path))
    typer.echo(f"Run {run_id}: launched {len(launched)} searches")
    for problem_target, pid, log_path in launched:
        typer.echo(f"  pid={pid} {problem_target}  log={log_path}")


@app.command()
def run(
    target: str,
    budget: str = typer.Option(None, help="Wall-clock budget, e.g. 2h / 30m"),
    backend: str = typer.Option(None, help="Operator backend: claude-code | dummy"),
    model: str = typer.Option(None, help="Model for operator calls, e.g. sonnet / opus"),
    holdout: bool = typer.Option(True, "--holdout/--no-holdout", help="Hidden selection holdout"),
    name: str = typer.Option(None, "--name", help="Run name shown in the TUI"),
    run_id: str = typer.Option(None, "--run-id", hidden=True),
    run_name: str = typer.Option(None, "--run-name", hidden=True),
):
    """Start a hillclimb run on a problem folder/name or a suite YAML."""
    config = Config.load(backend=backend, model=model)
    if not holdout:
        config.holdout.enabled = False
    resolved = resolve_target(target, config)
    if resolved.kind == "suite":
        _run_suite(target, config, budget, backend, model, holdout, name)
        return
    _run_problem(
        target,
        config,
        budget,
        run_id=run_id,
        run_name=run_name or name,
    )


@app.command()
def resume(search: str = typer.Argument("latest")):
    """Resume a parked or interrupted search (`<run-id>/<search-id>`,
    `<run-id>`, or `latest`)."""
    config = Config.load()
    search_dir = resolve_search_dir(config, search)
    meta = load_search_meta(search_dir)
    if meta is None:
        raise typer.BadParameter(f"No valid search.yaml in {search_dir}")
    config = Config.load(backend=meta.backend, model=meta.model)
    config.holdout.enabled = meta.holdout_enabled
    if meta.holdout_seed is not None:
        config.holdout.seed = meta.holdout_seed
    if meta.holdout_fraction is not None:
        config.holdout.fraction = meta.holdout_fraction
    problem = load_problem(meta.problem, config)
    journal = Journal(search_dir / "journal.jsonl")
    spent = spent_seconds(journal)
    typer.echo(
        f"Resuming {search_ref(search_dir)}: {len(journal.candidates)} candidates, ~{int(spent)}s spent"
    )
    _execute(
        config,
        problem,
        search_dir,
        BudgetManager(meta.budget_s, config.budget.stop_margin_s, spent_s=spent),
    )


@app.command()
def stop(search: str = typer.Argument("latest")):
    """Gracefully stop a running engine: it finishes the current operator
    call, then parks. Resume later with `hillclimb resume`."""
    config = Config.load()
    search_dir = resolve_search_dir(config, search)
    ref = search_ref(search_dir)
    outcome = request_stop(search_dir, source="cli")
    if outcome is None:
        typer.echo(f"Search {ref} is {effective_state(search_dir)}; nothing to stop.")
        raise typer.Exit(1)
    typer.echo(f"{outcome} (use `hillclimb kill {ref}` to interrupt now)")


@app.command()
def prune(
    search: str,
    candidate_id: str,
    reason: str = typer.Option("", help="Why this branch is being cut (recorded in the journal)"),
):
    """Prune a candidate and its whole subtree: the engine stops building on
    this lineage and it is excluded from selection. Statuses and scores stay
    visible in status/tree output."""
    config = Config.load()
    search_dir = resolve_search_dir(config, search)
    meta = load_search_meta(search_dir)
    lower = bool(meta.lower_is_better) if meta else False
    try:
        outcome = request_prune(
            search_dir,
            candidate_id,
            lower_is_better=lower,
            selection_mode=config.holdout.selection,
            reason=reason,
            source="cli",
        )
    except ValueError as exc:
        typer.echo(f"Cannot prune: {exc}")
        raise typer.Exit(1)
    typer.echo(outcome)


@app.command()
def kill(search: str = typer.Argument("latest")):
    """SIGTERM a running engine; it finalizes state and can be resumed.
    For a graceful stop that lets the current operator finish, use
    `hillclimb stop`."""
    config = Config.load()
    search_dir = resolve_search_dir(config, search)
    ref = search_ref(search_dir)
    state = effective_state(search_dir)
    if state != "running":
        typer.echo(f"Search {ref} is {state}; nothing to kill.")
        raise typer.Exit(1)
    engine_pid = read_status(search_dir).pid
    os.kill(engine_pid, signal.SIGTERM)
    typer.echo(f"Sent SIGTERM to engine pid {engine_pid} ({ref}).")
    typer.echo(f"Resume with: hillclimb resume {ref}")


@app.command()
def status(search: str = typer.Argument("latest")):
    """Show the candidate tree of a search."""
    config = Config.load()
    search_dir = resolve_search_dir(config, search)
    journal = Journal(search_dir / "journal.jsonl")
    search_status = read_status(search_dir)
    state = effective_state(search_dir)
    if search_status is not None:
        remaining = int(search_status.budget.remaining_s)
        line = f"state={state}  budget: {int(search_status.budget.spent_s)}s spent / {remaining}s left"
        if search_status.current is not None:
            line += (
                f"  current candidate: {search_status.current.candidate_id} "
                f"({search_status.current.operator}/{search_status.current.phase})"
            )
        typer.echo(line)
    typer.echo(f"Search {search_ref(search_dir)} — {len(journal.candidates)} candidates")
    for candidate in journal.candidates.values():
        score = f"{candidate.val_score:.5f}" if candidate.val_score is not None else "-"
        hold = f" hold={candidate.holdout_score:.5f}" if candidate.holdout_score is not None else ""
        marks = (" *SELECTED*" if candidate.is_selected else "") + (
            " *best-val*" if candidate.is_best else ""
        )
        if candidate.pruned:
            marks += " *PRUNED*"
        parent = f" <- {candidate.parent_id}" if candidate.parent_id else ""
        typer.echo(
            f"  {candidate.candidate_id} {candidate.operator:<9} {candidate.status:<9} "
            f"val={score}{hold}{marks}{parent}  {candidate.summary[:70]}"
        )
    scored = [
        c
        for c in journal.candidates.values()
        if c.val_score is not None and c.holdout_score is not None
    ]
    if scored:
        gaps = [abs(c.val_score - c.holdout_score) for c in scored]
        typer.echo(
            f"val→holdout gap: mean {sum(gaps)/len(gaps):.5g}, max {max(gaps):.5g} over {len(scored)} candidates"
        )


@app.command()
def tree(
    search: str = typer.Argument("latest"),
    out: Path = typer.Option(None, help="Output image path (.png/.svg/.pdf); default <search>/tree.png"),
):
    """Render the search's exploration tree (which candidates were created,
    built upon, or pruned) to an image."""
    try:
        from hillclimb.viz import render_tree
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb tree` needs the TUI extra: pip install 'rebase-hillclimb[tui]'"
        ) from exc

    config = Config.load()
    search_dir = resolve_search_dir(config, search)
    meta = load_search_meta(search_dir)
    lower = bool(meta.lower_is_better) if meta else False
    journal = Journal(search_dir / "journal.jsonl")
    out_path = out or (search_dir / "tree.png")
    if meta is not None:
        title = f"{meta.problem_id}  ({meta.model}, budget {meta.budget_s}s)"
    else:
        title = search_dir.name
    render_tree(journal, lower, out_path, title)
    typer.echo(f"Wrote {out_path} ({len(journal.candidates)} candidates)")


@app.command()
def watch():
    """Live TUI: runs, searches, candidates, and selected-candidate details.
    Keys: enter=open/details, esc=close/back, +/-=resize details, s=stop search,
    x=prune candidate, q=quit."""
    try:
        from hillclimb.watch import WatchApp
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb watch` needs the TUI extra: pip install 'rebase-hillclimb[tui]'"
        ) from exc

    WatchApp(Config.load()).run()


@app.command()
def smoke(
    target: str = typer.Argument("circle-packing"),
    model: str = typer.Option(None),
):
    """One real DRAFT call through the claude-code backend, then execute and
    report — verifies auth, JSON field names, and the filesystem contract."""
    config = Config.load(backend="claude-code", model=model)
    problem = load_problem(target, config)
    version = subprocess.run(["claude", "-v"], capture_output=True, text=True).stdout.strip()
    typer.echo(f"claude version: {version}")
    run_id = f"smoke-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = create_run_dir(config.paths.runs_dir, run_id)
    write_run_meta(
        run_dir,
        RunMeta(
            run_id=run_id,
            name=run_id,
            kind="problem",
            target=target,
            problem_ids=[problem.problem_id],
        ),
    )
    search_dir = create_search(config, problem, run_dir, run_id, total_s=1800)
    journal = Journal(search_dir / "journal.jsonl")
    searcher = GreedySearcher(
        problem=problem,
        config=config,
        journal=journal,
        backend=get_backend("claude-code"),
        executor=LocalExecutor(ensure_runtime_venv(config)),
        budget=BudgetManager(1800, stop_margin_s=0),
        search_dir=search_dir,
        log=typer.echo,
        holdout=_build_holdout(config, problem, search_dir),
    )
    candidate = searcher.run_operator("draft", None)
    trial = candidate.last_trial
    typer.echo(f"\ncandidate:   {candidate.candidate_id} status={candidate.status}")
    typer.echo(f"val_score:   {candidate.val_score}")
    typer.echo(f"holdout:     {candidate.holdout_score} (error: {trial.holdout_error if trial else '-'})")
    typer.echo(f"session_id:  {candidate.backend.session_id}")
    typer.echo(f"cost_usd:    {candidate.backend.cost_usd}")
    typer.echo(f"num_turns:   {candidate.backend.num_turns}")
    typer.echo(f"error_kind:  {candidate.backend.error_kind}")
    typer.echo(f"raw output:  {Path(candidate.workspace) / 'agent_raw.json'}")
    if candidate.backend.session_id is None and candidate.backend.error_kind is None:
        typer.echo("WARNING: session_id not parsed — check agent_raw.json for actual field names")


if __name__ == "__main__":
    app()
