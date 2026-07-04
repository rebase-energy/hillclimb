from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import typer
import yaml

from hillclimb.backends import get_backend
from hillclimb.budget import BudgetManager
from hillclimb.config import Config
from hillclimb.control import clear_stale_stops, request_prune, request_stop
from hillclimb.executor import LocalExecutor
from hillclimb.experiment import ExperimentMeta, write_experiment
from hillclimb.journal import Journal
from hillclimb.problem import (
    ProblemSpec,
    load_problem,
    resolve_target,
    suite_problem_targets,
)
from hillclimb.search import GreedySearcher, ParkedRun, StopRequested
from hillclimb.status import RunStatus, StatusWriter, effective_state, read_status
from hillclimb.workspace import create_run_dir

app = typer.Typer(help="rebase-hillclimb: auto-hillclimbing for verifier-defined problems")


def parse_budget(value: str) -> int:
    match = re.fullmatch(r"(\d+)\s*([hms]?)", value.strip())
    if not match:
        raise typer.BadParameter(f"Cannot parse budget {value!r} (use e.g. 2h, 30m, 3600s)")
    amount, unit = int(match.group(1)), match.group(2)
    return amount * {"h": 3600, "m": 60, "s": 1, "": 1}[unit]


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return slug or "experiment"


def _new_experiment_id(name: str) -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{_slug(name)}"


def ensure_runtime_venv(config: Config) -> Path:
    """Create the solution-script venv on first use."""
    python = config.paths.runtime_python.absolute()
    if python.exists():
        return python
    venv_dir = python.parents[1]
    typer.echo(f"Creating runtime venv at {venv_dir} ...")
    subprocess.run(["uv", "venv", "--python", "3.12", str(venv_dir)], check=True)
    subprocess.run(
        ["uv", "pip", "install", "-r", "runtime-requirements.txt", "--python", str(python)],
        check=True,
    )
    return python


def resolve_run_dir(config: Config, run_id: str | None) -> Path:
    runs_dir = config.paths.runs_dir
    if run_id and run_id != "latest":
        return runs_dir / run_id
    candidates = sorted(
        (d for d in runs_dir.iterdir() if (d / "run.yaml").exists()),
        key=lambda d: d.stat().st_mtime,
    )
    if not candidates:
        raise typer.BadParameter(f"No runs found in {runs_dir}")
    return candidates[-1]


def _build_holdout(config: Config, problem: ProblemSpec, run_dir: Path):
    if not config.holdout.enabled:
        return None
    from hillclimb.holdout import build_data_view

    info = build_data_view(
        problem.data_dir,
        run_dir,
        problem.sample_submission,
        config.holdout.fraction,
        config.holdout.seed,
        override=problem.holdout,
    )
    if info is None:
        typer.echo("holdout: disabled for this problem (no train.csv or targets not inferable)")
    else:
        typer.echo(
            f"holdout: {info.n_holdout} rows hidden "
            f"({info.strategy}; targets: {', '.join(info.target_cols)})"
        )
    return info


def _raise_stop_requested(signum, frame):
    raise StopRequested(f"signal {signal.Signals(signum).name}")


def _execute(config: Config, problem: ProblemSpec, run_dir: Path, budget: BudgetManager) -> None:
    clear_stale_stops(run_dir)
    journal = Journal(run_dir / "journal.jsonl")
    status = StatusWriter(
        run_dir,
        RunStatus(run_id=run_dir.name, state="running", pid=os.getpid()),
        budget=budget,
    )
    status.start_heartbeat()
    signal.signal(signal.SIGTERM, _raise_stop_requested)
    searcher = GreedySearcher(
        problem=problem,
        config=config,
        journal=journal,
        backend=get_backend(config.backend),
        executor=LocalExecutor(ensure_runtime_venv(config)),
        budget=budget,
        run_dir=run_dir,
        log=typer.echo,
        holdout=_build_holdout(config, problem, run_dir),
        status=status,
    )
    try:
        selected = searcher.run()
    except ParkedRun as exc:
        status.finalize("parked", last_error=str(exc)[:500])
        typer.echo(f"\nRate limited: {exc}")
        typer.echo(f"Resume later with: hillclimb resume {run_dir.name}")
        raise typer.Exit(2)
    except (StopRequested, KeyboardInterrupt) as exc:
        status.finalize("stopped", last_error=str(exc)[:500] or None)
        typer.echo("\nStopped.")
        typer.echo(f"Resume with: hillclimb resume {run_dir.name}")
        raise typer.Exit(2)
    except Exception as exc:
        status.finalize("failed", last_error=f"{type(exc).__name__}: {exc}"[:500])
        raise
    status.finalize("done")
    if selected is not None:
        scores = f"val_score={selected.val_score}"
        if selected.holdout_score is not None:
            scores += f", holdout={selected.holdout_score:.5g}"
        typer.echo(
            f"\nDone. Selected candidate {selected.node_id}: {scores} "
            f"({problem.metric_name}, {'lower' if problem.lower_is_better else 'higher'} is better)"
        )
    else:
        typer.echo("\nDone. No scored solution; best/ holds the baseline submission.")
    typer.echo(f"Submission: {run_dir / 'best' / 'submission.csv'}")
    typer.echo(f"Inspect with: hillclimb status {run_dir.name}")


def _run_problem(
    target: str,
    config: Config,
    budget: str | None,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> None:
    problem = load_problem(target, config)
    if experiment_id is None:
        experiment_name = experiment_name or problem.problem_id
        experiment_id = _new_experiment_id(experiment_name)
        write_experiment(
            config.paths.runs_dir,
            ExperimentMeta(
                experiment_id=experiment_id,
                name=experiment_name,
                kind="problem",
                target=target,
                problem_ids=[problem.problem_id],
            ),
        )
    else:
        experiment_name = experiment_name or experiment_id
    total_s = parse_budget(budget) if budget else problem.time_budget_s
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{problem.problem_id}"
    run_dir = create_run_dir(config.paths.runs_dir, run_id)
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "problem": str(problem.problem_dir),
                "problem_id": problem.problem_id,
                "backend": config.backend,
                "model": config.model,
                "metric": problem.metric_name,
                "lower_is_better": problem.lower_is_better,
                "budget_s": total_s,
                "holdout_enabled": config.holdout.enabled,
                "holdout_seed": config.holdout.seed,
                "holdout_fraction": config.holdout.fraction,
                "holdout_strategy": problem.holdout.strategy if problem.holdout else "random",
                "started_at": datetime.now().isoformat(),
            }
        )
    )
    typer.echo(
        f"Run {run_id} (experiment={experiment_name}, problem={problem.problem_id}, "
        f"backend={config.backend}, model={config.model}, budget={total_s}s)"
    )
    _execute(config, problem, run_dir, BudgetManager(total_s, config.budget.stop_margin_s))


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
    experiment_name = name or suite.suite_id
    experiment_id = _new_experiment_id(experiment_name)
    problem_targets = suite_problem_targets(suite, config)
    problem_ids = [load_problem(problem_target, config).problem_id for problem_target in problem_targets]
    write_experiment(
        config.paths.runs_dir,
        ExperimentMeta(
            experiment_id=experiment_id,
            name=experiment_name,
            kind="suite",
            target=target,
            problem_ids=problem_ids,
        ),
    )
    log_dir = config.paths.runs_dir / "suite-logs" / experiment_id
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
            "--experiment-id",
            experiment_id,
            "--experiment-name",
            experiment_name,
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
    typer.echo(f"Experiment {experiment_id}: launched {len(launched)} problem runs")
    for problem_target, pid, log_path in launched:
        typer.echo(f"  pid={pid} {problem_target}  log={log_path}")


@app.command()
def run(
    target: str,
    budget: str = typer.Option(None, help="Wall-clock budget, e.g. 2h / 30m"),
    backend: str = typer.Option(None, help="Operator backend: claude-code | dummy"),
    model: str = typer.Option(None, help="Model for operator calls, e.g. sonnet / opus"),
    holdout: bool = typer.Option(True, "--holdout/--no-holdout", help="Hidden selection holdout"),
    name: str = typer.Option(None, "--name", help="Experiment name shown in the TUI"),
    experiment_id: str = typer.Option(None, "--experiment-id", hidden=True),
    experiment_name: str = typer.Option(None, "--experiment-name", hidden=True),
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
        experiment_id=experiment_id,
        experiment_name=experiment_name or name,
    )


@app.command()
def resume(run_id: str = typer.Argument("latest")):
    """Resume a parked or interrupted run."""
    config = Config.load()
    run_dir = resolve_run_dir(config, run_id)
    meta = yaml.safe_load((run_dir / "run.yaml").read_text())
    config = Config.load(backend=meta["backend"], model=meta["model"])
    config.holdout.enabled = meta.get("holdout_enabled", False)  # legacy runs: off
    config.holdout.seed = meta.get("holdout_seed", config.holdout.seed)
    config.holdout.fraction = meta.get("holdout_fraction", config.holdout.fraction)
    problem = load_problem(meta["problem"], config)
    journal = Journal(run_dir / "journal.jsonl")
    spent = sum(
        (n.backend.agent_duration_s or 0) + (n.execution.duration_s or 0)
        for n in journal.nodes.values()
    )
    typer.echo(f"Resuming {run_dir.name}: {len(journal.nodes)} candidates, ~{int(spent)}s spent")
    _execute(
        config,
        problem,
        run_dir,
        BudgetManager(meta["budget_s"], config.budget.stop_margin_s, spent_s=spent),
    )


@app.command()
def stop(run_id: str = typer.Argument("latest")):
    """Gracefully stop a running engine: it finishes the current operator
    call, then parks. Resume later with `hillclimb resume`."""
    config = Config.load()
    run_dir = resolve_run_dir(config, run_id)
    outcome = request_stop(run_dir, source="cli")
    if outcome is None:
        typer.echo(f"Run {run_dir.name} is {effective_state(run_dir)}; nothing to stop.")
        raise typer.Exit(1)
    typer.echo(f"{outcome} (use `hillclimb kill {run_dir.name}` to interrupt now)")


@app.command()
def prune(
    run_id: str,
    node_id: str,
    reason: str = typer.Option("", help="Why this branch is being cut (recorded in the journal)"),
):
    """Prune a candidate and its whole subtree: the engine stops building on this
    lineage and it is excluded from selection. Statuses and scores stay
    visible in status/tree output."""
    config = Config.load()
    run_dir = resolve_run_dir(config, run_id)
    meta = yaml.safe_load((run_dir / "run.yaml").read_text())
    lower = bool(meta.get("lower_is_better", False))
    try:
        outcome = request_prune(
            run_dir,
            node_id,
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
def kill(run_id: str = typer.Argument("latest")):
    """SIGTERM a running engine; it finalizes state and can be resumed.
    For a graceful stop that lets the current operator finish, use
    `hillclimb stop`."""
    config = Config.load()
    run_dir = resolve_run_dir(config, run_id)
    state = effective_state(run_dir)
    if state != "running":
        typer.echo(f"Run {run_dir.name} is {state}; nothing to kill.")
        raise typer.Exit(1)
    engine_pid = read_status(run_dir).pid
    os.kill(engine_pid, signal.SIGTERM)
    typer.echo(f"Sent SIGTERM to engine pid {engine_pid} ({run_dir.name}).")
    typer.echo(f"Resume with: hillclimb resume {run_dir.name}")


@app.command()
def status(run_id: str = typer.Argument("latest")):
    """Show the candidate tree of a problem run."""
    config = Config.load()
    run_dir = resolve_run_dir(config, run_id)
    journal = Journal(run_dir / "journal.jsonl")
    run_status = read_status(run_dir)
    state = effective_state(run_dir)
    if run_status is not None:
        remaining = int(run_status.budget.remaining_s)
        line = f"state={state}  budget: {int(run_status.budget.spent_s)}s spent / {remaining}s left"
        if run_status.current is not None:
            line += (
                f"  current candidate: {run_status.current.node_id} "
                f"({run_status.current.operator}/{run_status.current.phase})"
            )
        typer.echo(line)
    typer.echo(f"Run {run_dir.name} — {len(journal.nodes)} candidates")
    for node in journal.nodes.values():
        score = f"{node.val_score:.5f}" if node.val_score is not None else "-"
        hold = f" hold={node.holdout_score:.5f}" if node.holdout_score is not None else ""
        marks = (" *SELECTED*" if node.is_selected else "") + (" *best-val*" if node.is_best else "")
        if node.pruned:
            marks += " *PRUNED*"
        parent = f" <- {node.parent_id}" if node.parent_id else ""
        typer.echo(
            f"  {node.node_id} {node.operator:<9} {node.status:<9} val={score}{hold}{marks}{parent}  {node.summary[:70]}"
        )
    scored = [n for n in journal.nodes.values() if n.val_score is not None and n.holdout_score is not None]
    if scored:
        gaps = [abs(n.val_score - n.holdout_score) for n in scored]
        typer.echo(f"val→holdout gap: mean {sum(gaps)/len(gaps):.5g}, max {max(gaps):.5g} over {len(scored)} candidates")


@app.command()
def tree(
    run_id: str = typer.Argument("latest"),
    out: Path = typer.Option(None, help="Output image path (.png/.svg/.pdf); default <run>/tree.png"),
):
    """Render the run's exploration tree (which candidates were created,
    built upon, or pruned) to an image."""
    from hillclimb.viz import render_tree

    config = Config.load()
    run_dir = resolve_run_dir(config, run_id)
    meta = yaml.safe_load((run_dir / "run.yaml").read_text())
    lower = bool(meta.get("lower_is_better", False))
    journal = Journal(run_dir / "journal.jsonl")
    out_path = out or (run_dir / "tree.png")
    title = f"{meta.get('problem_id', '?')}  ({meta.get('model', '?')}, budget {meta.get('budget_s', '?')}s)"
    render_tree(journal, lower, out_path, title)
    typer.echo(f"Wrote {out_path} ({len(journal.nodes)} candidates)")


@app.command()
def watch():
    """Live TUI: experiments, problem runs, candidates, and selected-candidate details.
    Keys: enter=open/details, esc=close/back, +/-=resize details, s=stop run,
    x=prune candidate, q=quit."""
    from hillclimb.watch import WatchApp

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
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": run_id,
                "problem": str(problem.problem_dir),
                "problem_id": problem.problem_id,
                "backend": "claude-code",
                "model": config.model,
                "metric": problem.metric_name,
                "lower_is_better": problem.lower_is_better,
                "budget_s": 1800,
                "started_at": datetime.now().isoformat(),
            }
        )
    )
    journal = Journal(run_dir / "journal.jsonl")
    searcher = GreedySearcher(
        problem=problem,
        config=config,
        journal=journal,
        backend=get_backend("claude-code"),
        executor=LocalExecutor(ensure_runtime_venv(config)),
        budget=BudgetManager(1800, stop_margin_s=0),
        run_dir=run_dir,
        log=typer.echo,
        holdout=_build_holdout(config, problem, run_dir),
    )
    node = searcher.run_operator("draft", None)
    typer.echo(f"\ncandidate:   {node.node_id} status={node.status}")
    typer.echo(f"val_score:   {node.val_score}")
    typer.echo(f"holdout:     {node.holdout_score} (error: {node.execution.holdout_error})")
    typer.echo(f"session_id:  {node.backend.session_id}")
    typer.echo(f"cost_usd:    {node.backend.cost_usd}")
    typer.echo(f"num_turns:   {node.backend.num_turns}")
    typer.echo(f"error_kind:  {node.backend.error_kind}")
    typer.echo(f"raw output:  {Path(node.workspace) / 'agent_raw.json'}")
    if node.backend.session_id is None and node.backend.error_kind is None:
        typer.echo("WARNING: session_id not parsed — check agent_raw.json for actual field names")


if __name__ == "__main__":
    app()
