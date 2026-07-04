from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import typer
import yaml

from hillclimb.backends import get_backend
from hillclimb.budget import BudgetManager
from hillclimb.config import Config
from hillclimb.executor import LocalExecutor
from hillclimb.grading import grade_submission
from hillclimb.journal import Journal
from hillclimb.search import GreedySearcher, ParkedRun
from hillclimb.task import TaskSpec, load_task
from hillclimb.workspace import create_run_dir

app = typer.Typer(help="rebase-hillclimb: auto-hillclimbing for ML tasks")


def parse_budget(value: str) -> int:
    match = re.fullmatch(r"(\d+)\s*([hms]?)", value.strip())
    if not match:
        raise typer.BadParameter(f"Cannot parse budget {value!r} (use e.g. 2h, 30m, 3600s)")
    amount, unit = int(match.group(1)), match.group(2)
    return amount * {"h": 3600, "m": 60, "s": 1, "": 1}[unit]


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


def _build_holdout(config: Config, task: TaskSpec, run_dir: Path):
    if not config.holdout.enabled:
        return None
    from hillclimb.holdout import build_data_view

    info = build_data_view(
        task.data_dir,
        run_dir,
        task.sample_submission,
        config.holdout.fraction,
        config.holdout.seed,
    )
    if info is None:
        typer.echo("holdout: disabled for this task (no train.csv or targets not inferable)")
    else:
        typer.echo(f"holdout: {info.n_holdout} rows hidden (targets: {', '.join(info.target_cols)})")
    return info


def _execute(config: Config, task: TaskSpec, run_dir: Path, budget: BudgetManager) -> None:
    journal = Journal(run_dir / "journal.jsonl")
    searcher = GreedySearcher(
        task=task,
        config=config,
        journal=journal,
        backend=get_backend(config.backend),
        executor=LocalExecutor(ensure_runtime_venv(config)),
        budget=budget,
        run_dir=run_dir,
        log=typer.echo,
        holdout=_build_holdout(config, task, run_dir),
    )
    try:
        selected = searcher.run()
    except ParkedRun as exc:
        typer.echo(f"\nRate limited: {exc}")
        typer.echo(f"Resume later with: hillclimb resume {run_dir.name}")
        raise typer.Exit(2)
    if selected is not None:
        scores = f"val_score={selected.val_score}"
        if selected.holdout_score is not None:
            scores += f", holdout={selected.holdout_score:.5g}"
        typer.echo(
            f"\nDone. Selected node {selected.node_id}: {scores} "
            f"({task.metric_name}, {'lower' if task.lower_is_better else 'higher'} is better)"
        )
    else:
        typer.echo("\nDone. No scored solution; best/ holds the baseline submission.")
    typer.echo(f"Submission: {run_dir / 'best' / 'submission.csv'}")
    typer.echo(f"Grade with: hillclimb grade {run_dir.name}")


@app.command()
def run(
    task: str,
    budget: str = typer.Option(None, help="Wall-clock budget, e.g. 2h / 30m"),
    backend: str = typer.Option(None, help="Operator backend: claude-code | dummy"),
    model: str = typer.Option(None, help="Model for operator calls, e.g. sonnet / opus"),
    holdout: bool = typer.Option(True, "--holdout/--no-holdout", help="Hidden selection holdout"),
):
    """Start a hillclimb run on a task."""
    config = Config.load(backend=backend, model=model)
    if not holdout:
        config.holdout.enabled = False
    spec = load_task(task, config)
    total_s = parse_budget(budget) if budget else spec.time_budget_s
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{spec.task_id}"
    run_dir = create_run_dir(config.paths.runs_dir, run_id)
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": run_id,
                "task": task,
                "backend": config.backend,
                "model": config.model,
                "budget_s": total_s,
                "holdout_enabled": config.holdout.enabled,
                "holdout_seed": config.holdout.seed,
                "holdout_fraction": config.holdout.fraction,
                "started_at": datetime.now().isoformat(),
            }
        )
    )
    typer.echo(f"Run {run_id} (backend={config.backend}, model={config.model}, budget={total_s}s)")
    _execute(config, spec, run_dir, BudgetManager(total_s, config.budget.stop_margin_s))


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
    spec = load_task(meta["task"], config)
    journal = Journal(run_dir / "journal.jsonl")
    spent = sum(
        (n.backend.agent_duration_s or 0) + (n.execution.duration_s or 0)
        for n in journal.nodes.values()
    )
    typer.echo(f"Resuming {run_dir.name}: {len(journal.nodes)} nodes, ~{int(spent)}s spent")
    _execute(
        config,
        spec,
        run_dir,
        BudgetManager(meta["budget_s"], config.budget.stop_margin_s, spent_s=spent),
    )


@app.command()
def status(run_id: str = typer.Argument("latest")):
    """Show the solution tree of a run."""
    config = Config.load()
    run_dir = resolve_run_dir(config, run_id)
    journal = Journal(run_dir / "journal.jsonl")
    typer.echo(f"Run {run_dir.name} — {len(journal.nodes)} nodes")
    for node in journal.nodes.values():
        score = f"{node.val_score:.5f}" if node.val_score is not None else "-"
        hold = f" hold={node.holdout_score:.5f}" if node.holdout_score is not None else ""
        marks = (" *SELECTED*" if node.is_selected else "") + (" *best-val*" if node.is_best else "")
        parent = f" <- {node.parent_id}" if node.parent_id else ""
        typer.echo(
            f"  {node.node_id} {node.operator:<9} {node.status:<9} val={score}{hold}{marks}{parent}  {node.summary[:70]}"
        )
    scored = [n for n in journal.nodes.values() if n.val_score is not None and n.holdout_score is not None]
    if scored:
        gaps = [abs(n.val_score - n.holdout_score) for n in scored]
        typer.echo(f"val→holdout gap: mean {sum(gaps)/len(gaps):.5g}, max {max(gaps):.5g} over {len(scored)} nodes")


@app.command()
def tree(
    run_id: str = typer.Argument("latest"),
    out: Path = typer.Option(None, help="Output image path (.png/.svg/.pdf); default <run>/tree.png"),
):
    """Render the run's exploration tree (which experiments were created,
    built upon, or pruned) to an image."""
    from hillclimb.viz import render_tree

    config = Config.load()
    run_dir = resolve_run_dir(config, run_id)
    meta = yaml.safe_load((run_dir / "run.yaml").read_text())
    task_yaml = config.paths.tasks_dir / f"{meta['task']}.yaml"
    lower = yaml.safe_load(task_yaml.read_text())["lower_is_better"] if task_yaml.exists() else False
    journal = Journal(run_dir / "journal.jsonl")
    out_path = out or (run_dir / "tree.png")
    title = f"{meta['task']}  ({meta.get('model', '?')}, budget {meta.get('budget_s', '?')}s)"
    render_tree(journal, lower, out_path, title)
    typer.echo(f"Wrote {out_path} ({len(journal.nodes)} nodes)")


@app.command()
def grade(run_id: str = typer.Argument("latest")):
    """Grade a run's best submission with mlebench."""
    config = Config.load()
    run_dir = resolve_run_dir(config, run_id)
    meta = yaml.safe_load((run_dir / "run.yaml").read_text())
    task_meta = yaml.safe_load((config.paths.tasks_dir / f"{meta['task']}.yaml").read_text())
    submission = run_dir / "best" / "submission.csv"
    report = grade_submission(submission, task_meta["comp_id"], config)
    typer.echo(json.dumps(report, indent=2))


@app.command()
def smoke(
    task: str = typer.Argument("spaceship-titanic"),
    model: str = typer.Option(None),
):
    """One real DRAFT call through the claude-code backend, then execute and
    report — verifies auth, JSON field names, and the filesystem contract."""
    config = Config.load(backend="claude-code", model=model)
    spec = load_task(task, config)
    version = subprocess.run(["claude", "-v"], capture_output=True, text=True).stdout.strip()
    typer.echo(f"claude version: {version}")
    run_id = f"smoke-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = create_run_dir(config.paths.runs_dir, run_id)
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump({"run_id": run_id, "task": task, "backend": "claude-code",
                        "model": config.model, "budget_s": 1800,
                        "started_at": datetime.now().isoformat()})
    )
    journal = Journal(run_dir / "journal.jsonl")
    searcher = GreedySearcher(
        task=spec,
        config=config,
        journal=journal,
        backend=get_backend("claude-code"),
        executor=LocalExecutor(ensure_runtime_venv(config)),
        budget=BudgetManager(1800, stop_margin_s=0),
        run_dir=run_dir,
        log=typer.echo,
        holdout=_build_holdout(config, spec, run_dir),
    )
    node = searcher.run_operator("draft", None)
    typer.echo(f"\nnode:        {node.node_id} status={node.status}")
    typer.echo(f"val_score:   {node.val_score}")
    typer.echo(f"holdout:     {node.holdout_score} (error: {node.execution.holdout_error})")
    typer.echo(f"session_id:  {node.backend.session_id}")
    typer.echo(f"cost_usd:    {node.backend.cost_usd}")
    typer.echo(f"num_turns:   {node.backend.num_turns}")
    typer.echo(f"error_kind:  {node.backend.error_kind}")
    typer.echo(f"raw output:  {Path(node.workspace) / 'agent_raw.json'}")
    if node.backend.session_id is None and node.backend.error_kind is None:
        typer.echo("WARNING: session_id not parsed — check agent_raw.json for actual field names")
