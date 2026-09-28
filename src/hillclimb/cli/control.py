"""`hillclimb ps|stop|prune|kill|reset`: controlling live engines."""

from __future__ import annotations

import signal

import typer

from hillclimb.cli import common
from hillclimb.cli._app import app
from hillclimb.cli.common import _m, say
from hillclimb.config import Config
from hillclimb.harness.control import request_prune, request_stop
from hillclimb.harness.store import (
    DataStore,
    SearchRecord,
    open_store,
    running_searches,
)


def _load_config_or_reap_orphans(all_: bool) -> Config:
    """`common.load_config()`, except that `--all` with no hillclimb dir in sight
    falls back to the live engines: the dir was deleted under them, so the
    control queue is gone and the only way to stop them is by signal."""
    from hillclimb.harness.orphans import kill_engines, orphan_engines
    from hillclimb.project import HillclimbDirNotFound

    try:
        return common.load_config(raise_not_found=True)
    except HillclimbDirNotFound as exc:
        if not all_:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        orphans = orphan_engines()
        if not orphans:
            typer.echo(str(exc), err=True)
            typer.echo("No orphaned engines running either.")
            raise typer.Exit(1)
        typer.echo("No hillclimb/ dir found, but engines whose hillclimb dir was deleted are still running:")
        for engine in orphans:
            typer.echo(f"  pid {engine.pid}  (was {engine.hillclimb_dir})")
        forced = kill_engines(orphans)
        typer.echo(
            f"Terminated {len(orphans)} engine process group(s) with their agents and verifiers"
            + (f"; {len(forced)} needed SIGKILL." if forced else ".")
        )
        raise typer.Exit(0)


def _search_targets(config: Config, search: str, all_: bool) -> tuple[DataStore, list[SearchRecord]]:
    """The searches a stop/kill applies to: every running one, or the ref."""
    if not all_:
        store, record = common.open_search(config, search)
        return store, [record]
    store = open_store(config)
    running = running_searches(store)
    if not running:
        typer.echo("No running searches.")
        raise typer.Exit(1)
    return store, running


@app.command()
def ps():
    """Every process hillclimb is responsible for on this machine.

    One block per live engine (`hillclimb run`), with its agents, verifiers
    and their children nested underneath. Engines whose hillclimb dir has
    been deleted are tagged `orphan` — `hillclimb stop --all` reaps those.
    """
    from hillclimb.harness.orphans import engine_trees, process_table

    table = process_table()
    trees = engine_trees(table)
    if not trees:
        typer.echo("No hillclimb engines running.")
        return
    total = 0
    for engine, kids in trees:
        proc = table[engine.pid]
        where = str(engine.hillclimb_dir) if engine.hillclimb_dir else "?"
        tag = "  [orphan: dir deleted]" if engine.hillclimb_dir and not engine.hillclimb_dir.exists() else ""
        argv = proc.command.split("hillclimb.cli run", 1)[-1].strip()
        typer.echo(f"engine pid {proc.pid}  up {proc.elapsed}  run {argv}")
        typer.echo(f"  dir {where}{tag}")
        for kid in kids:
            role = (
                "agent" if "claude -p" in kid.command or "claude --" in kid.command
                else "verifier" if "verifier.sh" in kid.command
                else "child"
            )
            typer.echo(
                f"  {role:8} pid {kid.pid:<6} cpu {kid.cpu:5.1f}%  mem {kid.rss_mb:6.0f}M  "
                f"up {kid.elapsed:>8}  {kid.command[:70]}"
            )
        total += 1 + len(kids)
    cpu = sum(table[e.pid].cpu for e, _ in trees) + sum(k.cpu for _, kids in trees for k in kids)
    mem = sum(table[e.pid].rss_mb for e, _ in trees) + sum(k.rss_mb for _, kids in trees for k in kids)
    typer.echo(f"{len(trees)} engine(s), {total} processes, {cpu:.0f}% cpu, {mem:.0f}M rss")


@app.command()
def stop(
    search: str = typer.Argument("latest"),
    all_: bool = typer.Option(False, "--all", help="Stop every running search"),
):
    """Gracefully stop a running engine.

    It finishes the current operator call, then parks. Resume later with
    `hillclimb resume`. `--all` stops every running search (e.g. the demo).
    """
    config = _load_config_or_reap_orphans(all_)
    store, targets = _search_targets(config, search, all_)
    for record in targets:
        ref = record.ref
        outcome = request_stop(store, record.key, source="cli")
        if outcome is None:
            typer.echo(f"Search {ref} is {record.state}; nothing to stop.")
            raise typer.Exit(1)
        typer.echo(f"{outcome} (use `hillclimb kill {ref}` to interrupt now)")


@app.command()
def prune(
    search: str,
    candidate_id: str,
    reason: str = typer.Option("", help="Why this branch is being cut (recorded in the journal)"),
):
    """Prune a candidate and its whole subtree.

    The engine stops building on this lineage and it is excluded from
    selection. Statuses and scores stay visible in status/tree output.
    """
    config = common.load_config()
    store, record = common.open_search(config, search)
    higher = bool(record.meta.higher_is_better)
    try:
        outcome = request_prune(
            store,
            record.key,
            candidate_id,
            higher_is_better=higher,
            selection_mode=config.holdout.selection,
            reason=reason,
            source="cli",
        )
    except ValueError as exc:
        typer.echo(f"Cannot prune: {exc}")
        raise typer.Exit(1)
    typer.echo(outcome)


@app.command()
def kill(
    search: str = typer.Argument("latest"),
    all_: bool = typer.Option(False, "--all", help="Kill every running search"),
):
    """SIGTERM a running engine; it can be resumed.

    State is finalized on the way out. For a graceful stop that lets the
    current operator finish, use `hillclimb stop`. `--all` kills every
    running search.
    """
    config = _load_config_or_reap_orphans(all_)
    store, targets = _search_targets(config, search, all_)
    for record in targets:
        ref = record.ref
        state = record.state
        if state != "running":
            typer.echo(f"Search {ref} is {state}; nothing to kill.")
            raise typer.Exit(1)
        engine_pid = store.read_status(record.key).pid
        from hillclimb.harness.oscompat import signal_pid

        signal_pid(engine_pid, signal.SIGTERM)
        typer.echo(f"Sent SIGTERM to engine pid {engine_pid} ({ref}).")
        typer.echo(f"Resume with: hillclimb resume {ref}")


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation"),
):
    """Kill every engine of THIS hillclimb dir and delete the dir.

    The hillclimb dir is the one found from the current directory (or
    `HILLCLIMB_DIR`). Only engines pinned to that exact dir are signalled —
    their agents and verifiers go with them — then the folder is removed.
    Searches of other folders on the machine are untouched. `runs_dir` or
    `problems_dir` configured outside the hillclimb dir are left in place and
    reported.
    """
    import shutil

    from hillclimb.harness.orphans import engines_for, kill_engines, live_engines

    config = common.load_config()
    root = config.hillclimb_dir
    if root is None:  # pragma: no cover - Config.load always sets it via discovery
        typer.echo("No hillclimb dir to reset.", err=True)
        raise typer.Exit(1)
    engines = live_engines()
    mine = engines_for(root, engines)
    unknown = [e for e in engines if e.hillclimb_dir is None]

    say(f"[head]Will delete[/] [path]{_m(root)}[/]")
    if mine:
        typer.echo(f"and terminate {len(mine)} engine(s) running against it (with their agents and verifiers):")
        for engine in mine:
            typer.echo(f"  pid {engine.pid}")
    else:
        typer.echo("No engines are running against it.")
    if unknown:
        typer.echo(
            f"Note: {len(unknown)} engine(s) whose hillclimb dir could not be read will be left alone: "
            + ", ".join(f"pid {e.pid}" for e in unknown)
        )
    outside = []
    for label, path in (("runs_dir", config.paths.runs_dir), ("problems_dir", config.paths.problems_dir)):
        if path.exists() and not path.resolve().is_relative_to(root.resolve()):
            outside.append((label, path))
    for label, path in outside:
        typer.echo(f"Note: {label} {path} lives outside the hillclimb dir and will be left in place.")
    if not yes and not typer.confirm("Proceed?", default=False):
        typer.echo("Aborted.")
        raise typer.Exit(1)

    if mine:
        forced = kill_engines(mine)
        typer.echo(
            f"Terminated {len(mine)} engine process tree(s)"
            + (f"; {len(forced)} needed SIGKILL." if forced else ".")
        )
    shutil.rmtree(root)
    say(f"[head]Deleted[/] [path]{_m(root)}[/]")
