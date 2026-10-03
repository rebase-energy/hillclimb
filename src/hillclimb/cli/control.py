"""`hillclimb ps|stop|prune|kill|reset`: controlling live engines."""

from __future__ import annotations

import os

import typer

from hillclimb.cli import common
from hillclimb.cli._app import app
from hillclimb.cli.common import _m, fail, say, warn
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
            common.say_no_hillclimb_dir(exc)
            raise typer.Exit(1) from exc
        orphans = orphan_engines()
        if not orphans:
            common.say_no_hillclimb_dir(exc)
            say("No orphaned engines running either.")
            raise typer.Exit(1)
        say("[head]No hillclimb.yaml found[/], but engines whose hillclimb dir was deleted are still running:")
        for engine in orphans:
            say(f"  pid [path]{engine.pid}[/]  [note](was {_m(engine.hillclimb_dir)})[/]")
        forced = kill_engines(orphans)
        say(
            f"[head]Terminated[/] {len(orphans)} engine process group(s) with their coding agents and verifiers"
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
        say("[head]No running searches.[/]")
        raise typer.Exit(1)
    return store, running


@app.command()
def ps():
    """Every process hillclimb is responsible for on this machine.

    One block per live engine (`hillclimb run`), with its coding agents, verifiers
    and their children nested underneath. Engines whose hillclimb dir has
    been deleted are tagged `orphan` — `hillclimb stop --all` reaps those.
    """
    from hillclimb.harness.orphans import engine_trees, process_table

    table = process_table()
    trees = engine_trees(table)
    if not trees:
        say("[head]No hillclimb engines running.[/]")
        return
    total = 0
    for engine, kids in trees:
        proc = table[engine.pid]
        where = str(engine.hillclimb_dir) if engine.hillclimb_dir else "?"
        tag = "  [orphan: dir deleted]" if engine.hillclimb_dir and not engine.hillclimb_dir.exists() else ""
        argv = proc.command.split("hillclimb.cli run", 1)[-1].strip()
        say(f"[head]engine pid {proc.pid}[/]  up {_m(proc.elapsed)}  run [cmd]{_m(argv)}[/]")
        say(f"  dir [path]{_m(where)}[/][warn]{_m(tag)}[/]")
        for kid in kids:
            role = (
                "agent" if "claude -p" in kid.command or "claude --" in kid.command
                else "verifier" if "verifier.sh" in kid.command
                else "child"
            )
            say(
                f"  {role:8} pid {kid.pid:<6} cpu {kid.cpu:5.1f}%  mem {kid.rss_mb:6.0f}M  "
                f"up {_m(kid.elapsed):>8}  [note]{_m(kid.command[:70])}[/]"
            )
        total += 1 + len(kids)
    cpu = sum(table[e.pid].cpu for e, _ in trees) + sum(k.cpu for _, kids in trees for k in kids)
    mem = sum(table[e.pid].rss_mb for e, _ in trees) + sum(k.rss_mb for _, kids in trees for k in kids)
    say(f"[head]{len(trees)} engine(s)[/], {total} processes, {cpu:.0f}% cpu, {mem:.0f}M rss")


@app.command()
def stop(
    search: str = typer.Argument("latest"),
    all_: bool = typer.Option(False, "--all", help="Stop every running search"),
    graceful: bool = typer.Option(
        False, "--graceful", "-g", help="Let the operators in flight finish and be scored first"
    ),
):
    """Stop a running engine now; it can be resumed.

    The operators in flight are aborted within about a second — their coding agents
    and verifiers are killed and their candidates journaled as abandoned (the
    tokens they spent are not recovered). `--graceful` instead starts no new
    work and parks once the operators in flight have finished and been
    scored. Resume later with `hillclimb resume`. `--all` stops every running
    search (e.g. a parallel run). An engine that does not respond: `hillclimb kill`.
    """
    config = _load_config_or_reap_orphans(all_)
    store, targets = _search_targets(config, search, all_)
    for record in targets:
        ref = record.ref
        outcome = request_stop(store, record.key, source="cli", graceful=graceful)
        if outcome is None:
            say(f"[head]Search [path]{_m(ref)}[/] is {_m(record.state)}[/]; nothing to stop.")
            raise typer.Exit(1)
        hint = (
            "(drop [cmd]--graceful[/] to abort them now)"
            if graceful
            else "(use [cmd]--graceful[/] to let them finish)"
        )
        say(f"[head]{_m(outcome)}[/] [note]{hint}[/]")
    say(f"Resume with: [cmd]hillclimb resume {_m(targets[0].ref if len(targets) == 1 else '--all')}[/]")


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
        fail(f"Cannot prune: {_m(exc)}")
        raise typer.Exit(1)
    say(f"[head]{_m(outcome)}[/]")


@app.command()
def kill(
    search: str = typer.Argument("latest"),
    all_: bool = typer.Option(False, "--all", help="Kill every running search"),
    grace: float = typer.Option(5.0, "--grace", help="Seconds between SIGTERM and SIGKILL"),
):
    """Last resort for an engine that does not respond to `hillclimb stop`.

    SIGTERMs the engine with its coding agents and verifiers, then SIGKILLs whatever
    is still alive after `--grace` seconds. The search stays resumable: a
    candidate left pending is recovered as abandoned on resume. `--all` kills
    every running search.
    """
    from hillclimb.harness.orphans import Engine, kill_engines, live_engines
    from hillclimb.harness.oscompat import IS_WINDOWS

    config = _load_config_or_reap_orphans(all_)
    store, targets = _search_targets(config, search, all_)
    engines: dict[int, Engine] = {}
    listed = {engine.pid: engine for engine in live_engines()}
    for record in targets:
        ref = record.ref
        state = record.state
        if state != "running":
            say(f"[head]Search [path]{_m(ref)}[/] is {_m(state)}[/]; nothing to kill.")
            raise typer.Exit(1)
        pid = store.read_status(record.key).pid
        if pid is None:
            continue
        engine = listed.get(pid)
        if engine is None:  # not recognizable as an engine in ps: its own group
            try:
                pgid = pid if IS_WINDOWS else os.getpgid(pid)
            except ProcessLookupError:
                continue  # died since the status was read
            engine = Engine(pid=pid, pgid=pgid, hillclimb_dir=None)
        engines[pid] = engine
    forced = kill_engines(list(engines.values()), grace_s=grace)
    for record in targets:
        say(f"[head]Killed[/] engine of [path]{_m(record.ref)}[/].")
    if forced:
        say(f"[note]{len(forced)} engine(s) ignored SIGTERM and needed SIGKILL.[/]")
    say(f"Resume with: [cmd]hillclimb resume {_m(targets[0].ref if len(targets) == 1 else '--all')}[/]")


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation"),
):
    """Kill every engine of THIS hillclimb dir and delete what hillclimb made in it.

    The hillclimb dir is the one found from the current directory (or
    `HILLCLIMB_DIR`). Only engines pinned to that exact dir are signalled —
    their coding agents and verifiers go with them — then hillclimb.yaml and the
    folders beside it that hillclimb owns (problems/, runs/, knowledge/,
    climbers/, experiments/, the sqlite store) are removed. Anything else in
    the folder — your code, .env, .gitignore — stays. Searches of other
    folders on the machine are untouched. `runs_dir` or `problems_dir`
    configured outside the hillclimb dir are left in place and reported.
    """
    import shutil

    from hillclimb.harness.orphans import engines_for, kill_engines, live_engines

    config = common.load_config()
    root = config.hillclimb_dir
    if root is None:  # pragma: no cover - Config.load always sets it via discovery
        fail("No hillclimb dir to reset.")
        raise typer.Exit(1)
    engines = live_engines()
    mine = engines_for(root, engines)
    unknown = [e for e in engines if e.hillclimb_dir is None]

    owned = common.owned_paths(root, config)
    say(f"[head]Will delete[/] from [path]{_m(root)}[/]:")
    for path in owned:
        say(f"  [path]{_m(path.relative_to(root))}{'/' if path.is_dir() else ''}[/]")
    if mine:
        say(f"and terminate {len(mine)} engine(s) running against it [note](with their coding agents and verifiers)[/]:")
        for engine in mine:
            say(f"  pid [path]{engine.pid}[/]")
    else:
        say("No engines are running against it.")
    if unknown:
        say(
            f"[note]Note: {len(unknown)} engine(s) whose hillclimb dir could not be read will be left alone: "
            + ", ".join(f"pid {e.pid}" for e in unknown)
            + "[/]"
        )
    outside = []
    for label, path in (("runs_dir", config.paths.runs_dir), ("problems_dir", config.paths.problems_dir)):
        if path.exists() and not path.resolve().is_relative_to(root.resolve()):
            outside.append((label, path))
    for label, path in outside:
        say(f"[note]Note: {_m(label)} [path]{_m(path)}[/] lives outside the hillclimb dir and will be left in place.[/]")
    if not yes and not typer.confirm("Proceed?", default=False):
        say("[head]Aborted.[/]")
        raise typer.Exit(1)

    if mine:
        forced = kill_engines(mine)
        say(
            f"[head]Terminated[/] {len(mine)} engine process tree(s)"
            + (f"; {len(forced)} needed SIGKILL." if forced else ".")
        )
    for path in owned:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    say(f"[head]Reset[/] [path]{_m(root)}[/] [note](it is no longer a hillclimb dir)[/]")
