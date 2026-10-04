"""`hillclimb ps|top|stop|prune|kill|reset`: controlling live engines."""

from __future__ import annotations

import os

import typer

from hillclimb.cli import common
from hillclimb.cli._app import app
from hillclimb.cli.common import _m, fail, say, warn
from hillclimb.config import Config
from hillclimb import terms
from hillclimb.terms import ENGINE
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
            say(f"No orphaned {ENGINE.pl} running either.")
            raise typer.Exit(1)
        say(f"[head]No hillclimb.yaml found[/], but {ENGINE.pl} whose hillclimb dir was deleted are still running:")
        for engine in orphans:
            say(f"  pid [path]{engine.pid}[/]  [note](was {_m(engine.hillclimb_dir)})[/]")
        forced = kill_engines(orphans)
        say(
            f"[head]Terminated[/] {len(orphans)} {ENGINE} process group(s) with their coding agents and verifiers"
            + (f"; {len(forced)} needed SIGKILL." if forced else ".")
        )
        raise typer.Exit(0)


def _stop_orphaned_children(records) -> int:
    """Stop what dead engines left running for these searches (their coding
    agents and verifiers, still billing). Returns how many were stopped."""
    from hillclimb.harness.orphans import stop_orphaned_children

    total = 0
    for record in records:
        if record.state == "running":
            continue  # a live engine stops its own
        stopped = stop_orphaned_children(record.search_dir)
        if stopped:
            names = ", ".join(sorted({entry.get("program") or "?" for entry in stopped}))
            say(f"[head]Stopped[/] {len(stopped)} process(es) left running after [path]{_m(record.ref)}[/] died [note]({_m(names)})[/]")
            total += len(stopped)
    return total


def _search_targets(config: Config, search: str, all_: bool) -> tuple[DataStore, list[SearchRecord]]:
    """The searches a stop/kill applies to: every running one, or the ref.
    Searches whose engine died first have what it left running stopped."""
    if not all_:
        store, record = common.open_search(config, search)
        if record.state != "running" and _stop_orphaned_children([record]):
            raise typer.Exit(0)
        return store, [record]
    store = open_store(config)
    _stop_orphaned_children(store.searches())
    running = running_searches(store)
    if not running:
        say("[head]No running searches.[/]")
        raise typer.Exit(1)
    return store, running


def _ps_frame(width: int, *, max_height: int | None = None, footer: str | None = None, table=None):
    """One `hillclimb ps` frame: the boxed machine + processes view."""
    from hillclimb.tui.machine import scan
    from hillclimb.tui.psview import render_ps

    try:
        slots = Config.load(require_dir=False).concurrency.effective_machine_max_agents()
    except Exception:
        slots = None
    # compute only: no search records (that is `watch`'s), the engine as the
    # root of its own tree
    machine, engines = scan(table=table, agent_slots=slots, read_searches=False, root_guides=True)
    return render_ps(machine, engines, width, max_height=max_height, footer=footer)


def _watch_ps(interval: float) -> None:
    """Redraw the frame in place every `interval` seconds until ctrl+c —
    plain terminal output, like `watch hillclimb ps`, sized to the terminal
    so it stays on one screen; `hillclimb top` is the interactive one."""
    import time

    from rich.live import Live

    console = common._console()
    footer = f"[note] every {interval:g}s · ctrl+c to quit · [cmd]hillclimb top[/] to sort, stop or open them[/]"

    def frame():
        return _ps_frame(console.width, max_height=console.height, footer=footer)

    try:
        with Live(frame(), console=console, auto_refresh=False, vertical_overflow="crop") as live:
            while True:
                time.sleep(interval)
                live.update(frame(), refresh=True)
    except KeyboardInterrupt:
        pass


@app.command()
@terms.doc
def ps(
    watch: bool = typer.Option(False, "--watch", "-w", help="Keep redrawing it in place until ctrl+c"),
    interval: float = typer.Option(1.0, "--interval", "-n", help="Seconds between redraws with --watch"),
):
    """Every process hillclimb is responsible for on this machine — compute only.

    One box, sized to the terminal: the machine at a glance ({engines}, coding
    agents against the machine's slot cap, cpu, memory) above one
    process table — each {engine} on a row of its own (its problem and
    hillclimb dir), its processes nested under it. A hillclimb command
    computing in a terminal (`verify`, `grade`, `run --no-detach`) gets a
    block of its own the same way. Roles: agent (a coding
    agent), tool (a command the coding agent runs itself, e.g. trying its
    solution), mcp (a server it loaded from your own Claude config),
    verifier (a scored run) and solution (what the verifier runs). {Engines}
    whose hillclimb dir has been deleted are marked orphan — `hillclimb stop
    --all` reaps those. `--watch` redraws it every second; search progress
    is `hillclimb watch`, and `hillclimb top` sorts, stops and opens them.
    """
    if watch:
        if interval <= 0:
            fail("--interval must be positive.")
            raise typer.Exit(1)
        _watch_ps(interval)
        return
    console = common._console()
    console.print(_ps_frame(console.width))


@app.command()
@terms.doc
def top():
    """The control pane for every hillclimb process on this machine.

    `hillclimb ps`, live and with controls: the machine at a glance
    ({engines}, coding agents against the machine's slot cap, cpu, memory)
    above one table where each {engine} heads its own process tree — coding
    agents, their tool shells and MCP servers, verifiers and the solution
    they score. Works from any folder. Keys act on the highlighted row:
    s=stop its {engine} (resumable), g=stop gracefully, k=kill that process
    and what it started (on {an_engine} row: the whole {engine}), o=change the
    order of the {engines}, r=reverse it, q=quit. Search progress is
    `hillclimb watch`.
    """
    try:
        from hillclimb.tui.top import TopApp
    except ModuleNotFoundError as exc:
        raise typer.BadParameter("`hillclimb top` needs the TUI extra: pip install 'hillclimb[tui]'") from exc
    try:
        slots = Config.load(require_dir=False).concurrency.effective_machine_max_agents()
    except Exception:
        slots = None
    TopApp(agent_slots=slots).run()


@app.command()
@terms.doc
def stop(
    search: str = typer.Argument("latest"),
    all_: bool = typer.Option(False, "--all", help="Stop every running search"),
    graceful: bool = typer.Option(
        False, "--graceful", "-g", help="Let the operators in flight finish and be scored first"
    ),
):
    """Stop a running {engine} now; it can be resumed.

    The operators in flight are aborted within about a second — their coding agents
    and verifiers are killed and their candidates journaled as abandoned (the
    tokens they spent are not recovered). `--graceful` instead starts no new
    work and parks once the operators in flight have finished and been
    scored. Resume later with `hillclimb resume`. `--all` stops every running
    search (e.g. a parallel run). One that does not respond: `hillclimb kill`.
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
@terms.doc
def prune(
    search: str,
    candidate_id: str,
    reason: str = typer.Option("", help="Why this branch is being cut (recorded in the journal)"),
):
    """Prune a candidate and its whole subtree.

    The {engine} stops building on this lineage and it is excluded from
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
@terms.doc
def kill(
    search: str = typer.Argument("latest"),
    all_: bool = typer.Option(False, "--all", help="Kill every running search"),
    grace: float = typer.Option(5.0, "--grace", help="Seconds between SIGTERM and SIGKILL"),
):
    """Last resort for {an_engine} that does not respond to `hillclimb stop`.

    SIGTERMs its process with its coding agents and verifiers, then SIGKILLs whatever
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
        say(f"[head]Killed[/] [path]{_m(record.ref)}[/].")
    if forced:
        say(f"[note]{ENGINE.n(len(forced))} ignored SIGTERM and needed SIGKILL.[/]")
    say(f"Resume with: [cmd]hillclimb resume {_m(targets[0].ref if len(targets) == 1 else '--all')}[/]")


@app.command()
@terms.doc
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation"),
    runs: bool = typer.Option(
        False, "--runs",
        help="Delete only what searches produced (runs/, the sqlite store, knowledge/) and keep the "
        "hillclimb dir: hillclimb.yaml, problems/ and climbers/ stay",
    ),
):
    """Kill every {engine} of THIS hillclimb dir and delete what hillclimb made in it.

    The hillclimb dir is the one found from the current directory (or
    `HILLCLIMB_DIR`). Only {engines} pinned to that exact dir are signalled —
    their coding agents and verifiers go with them — then hillclimb.yaml, the
    sqlite store and the folders hillclimb created beside it (each carries a
    hidden `.hillclimb` file: problems/, runs/, knowledge/, climbers/) are
    removed. A folder of the same name without the marker is yours and stays,
    and so does everything else in the folder — your code, .env, .gitignore.
    Without --yes it lists what goes and what stays first. Searches of other
    folders on the machine are untouched. `runs_dir` or `problems_dir`
    configured outside the hillclimb dir are left in place and reported.

    `--runs` starts the folder's searches over without setting it up again:
    the {engines} are killed and runs/, the sqlite store and knowledge/ (what
    the searches learned, and any papers distilled into it) are removed,
    while hillclimb.yaml, problems/ and climbers/ stay.
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

    owned, kept = common.owned_paths(root, config, runs_only=runs)
    say(f"[head]Will delete[/] from [path]{_m(root)}[/] [note](what hillclimb created)[/]:")
    for path in owned:
        say(f"  [path]{_m(path.relative_to(root))}{'/' if path.is_dir() else ''}[/]")
    if kept:
        say("[head]Will keep[/] [note](no .hillclimb marker: yours, or made before hillclimb marked its folders)[/]:")
        for path in kept:
            say(f"  [path]{_m(path.relative_to(root))}/[/]")
    if mine:
        say(f"and terminate {ENGINE.n(len(mine))} running against it [note](with their coding agents and verifiers)[/]:")
        for engine in mine:
            say(f"  pid [path]{engine.pid}[/]")
    else:
        say(f"No {ENGINE.pl} are running against it.")
    if unknown:
        say(
            f"[note]Note: {ENGINE.n(len(unknown))} whose hillclimb dir could not be read will be left alone: "
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
            f"[head]Terminated[/] {len(mine)} {ENGINE} process tree(s)"
            + (f"; {len(forced)} needed SIGKILL." if forced else ".")
        )
    for path in owned:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    from hillclimb.project import OWNED_MARKER, ensure_owned_dir, is_owned_dir

    if runs:
        if config.paths.runs_dir in owned:
            ensure_owned_dir(config.paths.runs_dir)  # as `hillclimb init` left it
        say(f"[head]Reset the runs[/] of [path]{_m(root)}[/] [note](problems and config kept)[/]")
        return
    left = [p.name for p in root.iterdir() if p.name not in (OWNED_MARKER, ".gitignore")] if root.is_dir() else []
    if is_owned_dir(root) and not left:
        shutil.rmtree(root)  # a hillclimb/ subfolder hillclimb made, now empty of anything else
        say(f"[head]Reset[/]: removed [path]{_m(root)}[/]")
        return
    say(f"[head]Reset[/] [path]{_m(root)}[/] [note](it is no longer a hillclimb dir)[/]")
