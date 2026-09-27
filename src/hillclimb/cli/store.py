"""`hillclimb store …`: the record store behind the cross-run views."""

from __future__ import annotations

import typer

from hillclimb.cli import common
from hillclimb.cli._app import HillclimbGroup, app
from hillclimb.cli.common import _m, say
from hillclimb.harness.journal import Journal

store_app = typer.Typer(
    cls=HillclimbGroup,
    help="The record store behind the cross-run views (`store.backend` in config.yaml: files | sqlite)",
)


app.add_typer(store_app, name="store")


@store_app.command("sync")
def store_sync():
    """Import the hillclimb folder's searches into the configured store.

    Searches the store already holds are left alone, so this is safe to
    repeat. Run it after switching `store.backend` to `sqlite` so history
    written as files shows up in the chart and best-ever views.
    """
    from hillclimb.harness.store import FileDataStore, open_store, sync_store

    config = common.load_config()
    if config.store.backend == "files":
        say("[head]store.backend is `files`[/]: the hillclimb folder is the store, nothing to import")
        return
    store = open_store(config)
    try:
        counts = sync_store(FileDataStore(config.paths.runs_dir), store)
    finally:
        store.close()
    say(
        f"[head]imported[/] {counts['runs']} run(s), {counts['searches']} search(es), "
        f"{counts['records']} journal record(s) into [path]{_m(config.store.sqlite_path)}[/]"
    )


@store_app.command("searches")
def store_searches(
    problem: str | None = typer.Option(None, "--problem", help="Only searches on this problem key"),
):
    """List the searches the store knows, best score per search."""
    from hillclimb.harness.direction import better
    from hillclimb.harness.store import open_store

    config = common.load_config()
    store = open_store(config)
    try:
        records = store.searches(problem_key=problem)
        if not records:
            say("[head]no searches recorded[/]" + (f" for [path]{_m(problem)}[/]" if problem else ""))
            return
        say(f"[head]{'search':40} {'problem':24} {'state':8} {'best':>12}  run[/]")
        for record in records:
            best = None
            for cand in Journal(store.journal(record.key)).candidates.values():
                if cand.pruned or cand.val_score is None:
                    continue
                if best is None or better(cand.val_score, best, record.meta.higher_is_better):
                    best = cand.val_score
            shown = f"{best:.6g}" if best is not None else "-"
            say(
                f"[path]{_m(record.ref):40}[/] {_m(record.meta.problem_key):24} {_m(record.state):8} "
                f"{shown:>12}  {_m(record.run_name)}"
            )
    finally:
        store.close()
