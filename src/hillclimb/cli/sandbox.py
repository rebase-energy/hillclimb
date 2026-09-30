"""`hillclimb sandbox …`: the sandbox agents and verifiers run in."""

from __future__ import annotations

import json

import typer

from hillclimb.cli import common
from hillclimb.cli._app import HillclimbGroup, app
from hillclimb.cli.common import _m, fail, say, warn

sandbox_app = typer.Typer(
    cls=HillclimbGroup,
    help="The sandbox agents and solutions run in: what it stops, shown on this machine",
)


app.add_typer(sandbox_app, name="sandbox")

OUTCOME_STYLE = {"blocked": "ok", "allowed": "ok", "skipped": "note"}


@sandbox_app.command("check")
def sandbox_check(
    as_json: bool = typer.Option(False, "--json", help="Print the attempts as JSON"),
):
    """Try to break out of the sandbox, and show what happened.

    Runs a script that behaves like a hostile solution inside the sandbox a
    search uses: it writes outside its folder, reads your keys, connects to
    the internet and signals other processes. Every attempt is listed with
    its outcome. Exit code 1 when one got through, or when there is no
    sandbox to check.
    """
    from hillclimb.config import Config
    from hillclimb.harness import sandbox
    from hillclimb.harness.sandbox_check import as_dicts, run_check

    config = Config.load(require_dir=False)
    try:
        attempts = run_check(config)
        kind = sandbox.backend() if attempts else None
    except sandbox.SandboxUnavailable as exc:
        fail(f"No sandbox: {_m(exc)}")
        raise typer.Exit(1) from exc
    if as_json:
        typer.echo(json.dumps({"sandbox": kind, "attempts": as_dicts(attempts)}, indent=2))
        raise typer.Exit(0 if attempts and all(a.holds for a in attempts) else 1)
    if not attempts:
        reason = "off" if not sandbox.enabled(config) else "none exists for this operating system"
        warn(f"sandbox: {reason} — agents and solutions run with your full user rights")
        raise typer.Exit(1)
    tool = {"seatbelt": "sandbox-exec", "bwrap": "bubblewrap"}[kind]
    say()
    say(f"[head]A hostile script, run inside the sandbox[/] [note]({_m(tool)})[/]")
    group = None
    rows = []
    for item in attempts:
        if item.group != group:
            group = item.group
            rows.append(("", f"[head]{_m(group)}[/]", "", ""))
        style = OUTCOME_STYLE[item.outcome] if item.holds else "bad"
        mark = "[ok]✓[/]" if item.holds and item.outcome != "skipped" else ("[note]·[/]" if item.holds else "[bad]✗[/]")
        rows.append((mark, _m(item.label), f"[{style}]{_m(item.outcome)}[/]",
                     f"[note]{_m(item.detail)}[/]" if item.outcome == "skipped" or not item.holds else ""))
    say()
    common.table([("", None), ("attempt", None), ("outcome", None), ("", None)], rows)
    say()
    broken = [a for a in attempts if not a.holds]
    if broken:
        fail(
            f"{len(broken)} of {len(attempts)} attempts did not end as they should: "
            + "; ".join(f"{a.label} was {a.outcome}" for a in broken)
        )
        raise typer.Exit(1)
    stopped = sum(a.outcome == "blocked" for a in attempts)
    say(f"  [ok]The sandbox holds:[/] {stopped} attempts blocked, and its own folder stays writable.")
    say("  [note]What it does not cover:[/] [path]https://docs.hillclimb.sh/docs/sandbox[/]")
    say()
