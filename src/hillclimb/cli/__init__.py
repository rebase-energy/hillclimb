"""hillclimb's command line: `hillclimb <command>`, and `python -m hillclimb.cli`
for the engine children. Importing this package registers every command:
`_app` holds the typer app, `common` what commands share, and one module per
command group below — a module missing from the import list has no commands
(tests/test_cli_help.py catches that).
"""

# ruff: isort: skip_file

from __future__ import annotations

import sys

import typer

from hillclimb.cli import common  # noqa: F401 — first, so every command module finds it loaded
from hillclimb.cli._app import (  # noqa: F401 — re-exported
    BANNER_LINES,
    LOGO_LINES,
    WORDMARK_LINES,
    HillclimbGroup,
    app,
    print_banner,
)
from hillclimb.cli.common import legend, next_steps, say  # noqa: F401 — re-exported
from hillclimb.cli import (  # noqa: F401 — importing a module registers its commands on `app`
    climber,
    connect,
    control,
    experiment,
    knowledge,
    problem,
    run,
    store,
    views,
)


@app.command(hidden=True)
def intro(
    reset: bool = typer.Option(
        False, "--reset", help="Forget that the intro played; it plays again on the next run"
    ),
):
    """Replay the first-run 3D intro animation."""
    from hillclimb.tui.intro import intro_marker_path, play_intro

    if reset:
        intro_marker_path().unlink(missing_ok=True)
        typer.echo("Intro re-armed: it plays on the next hillclimb command.")
        return
    play_intro()


def main(argv: list[str] | None = None) -> None:
    """Console-script entry: front the help screen with the wordmark.

    A bare `hillclimb` is a request to see what the tool can do, not a usage
    error — so it prints the banner and the full command list instead of
    typer's "Missing command".
    """
    args = list(sys.argv[1:] if argv is None else argv)
    from hillclimb.harness.oscompat import ensure_utf8_mode

    ensure_utf8_mode(args)
    if "--skip-intro" in args:
        # Opt out of the first-run intro for good; `hillclimb intro` still plays it.
        from hillclimb.tui.intro import mark_intro_shown

        args = [arg for arg in args if arg != "--skip-intro"]
        mark_intro_shown()
    elif not args or args[0] != "intro":  # `hillclimb intro` plays it itself
        from hillclimb.tui.intro import maybe_play_intro

        maybe_play_intro()
    if not args or args in (["--help"], ["-h"]):
        print_banner()
        args = ["--help"]
    app(args=args, prog_name="hillclimb")


__all__ = ["app", "main", "HillclimbGroup", "BANNER_LINES", "LOGO_LINES", "WORDMARK_LINES",
           "print_banner", "say", "legend", "next_steps"]
