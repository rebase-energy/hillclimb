"""The typer app and its group class — a leaf every command module imports."""

from __future__ import annotations

import typer
import typer.core
import typer.rich_utils

from hillclimb.tui.banner import (  # noqa: F401 — re-exported
    BANNER_LINES,
    LOGO_LINES,
    WORDMARK_LINES,
    print_banner,
)

# Typer's default rich theme paints "Usage:" and every `<...>` metavar yellow,
# which clashes with the cyan command/option column. Repaint both in the same
# cyan family so the help screen reads as one palette. These are module-level
# globals that typer.rich_utils reads at render time, so assigning them here
# (before any help is formatted) is enough.
typer.rich_utils.STYLE_USAGE = "bold cyan"


typer.rich_utils.STYLE_TYPES = "cyan"


# The top-level commands `hillclimb --help` lists: the README's get-started
# flow. Every other command still runs; `hillclimb --help --all` lists it.
CORE_COMMANDS = frozenset({
    "init", "connect", "problem", "verify", "run", "watch", "chart", "tree",
    "stop", "resume", "summit", "climber", "experiment",
})


class HillclimbGroup(typer.core.TyperGroup):
    """Command listing and the completion flags, the way this CLI wants them."""

    # Set by `main()` for `hillclimb --help --all`.
    show_all = False

    def format_help(self, ctx, formatter) -> None:
        """At the top level, list only CORE_COMMANDS unless `--all` asked for
        every one, and say in a footer that there are more."""
        if ctx.parent is not None or HillclimbGroup.show_all:
            return super().format_help(ctx, formatter)
        extra = [c for name, c in self.commands.items() if name not in CORE_COMMANDS and not c.hidden]
        for command in extra:
            command.hidden = True
        try:
            super().format_help(ctx, formatter)
        finally:
            for command in extra:
                command.hidden = False
        # Printed here rather than as the epilog: typer pads the epilog with a
        # blank line below, and click's echo of the (empty) help text adds a
        # second one before the prompt. `end=""` leaves that echo to close the line.
        console = typer.rich_utils._get_rich_console()
        console.print()
        console.print(typer.rich_utils.highlighter(f" {len(extra)} more commands: hillclimb --help --all"), end="")

    # `ctx` is a click Context and get_params returns click Parameters, but
    # typer >=0.27 vendors click as `typer._click` and hillclimb does not
    # depend on the standalone package — so these stay unannotated rather than
    # importing a module that is not guaranteed to be installed.
    def list_commands(self, ctx) -> list[str]:
        """Typer lists commands in declaration order; list them alphabetically."""
        return sorted(self.commands)

    def get_params(self, ctx) -> list:
        """Drop `--show-completion`, and keep `--install-completion` working but
        unlisted — shell completion is a one-time setup step documented in the
        README, not something worth a third of the top-level options panel."""
        params = []
        for param in super().get_params(ctx):
            if param.name == "show_completion":
                continue
            if param.name == "install_completion":
                param.hidden = True
            params.append(param)
        return params


app = typer.Typer(
    cls=HillclimbGroup,
    help="Hillclimbing on verifier-defined problems: a code-generation harness for model development with long-running coding agents.",
    no_args_is_help=True,
    # Subcommands inherit help_option_names from the parent click Context, so
    # `-h` works on every command in the tree, not just the top level.
    context_settings={"help_option_names": ["--help", "-h"]},
)


def _print_version(value: bool) -> None:
    if value:
        from hillclimb import __version__

        typer.echo(f"hillclimb {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_print_version, is_eager=True,
        help="Show hillclimb's version and exit.",
    ),
) -> None:
    pass
