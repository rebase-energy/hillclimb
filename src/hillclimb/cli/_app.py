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


class HillclimbGroup(typer.core.TyperGroup):
    """Command listing and the completion flags, the way this CLI wants them."""

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
    help="Hillclimbing on verifier-defined problems: a code-generation harness for model development with long-running agents.",
    no_args_is_help=True,
    # Subcommands inherit help_option_names from the parent click Context, so
    # `-h` works on every command in the tree, not just the top level.
    context_settings={"help_option_names": ["--help", "-h"]},
)
