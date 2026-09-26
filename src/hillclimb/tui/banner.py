"""The CLI banner: the ANSI-shadow "HILLCLIMB" wordmark and the rising-trend mark
beside it, printed above the command list on a bare `hillclimb` and on `--help`.
The intro downscales the same art into its pixel logotype, and the website's
wordmark SVG is generated from it — this file is the one source of the glyphs.
"""

from __future__ import annotations

from itertools import zip_longest

# Ridge-line mark + ANSI-shadow "HILLCLIMB", printed above the command list on a bare `hillclimb`
# and on `--help`, the way `rebase` fronts the toolkit CLI. Same bold cyan as
# the command and option columns below it, so the whole help screen reads as
# one palette.
BANNER_STYLE = "bold cyan"
# Rising-trend arrow to the right of the wordmark, in the same ANSI-shadow
# style as the letters: a climb, a small dip, then a climb into the arrowhead.
# Diagonals step one column per row so adjacent cells share an edge, not just
# a corner. The seventh row closes the shadow below the lowest step.
LOGO_LINES = [
    "        ██████╗",
    "    ██╗ ╚═████║",
    "   ████╗ ██╔██║",
    "  ██╔═████╔╝╚═╝",
    " ██╔╝ ╚██╔╝    ",
    "██╔╝   ╚═╝     ",
    "╚═╝            ",
]
WORDMARK_LINES = [
    "██╗  ██╗ ██╗ ██╗      ██╗       ██████╗ ██╗      ██╗ ███╗   ███╗ ██████╗ ",
    "██║  ██║ ██║ ██║      ██║      ██╔════╝ ██║      ██║ ████╗ ████║ ██╔══██╗",
    "███████║ ██║ ██║      ██║      ██║      ██║      ██║ ██╔████╔██║ ██████╔╝",
    "██╔══██║ ██║ ██║      ██║      ██║      ██║      ██║ ██║╚██╔╝██║ ██╔══██╗",
    "██║  ██║ ██║ ███████╗ ███████╗ ╚██████╗ ███████╗ ██║ ██║ ╚═╝ ██║ ██████╔╝",
    "╚═╝  ╚═╝ ╚═╝ ╚══════╝ ╚══════╝  ╚═════╝ ╚══════╝ ╚═╝ ╚═╝     ╚═╝ ╚═════╝ ",
]
WORDMARK_WIDTH = max(len(line) for line in WORDMARK_LINES)
LOGO_WIDTH = max(len(line) for line in LOGO_LINES)
BANNER_LINES = [
    f"{word:<{WORDMARK_WIDTH}}  {logo:<{LOGO_WIDTH}}"
    for word, logo in zip_longest(WORDMARK_LINES, LOGO_LINES, fillvalue="")
]
BANNER_WIDTH = max(len(line) for line in BANNER_LINES)



def print_banner() -> None:
    """Print the mark + wordmark; drop the mark, then the art, as the terminal narrows."""
    from rich.console import Console

    console = Console(highlight=False)
    console.print()
    if console.width >= BANNER_WIDTH:
        lines = BANNER_LINES
    elif console.width >= WORDMARK_WIDTH:
        lines = WORDMARK_LINES
    else:
        lines = ["hillclimb"]
    for line in lines:
        console.print(line, style=BANNER_STYLE)
    console.print()
