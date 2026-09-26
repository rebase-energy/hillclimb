"""The one TUI palette, shared with hillclimb.sh.

The website is the design of record: one flat near-black page, a single
subtle lift for chrome (its `--panel`), neutral borders, and cyan reserved
for the things meant to catch the eye. The tokens below are its CSS
variables, so the terminal and the browser render the same product.

Cyan carries the *chrome* — borders, cursors, footer keys, dividers, the
panels that only frame content. Categorical color stays categorical: run
status in watch.py and node/edge/concept color in graphview.py encode data,
and flattening those to cyan would delete the encoding, so they keep their
own literals (drawn from the same vivid palette the website's graph uses).

Textual resolves every `$token` in the screens' CSS from the registered
theme, so all three apps get the chrome recolored by applying this one
object. `HILLCLIMB_CSS` goes on each App as its `CSS`.
"""

from __future__ import annotations

from textual.scrollbar import ScrollBar, ScrollBarRender
from textual.app import App
from textual.theme import Theme

HILLCLIMB_THEME = Theme(
    name="hillclimb",
    primary="#2EE6E6",    # borders, block cursors, table cursor — the site's --cyan
    secondary="#2B7F86",  # dividers, second-rank chrome
    accent="#7FF3F3",     # footer keys — the one thing meant to catch the eye
    foreground="#E8EAEC",  # --fg
    background="#0B0D0E",  # --bg
    surface="#131719",     # --panel: the site's one lift, for chrome bars
    panel="#1A2024",       # a touch above surface, for framed panels
    # Status colors are read as meaning, not decoration — they stay themselves,
    # matching the vivid palette graphview paints the knowledge graph with.
    success="#22C55E",
    warning="#EAB308",
    error="#EF4444",
    dark=True,
)

# Textual's DataTable, RichLog and OptionList paint themselves on `$surface`
# and add `background-tint: $foreground 5%` when focused — so the panel a
# reader is actually looking at is the one that visibly lightens, and moving
# focus makes the page flicker between two greys. The website keeps one flat
# page color and lifts only thin chrome bars, so do that here: content sits
# on the background and stays put when focused.
class WholeCellScrollBarRender(ScrollBarRender):
    """Scrollbar thumb snapped to whole cells. Textual positions the thumb
    with sub-cell precision and draws its fractional ends with eighth-block
    glyphs (▁▂▃…) in thumb-on-track colours — at a 1-2 cell width that
    reads as a second, smaller rectangle stuck to the thumb. With one glyph
    per table the start/end remainders are always 0, so every cell is either
    all thumb or all track."""

    VERTICAL_BARS = [" "]
    HORIZONTAL_BARS = [" "]


# every ScrollBar in the process (all hillclimb TUIs share this module)
ScrollBar.renderer = WholeCellScrollBarRender


HILLCLIMB_CSS = """
Screen { background: $background; }

/* scrollbars: one cell wide, a crisp thumb on an invisible track */
* {
    scrollbar-size-vertical: 1;
    scrollbar-background: $background;
    scrollbar-background-hover: $background;
    scrollbar-background-active: $background;
    scrollbar-color: $primary-darken-1;
    scrollbar-color-hover: $primary;
    scrollbar-color-active: $primary-lighten-1;
}

DataTable, RichLog, OptionList {
    background: $background;
}

DataTable:focus, RichLog:focus, OptionList:focus {
    background-tint: $foreground 0%;
}

DataTable > .datatable--header {
    background: $background;
    color: $foreground;
}

DataTable:focus > .datatable--header {
    background-tint: $foreground 0%;
}

/* Textual already tracks the row under the pointer separately from its
   keyboard cursor. Give that hover row the site's subtle panel lift so it is
   easy to aim at, while leaving the cyan cursor as the stronger selection. */
DataTable > .datatable--hover {
    background: $panel;
}
"""


def apply_theme(app: App) -> None:
    """Register and select the hillclimb theme — call from an App.on_mount."""
    app.register_theme(HILLCLIMB_THEME)
    app.theme = HILLCLIMB_THEME.name


# The website's chart chrome (index.html `.climb-plot`), so a plotui canvas
# in a hillclimb TUI reads as the same figure: grid and axes that recede into
# the page, tick labels in the faint ink, data in the site's cyan.
PLOT_BG = (14, 17, 19)          # #0e1113 — the site's .climb box
PLOT_GRID = (26, 32, 36)        # #1a2024
PLOT_FRAME = (43, 50, 55)       # #2b3237
PLOT_INK = (103, 111, 118)      # #676f76 — --faint
PLOT_INK_BRIGHT = (144, 153, 160)  # #9099a0 — --muted
CYAN = (46, 230, 230)           # #2ee6e6 — --cyan


def themed_plot():
    """A plotui Plot with the site's chrome. Falls back to plotui's own
    palette on a build that predates `set_chrome`."""
    from plotui import Plot

    plot = Plot()
    if hasattr(plot, "set_chrome"):
        plot.set_chrome(bg=PLOT_BG, frame=PLOT_FRAME, grid=PLOT_GRID, ink=PLOT_INK, ink_bright=PLOT_INK_BRIGHT)
    return plot
