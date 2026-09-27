"""ChartPlotWidget row backgrounds per render mode (see its docstring)."""

import pytest
from textual.app import App, ComposeResult

from hillclimb.tui.chart import ChartPlotWidget, ChartScreen
from hillclimb.tui.theme import PLOT_BG, themed_plot


class _Host(App):
    def __init__(self, mode: str):
        super().__init__()
        self._mode = mode

    def compose(self) -> ComposeResult:
        plot = themed_plot()
        plot.add_line([0.0, 1.0], [0.0, 1.0], name="x")
        yield ChartPlotWidget(plot, render_mode=self._mode, id="plot")


def _row_cells(app: App) -> list:
    """(text, bgcolor triplet or None) per cell of a row without the upload
    control segment."""
    widget = app.query_one("#plot", ChartPlotWidget)
    cells = []
    for seg in widget.render_line(1):
        if seg.control or not seg.text:
            continue
        bg = seg.style.bgcolor.triplet if seg.style and seg.style.bgcolor else None
        cells.extend((ch, bg) for ch in seg.text)
    return cells


def _row_bgcolors(app: App) -> set:
    return {bg for _ch, bg in _row_cells(app)}


@pytest.mark.asyncio
async def test_direct_mode_paints_only_the_last_cell_of_each_row():
    """iTerm2 extends a row's last cell background across its right window
    margin — else the margin shows the profile colour as a stripe beside
    the opaque canvas. Only that cell may carry a background: the image is
    placed below cell backgrounds, so a painted cell hides the plot."""
    async with _Host("direct").run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        cells = _row_cells(pilot.app)
        assert len(cells) == 40
        assert cells[-1] == (" ", PLOT_BG)
        assert all(bg is None for _ch, bg in cells[:-1])


@pytest.mark.asyncio
async def test_placeholder_mode_rows_stay_unpainted():
    """Kitty draws the image below cells with an explicit background (the
    annotation tags rely on it), so placeholder rows keep the default."""
    async with _Host("placeholder").run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert all(c is None for c in _row_bgcolors(pilot.app))


def test_chart_screen_never_paints_a_text_selection():
    """A double click on a legend entry toggles that series; Textual's text
    selection must not light up the band instead."""
    assert ChartScreen.ALLOW_SELECT is False


def test_readout_ranks_best_first_in_the_metric_direction(monkeypatch):
    """The hover readout is the leaderboard: rows ordered best-first, so the
    direction of the metric picks the order."""
    from hillclimb.tui.chart import Climb, ClimbEvent, build_climb_plot

    class Spy:
        legend_visible = True
        order = None

        def add_line(self, *a, **kw): pass
        def add_scatter(self, *a, **kw): pass
        def set_readout_order(self, order): self.order = order

    climb = Climb(events=[ClimbEvent(1.0, 0.6, True, "r", "draft")], extent=1.0)
    for higher, expected in ((True, "descending"), (False, "ascending")):
        spy = Spy()
        monkeypatch.setattr("hillclimb.tui.chart.themed_plot", lambda: spy)
        build_climb_plot(climb, {"OpenEvolve": 0.7}, higher_is_better=higher)
        assert spy.order == expected


@pytest.mark.asyncio
async def test_notifications_wear_the_hillclimb_panel():
    """A toast is framed like the plot readout — rounded neutral rule on the
    surface lift, hugging its text — not Textual's grey slab with a green
    bar; a warning colours the frame in its status colour."""
    from textual.widgets._toast import Toast

    from hillclimb.tui.theme import HILLCLIMB_CSS, apply_theme

    class Host(App):
        CSS = HILLCLIMB_CSS

        def on_mount(self) -> None:
            apply_theme(self)
            self.notify("only one problem in this folder")
            self.notify("careful", severity="warning")

    async with Host().run_test(size=(80, 24), notifications=True) as pilot:
        for _ in range(20):  # toasts mount on a later frame
            await pilot.pause(0.05)
            toasts = list(pilot.app.query(Toast))
            if len(toasts) == 2:
                break
        assert len(toasts) == 2
        info, warning = toasts
        assert info.styles.background.hex.lower() == "#131719"
        assert info.styles.border_left == ("round", info.styles.border_left[1])
        assert info.styles.border_left[1].hex.lower() == "#2b3237"
        assert info.styles.width is not None and info.styles.width.is_auto
        # $warning comes back through Textual's colour space, a unit off
        warn_rgb = warning.styles.border_left[1].rgb
        assert all(abs(a - b) <= 2 for a, b in zip(warn_rgb, (0xEA, 0xB3, 0x08)))
