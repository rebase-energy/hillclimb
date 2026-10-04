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
        split = None

        def add_line(self, *a, **kw): pass
        def add_scatter(self, *a, **kw): pass
        def set_readout_order(self, order): self.order = order
        def set_readout_split_axes(self, on): self.split = on

    climb = Climb(events=[ClimbEvent(1.0, 0.6, True, "r", "draft")], extent=1.0)
    for higher, expected in ((True, "descending"), (False, "ascending")):
        spy = Spy()
        monkeypatch.setattr("hillclimb.tui.chart.themed_plot", lambda: spy)
        build_climb_plot(climb, {"OpenEvolve": 0.7}, higher_is_better=higher)
        assert spy.order == expected
        # the cost overlay's y2/y3 rows sit under a rule of their own
        assert spy.split is True


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


def test_y_axis_names_the_metric_and_its_direction(monkeypatch):
    """The y axis says what the problem scores and which way is up, and
    the holdout view says so up front; the plot gets it as its y title."""
    from hillclimb.tui.chart import Climb, ClimbEvent, build_climb_plot, y_axis_title

    assert y_axis_title("normalized-min-triangle-area", True) == (
        "normalized-min-triangle-area (higher is better)"
    )
    assert y_axis_title("rmse", False) == "rmse (lower is better)"
    assert y_axis_title("rmse", False, holdout=True) == "holdout rmse (lower is better)"

    class Spy:
        legend_visible = True
        title = None

        def add_line(self, *a, **kw): pass
        def add_scatter(self, *a, **kw): pass
        def set_y_title(self, text): self.title = text

    spy = Spy()
    monkeypatch.setattr("hillclimb.tui.chart.themed_plot", lambda: spy)
    climb = Climb(events=[ClimbEvent(1.0, 0.6, True, "r", "draft")], extent=1.0)
    build_climb_plot(climb, {}, y_title=y_axis_title("score", True))
    assert spy.title == "score (higher is better)"


def test_footer_offers_switch_problem_and_holdout_only_where_they_apply():
    """`p` is shown only when the folder holds a second problem to switch
    to, `h` only for a problem that scores a holdout — a key that could
    only say "no" stays out of the footer."""
    from types import SimpleNamespace

    from hillclimb.config import Config

    screen = ChartScreen(Config())
    assert {b.action: b.description for b in ChartScreen.BINDINGS if b.key == "p"} == {
        "next_problem": "switch problem"
    }
    screen._problem_keys = ["circle-packing"]
    screen._anchor = SimpleNamespace(holdout_enabled=False)
    assert screen.check_action("next_problem", ()) is False
    assert screen.check_action("toggle_holdout", ()) is False
    screen._problem_keys = ["circle-packing", "heilbronn-convex-13"]
    screen._anchor = SimpleNamespace(holdout_enabled=True)
    assert screen.check_action("next_problem", ()) is True
    assert screen.check_action("toggle_holdout", ()) is True
    # the anchor is not resolved yet: nothing to toggle holdout on
    screen._anchor = None
    assert screen.check_action("toggle_holdout", ()) is False
    assert screen.check_action("toggle_cost", ()) is True


@pytest.mark.asyncio
async def test_cost_toggle_puts_the_overlay_on_its_own_legend_row(config):
    """`c` overlays the cost series and the legend band gains a `Cost:` row
    for them — the whole refresh path runs with the four-element entries."""
    from hillclimb.tui.chart import COST_GROUP, COST_TOKENS_LABEL, ChartApp, ChartLegend
    from tests.test_watch import make_run_with_search

    make_run_with_search(config.paths.runs_dir, "r1")
    app = ChartApp(config, search="r1/circle-packing")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        assert isinstance(app.screen, ChartScreen)
        await pilot.press("c")
        await pilot.pause(0.3)
        entries = app.screen._legend_entries
        cost_rows = [e[0] for e in entries if len(e) > 3 and e[3] == COST_GROUP]
        assert COST_TOKENS_LABEL in cost_rows
        band = str(app.screen.query_one("#chart-legend", ChartLegend).render())
        assert f"{COST_GROUP}: " in band


def test_hiding_a_legend_series_holds_the_view_until_a_recentre():
    """Toggling a legend entry must not rescale the axes — the reader could
    not tell what went from what stayed. With `unfit` (the hold-still rule)
    a hidden series still counts toward the axes; only the entries hidden at
    the last recentre (`f`) are left out of the fit."""
    from hillclimb.tui.chart import Climb, ClimbEvent, build_climb_plot
    from hillclimb.tui.theme import CYAN

    climb = Climb(
        events=[ClimbEvent(1.0, 0.50, True, "r", "draft"), ClimbEvent(2.0, 0.52, False, "r", "improve"),
                ClimbEvent(3.0, 0.60, True, "r", "improve")],
        extent=3.0,
    )
    baselines = {"far above": 5.0}  # a reference well outside the climb's own range

    def cyan_rows(plot) -> set[int]:
        """The pixel rows the staircase is drawn on: where it sits on the y axis."""
        w, h = 320, 200
        rgba = plot.render_rgba(w, h)
        return {
            y for y in range(h) for x in range(w)
            if tuple(rgba[(y * w + x) * 4:(y * w + x) * 4 + 3]) == CYAN
        }

    shown = cyan_rows(build_climb_plot(climb, baselines, show_legend=False, unfit=frozenset()))
    assert shown
    hidden = {"far above"}
    held = cyan_rows(build_climb_plot(climb, baselines, show_legend=False, hidden=hidden, unfit=frozenset()))
    assert held == shown  # the reference is gone, the staircase has not moved
    refit = cyan_rows(build_climb_plot(climb, baselines, show_legend=False, hidden=hidden, unfit=frozenset(hidden)))
    assert refit != shown  # after a recentre the axes fit what is shown
    assert refit == cyan_rows(build_climb_plot(climb, baselines, show_legend=False, hidden=hidden))  # the old rule, for other callers
    # shown again after the recentre: drawn, but it does not move the view either
    back = cyan_rows(build_climb_plot(climb, baselines, show_legend=False, unfit=frozenset(hidden)))
    assert back == refit
