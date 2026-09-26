"""`hillclimb archive` — the archive tree (left) beside the progress chart
(right), live: the Darwin Gödel Machine paper's two-panel figure, on one
search's journal. Encodings in `tree2.py` (left) and `archive.py` (right).

A skin over `tree2view.py`: the screen keeps the tree widget, the time
scrubber, the candidate detail dock, n/p between searches and enter into
the candidate screen, and adds a second plot on the right that is rebuilt
from the same scrubbed tree every time the left one is. So `j`/`k` step
both panels through time together — a node appears in the tree as its dot
lands on the chart — and the lineage of the final best is the bold path on
the left and the thick line on the right, one circle and one dot per
candidate number on both. Clicking a node in the tree rings its dot on the chart.

Two plots on one screen need two pairs of Kitty image ids (the terminal
keeps one picture per id): the chart takes plotui's `image_slot=1`, the
tree keeps the default slot.
"""

from __future__ import annotations

from rich.style import Style
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Footer, Label, RichLog

from hillclimb.tui.archive import (
    SERIES, build_progress_plot, cursor_number, legend_entry_at, legend_series, legend_spans,
    progress_points,
)
from hillclimb.tui.chart import ChartPlotWidget
from hillclimb.config import Config
from hillclimb.tui.graphview import TimeScrubber
from hillclimb.tui.header import HillclimbHeader, TimezoneMixin
from hillclimb.tui.theme import HILLCLIMB_CSS, apply_theme, themed_plot
from hillclimb.tui.tree import SearchTree
from hillclimb.tui.tree2 import LEGEND_ENTRIES as TREE_LEGEND
from hillclimb.tui.tree2view import Tree2Keys, Tree2PlotWidget, Tree2Screen
from hillclimb.tui.treeview import TreePlotWidget
from hillclimb.tui.watch import _mouse_event_x, _mouse_event_y

PROGRESS_SLOT = 1  # the chart's Kitty image-id slot; the tree has slot 0
CHART_FIRST_KEY = len(TREE_LEGEND) + 1  # the tree's legend takes 1-5, the chart's series 6-9


class ProgressPlotWidget(ChartPlotWidget):
    """The right panel: a static figure (no zoom or pan, like `chart`)
    rebuilt from whatever tree the screen hands it. Remembers what it was
    last shown so the screen can re-ring a selection without re-deriving
    the rest. The legend is a text overlay in the corner the climb leaves
    empty (`archive.legend_spans`); clicking an entry toggles its series."""

    class SeriesToggled(Message):
        def __init__(self, series: str):
            super().__init__()
            self.series = series

    def __init__(self, **kwargs):
        kwargs.setdefault("image_slot", PROGRESS_SLOT)
        super().__init__(themed_plot(), **kwargs)
        self._tree: SearchTree | None = None
        self._frame: SearchTree | None = None
        self.higher_is_better = True
        self.metric = "score"
        self.cursor: int | None = None
        self.until: str | None = None
        self.selected: str | None = None
        self.hidden: frozenset[str] = frozenset()

    def show(
        self, tree: SearchTree, *, frame: SearchTree | None, cursor: int | None, selected: str | None,
        until: str | None = None,
    ) -> None:
        self._tree, self._frame, self.cursor, self.selected, self.until = tree, frame, cursor, selected, until
        self.rebuild()

    def rebuild(self) -> None:
        if self._tree is None:
            return
        self._plot = build_progress_plot(
            self._tree, self.higher_is_better, frame=self._frame, cursor=self.cursor,
            selected=self.selected, hidden=self.hidden, metric=self.metric,
        )
        self.invalidate()
        self._sync_annotations()

    def _sync_annotations(self) -> None:
        # the chart widget's overlay hook (also called on resize): the legend
        if self._mode == "unsupported":
            return
        points = progress_points(self._tree, self.higher_is_better) if self._tree is not None else None
        spans = legend_spans(
            points, self.higher_is_better, hidden=self.hidden, rows=max(self.size.height, 1),
            first_key=CHART_FIRST_KEY,
        )
        self.set_overlay([(r, c, t, Style.parse(st)) for r, c, t, st in spans])

    def on_click_at(self, event: events.MouseUp) -> None:
        series = legend_entry_at(
            _mouse_event_x(event) - self.region.x, _mouse_event_y(event) - self.region.y,
            self.higher_is_better, rows=max(self.size.height, 1),
        )
        if series is not None:
            self.post_message(self.SeriesToggled(series))


class ArchiveKeys(Tree2Keys):
    POINTER = [
        ("drag", "pan the tree"),
        ("scroll", "zoom the tree"),
        ("arrows", "pan the tree"),
        ("r", "reset view"),
        ("click", "select (rings its dot on the chart)"),
        ("click again", "open"),
        ("click tree legend", "hide a stage, the best or the lineage (1-5 too)"),
        ("click chart legend", "hide a series (6-9 too)"),
        ("n / p", "next / previous search"),
    ]


class ArchiveScreen(Tree2Screen):
    """Tree + progress chart + shared time scrubber + candidate detail.
    Reached via `hillclimb archive [search]`."""

    BINDING_GROUP_TITLE = "archive"
    BINDINGS = [
        Binding("j", "scrub_back", "back in time", show=False, tooltip="one tick per landed result, both panels"),
        Binding("k", "scrub_forward", "forward", show=False),
        # the chart legend's keys, after the tree legend's 1-5
        Binding(str(CHART_FIRST_KEY), f"toggle_type({CHART_FIRST_KEY - 1})", "hide/show a chart series", show=False, key_display="6-9"),
        *(Binding(str(k % 10), f"toggle_type({k - 1})", show=False) for k in range(CHART_FIRST_KEY + 1, CHART_FIRST_KEY + len(SERIES))),
    ]

    DEFAULT_CSS = """
    ArchiveScreen #archive-panes { width: 1fr; height: 1fr; }
    ArchiveScreen #tree-canvas { width: 1fr; height: 1fr; }
    ArchiveScreen #progress-canvas { width: 1fr; height: 1fr; background: #0e1113; }
    """

    def compose(self) -> ComposeResult:
        yield HillclimbHeader()
        yield Label(id="treeline")
        yield RichLog(id="node-detail", wrap=True, markup=False, auto_scroll=False, min_width=1)
        with Vertical(id="tree-stage"):
            with Horizontal(id="archive-panes"):
                yield Tree2PlotWidget(id="tree-canvas")
                yield ProgressPlotWidget(id="progress-canvas")
            yield TimeScrubber(id="time-scrubber")
        yield Footer()

    # -- the right panel follows the left --

    def _progress(self) -> ProgressPlotWidget:
        return self.query_one("#progress-canvas", ProgressPlotWidget)

    def _apply_view(self) -> None:
        super()._apply_view()
        self._sync_progress()

    def _sync_progress(self) -> None:
        if self._record is None or self._tree is None:
            return
        scrubber = self.query_one("#time-scrubber", TimeScrubber)
        until = None
        if scrubber.index is not None and scrubber.events_list:
            until = scrubber.events_list[scrubber.index]
        chart = self._progress()
        chart.higher_is_better = bool(self._record.meta.higher_is_better)
        chart.metric = self._record.meta.metric
        chart.show(
            self._tree, frame=self._live_tree,
            cursor=cursor_number(self._tree, until), selected=self._canvas().selected, until=until,
        )

    def on_tree_plot_widget_node_selected(self, message: TreePlotWidget.NodeSelected) -> None:
        super().on_tree_plot_widget_node_selected(message)
        chart = self._progress()
        if chart.selected != message.node_id:
            chart.selected = message.node_id
            chart.rebuild()

    def on_progress_plot_widget_series_toggled(self, message: ProgressPlotWidget.SeriesToggled) -> None:
        self.action_toggle_series(message.series)

    def action_toggle_type(self, index: int) -> None:
        # one row of hotkeys for two legends: 1-5 the tree's (inherited),
        # 6-9 the chart's series
        if 0 <= index < len(TREE_LEGEND):
            super().action_toggle_type(index)
        elif 0 <= index - len(TREE_LEGEND) < len(SERIES):
            self.action_toggle_series(SERIES[index - len(TREE_LEGEND)])

    def action_toggle_lineage(self) -> None:
        """`l`: the lineage in both panels at once — shown if either hides it,
        hidden when both show it, so one press always brings them in step."""
        canvas, chart = self._canvas(), self._progress()
        show = "lineage" in canvas.hidden or "lineage" in chart.hidden
        self._hidden = (self._hidden - {"lineage"}) if show else (self._hidden | {"lineage"})
        canvas.set_hidden(self._hidden)
        chart.hidden = (chart.hidden - {"lineage"}) if show else (chart.hidden | {"lineage"})
        chart.rebuild()

    def action_toggle_series(self, label: str) -> None:
        """Show or hide one chart series — a legend click or its hotkey."""
        chart = self._progress()
        chart.hidden = chart.hidden ^ {legend_series(label)}
        chart.rebuild()

    def action_toggle_help(self) -> None:
        panel = self.query(ArchiveKeys)
        if panel:
            panel.remove()
        else:
            self.mount(ArchiveKeys())


class ArchiveApp(TimezoneMixin, App):
    """Standalone shell for `hillclimb archive`."""

    BINDINGS = [Binding("t", "choose_timezone", "time zone", show=False)]
    CSS = HILLCLIMB_CSS

    def __init__(self, config: Config | None = None, search: str | None = None):
        super().__init__()
        self.config = config or Config.load()
        self.search = search

    def on_mount(self) -> None:
        apply_theme(self)
        self._init_timezone()
        self.push_screen(ArchiveScreen(self.config, self.search))
