"""`hillclimb archive` — the archive tree (left) beside the progress chart
(right), live: the Darwin Gödel Machine paper's two-panel figure, on one
search's journal. Encodings in `tree2.py` (left) and `archive.py` (right).

A skin over `tree2view.py`: the screen keeps the tree widget, the time
scrubber, the candidate detail dock, n/p between searches and enter into
the candidate screen, and adds a second plot on the right that is rebuilt
from the same scrubbed tree every time the left one is. So `j`/`k` step
both panels through time together — a node appears in the tree as its dot
lands on the chart — and the lineage of the final best is the bold path on
the left and the thick line on the right, one candidate per iteration
number on both. Clicking a node in the tree rings its dot on the chart.

Two plots on one screen need two pairs of Kitty image ids (the terminal
keeps one picture per id): the chart takes plotui's `image_slot=1`, the
tree keeps the default slot.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Label, RichLog

from hillclimb.archive import (
    build_progress_plot, cursor_iteration, legend_series, progress_legend, progress_points,
)
from hillclimb.chart import ChartLegend, ChartPlotWidget
from hillclimb.config import Config
from hillclimb.graphview import TimeScrubber
from hillclimb.header import HillclimbHeader, TimezoneMixin
from hillclimb.theme import HILLCLIMB_CSS, apply_theme, themed_plot
from hillclimb.tree import SearchTree
from hillclimb.tree2view import Tree2Keys, Tree2PlotWidget, Tree2Screen
from hillclimb.treeview import TreePlotWidget

PROGRESS_SLOT = 1  # the chart's Kitty image-id slot; the tree has slot 0


class ProgressPlotWidget(ChartPlotWidget):
    """The right panel: a static figure (no zoom or pan, like `chart`)
    rebuilt from whatever tree the screen hands it. Remembers what it was
    last shown so the screen can re-ring a selection without re-deriving
    the rest."""

    def __init__(self, **kwargs):
        kwargs.setdefault("image_slot", PROGRESS_SLOT)
        super().__init__(themed_plot(), **kwargs)
        self._tree: SearchTree | None = None
        self._frame: SearchTree | None = None
        self.higher_is_better = True
        self.metric = "score"
        self.cursor: int | None = None
        self.selected: str | None = None
        self.hidden: frozenset[str] = frozenset()

    def show(
        self, tree: SearchTree, *, frame: SearchTree | None, cursor: int | None, selected: str | None,
    ) -> None:
        self._tree, self._frame, self.cursor, self.selected = tree, frame, cursor, selected
        self.rebuild()

    def rebuild(self) -> None:
        if self._tree is None:
            return
        self._plot = build_progress_plot(
            self._tree, self.higher_is_better, frame=self._frame, cursor=self.cursor,
            selected=self.selected, hidden=self.hidden, metric=self.metric,
        )
        self.invalidate()


class ArchiveKeys(Tree2Keys):
    POINTER = [
        ("drag", "pan the tree"),
        ("scroll", "zoom the tree"),
        ("arrows", "pan the tree"),
        ("r", "reset view"),
        ("click", "select (rings its dot on the chart)"),
        ("click again", "open"),
        ("click chart legend", "hide a series"),
        ("n / p", "next / previous search"),
    ]


class ArchiveScreen(Tree2Screen):
    """Tree + progress chart + shared time scrubber + candidate detail.
    Reached via `hillclimb archive [search]`."""

    BINDING_GROUP_TITLE = "archive"
    BINDINGS = [
        Binding("j", "scrub_back", "back in time", show=False, tooltip="one tick per landed result, both panels"),
        Binding("k", "scrub_forward", "forward", show=False),
    ]

    DEFAULT_CSS = """
    ArchiveScreen #archive-panes { width: 1fr; height: 1fr; }
    ArchiveScreen #tree-canvas { width: 1fr; height: 1fr; }
    ArchiveScreen #progress-pane { width: 1fr; height: 1fr; background: #0e1113; }
    ArchiveScreen #progress-canvas { width: 1fr; height: 1fr; }
    ArchiveScreen #progress-x-label {
        width: 1fr; height: 1; text-align: center; color: $secondary; background: #0e1113;
    }
    ArchiveScreen #progress-legend {
        width: 1fr; height: auto; min-height: 1; padding: 0 2;
        text-align: center; background: #0e1113;
    }
    """

    def compose(self) -> ComposeResult:
        yield HillclimbHeader()
        yield Label(id="treeline")
        yield RichLog(id="node-detail", wrap=True, markup=False, auto_scroll=False, min_width=1)
        with Vertical(id="tree-stage"):
            with Horizontal(id="archive-panes"):
                yield Tree2PlotWidget(id="tree-canvas")
                with Vertical(id="progress-pane"):
                    yield ProgressPlotWidget(id="progress-canvas")
                    yield Label("iteration", id="progress-x-label")
                    yield ChartLegend(id="progress-legend")
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
            cursor=cursor_iteration(self._tree, until), selected=self._canvas().selected,
        )
        self.query_one("#progress-x-label", Label).update(
            "iteration" if until is None else f"iteration  ·  at {until[11:19]}"
        )
        self.query_one("#progress-legend", ChartLegend).set_entries(
            progress_legend(progress_points(self._tree, chart.higher_is_better)),
            {label for label in chart.hidden},
        )

    def on_tree_plot_widget_node_selected(self, message: TreePlotWidget.NodeSelected) -> None:
        super().on_tree_plot_widget_node_selected(message)
        chart = self._progress()
        if chart.selected != message.node_id:
            chart.selected = message.node_id
            chart.rebuild()

    def action_toggle_series(self, label: str) -> None:
        """Show or hide one chart series — reached by clicking its legend entry."""
        chart = self._progress()
        series = legend_series(label)
        chart.hidden = chart.hidden ^ {series}
        self._sync_progress()

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
