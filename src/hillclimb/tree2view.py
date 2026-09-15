"""`hillclimb tree2` — the archive tree of one search, live: the Darwin
Gödel Machine's archive picture on hillclimb's journal (encoding in
tree2.py).

A thin skin over `treeview.py`: the widget swaps the plot builder and the
legend through the hooks `TreePlotWidget` exposes and keeps its picking,
camera, hover and selection; the screen swaps the widget and drops the
legend filter (the ring ladder is a key, not a switch). Scrubbing through
time, the candidate detail dock, n/p between searches and enter into the
candidate screen are inherited unchanged.

The one thing this widget adds: marks are sized to the projection. Every
zoom, reset and resize rebuilds the plot with the radius `tree2.fit_radius`
derives from the closest pair of centres, so circles never overlap and the
numbers inside them (plotui draws those, only where they fit) come and go
with the zoom. The hovered or selected node is also named beside its mark,
for the fit view where the marks are too small to carry a number.
"""

from __future__ import annotations

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Label, RichLog

from hillclimb.config import Config
from hillclimb.graphview import GraphKeys, TimeScrubber, VNode, place_labels_by_node
from hillclimb.header import HillclimbHeader, TimezoneMixin
from hillclimb.theme import HILLCLIMB_CSS, apply_theme
from hillclimb.tree import SearchTree
from hillclimb.tree2 import (
    BEST_SIZE_SCALE, DEFAULT_RADIUS, build_tree2_plot, fit_radius, label_nodes, legend_spans,
)
from hillclimb.treeview import TreePlotWidget, TreeScreen


class Tree2PlotWidget(TreePlotWidget):
    """The archive-tree encoding on the tree widget's plumbing. The screen
    tells it the metric's name and direction (`higher_is_better`, `metric`)
    before handing it a tree; both feed the fill ramp and its legend."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.higher_is_better = True
        self.metric = "score"
        self.radius = DEFAULT_RADIUS  # the last fitted mark radius, plotui units
        self.star = BEST_SIZE_SCALE   # the star's size over the discs', as last fitted

    def _build_plot(self, tree: SearchTree, selected: str | None, frame: SearchTree | None):
        # build once to project at the camera the widget holds now, size
        # the marks to that projection, then build the plot that is shown
        def build(radius: float, star: float):
            return build_tree2_plot(
                tree, self.higher_is_better, selected=selected, frame=frame, radius=radius, star=star,
            )

        probe, ids = build(self.radius, self.star)
        if ids and self.size.width > 0:
            _yaw, _pitch, zoom, pan_x, pan_y = self._plot.camera_state()
            probe.set_camera_state(0.0, 0.0, zoom, pan_x, pan_y)
            best = ids.index(tree.best_id) if tree.best_id in ids else None
            self.radius, self.star = fit_radius(probe, *self._px_dims(), best_index=best)
            return build(self.radius, self.star)
        return probe, ids

    def _label_nodes(self, tree: SearchTree) -> list[VNode]:
        return label_nodes(tree, frame=self._frame)

    def _place_labels(self, projected):
        # zoom 0: only the selected/hovered node is named beside its mark
        return place_labels_by_node(
            self._labels, projected,
            cols=self.size.width, rows=self.size.height, cell_px=(self._cell_w, self._cell_h),
            zoom=0.0, selected=self.selected, hovered=self._hover,
        )

    def _legend_spans(self) -> list[tuple[int, int, str, str]]:
        return legend_spans(self._tree, self.metric, self.higher_is_better, cols=self.size.width)

    def _legend_entry_at(self, col: int, row: int) -> str | None:
        return None  # the ring ladder is not a filter

    # -- the mark radius follows the projection: rebuild on every change of
    # scale (zoom, reset, resize); a pan keeps the spacing --

    def apply_zoom(self, factor: float) -> None:
        self._plot.zoom_by(factor)
        self.rebuild()

    def apply_reset(self) -> None:
        self._plot.set_camera_state(0.0, 0.0, 1.0, 0.0, 0.0)
        self.rebuild()

    def on_resize(self, event: events.Resize) -> None:
        self.rebuild()


class Tree2Keys(GraphKeys):
    POINTER = [
        ("drag", "pan"),
        ("scroll", "zoom (numbers appear as the circles grow)"),
        ("arrows", "pan"),
        ("r", "reset view"),
        ("click", "select"),
        ("click again", "open"),
        ("n / p", "next / previous search"),
    ]


class Tree2Screen(TreeScreen):
    """Canvas + time scrubber + candidate detail, archive-tree encoding.
    Reached via `hillclimb tree2 [search]`."""

    BINDING_GROUP_TITLE = "tree2"
    BINDINGS = [
        Binding("escape", "dismiss_or_back", "back"),
        Binding("enter", "activate", "open", show=False, priority=True),
        Binding("+,=", "zoom_in", "zoom in", show=False),
        Binding("-", "zoom_out", "zoom out", show=False),
        Binding("f,0", "fit", "fit", show=False, tooltip="frame the whole tree"),
        Binding("b", "select_best", "best", tooltip="select the current best"),
        Binding("n", "next_search", "next search", tooltip="the next search in the store"),
        Binding("p", "prev_search", "prev search", show=False, tooltip="the previous search"),
        Binding("j", "scrub_back", "back in time", show=False, tooltip="one tick per landed result"),
        Binding("k", "scrub_forward", "forward", show=False),
        Binding("end", "scrub_live", "live", show=False, tooltip="jump to now"),
        Binding("question_mark", "toggle_help", "keys"),
        Binding("q", "app.quit", "quit"),
    ]

    def compose(self) -> ComposeResult:
        yield HillclimbHeader()
        yield Label(id="treeline")
        yield RichLog(id="node-detail", wrap=True, markup=False, auto_scroll=False, min_width=1)
        with Vertical(id="tree-stage"):
            yield Tree2PlotWidget(id="tree-canvas")
            yield TimeScrubber(id="time-scrubber")
        yield Footer()

    def _apply_view(self) -> None:
        if self._record is not None:
            canvas = self._canvas()
            canvas.higher_is_better = bool(self._record.meta.higher_is_better)
            canvas.metric = self._record.meta.metric
        super()._apply_view()

    def action_toggle_type(self, index: int) -> None:
        return  # no legend filter in this view

    def action_toggle_help(self) -> None:
        panel = self.query(Tree2Keys)
        if panel:
            panel.remove()
        else:
            self.mount(Tree2Keys())


class Tree2App(TimezoneMixin, App):
    """Standalone shell for `hillclimb tree2`."""

    BINDINGS = [Binding("t", "choose_timezone", "time zone", show=False)]
    CSS = HILLCLIMB_CSS

    def __init__(self, config: Config | None = None, search: str | None = None):
        super().__init__()
        self.config = config or Config.load()
        self.search = search

    def on_mount(self) -> None:
        apply_theme(self)
        self._init_timezone()
        self.push_screen(Tree2Screen(self.config, self.search))
