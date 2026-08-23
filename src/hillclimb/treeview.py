"""`hillclimb tree` — the exploration tree of one search, live.

Roots (baseline, drafts) across the top, one row per operator step below,
every candidate a mark whose colour is its operator and whose silhouette is
its fate (see tree.py): filled = expanded, ring = scored but never built
on, diamond = the current best, dot = failed, faded = pruned. The accepted
lineage is drawn bold. Scrub back through time to watch the tree grow;
click a node for the candidate's detail; enter opens it in the candidate
screen.

Same architecture as graphview.py: pure functions up top (layout lives in
tree.py, the plotui adapter and label geometry here), thin Textual shells
below. The plot is a plotui Graph3d laid flat in the z=0 plane with the
orbit camera locked face-on — that keeps node picking, hover, selection and
labels identical to the knowledge graph without a second renderer.
"""

from __future__ import annotations

from rich.style import Style

from hillclimb.config import Config
from hillclimb.graphview import LABEL_ZOOM, VNode
from hillclimb.tree import FATES, SearchTree, TreeNode

ZOOM_FACTOR = 1.25
NODE_SIZE = 3.5
# world units per column / per depth row — rows a bit taller than columns so
# the fans of siblings read as fans, not as a grid
COL_W = 1.0
ROW_H = 1.6

OPERATOR_RGB = {
    "baseline": (138, 143, 152),
    "draft": (76, 141, 255),
    "debug": (255, 176, 46),
    "improve": (47, 191, 113),
    "ensemble": (197, 91, 255),
}
OPERATOR_STYLE = {
    "baseline": "bright_black", "draft": "blue", "debug": "yellow",
    "improve": "green", "ensemble": "magenta",
}
FATE_SHAPE = {
    "expanded": "disc", "best": "diamond", "discontinued": "ring",
    "failed": "dot", "pruned": "dot", "pending": "triangle",
}
FATE_GLYPH = {
    "expanded": "●", "best": "◆", "discontinued": "◯", "failed": "•", "pruned": "·", "pending": "▲",
}
DIM = 0.35  # colour factor for edges/nodes that lead nowhere
BEST_LABEL_STYLE = "bold rgb(255,200,40)"  # the star and name of the best: gold, never dimmed


def dim_rgb(rgb: tuple[int, int, int], factor: float = DIM) -> tuple[int, int, int]:
    return tuple(int(c * factor) for c in rgb)  # type: ignore[return-value]


def node_rgb(node: TreeNode, on_path: bool) -> tuple[int, int, int]:
    rgb = OPERATOR_RGB.get(node.operator, (160, 160, 160))
    if node.fate in ("failed", "pruned"):
        return dim_rgb(rgb, 0.5)
    if node.fate == "discontinued" and not on_path:
        return dim_rgb(rgb, 0.75)
    return rgb


def node_size(node: TreeNode, on_path: bool) -> float:
    if node.fate == "best":
        return NODE_SIZE * 1.7
    if on_path or node.fate == "expanded":
        return NODE_SIZE * 1.3
    if node.fate in ("failed", "pruned"):
        return NODE_SIZE * 0.8
    return NODE_SIZE


def node_label(node: TreeNode) -> str:
    """`id score`; the current best wears a star so it is the first thing the
    eye finds in a crowded tree."""
    name = f"{node.id} ★" if node.fate == "best" else node.id
    if node.score is None:
        return name
    return f"{name} {node.score:.4g}"


def world_xy(node: TreeNode) -> tuple[float, float]:
    """Tree column/depth → plot plane; +y is up on screen, so depth goes down."""
    return node.x * COL_W, -node.depth * ROW_H


def tree_extent(tree: SearchTree) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """The world-plane box `((x0, y0), (x1, y1))` the tree's nodes span."""
    if not tree.nodes:
        return None
    xs, ys = zip(*(world_xy(n) for n in tree.nodes))
    return (min(xs), min(ys)), (max(xs), max(ys))


def build_tree_plot(tree: SearchTree, *, selected: str | None = None, frame: SearchTree | None = None):
    """SearchTree → a fresh plotui Plot (one Graph3d trace in the z=0 plane,
    camera face-on at zoom 1) plus the flat-index → node-id list. Zoom and
    pan are the caller's to restore. `frame` pins the view to another tree's
    extent (the live tree, while scrubbing), so drawing a subset of its
    nodes does not re-centre the picture."""
    from hillclimb.theme import themed_plot

    ids = [n.id for n in tree.nodes]
    index_of = {node_id: i for i, node_id in enumerate(ids)}
    on_path = set(tree.accepted)
    plot = themed_plot()
    plot.set_show_box(False)
    plot.set_camera_state(0.0, 0.0, 1.0, 0.0, 0.0)
    extent = tree_extent(frame) if frame is not None else None
    if extent is not None:
        (x0, y0), (x1, y1) = extent
        plot.set_bounds((x0, y0, 0.0), (x1, y1, 0.0))
    if tree.nodes:
        xs, ys = zip(*(world_xy(n) for n in tree.nodes))
        edges = [e for e in tree.edges if e.src in index_of and e.dst in index_of]
        by_id = {n.id: n for n in tree.nodes}
        edge_colors = []
        for edge in edges:
            child = by_id[edge.dst]
            rgb = OPERATOR_RGB.get(child.operator, (160, 160, 160))
            if edge.kind == "ensemble-input":
                edge_colors.append(dim_rgb(OPERATOR_RGB["ensemble"], 0.6))
            elif edge.on_path:
                edge_colors.append(rgb)
            elif child.fate in ("failed", "pruned"):
                edge_colors.append(dim_rgb(rgb, 0.25))
            else:
                edge_colors.append(dim_rgb(rgb))
        plot.add_graph3d(
            list(xs), list(ys), [0.0] * len(ids),
            edges=[(index_of[e.src], index_of[e.dst]) for e in edges],
            node_colors=[node_rgb(n, n.id in on_path) for n in tree.nodes],
            size=NODE_SIZE,
            node_sizes=[node_size(n, n.id in on_path) for n in tree.nodes],
            edge_colors=edge_colors,
            node_shapes=[FATE_SHAPE.get(n.fate, "disc") for n in tree.nodes],
        )
        if selected in index_of:
            plot.set_selected(index_of[selected])
    return plot, ids


def label_nodes(tree: SearchTree) -> list[VNode]:
    """The tree as graphview VNodes, so place_labels can lay the labels out
    with the same collision rules the knowledge graph uses. `count` carries
    label priority: the accepted lineage wins a crowded row."""
    on_path = set(tree.accepted)
    out = []
    for node in tree.nodes:
        x, y = world_xy(node)
        out.append(VNode(
            id=node.id, type="candidate", label=node_label(node), x=x, y=y, z=0.0,
            count=3 if node.fate == "best" else 2 if node.id in on_path else 1,
            style=BEST_LABEL_STYLE if node.fate == "best" else "",
        ))
    return out


# The legend: operators (colour) then fates (silhouette), one line each from
# LEGEND_ROW down at the top left — same overlay mechanism and hit-test as the
# knowledge graph's, so it costs the canvas no columns. Each entry toggles its
# operator/fate off and on (hotkey 1-9, 0 for the tenth; click works too).
LEGEND_ENTRIES = tuple(OPERATOR_RGB) + FATES
LEGEND_KEYS = "1234567890"
LEGEND_ROW, LEGEND_COL = 1, 1
LEGEND_WIDTH = 2 + 2 + max(len(e) for e in LEGEND_ENTRIES) + 4  # "1 " + glyph + " " + name + count


def filter_hidden(tree: SearchTree, hidden: frozenset[str]) -> SearchTree:
    """Legend filter: drop every node whose operator or fate is hidden, and
    the edges that touched one. Positions are kept, so the rest of the tree
    stays put and the gaps show what is not drawn."""
    if not hidden:
        return tree
    nodes = tuple(n for n in tree.nodes if n.operator not in hidden and n.fate not in hidden)
    ids = {n.id for n in nodes}
    edges = tuple(e for e in tree.edges if e.src in ids and e.dst in ids)
    return SearchTree(nodes=nodes, edges=edges, accepted=tree.accepted)


def legend_spans(tree: SearchTree | None, hidden: frozenset[str] = frozenset()) -> list[tuple[int, int, str, str]]:
    """The legend as overlay spans `(row, col, text, style)`: hotkey, glyph,
    name, and for fates the count in the (unfiltered) tree. Hidden entries go
    dim and struck through, so the legend shows what the canvas is not
    drawing."""
    counts = tree.fate_counts() if tree is not None else {}
    spans: list[tuple[int, int, str, str]] = []
    for index, entry in enumerate(LEGEND_ENTRIES):
        row = LEGEND_ROW + index
        spans.append((row, LEGEND_COL, f"{LEGEND_KEYS[index]} ", "dim"))
        count = f" {counts[entry]}" if counts.get(entry) else ""
        if entry in hidden:
            spans.append((row, LEGEND_COL + 2, f"  {entry}{count}", "dim strike"))
        elif entry in OPERATOR_RGB:
            r, g, b = OPERATOR_RGB[entry]
            spans.append((row, LEGEND_COL + 2, f"● {entry}", f"rgb({r},{g},{b})"))
        else:
            style = "bright_black" if entry in ("failed", "pruned") else "white"
            spans.append((row, LEGEND_COL + 2, f"{FATE_GLYPH[entry]} {entry}{count}", style))
    return spans


def legend_entry_at(col: int, row: int) -> str | None:
    """The legend entry under canvas cell `(col, row)`, or None off the legend."""
    index = row - LEGEND_ROW
    if 0 <= index < len(LEGEND_ENTRIES) and LEGEND_COL <= col < LEGEND_COL + LEGEND_WIDTH:
        return LEGEND_ENTRIES[index]
    return None


def statusline(
    ref: str, state: str, tree: SearchTree, metric: str, higher_is_better: bool,
    position: tuple[int, int] | None = None,
) -> str:
    """The line above the canvas: where we are and how the exploration went.
    `position` is `(index, total)` among the store's searches, so the n/p
    cycling has a readout."""
    from hillclimb.watch import STATE_STYLE

    parts = [f"[bold]{ref}[/] [{STATE_STYLE.get(state, '')}]{state}[/]"]
    if position is not None and position[1] > 1:
        parts[0] += f" [dim]({position[0] + 1}/{position[1]}, n/p to switch)[/]"
    parts.append(f"{len(tree.nodes)} candidates · depth {tree.depth}")
    best = tree.node(tree.best_id) if tree.best_id else None
    direction = "higher" if higher_is_better else "lower"
    if best is not None and best.score is not None:
        parts.append(f"best [bold]{best.id} ★[/] {metric}={best.score:.5g} ({direction} is better)")
    return "  ·  ".join(parts)


# --- Textual widgets ---

from plotui import Plot  # noqa: E402
from plotui.textual import PlotWidget  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual import events  # noqa: E402
from textual.containers import Vertical  # noqa: E402
from textual.message import Message  # noqa: E402
from textual.widgets import Footer, Label, RichLog  # noqa: E402

from hillclimb.graphview import GraphKeys, TimeScrubber, place_labels_by_node  # noqa: E402
from hillclimb.header import HillclimbHeader, TimezoneMixin  # noqa: E402
from hillclimb.journal import Journal  # noqa: E402
from hillclimb.store import DataStore, SearchRecord, open_store, resolve_search  # noqa: E402
from hillclimb.theme import HILLCLIMB_CSS, apply_theme, themed_plot  # noqa: E402
from hillclimb.tree import build_tree, candidates_until, tree_events  # noqa: E402
from hillclimb.watch import (  # noqa: E402
    LiveScreen, candidate_detail_renderables, _mouse_event_x, _mouse_event_y,
)


class TreePlotWidget(PlotWidget):
    """The flat plotui view of one search tree. Camera is face-on and stays
    that way (a drag pans instead of rotating); scroll and +/- zoom, click
    selects, re-click activates. The plot is rebuilt from the
    screen-supplied tree on every change with zoom/pan carried over."""

    class NodeSelected(Message):
        def __init__(self, node_id: str | None):
            super().__init__()
            self.node_id = node_id

    class NodeActivated(Message):
        def __init__(self, node_id: str):
            super().__init__()
            self.node_id = node_id

    class TypeToggled(Message):
        """A legend entry was clicked."""

        def __init__(self, entry: str):
            super().__init__()
            self.entry = entry

    def __init__(self, **kwargs):
        super().__init__(themed_plot(), **kwargs)
        self.selected: str | None = None
        self.hidden: frozenset[str] = frozenset()
        self._tree: SearchTree | None = None      # unfiltered (legend counts)
        self._frame: SearchTree | None = None     # whose extent the view is pinned to
        self._visible: SearchTree | None = None   # what is drawn
        self._labels: list[VNode] = []
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._label_cells: dict[tuple[int, int], str] = {}  # (row, col) -> node id
        self._hover: str | None = None

    # -- data --

    def set_tree(self, tree: SearchTree, frame: SearchTree | None = None) -> None:
        """`frame` (the live tree while scrubbing) fixes the picture's extent,
        so `tree` — a subset of it — draws every node at the same place."""
        self._tree = tree
        self._frame = frame
        if self.selected is not None and tree.node(self.selected) is None:
            self.selected = None
            self.post_message(self.NodeSelected(None))
        self.rebuild()

    def rebuild(self) -> None:
        if self._tree is None:
            return
        _yaw, _pitch, zoom, pan_x, pan_y = self._plot.camera_state()
        self._visible = filter_hidden(self._tree, self.hidden)
        if self.selected is not None and self._visible.node(self.selected) is None:
            self.selected = None
            self.post_message(self.NodeSelected(None))
        plot, self._ids = build_tree_plot(self._visible, selected=self.selected, frame=self._frame)
        self._index = {node_id: i for i, node_id in enumerate(self._ids)}
        self._labels = label_nodes(self._visible)
        if self._hover is not None:
            if self._hover in self._index:
                plot.set_hovered(self._index[self._hover])
            else:
                self._hover = None
        plot.set_camera_state(0.0, 0.0, zoom, pan_x, pan_y)  # face-on, always
        self._plot = plot
        self._refresh_overlay()
        self.invalidate()

    # -- geometry --

    def _px_dims(self) -> tuple[int, int]:
        w, h = max(self.size.width, 1), max(self.size.height, 1)
        return w * self._cell_w, h * self._cell_h

    def _node_at(self, event: events.MouseEvent) -> str | None:
        x = _mouse_event_x(event) - self.region.x
        y = _mouse_event_y(event) - self.region.y
        px_w, px_h, px, py, radius = self._pixel_geometry(x, y)
        flat = self._plot.pick_px(px_w, px_h, px, py, radius)
        if flat is not None and flat < len(self._ids):
            return self._ids[flat]
        # not on a mark: the candidate's label next to it counts as the node too
        return self._label_cells.get((int(y), int(x)))

    def _refresh_overlay(self) -> None:
        if self._mode == "unsupported":
            return
        legend = legend_spans(self._tree, self.hidden)
        self._label_cells = {}
        if self._tree is None or not self._ids or self.size.width <= 0:
            self.set_overlay([(r, c, t, Style.parse(st)) for r, c, t, st in legend])
            return
        # legend first: set_overlay keeps the first span where two overlap, so
        # a node label never paints over a legend line. Labels always on (zoom
        # pinned past the graph's threshold); the collision mask thins a crowd
        placed = place_labels_by_node(
            self._labels,
            self._plot.project_nodes(*self._px_dims()),
            cols=self.size.width,
            rows=self.size.height,
            cell_px=(self._cell_w, self._cell_h),
            zoom=LABEL_ZOOM,
            selected=self.selected,
            hovered=self._hover,
        )
        for (row, col, text, _style), node_id in placed:
            for x in range(col, col + len(text)):
                self._label_cells[(row, x)] = node_id
        spans = legend + [span for span, _node_id in placed]
        self.set_overlay([(r, c, t, Style.parse(st)) for r, c, t, st in spans])

    def on_resize(self, event: events.Resize) -> None:
        self._refresh_overlay()

    # -- interaction hooks (PlotWidget routes all input through these) --

    def apply_zoom(self, factor: float) -> None:
        super().apply_zoom(factor)
        self._refresh_overlay()

    # PlotWidget turns one dragged cell (or an arrow key) into this much yaw/
    # pitch; undoing it recovers the cell delta so a plain drag pans exactly
    # as far as the pointer moved — the tree stays under the mouse.
    ROTATE_PER_CELL = 0.03

    def apply_rotate(self, d_yaw: float, d_pitch: float) -> None:
        # a tree is flat: a drag pans instead of tilting it
        self.apply_pan(
            d_yaw / self.ROTATE_PER_CELL * self._cell_w,
            d_pitch / self.ROTATE_PER_CELL * self._cell_h,
        )

    def apply_pan(self, dx: float, dy: float) -> None:
        super().apply_pan(dx, dy)
        self._refresh_overlay()

    def apply_reset(self) -> None:
        self._plot.set_camera_state(0.0, 0.0, 1.0, 0.0, 0.0)
        self.invalidate()
        self._refresh_overlay()

    def set_hidden(self, hidden: frozenset[str]) -> None:
        self.hidden = hidden
        self.rebuild()

    def on_click_at(self, event: events.MouseUp) -> None:
        entry = legend_entry_at(
            _mouse_event_x(event) - self.region.x, _mouse_event_y(event) - self.region.y
        )
        if entry is not None:
            self.post_message(self.TypeToggled(entry))
            return
        node_id = self._node_at(event)
        if node_id != self.selected:
            self.selected = node_id
            self._plot.set_selected(self._index.get(node_id) if node_id else None)
            self.invalidate()
            self._refresh_overlay()
            self.post_message(self.NodeSelected(node_id))
        elif node_id is not None:
            self.post_message(self.NodeActivated(node_id))

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._dragging:
            return
        hover = self._node_at(event)
        if hover != self._hover:
            self._hover = hover
            element = self._index.get(hover) if hover is not None else None
            if self._plot.set_hovered(element):
                self.invalidate()
            self._refresh_overlay()

    # -- the screen's camera contract --

    def zoom(self, factor: float) -> None:
        self.apply_zoom(factor)

    def fit(self) -> None:
        self.apply_reset()

    def select(self, node_id: str | None) -> None:
        self.selected = node_id
        self.rebuild()
        self.post_message(self.NodeSelected(node_id))


class TreeKeys(GraphKeys):
    POINTER = [
        ("drag", "pan"),
        ("scroll", "zoom"),
        ("arrows", "pan"),
        ("r", "reset view"),
        ("click", "select (mark or name)"),
        ("click again", "open"),
        ("click legend", "hide an operator/fate"),
        ("n / p", "next / previous search"),
    ]


NODE_DETAIL_WIDTH = 48  # the slide-out detail dock; content renders to fit it


class TreeScreen(LiveScreen):
    """Canvas + time scrubber + candidate detail for one search. Reached via
    `hillclimb tree [search]`."""

    BINDING_GROUP_TITLE = "tree"
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
        # One binding per legend slot; only the first is described, so the
        # `?` panel shows a single "1-9 0" row for the lot.
        Binding("1", "toggle_type(0)", "hide/show an operator or fate", show=False, key_display="1-9 0"),
        *(Binding(LEGEND_KEYS[i], f"toggle_type({i})", show=False) for i in range(1, len(LEGEND_ENTRIES))),
        Binding("question_mark", "toggle_help", "keys"),
        Binding("q", "app.quit", "quit"),
    ]

    DEFAULT_CSS = """
    TreeScreen #treeline { height: 1; padding: 0 1; background: $surface; }
    TreeScreen #node-detail { dock: right; width: 48; display: none; padding: 0 1; }
    TreeScreen #tree-stage { width: 1fr; height: 1fr; }
    TreeScreen #time-scrubber { height: 2; background: $surface; padding: 0 1; }
    TreeScreen #tree-canvas { width: 1fr; height: 1fr; }
    """

    def __init__(self, config: Config, search: str | None = None):
        super().__init__()
        self.config = config
        self.search = search
        self._store: DataStore | None = None
        self._record: SearchRecord | None = None
        self._journal: Journal | None = None
        self._fingerprint: tuple | None = None
        self._tree: SearchTree | None = None
        self._live_tree: SearchTree | None = None  # the full tree: the layout every scrub step keeps
        self._hidden: frozenset[str] = frozenset()
        self._position: tuple[int, int] | None = None  # (index, total) among the store's searches

    def compose(self) -> ComposeResult:
        yield HillclimbHeader()
        yield Label(id="treeline")
        yield RichLog(id="node-detail", wrap=True, markup=False, auto_scroll=False, min_width=1)
        with Vertical(id="tree-stage"):
            yield TreePlotWidget(id="tree-canvas")
            yield TimeScrubber(id="time-scrubber")
        yield Footer()

    def on_mount(self) -> None:
        self.start_live()
        self._canvas().focus()

    # -- data --

    def _resolve(self) -> SearchRecord | None:
        if self._store is None:
            self._store = open_store(self.config)
        try:
            record = resolve_search(self._store, self.search)
        except LookupError as exc:
            self.query_one("#treeline", Label).update(str(exc))
            return None
        self._record = record
        return record

    def _all_searches(self) -> list[SearchRecord]:
        return self._store.searches() if self._store is not None else []

    def _switch_search(self, step: int) -> None:
        """Move to the next/previous search (by start time, wrapping) and
        start over on it: fresh journal, live time, nothing selected; the
        legend filter and camera carry across so trees compare like for like."""
        if self._record is None:
            return
        records = self._all_searches()
        keys = [r.key for r in records]
        if self._record.key not in keys or len(records) < 2:
            return
        target = records[(keys.index(self._record.key) + step) % len(records)]
        self._record = target
        self.search = target.ref
        self._journal = None
        self._fingerprint = None
        canvas = self._canvas()
        canvas.selected = None
        canvas._hover = None
        self._show_detail(None)
        self.query_one("#time-scrubber", TimeScrubber).set_index(None)
        self.refresh_data()

    def refresh_data(self) -> None:
        canvas = self._canvas()
        if canvas.dragging:
            return
        record = self._record if self._record is not None else self._resolve()
        if record is None:
            return
        if self._store is not None:
            fresh = self._store.search(record.key)
            if fresh is not None:
                record = self._record = fresh
        self._journal = Journal(self._store.journal(record.key))  # type: ignore[union-attr]
        candidates = list(self._journal.candidates.values())
        fingerprint = (
            record.state,
            tuple((c.candidate_id, c.status, c.val_score, c.pruned, c.finished_at) for c in candidates),
        )
        if fingerprint == self._fingerprint:
            return
        self._fingerprint = fingerprint
        self._live_tree = build_tree(candidates, bool(record.meta.higher_is_better))
        keys = [r.key for r in self._all_searches()]
        self._position = (keys.index(record.key), len(keys)) if record.key in keys else None
        self.query_one("#time-scrubber", TimeScrubber).set_events(tree_events(candidates))
        self._apply_view()

    def _apply_view(self) -> None:
        if self._record is None or self._journal is None:
            return
        scrubber = self.query_one("#time-scrubber", TimeScrubber)
        until = None
        if scrubber.index is not None and scrubber.events_list:
            until = scrubber.events_list[scrubber.index]
        candidates = candidates_until(list(self._journal.candidates.values()), until)
        higher = bool(self._record.meta.higher_is_better)
        self._tree = build_tree(candidates, higher, layout=self._live_tree)
        self.query_one("#treeline", Label).update(
            statusline(
                self._record.ref, self._record.state, self._tree, self._record.meta.metric, higher,
                position=self._position,
            )
        )
        self._canvas().set_tree(self._tree, frame=self._live_tree)
        if self._canvas().selected is not None:
            self._show_detail(self._canvas().selected)

    def _show_detail(self, node_id: str | None) -> None:
        detail = self.query_one("#node-detail", RichLog)
        if node_id is None or self._record is None or self._journal is None:
            detail.styles.display = "none"
            return
        detail.clear()
        # render at the dock's content width (48 minus padding and scrollbar):
        # RichLog's default min_width is 78, which forces horizontal scroll
        width = NODE_DETAIL_WIDTH - 3
        for renderable in candidate_detail_renderables(self._record, self._journal, node_id):
            detail.write(renderable, width=width, expand=True)
        detail.styles.display = "block"

    # -- messages --

    def on_tree_plot_widget_node_selected(self, message: TreePlotWidget.NodeSelected) -> None:
        self._show_detail(message.node_id)

    def on_tree_plot_widget_node_activated(self, message: TreePlotWidget.NodeActivated) -> None:
        self._open_candidate(message.node_id)

    def on_time_scrubber_time_changed(self, message: TimeScrubber.TimeChanged) -> None:
        self._apply_view()

    def on_tree_plot_widget_type_toggled(self, message: TreePlotWidget.TypeToggled) -> None:
        self._toggle_type(message.entry)

    def action_toggle_type(self, index: int) -> None:
        if 0 <= index < len(LEGEND_ENTRIES):
            self._toggle_type(LEGEND_ENTRIES[index])

    def _toggle_type(self, entry: str) -> None:
        self._hidden = self._hidden ^ {entry}
        self._canvas().set_hidden(self._hidden)

    # -- actions --

    def _canvas(self) -> TreePlotWidget:
        return self.query_one("#tree-canvas", TreePlotWidget)

    def action_zoom_in(self) -> None:
        self._canvas().zoom(ZOOM_FACTOR)

    def action_zoom_out(self) -> None:
        self._canvas().zoom(1 / ZOOM_FACTOR)

    def action_fit(self) -> None:
        self._canvas().fit()

    def action_select_best(self) -> None:
        if self._tree is not None and self._tree.best_id:
            self._canvas().select(self._tree.best_id)

    def action_next_search(self) -> None:
        self._switch_search(1)

    def action_prev_search(self) -> None:
        self._switch_search(-1)

    def action_scrub_back(self) -> None:
        self.query_one("#time-scrubber", TimeScrubber).step(-1)

    def action_scrub_forward(self) -> None:
        self.query_one("#time-scrubber", TimeScrubber).step(1)

    def action_scrub_live(self) -> None:
        self.query_one("#time-scrubber", TimeScrubber).set_index(None)
        self._apply_view()

    def action_toggle_help(self) -> None:
        panel = self.query(TreeKeys)
        if panel:
            panel.remove()
        else:
            self.mount(TreeKeys())

    def action_activate(self) -> None:
        canvas = self._canvas()
        if canvas.selected is not None:
            self._open_candidate(canvas.selected)

    def _open_candidate(self, node_id: str) -> None:
        if self._record is None:
            return
        from hillclimb.watch import CandidateScreen

        self.app.push_screen(CandidateScreen(self.config, self._record.search_dir, open_candidate_id=node_id))

    def action_dismiss_or_back(self) -> None:
        canvas = self._canvas()
        if canvas.selected is not None:
            canvas.select(None)
            return
        if len(self.app.screen_stack) > 2:
            self.app.pop_screen()
        else:
            self.app.exit()


class TreeApp(TimezoneMixin, App):
    """Standalone shell for `hillclimb tree`."""

    BINDINGS = [Binding("t", "choose_timezone", "time zone", show=False)]
    CSS = HILLCLIMB_CSS

    def __init__(self, config: Config | None = None, search: str | None = None):
        super().__init__()
        self.config = config or Config.load()
        self.search = search

    def on_mount(self) -> None:
        apply_theme(self)
        self._init_timezone()
        self.push_screen(TreeScreen(self.config, self.search))
