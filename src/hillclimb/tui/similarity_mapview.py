"""`hillclimb similarity map` — candidates embedded by pairwise distance.

Every candidate is a node of one 3D graph laid out by `similarity_map`:
nearby dots really are similar solutions (the reference cube next door
measures everything from one candidate instead). Lineage edges join parent
to child, dimmed, so a long edge is an operator that jumped far and a
knot of short ones is an agent fiddling; the best-so-far sequence is
drawn as a gold trail through the cloud.

Search scope: colour is the score's rank bin (cold→hot), size grows with
rank, the champion is a gold diamond, the origin a white open diamond,
shape carries the tree's fate. Run scope (every search of the problem in
a study run): colour is the experiment, as in `chart`.

Keys: `m` cycles the metric (behavioral → structural → blend), `v` swaps
to the reference cube (`r` is plotui's own reset-view key), `space` replays the search growing candidate by
candidate, `s` toggles the idle spin (any drag stops it too). Hovering a
node puts its id, operator, score and its three distances to the
selected node (else the origin) in the statusline; clicking selects it
and dims everything outside its ancestors and descendants.

Pure functions up top (styles, highlight colours, replay prefix, extent),
thin Textual shells below. Screens derive from similarityview's
`SimilarityBase`/`RunScopeMixin`, so store resolution and n/p stepping
are the cube's.
"""

from __future__ import annotations

import math
from dataclasses import replace

from plotui import Plot
from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Label

from hillclimb.config import Config
from hillclimb.tui.header import HillclimbHeader
from hillclimb.harness.journal import Journal
from hillclimb.tui.similarity_map import METRICS, MapNode, MapView, build_map, build_run_map
from hillclimb.tui.similarityview import (
    BEST_RGB,
    BEST_SIZE,
    REFERENCE_RGB,
    SCORE_RGB,
    START_CAMERA,
    UNSCORED_RGB,
    UNSCORED_SIZE,
    RunScopeMixin,
    SimilarityBase,
    experiment_colour,
    experiment_legend,
    rank_size,
    search_inputs,
    run_state,
)
from hillclimb.harness.store import SearchRecord
from hillclimb.tui.theme import themed_plot
from hillclimb.tui.treeview import FATE_SHAPE, dim_rgb
from hillclimb.tui.watch import STATE_STYLE

from plotui.textual import PlotWidget

RGB = tuple[int, int, int]

TRAIL_RGB = (255, 200, 40)      # the champion's gold, as a line
EDGE_DIM = 0.45                 # lineage edges sit behind the nodes
HIGHLIGHT_DIM = 0.22            # everything outside the selected lineage
SPIN_STEP = 0.006               # radians per tick at SPIN_HZ
SPIN_HZ = 30
REPLAY_S = 0.12                 # seconds per candidate when replaying growth
_NICE = (1.0, 2.0, 5.0)


# ---------------------------------------------------------------------------
# pure


def node_colour(view: MapView, node: MapNode) -> RGB:
    if node.origin:
        return REFERENCE_RGB
    if node.best:
        return BEST_RGB
    if view.scope == "run":
        return experiment_colour(view, node.experiment)  # type: ignore[arg-type]
    if node.bin is None:
        return UNSCORED_RGB
    return SCORE_RGB[node.bin]


def node_size(node: MapNode) -> float:
    if node.origin or node.best:
        return BEST_SIZE
    return rank_size(node.bin)


def node_shape(node: MapNode) -> str:
    if node.best:
        return "diamond"
    if node.origin:
        return "diamond-open"
    return FATE_SHAPE.get(node.fate, "disc")


def node_styles(view: MapView) -> tuple[list[RGB], list[float], list[str]]:
    return (
        [node_colour(view, n) for n in view.nodes],
        [node_size(n) for n in view.nodes],
        [node_shape(n) for n in view.nodes],
    )


def edge_colours(view: MapView, colours: list[RGB]) -> list[RGB]:
    """An edge takes its child's colour, dimmed; edges on the accepted
    path a little less so the climb reads through the cloud."""
    out = []
    for parent, child in view.edges:
        node = view.nodes[child]
        factor = EDGE_DIM * 1.5 if node.on_path else EDGE_DIM
        out.append(dim_rgb(colours[child], min(factor, 1.0)))
    return out


def lineage_of(view: MapView, index: int) -> set[int]:
    """The node, its ancestors and its descendants — what stays lit when
    it is selected."""
    by_id = {n.id: i for i, n in enumerate(view.nodes)}
    keep = {index}
    node = view.nodes[index]
    while node.parent_id is not None and node.parent_id in by_id and by_id[node.parent_id] not in keep:
        keep.add(by_id[node.parent_id])
        node = view.nodes[by_id[node.parent_id]]
    children: dict[str, list[int]] = {}
    for i, n in enumerate(view.nodes):
        if n.parent_id is not None:
            children.setdefault(n.parent_id, []).append(i)
    frontier = [index]
    while frontier:
        current = frontier.pop()
        for child in children.get(view.nodes[current].id, []):
            if child not in keep:
                keep.add(child)
                frontier.append(child)
    return keep


def highlight_colours(
    view: MapView, colours: list[RGB], edges: list[RGB], selected: int | None,
) -> tuple[list[RGB], list[RGB]]:
    """The colour lists with everything outside the selected node's lineage
    dimmed; unchanged when nothing is selected."""
    if selected is None:
        return colours, edges
    keep = lineage_of(view, selected)
    nodes = [c if i in keep else dim_rgb(c, HIGHLIGHT_DIM) for i, c in enumerate(colours)]
    lit = [
        c if (a in keep and b in keep) else dim_rgb(c, HIGHLIGHT_DIM)
        for (a, b), c in zip(view.edges, edges)
    ]
    return nodes, lit


def nice_extent(view: MapView) -> float:
    """Half the cube edge: the layout's largest coordinate rounded up to a
    1/2/5 step, so the frame only jumps when the cloud outgrows it and dots
    keep their pixels between rebuilds."""
    biggest = max((abs(v) for n in view.nodes for v in (n.x, n.y, n.z)), default=0.0)
    if biggest <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(biggest))
    for step in _NICE:
        candidate = step * 10.0 ** exponent
        if candidate >= biggest:
            return candidate
    return 10.0 ** (exponent + 1)


def prefix_view(view: MapView, n: int) -> MapView:
    """The first `n` nodes of a view at their full-view positions — one
    frame of the growth replay; edges and trail restricted to them."""
    n = max(0, min(n, len(view.nodes)))
    return replace(
        view,
        nodes=view.nodes[:n],
        edges=tuple((a, b) for a, b in view.edges if a < n and b < n),
        trail=tuple(i for i in view.trail if i < n),
        behavioral=view.behavioral[:n, :n] if view.behavioral is not None else None,
        structural=view.structural[:n, :n] if view.structural is not None else None,
        lineage=view.lineage[:n, :n] if view.lineage is not None else None,
    )


def hover_readout(view: MapView, hovered: int, anchor: int | None) -> str:
    """`id operator score · to <anchor>: behav struct lineage`."""
    node = view.nodes[hovered]
    parts = [f"[bold]{node.id}[/] {node.operator}"]
    if node.score is not None:
        parts.append(f"{node.score:.4g}")
    if anchor is None:
        anchor = next((i for i, n in enumerate(view.nodes) if n.origin), None)
    if (
        anchor is not None and anchor != hovered
        and view.behavioral is not None and view.structural is not None and view.lineage is not None
    ):
        parts.append(
            f"· to {view.nodes[anchor].id}: behav {view.behavioral[hovered, anchor]:.3g} "
            f"struct {view.structural[hovered, anchor]:.2f} lineage {view.lineage[hovered, anchor]:.0f}"
        )
    return " ".join(parts)


def build_map_plot(view: MapView, selected: int | None = None) -> tuple[Plot, int | None]:
    """The plot and the graph trace's handle (None when there is nothing
    to draw). Node index i of `view.nodes` is flat node index i of the
    plot: the graph is its only pickable trace, and the trail is a line."""
    plot = themed_plot()
    if not view.nodes:
        return plot, None
    colours, sizes, shapes = node_styles(view)
    edges = edge_colours(view, colours)
    colours, edges = highlight_colours(view, colours, edges, selected)
    handle = plot.add_graph3d(
        [n.x for n in view.nodes], [n.y for n in view.nodes], [n.z for n in view.nodes],
        edges=list(view.edges), node_colors=colours, size=UNSCORED_SIZE,
        node_sizes=sizes, edge_colors=edges, node_shapes=shapes,
    )
    if len(view.trail) > 1:
        trail = [view.nodes[i] for i in view.trail]
        plot.add_line3d(
            [n.x for n in trail], [n.y for n in trail], [n.z for n in trail],
            color=TRAIL_RGB, width=2.0, name="best-so-far",
        )
    if selected is not None:
        plot.set_selected(selected)
    extent = nice_extent(view)
    plot.set_bounds((-extent, -extent, -extent), (extent, extent, extent))
    return plot, handle


def statusline(
    ref: str, state: str, view: MapView | None, position: tuple[int, int] | None = None,
    hover: str = "", spinning: bool = False, replaying: int | None = None,
) -> str:
    parts = [f"[bold]{ref}[/] [{STATE_STYLE.get(state, '')}]{state}[/]"]
    if position is not None and position[1] > 1:
        parts.append(f"({position[0] + 1}/{position[1]})")
    if view is None:
        return "  ".join(parts)
    if view.unavailable is not None:
        parts.append(view.unavailable)
        return "  ".join(parts)
    parts.append(f"map by [bold]{view.metric}[/] ({view.mode})")
    if replaying is not None:
        parts.append(f"[bold]replay {replaying}/{len(view.nodes)}[/]")
    elif view.scope == "run":
        parts.append(f"{len(view.nodes)} placed over {view.n_searches} searches")
    else:
        parts.append(f"{len(view.nodes)} placed")
    if view.n_unpositioned:
        parts.append(f"[dim]{view.n_unpositioned} unpositioned[/]")
    parts.append(f"[dim]stress {view.stress:.2f}[/]")
    if hover:
        parts.append(hover)
    elif view.scope == "run" and view.experiments:
        parts.append(experiment_legend(view))  # type: ignore[arg-type]
    if spinning:
        parts.append("[dim]spinning[/]")
    return "  ".join(parts)


def next_metric(metric: str) -> str:
    return METRICS[(METRICS.index(metric) + 1) % len(METRICS)]


# ---------------------------------------------------------------------------
# widgets


class MapPlotWidget(PlotWidget):
    """Free-orbit pickable graph that survives data refreshes: the camera
    carries across rebuilds, selection and hover are re-applied by node
    id, and the idle spin stops on the first drag."""

    def __init__(self, **kwargs):
        super().__init__(themed_plot(), pickable=True, **kwargs)
        self._view: MapView | None = None
        self._handle: int | None = None
        self.selected: str | None = None
        self.spinning = True

    @property
    def view(self) -> MapView | None:
        return self._view

    def set_view(self, view: MapView) -> None:
        camera = self._plot.camera_state() if self._view is not None else START_CAMERA
        selected = view.index(self.selected) if self.selected else None
        if self.selected and selected is None:
            self.selected = None
        plot, self._handle = build_map_plot(view, selected)
        plot.set_camera_state(*camera)
        self._plot = plot
        self._view = view
        self.invalidate()

    def clear_view(self) -> None:
        if self._view is None:
            return
        self._plot = themed_plot()
        self._view = None
        self._handle = None
        self.invalidate()

    def select(self, index: int | None) -> None:
        """Select by flat index (what a pick reports) and re-light the
        graph: the lineage stays bright, the rest dims."""
        if self._view is None or self._handle is None:
            return
        self.selected = self._view.nodes[index].id if index is not None else None
        colours, _sizes, _shapes = node_styles(self._view)
        edges = edge_colours(self._view, colours)
        colours, edges = highlight_colours(self._view, colours, edges, index)
        self._plot.set_graph_colors(self._handle, colours, edges)
        self._plot.set_selected(index)
        self.invalidate()

    def spin_step(self) -> None:
        if self.spinning and not self.dragging and self._view is not None:
            self._plot.spin(SPIN_STEP)
            self.invalidate()

    # -- the camera contract: a drag ends the idle spin --

    def apply_rotate(self, d_yaw: float, d_pitch: float) -> None:
        self.spinning = False
        super().apply_rotate(d_yaw, d_pitch)

    def on_click_at(self, event) -> None:
        # the default picks and posts ElementPicked; the screen then calls
        # `select`, which is where the dimming happens
        self.spinning = False
        super().on_click_at(event)


class MapScreen(SimilarityBase):
    """Canvas + statusline for one search's pairwise map. Reached via
    `hillclimb similarity map [search]` or `m` from the reference cube."""

    BINDINGS = [
        Binding("m", "cycle_metric", "metric", tooltip="behavioral → structural → blend"),
        Binding("v", "open_reference", "reference view", tooltip="the same candidates as distances from one reference"),
        Binding("space", "replay", "replay", tooltip="watch the search grow candidate by candidate"),
        Binding("s", "toggle_spin", "spin", show=False, tooltip="start or stop the idle spin"),
    ]

    def __init__(self, config: Config, search: str | None = None, metric: str = "behavioral"):
        super().__init__(config, search)
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
        self.metric = metric
        self._view: MapView | None = None
        self._previous: dict[str, tuple[float, float, float]] | None = None
        self._hover: str = ""
        self._ref: str = ""
        self._state: str = ""
        self._replay: int | None = None

    def compose(self) -> ComposeResult:
        yield HillclimbHeader()
        yield Label(id="similarityline")
        yield MapPlotWidget(id="similarity-canvas")
        yield Footer()

    def on_mount(self) -> None:
        # Textual dispatches on_mount for every class in the MRO, so the
        # base's start_live runs on its own — only the spin is ours.
        self.set_interval(1 / SPIN_HZ, self._canvas().spin_step)

    def _canvas(self) -> MapPlotWidget:
        return self.query_one("#similarity-canvas", MapPlotWidget)

    # -- actions --

    def action_cycle_metric(self) -> None:
        self.metric = next_metric(self.metric)
        self._fingerprint = None
        self.refresh_data()

    def action_open_reference(self) -> None:
        from hillclimb.tui.similarityview import SimilarityScreen

        self.app.switch_screen(SimilarityScreen(self.config, self.search))

    def action_toggle_spin(self) -> None:
        canvas = self._canvas()
        canvas.spinning = not canvas.spinning
        self._update_statusline()

    def action_replay(self) -> None:
        if self._view is None or self._view.unavailable is not None or not self._view.nodes:
            return
        if self._replay is not None:
            self._end_replay()
            return
        self._replay = 0
        self._replay_timer = self.set_interval(REPLAY_S, self._replay_step)
        self._replay_step()

    def _replay_step(self) -> None:
        if self._view is None or self._replay is None:
            return
        self._replay += 1
        self._canvas().set_view(prefix_view(self._view, self._replay))
        self._update_statusline()
        if self._replay >= len(self._view.nodes):
            self._end_replay()

    def _end_replay(self) -> None:
        timer = getattr(self, "_replay_timer", None)
        if timer is not None:
            timer.stop()
        self._replay = None
        if self._view is not None:
            self._canvas().set_view(self._view)
        self._update_statusline()

    def _on_switch(self) -> None:
        self._previous = None
        self._hover = ""
        self._canvas().selected = None
        if self._replay is not None:
            self._end_replay()
        super()._on_switch()

    # -- pointer --

    def on_plot_widget_element_hovered(self, message: PlotWidget.ElementHovered) -> None:
        view = self._canvas().view
        element = message.element
        if view is None or element is None or element[0] != "node" or element[1] >= len(view.nodes):
            self._hover = ""
        else:
            anchor = view.index(self._canvas().selected) if self._canvas().selected else None
            self._hover = hover_readout(view, element[1], anchor)
        self._update_statusline()

    def on_plot_widget_element_picked(self, message: PlotWidget.ElementPicked) -> None:
        element = message.element
        index = element[1] if element is not None and element[0] == "node" else None
        self._canvas().select(index)
        self._update_statusline()

    # -- data --

    def _update_statusline(self) -> None:
        self._statusline().update(statusline(
            self._ref, self._state, self._view, position=self._position, hover=self._hover,
            spinning=self._canvas().spinning, replaying=self._replay,
        ))

    def _show(self, view: MapView, ref: str, state: str) -> None:
        self._view = view
        self._ref, self._state = ref, state
        canvas = self._canvas()
        if view.unavailable is not None:
            canvas.clear_view()
        else:
            self._previous = view.positions()
            canvas.set_view(view)
        self._update_statusline()

    def refresh_data(self) -> None:
        if self._canvas().dragging or self._replay is not None:
            return
        record = self._record if self._record is not None else self._resolve()
        if record is None:
            return
        if self._store is not None:
            fresh = self._store.search(record.key)
            if fresh is not None:
                record = self._record = fresh
        keys = [r.key for r in (self._store.searches() if self._store else [])]
        self._position = (keys.index(record.key), len(keys)) if record.key in keys else None

        journal = Journal(self._store.journal(record.key))  # type: ignore[union-attr]
        candidates = list(journal.candidates.values())
        fingerprint = (
            record.key, record.state, self.metric,
            tuple((c.candidate_id, c.status, c.val_score, c.pruned, c.finished_at) for c in candidates),
        )
        if fingerprint == self._fingerprint:
            return
        self._fingerprint = fingerprint

        view = build_map(
            candidates, record.search_dir, bool(record.meta.higher_is_better), metric=self.metric,
            output_artifacts=record.meta.output_artifacts,
            fingerprint_path=self._fingerprint_path(record), previous=self._previous,
        )
        self._show(view, record.ref, record.state)


class RunMapScreen(RunScopeMixin, MapScreen):
    """Run scope: every search of one problem in a study run in one
    map, coloured by experiment. `n`/`p` step through the run's problems."""

    BINDINGS = [
        Binding("n", "next_search", "next problem", tooltip="the run's next problem"),
        Binding("p", "prev_search", "prev problem", show=False, tooltip="the run's previous problem"),
    ]

    def __init__(self, config: Config, run_id: str, problem_key: str, metric: str = "behavioral"):
        super().__init__(config, search=None, metric=metric)
        self.run_id = run_id
        self.problem_key = problem_key

    def action_open_reference(self) -> None:
        from hillclimb.tui.similarityview import RunSimilarityScreen

        self.app.switch_screen(RunSimilarityScreen(self.config, self.run_id, self.problem_key))

    def refresh_data(self) -> None:
        if self._canvas().dragging or self._replay is not None:
            return
        records, ref = self._run_records()
        if not records:
            self._canvas().clear_view()
            self._statusline().update(f"no searches for {self.problem_key} in run {self.run_id}")
            return
        inputs = search_inputs(self._store, records)  # type: ignore[arg-type]
        fingerprint = self._run_fingerprint(records, inputs, self.run_id, self.problem_key, self.metric)
        if fingerprint == self._fingerprint:
            return
        self._fingerprint = fingerprint

        first: SearchRecord = records[0]
        view = build_run_map(
            inputs, bool(first.meta.higher_is_better), metric=self.metric,
            output_artifacts=first.meta.output_artifacts,
            fingerprint_path=self._fingerprint_path(first), previous=self._previous,
            problem_key=self.problem_key,
        )
        self._show(view, ref, run_state(records))
