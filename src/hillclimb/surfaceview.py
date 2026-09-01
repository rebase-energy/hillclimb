"""`hillclimb surface` — one search's candidates on the problem's terrain.

The problem's `landscape.py` supplies the ground truth surface; every
candidate whose verifier journaled a position (`surface_metrics` keys in
Trial.metrics) is a mark on it, coloured by its tree.py fate — gold diamond
of the tree becomes a gold dot here — and the accepted lineage is a white
path draped along the terrain. A white marker sits on the grid's summit, so
an unclimbed global peak is legible at a glance.

Same architecture as treeview.py: pure functions up top, thin Textual
shells below. The camera is plotui's free orbit — the one deliberate
difference from the tree's locked face-on view — starting from above
(negative pitch: plotui's default pitch would view the terrain from
underneath and render the peaks as stalactites).

`hillclimb similarity` (`similarityview.py`) shares this screen shape: three
candidate-distance coordinates, score as the colour scale, no terrain.
"""

from __future__ import annotations

from plotui import Plot
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Label

from hillclimb.config import Config
from hillclimb.header import HillclimbHeader, TimezoneMixin
from hillclimb.journal import Journal
from hillclimb.problem import ProblemSpec, load_problem
from hillclimb.store import DataStore, SearchRecord, open_store, resolve_search
from hillclimb.surface import LandscapeError, SurfaceView, build_surface, load_landscape
from hillclimb.theme import HILLCLIMB_CSS, apply_theme, themed_plot
from hillclimb.watch import STATE_STYLE, LiveScreen

from plotui.textual import PlotWidget

# Fate colours (tree.py vocabulary). The best keeps the tree's gold; the
# summit marker is white so the two never read as one another.
FATE_RGB = {
    "best": (255, 200, 40),
    "expanded": (47, 191, 113),
    "discontinued": (110, 140, 180),
    "failed": (200, 80, 80),
    "pruned": (90, 90, 90),
    "pending": (235, 235, 235),
}
NODE_SIZE = 5.0
BEST_SIZE = 7.0
SUMMIT_RGB = (255, 255, 255)
LINEAGE_RGB = (255, 255, 255)
# Start above the terrain: plotui's default pitch (+0.5) orbits underneath a
# surface and shows the peaks hanging downward.
START_CAMERA = (0.9, -0.8, 1.0, 0.0, 0.0)


def build_surface_plot(view: SurfaceView) -> Plot:
    plot = themed_plot()
    plot.add_surface3d(list(view.xs), list(view.ys), [list(row) for row in view.zgrid])
    if view.lineage.xs:
        plot.add_line3d(
            list(view.lineage.xs), list(view.lineage.ys), list(view.lineage.zs),
            color=LINEAGE_RGB, width=2.0, name="accepted path",
        )
    for fate, rgb in FATE_RGB.items():
        nodes = [n for n in view.nodes if n.fate == fate]
        if not nodes:
            continue
        plot.add_scatter3d(
            [n.x for n in nodes], [n.y for n in nodes], [n.z for n in nodes],
            color=rgb, size=BEST_SIZE if fate == "best" else NODE_SIZE,
        )
    px, py, pz = view.peak
    plot.add_scatter3d([px], [py], [pz], color=SUMMIT_RGB, size=BEST_SIZE)
    return plot


def statusline(
    ref: str, state: str, view: SurfaceView | None, metric: str, higher: bool,
    reason: str | None = None, position: tuple[int, int] | None = None,
) -> str:
    parts = [f"[bold]{ref}[/] [{STATE_STYLE.get(state, '')}]{state}[/]"]
    if position is not None and position[1] > 1:
        parts.append(f"({position[0] + 1}/{position[1]})")
    if reason is not None:
        parts.append(reason)
        return "  ".join(parts)
    if view is not None:
        arrow = "↑" if higher else "↓"
        parts.append(f"{len(view.nodes)} on the surface")
        if view.n_unpositioned:
            parts.append(f"[dim]{view.n_unpositioned} unpositioned[/]")
        best = [n for n in view.nodes if n.fate == "best"]
        if best and best[0].score is not None:
            parts.append(f"best {metric} {arrow} [bold rgb(255,200,40)]{best[0].score:g}[/]")
    return "  ".join(parts)


class SurfacePlotWidget(PlotWidget):
    """Free-orbit plot that survives data refreshes: the camera the user set
    carries across every rebuild; only the first plot gets START_CAMERA."""

    def __init__(self, **kwargs):
        super().__init__(themed_plot(), **kwargs)
        self._has_view = False

    def set_view(self, view: SurfaceView) -> None:
        camera = self._plot.camera_state() if self._has_view else START_CAMERA
        plot = build_surface_plot(view)
        plot.set_camera_state(*camera)
        self._plot = plot
        self._has_view = True
        self.invalidate()

    def clear_view(self) -> None:
        if not self._has_view:
            return
        self._plot = themed_plot()
        self._has_view = False
        self.invalidate()


class SurfaceScreen(LiveScreen):
    """Canvas + statusline for one search's surface. Reached via
    `hillclimb surface [search]`."""

    BINDING_GROUP_TITLE = "surface"
    BINDINGS = [
        Binding("n", "next_search", "next search", tooltip="the next search in the store"),
        Binding("p", "prev_search", "prev search", show=False, tooltip="the previous search"),
        Binding("q", "app.quit", "quit"),
    ]

    DEFAULT_CSS = """
    SurfaceScreen #surfaceline { height: 1; padding: 0 1; background: $surface; }
    SurfaceScreen #surface-canvas { width: 1fr; height: 1fr; }
    """

    def __init__(self, config: Config, search: str | None = None):
        super().__init__()
        self.config = config
        self.search = search
        self._store: DataStore | None = None
        self._record: SearchRecord | None = None
        self._fingerprint: tuple | None = None
        self._problems: dict[str, ProblemSpec | None] = {}  # meta.problem -> spec (None = unloadable)
        self._position: tuple[int, int] | None = None

    def compose(self) -> ComposeResult:
        yield HillclimbHeader()
        yield Label(id="surfaceline")
        yield SurfacePlotWidget(id="surface-canvas")
        yield Footer()

    def on_mount(self) -> None:
        self.start_live()
        self._canvas().focus()

    def _canvas(self) -> SurfacePlotWidget:
        return self.query_one("#surface-canvas", SurfacePlotWidget)

    # -- data --

    def _resolve(self) -> SearchRecord | None:
        if self._store is None:
            self._store = open_store(self.config)
        try:
            record = resolve_search(self._store, self.search)
        except LookupError as exc:
            self.query_one("#surfaceline", Label).update(str(exc))
            return None
        self._record = record
        return record

    def _problem(self, record: SearchRecord) -> ProblemSpec | None:
        key = record.meta.problem
        if key not in self._problems:
            try:
                self._problems[key] = load_problem(key, self.config)
            except Exception:  # noqa: BLE001 — a moved/deleted problem dir
                self._problems[key] = None
        return self._problems[key]

    def _switch_search(self, step: int) -> None:
        if self._record is None or self._store is None:
            return
        records = self._store.searches()
        keys = [r.key for r in records]
        if self._record.key not in keys or len(records) < 2:
            return
        target = records[(keys.index(self._record.key) + step) % len(records)]
        self._record = target
        self.search = target.ref
        self._fingerprint = None
        self.refresh_data()

    def action_next_search(self) -> None:
        self._switch_search(1)

    def action_prev_search(self) -> None:
        self._switch_search(-1)

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
        keys = [r.key for r in (self._store.searches() if self._store else [])]
        self._position = (keys.index(record.key), len(keys)) if record.key in keys else None
        line = self.query_one("#surfaceline", Label)
        higher = bool(record.meta.higher_is_better)

        problem, reason = self._problem(record), None
        if problem is None:
            reason = f"problem {record.meta.problem!r} is not loadable from here"
        elif problem.landscape_path is None:
            reason = surface_unavailable(problem)
        if reason is not None:
            canvas.clear_view()
            self._fingerprint = None
            line.update(statusline(record.ref, record.state, None, record.meta.metric, higher,
                                   reason=reason, position=self._position))
            return

        journal = Journal(self._store.journal(record.key))  # type: ignore[union-attr]
        candidates = list(journal.candidates.values())
        fingerprint = (
            record.key, record.state,
            tuple((c.candidate_id, c.status, c.val_score, c.pruned, c.finished_at) for c in candidates),
        )
        if fingerprint == self._fingerprint:
            return
        self._fingerprint = fingerprint
        try:
            landscape = load_landscape(problem.landscape_path)
        except LandscapeError as exc:
            canvas.clear_view()
            line.update(statusline(record.ref, record.state, None, record.meta.metric, higher,
                                   reason=str(exc), position=self._position))
            return
        view = build_surface(
            candidates, higher, landscape, metric_keys=tuple(problem.surface_metrics[:2]),
        )
        canvas.set_view(view)
        line.update(statusline(record.ref, record.state, view, record.meta.metric, higher,
                               position=self._position))


def surface_unavailable(problem: ProblemSpec) -> str:
    """Why `hillclimb surface` has nothing to draw for this problem."""
    return (
        f"{problem.problem_id} defines no surface — add a landscape.py "
        "(elevation(x, y) + grid(n)) to the problem dir and journal each "
        "candidate's position as extra numeric keys next to the score"
    )


class SurfaceApp(TimezoneMixin, App):
    """Standalone shell for `hillclimb surface`."""

    BINDINGS = [Binding("t", "choose_timezone", "time zone", show=False)]
    CSS = HILLCLIMB_CSS

    def __init__(self, config: Config | None = None, search: str | None = None):
        super().__init__()
        self.config = config or Config.load()
        self.search = search

    def on_mount(self) -> None:
        apply_theme(self)
        self._init_timezone()
        self.push_screen(SurfaceScreen(self.config, self.search))
