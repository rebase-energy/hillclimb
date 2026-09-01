"""`hillclimb similarity` — one search's candidates as a 3D distance scatter.

Each candidate sits at (behavioral, structural, lineage) distance from the
reference candidate — the baseline by default, the current champion on `c` —
inside a unit cube (each axis normalized by its own 95th percentile; the raw
cube-edge values ride in the statusline). Colour is the score's rank bin on
a cold→hot ramp; the current champion is gold, the reference an anchoring
white dot at the origin, positioned-but-unscored candidates a dim grey.

The reading this view is built for: near the structure axis and far on
behavior = a sensitive knob was found; far on structure and near on
behavior = agents refactored without changing behavior (wasted operators).

Same architecture as surfaceview.py: pure functions up top, thin Textual
shells below, free orbit camera starting from the shared START_CAMERA.
"""

from __future__ import annotations

from plotui import Plot
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Label

from hillclimb.config import Config
from hillclimb.header import HillclimbHeader, TimezoneMixin
from hillclimb.journal import Journal
from hillclimb.similarity import N_BINS, SimilarityView, build_similarity, clear_caches
from hillclimb.store import DataStore, SearchRecord, open_store, resolve_search
from hillclimb.theme import HILLCLIMB_CSS, apply_theme, themed_plot
from hillclimb.watch import STATE_STYLE, LiveScreen

from plotui.textual import PlotWidget

# Rank-bin ramp, worst → best: cold purple through hot orange to pale
# yellow — reads as temperature on the dark theme and stays clear of both
# the cyan chrome and the champion's gold.
SCORE_RGB = [
    (58, 44, 92), (110, 50, 122), (176, 62, 100),
    (222, 92, 68), (246, 152, 58), (250, 222, 134),
]
assert len(SCORE_RGB) == N_BINS
BEST_RGB = (255, 200, 40)       # the champion — same gold as tree/surface
REFERENCE_RGB = (255, 255, 255)  # the anchor — white, like surface's summit
UNSCORED_RGB = (90, 90, 90)
NODE_SIZE = 5.0
BEST_SIZE = 7.0
UNSCORED_SIZE = 3.5
# From above (negative pitch, as in surfaceview) but zoomed out enough that
# the unit cube's corners — where extreme candidates pin — stay in frame.
START_CAMERA = (0.9, -0.8, 0.75, 0.0, 0.0)
# The axes are normalized to [0, 1], so pin the frame with a small margin:
# dots keep their pixels across rebuilds and corner points aren't flush
# against the wireframe.
BOUNDS = ((-0.05, -0.05, -0.05), (1.05, 1.05, 1.05))


def build_similarity_plot(view: SimilarityView) -> Plot:
    plot = themed_plot()
    groups: dict[tuple[tuple[int, int, int], float], list] = {}
    for node in view.nodes:
        if node.id == view.reference_id:
            key = (REFERENCE_RGB, BEST_SIZE)
        elif node.best:
            key = (BEST_RGB, BEST_SIZE)
        elif node.bin is None:
            key = (UNSCORED_RGB, UNSCORED_SIZE)
        else:
            key = (SCORE_RGB[node.bin], NODE_SIZE)
        groups.setdefault(key, []).append(node)
    for (rgb, size), nodes in groups.items():
        plot.add_scatter3d(
            [n.x for n in nodes], [n.y for n in nodes], [n.z for n in nodes],
            color=rgb, size=size,
        )
    plot.set_bounds(*BOUNDS)
    return plot


def statusline(
    ref: str, state: str, view: SimilarityView | None, metric: str, higher: bool,
    position: tuple[int, int] | None = None,
) -> str:
    parts = [f"[bold]{ref}[/] [{STATE_STYLE.get(state, '')}]{state}[/]"]
    if position is not None and position[1] > 1:
        parts.append(f"({position[0] + 1}/{position[1]})")
    if view is None:
        return "  ".join(parts)
    if view.unavailable is not None:
        parts.append(view.unavailable)
        return "  ".join(parts)
    parts.append(f"vs [bold]{view.reference}[/] {view.reference_id} ({view.mode})")
    parts.append(f"{len(view.nodes)} placed")
    if view.n_unpositioned:
        parts.append(f"[dim]{view.n_unpositioned} unpositioned[/]")
    b, s, l = view.scales
    parts.append(f"[dim]cube: behav≤{b:.3g} struct≤{s:.3g} lineage≤{l:.3g}[/]")
    return "  ".join(parts)


class SimilarityPlotWidget(PlotWidget):
    """Free-orbit scatter that survives data refreshes: the camera the user
    set carries across rebuilds; only the first plot gets START_CAMERA."""

    def __init__(self, **kwargs):
        super().__init__(themed_plot(), **kwargs)
        self._has_view = False

    def set_view(self, view: SimilarityView) -> None:
        camera = self._plot.camera_state() if self._has_view else START_CAMERA
        plot = build_similarity_plot(view)
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


class SimilarityScreen(LiveScreen):
    """Canvas + statusline for one search's distance scatter. Reached via
    `hillclimb similarity [search]`."""

    BINDING_GROUP_TITLE = "similarity"
    BINDINGS = [
        Binding("c", "toggle_reference", "baseline/champion",
                tooltip="measure from the baseline or from the current best"),
        Binding("n", "next_search", "next search", tooltip="the next search in the store"),
        Binding("p", "prev_search", "prev search", show=False, tooltip="the previous search"),
        Binding("q", "app.quit", "quit"),
    ]

    DEFAULT_CSS = """
    SimilarityScreen #similarityline { height: 1; padding: 0 1; background: $surface; }
    SimilarityScreen #similarity-canvas { width: 1fr; height: 1fr; }
    """

    def __init__(self, config: Config, search: str | None = None, reference: str = "baseline"):
        super().__init__()
        self.config = config
        self.search = search
        self.reference = reference
        self._store: DataStore | None = None
        self._record: SearchRecord | None = None
        self._fingerprint: tuple | None = None
        self._position: tuple[int, int] | None = None

    def compose(self) -> ComposeResult:
        yield HillclimbHeader()
        yield Label(id="similarityline")
        yield SimilarityPlotWidget(id="similarity-canvas")
        yield Footer()

    def on_mount(self) -> None:
        self.start_live()
        self._canvas().focus()

    def _canvas(self) -> SimilarityPlotWidget:
        return self.query_one("#similarity-canvas", SimilarityPlotWidget)

    def action_toggle_reference(self) -> None:
        self.reference = "champion" if self.reference == "baseline" else "baseline"
        self._fingerprint = None
        self.refresh_data()

    def action_next_search(self) -> None:
        self._switch_search(1)

    def action_prev_search(self) -> None:
        self._switch_search(-1)

    # -- data --

    def _resolve(self) -> SearchRecord | None:
        if self._store is None:
            self._store = open_store(self.config)
        try:
            record = resolve_search(self._store, self.search)
        except LookupError as exc:
            self.query_one("#similarityline", Label).update(str(exc))
            return None
        self._record = record
        return record

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
        clear_caches()
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
        keys = [r.key for r in (self._store.searches() if self._store else [])]
        self._position = (keys.index(record.key), len(keys)) if record.key in keys else None

        journal = Journal(self._store.journal(record.key))  # type: ignore[union-attr]
        candidates = list(journal.candidates.values())
        fingerprint = (
            record.key, record.state, self.reference,
            tuple((c.candidate_id, c.status, c.val_score, c.pruned, c.finished_at) for c in candidates),
        )
        if fingerprint == self._fingerprint:
            return
        self._fingerprint = fingerprint

        higher = bool(record.meta.higher_is_better)
        view = build_similarity(
            candidates, record.search_dir, higher, reference=self.reference,
            output_artifacts=record.meta.output_artifacts,
        )
        if view.unavailable is not None:
            canvas.clear_view()
        else:
            canvas.set_view(view)
        self.query_one("#similarityline", Label).update(
            statusline(record.ref, record.state, view, record.meta.metric, higher,
                       position=self._position)
        )


class SimilarityApp(TimezoneMixin, App):
    """Standalone shell for `hillclimb similarity`."""

    BINDINGS = [Binding("t", "choose_timezone", "time zone", show=False)]
    CSS = HILLCLIMB_CSS

    def __init__(self, config: Config | None = None, search: str | None = None,
                 reference: str = "baseline"):
        super().__init__()
        self.config = config or Config.load()
        self.search = search
        self.reference = reference

    def on_mount(self) -> None:
        apply_theme(self)
        self._init_timezone()
        self.push_screen(SimilarityScreen(self.config, self.search, reference=self.reference))
