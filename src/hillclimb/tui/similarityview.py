"""`hillclimb similarity reference` — candidates as a 3D distance scatter.

Each candidate sits at (behavioral, structural, lineage) distance from the
reference candidate inside a unit cube (each axis normalized by its own
95th percentile; the raw cube-edge values ride in the statusline).

**Search scope** (one search): the reference is the origin the search grew
from — its seed, else its baseline — and `c` switches to the current
champion. Colour is the score's rank bin on a cold→hot ramp; the champion
is gold, the reference an anchoring white dot at the origin,
positioned-but-unscored candidates a dim grey.

**Run scope** (every search of one problem in a study run, opened
automatically when the anchored search is one experiment of a study): each search
is measured from its own seed — one shared seed file, so one origin — and
colour is the experiment (the chart's palette, so an experiment looks the same
in both views); size carries the score rank. `c` re-anchors on the run's
best candidate; `n`/`p` step through the run's problems.

The reading this view is built for: near the structure axis and far on
behavior = a sensitive knob was found; far on structure and near on
behavior = coding agents refactored without changing behavior (wasted operators).
In run scope: experiments whose clouds overlap explored the same way.

Same architecture as surfaceview.py: pure functions up top, thin Textual
shells below, free orbit camera starting from the shared START_CAMERA.
`SimilarityBase` and `RunScopeMixin` are shared with `similarity_mapview`
(the pairwise map, `v` from here) — the two views are one command with
two layouts.
"""

from __future__ import annotations

from plotui import Plot
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Label

from hillclimb.tui.chart import EXPERIMENT_PALETTE
from hillclimb.config import Config
from hillclimb.tui.header import APP_TITLE, HillclimbHeader, TimezoneMixin
from hillclimb.harness.journal import Journal
from hillclimb.problem import ProblemSpec, load_problem
from hillclimb.tui.similarity import (
    N_BINS,
    SearchInput,
    SimilarityView,
    build_run_similarity,
    build_similarity,
    clear_caches,
)
from hillclimb.harness.store import DataStore, SearchRecord, open_store, resolve_search
from hillclimb.tui.theme import HILLCLIMB_CSS, apply_theme, themed_plot
from hillclimb.tui.watch import STATE_STYLE, LiveScreen

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


def experiment_colour(view: SimilarityView, experiment: str | None) -> tuple[int, int, int]:
    """The chart's colour for an experiment — first-seen order over the run's
    searches, so `chart` and `similarity` agree on which experiment is which."""
    if experiment is None or experiment not in view.experiments:
        return UNSCORED_RGB
    return EXPERIMENT_PALETTE[view.experiments.index(experiment) % len(EXPERIMENT_PALETTE)]


def rank_size(bin_index: int | None) -> float:
    """Run scope carries the score rank in dot size: worst bin just above
    the unscored size, best bin just under the champion's."""
    if bin_index is None:
        return UNSCORED_SIZE
    return round(UNSCORED_SIZE + (bin_index + 1) / N_BINS * (NODE_SIZE + 1.0 - UNSCORED_SIZE), 2)


def build_similarity_plot(view: SimilarityView) -> Plot:
    plot = themed_plot()
    groups: dict[tuple[tuple[int, int, int], float], list] = {}
    for node in view.nodes:
        if node.id in view.reference_ids:
            key = (REFERENCE_RGB, BEST_SIZE)
        elif node.best:
            key = (BEST_RGB, BEST_SIZE)
        elif view.scope == "run":
            key = (experiment_colour(view, node.experiment), rank_size(node.bin))
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


def experiment_legend(view: SimilarityView) -> str:
    return "  ".join(
        f"[rgb({r},{g},{b})]■[/] {experiment}" for experiment in view.experiments for (r, g, b) in [experiment_colour(view, experiment)]
    )


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
    label = view.reference_label or view.reference
    if view.scope == "run":
        parts.append(f"vs [bold]{label}[/] ({view.mode})")
        parts.append(f"{len(view.nodes)} placed over {view.n_searches} searches")
    else:
        parts.append(f"vs [bold]{label}[/] {view.reference_id} ({view.mode})")
        parts.append(f"{len(view.nodes)} placed")
    if view.n_unpositioned:
        parts.append(f"[dim]{view.n_unpositioned} unpositioned[/]")
    if view.scope == "run" and view.experiments:
        parts.append(experiment_legend(view))
    if view.lineage_note:
        parts.append(f"[dim]{view.lineage_note}[/]")
    b, s, l = view.scales
    parts.append(f"[dim]cube: behav≤{b:.3g} struct≤{s:.3g} lineage≤{l:.3g}[/]")
    return "  ".join(parts)


def run_inputs(store: DataStore, run_id: str, problem_key: str) -> list[SearchRecord]:
    """The run's searches on one problem, oldest first (the experiment order the
    chart colours by)."""
    return [r for r in store.searches(run_id=run_id) if r.meta.problem_key == problem_key]


def search_inputs(store: DataStore, records: list[SearchRecord]) -> list[SearchInput]:
    return [
        SearchInput(
            search_id=r.search_id, search_dir=r.search_dir,
            candidates=list(Journal(store.journal(r.key)).candidates.values()),
            experiment=r.meta.experiment, seed_from=r.meta.seed_from,
            seed_sha256=getattr(r.meta, "seed_sha256", None),
        )
        for r in records
    ]


def run_state(records: list[SearchRecord]) -> str:
    states = {r.state for r in records}
    if "running" in states:
        return "running"
    return states.pop() if len(states) == 1 else "mixed"


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


class SimilarityBase(LiveScreen):
    """What the reference cube and the map share: the store, the anchored
    search, n/p stepping through the store's searches, and the problem
    lookup for its optional fingerprint.py. Subclasses compose their own
    canvas and own `refresh_data`."""

    BINDING_GROUP_TITLE = "similarity"
    BINDINGS = [
        Binding("n", "next_search", "next search", tooltip="the next search in the store"),
        Binding("p", "prev_search", "prev search", show=False, tooltip="the previous search"),
        Binding("q", "app.quit", "quit"),
    ]

    DEFAULT_CSS = """
    SimilarityBase #similarityline { height: 1; padding: 0 1; background: $surface; }
    SimilarityBase #similarity-canvas { width: 1fr; height: 1fr; }
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
        self._problems: dict[str, ProblemSpec | None] = {}

    def on_mount(self) -> None:
        self.start_live()
        self._canvas().focus()

    def _canvas(self):
        return self.query_one("#similarity-canvas", PlotWidget)

    def _statusline(self) -> Label:
        return self.query_one("#similarityline", Label)

    def action_next_search(self) -> None:
        self._switch_search(1)

    def action_prev_search(self) -> None:
        self._switch_search(-1)

    def _on_switch(self) -> None:
        """A different search or problem is now anchored: forget what was
        derived from the old one and redraw."""
        self._fingerprint = None
        clear_caches()
        self.refresh_data()

    # -- data --

    def _problem(self, record: SearchRecord) -> ProblemSpec | None:
        """The problem behind a search, for its optional fingerprint.py; None
        when it is not loadable from here (a moved or provider problem) —
        the view then falls back to the submission file."""
        key = record.meta.problem
        if key not in self._problems:
            try:
                self._problems[key] = load_problem(key, self.config)
            except Exception:  # noqa: BLE001 — a moved/deleted problem dir
                self._problems[key] = None
        return self._problems[key]

    def _fingerprint_path(self, record: SearchRecord):
        problem = self._problem(record)
        return problem.fingerprint_path if problem is not None else None

    def _resolve(self) -> SearchRecord | None:
        if self._store is None:
            self._store = open_store(self.config)
        try:
            record = resolve_search(self._store, self.search)
        except LookupError as exc:
            self._statusline().update(str(exc))
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
        self._on_switch()

    def refresh_data(self) -> None:  # pragma: no cover — subclasses implement
        raise NotImplementedError


class SimilarityScreen(SimilarityBase):
    """Canvas + statusline for one search's distance scatter. Reached via
    `hillclimb similarity reference [search]`; `v` swaps to the map."""

    BINDINGS = [
        Binding("c", "toggle_reference", "origin/champion",
                tooltip="measure from the seed or baseline, or from the current best"),
        Binding("v", "open_map", "map view", tooltip="the same candidates embedded by pairwise distance"),
    ]

    def compose(self) -> ComposeResult:
        yield HillclimbHeader()
        yield Label(id="similarityline")
        yield SimilarityPlotWidget(id="similarity-canvas")
        yield Footer()

    def _canvas(self) -> SimilarityPlotWidget:
        return self.query_one("#similarity-canvas", SimilarityPlotWidget)

    def action_toggle_reference(self) -> None:
        self.reference = "champion" if self.reference == "baseline" else "baseline"
        self._fingerprint = None
        self.refresh_data()

    def action_open_map(self) -> None:
        from hillclimb.tui.similarity_mapview import MapScreen

        self.app.switch_screen(MapScreen(self.config, self.search))

    def _show(self, view: SimilarityView, ref: str, state: str, metric: str, higher: bool) -> None:
        canvas = self._canvas()
        if view.unavailable is not None:
            canvas.clear_view()
        else:
            canvas.set_view(view)
        self._statusline().update(statusline(ref, state, view, metric, higher, position=self._position))

    def refresh_data(self) -> None:
        if self._canvas().dragging:
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
            fingerprint_path=self._fingerprint_path(record),
        )
        self._show(view, record.ref, record.state, record.meta.metric, higher)


class RunScopeMixin:
    """Run scope for either view: anchored on (run_id, problem_key), `n`/`p`
    step through the run's problems. Mixed in before a SimilarityBase
    subclass, which supplies `_store`, `_fingerprint`, `_position`,
    `_on_switch` and `refresh_data`."""

    run_id: str
    problem_key: str

    def _problem_keys(self) -> list[str]:
        assert self._store is not None  # type: ignore[attr-defined]
        return list(dict.fromkeys(
            r.meta.problem_key for r in self._store.searches(run_id=self.run_id)  # type: ignore[attr-defined]
        ))

    def _switch_search(self, step: int) -> None:
        if self._store is None:  # type: ignore[attr-defined]
            return
        keys = self._problem_keys()
        if self.problem_key not in keys or len(keys) < 2:
            return
        self.problem_key = keys[(keys.index(self.problem_key) + step) % len(keys)]
        self._on_switch()  # type: ignore[attr-defined]

    def _run_records(self) -> tuple[list[SearchRecord], str]:
        """This problem's searches in the run, and the ref the statusline
        names; also refreshes the (position/count) readout."""
        if self._store is None:  # type: ignore[attr-defined]
            self._store = open_store(self.config)  # type: ignore[attr-defined]
        records = run_inputs(self._store, self.run_id, self.problem_key)  # type: ignore[attr-defined]
        keys = self._problem_keys()
        self._position = (  # type: ignore[attr-defined]
            (keys.index(self.problem_key), len(keys)) if self.problem_key in keys else None
        )
        return records, f"{self.run_id} {self.problem_key}"

    @staticmethod
    def _run_fingerprint(records: list[SearchRecord], inputs: list[SearchInput], *extra) -> tuple:
        return (
            *extra,
            tuple(
                (r.key, r.state, tuple((c.candidate_id, c.status, c.val_score, c.pruned, c.finished_at)
                                       for c in s.candidates))
                for r, s in zip(records, inputs)
            ),
        )


class RunSimilarityScreen(RunScopeMixin, SimilarityScreen):
    """Run scope: every search of one problem in a study run in one
    cube, coloured by experiment. `n`/`p` step through the run's problems."""

    BINDINGS = [
        Binding("c", "toggle_reference", "seed/champion",
                tooltip="measure from the shared seed, or from the run's best candidate"),
        Binding("n", "next_search", "next problem", tooltip="the run's next problem"),
        Binding("p", "prev_search", "prev problem", show=False, tooltip="the run's previous problem"),
    ]

    def __init__(self, config: Config, run_id: str, problem_key: str, reference: str = "seed"):
        super().__init__(config, search=None, reference=reference)
        self.run_id = run_id
        self.problem_key = problem_key

    def action_toggle_reference(self) -> None:
        self.reference = "champion" if self.reference == "seed" else "seed"
        self._fingerprint = None
        self.refresh_data()

    def action_open_map(self) -> None:
        from hillclimb.tui.similarity_mapview import RunMapScreen

        self.app.switch_screen(RunMapScreen(self.config, self.run_id, self.problem_key))

    def refresh_data(self) -> None:
        if self._canvas().dragging:
            return
        records, ref = self._run_records()
        if not records:
            self._canvas().clear_view()
            self._statusline().update(f"no searches for {self.problem_key} in run {self.run_id}")
            return
        inputs = search_inputs(self._store, records)  # type: ignore[arg-type]
        fingerprint = self._run_fingerprint(records, inputs, self.run_id, self.problem_key, self.reference)
        if fingerprint == self._fingerprint:
            return
        self._fingerprint = fingerprint

        higher = bool(records[0].meta.higher_is_better)
        view = build_run_similarity(
            inputs, higher, reference=self.reference,
            output_artifacts=records[0].meta.output_artifacts,
            fingerprint_path=self._fingerprint_path(records[0]),
            problem_key=self.problem_key,
        )
        self._show(view, ref, run_state(records), records[0].meta.metric, higher)


class SimilarityApp(TimezoneMixin, App):
    """Standalone shell for `hillclimb similarity`."""
    TITLE = f"{APP_TITLE} similarity"  # the header names the command that opened it

    BINDINGS = [Binding("t", "choose_timezone", "time zone", show=False)]
    CSS = HILLCLIMB_CSS

    def __init__(self, config: Config | None = None, search: str | None = None,
                 reference: str = "baseline", run: tuple[str, str] | None = None,
                 view: str = "reference", metric: str = "behavioral"):
        super().__init__()
        self.config = config or Config.load()
        self.search = search
        self.reference = reference
        self.run = run  # (run_id, problem_key) -> run scope
        self.view = view  # "reference" (the cube) | "map"
        self.metric = metric  # the map's opening metric

    def on_mount(self) -> None:
        apply_theme(self)
        self._init_timezone()
        self.push_screen(self.first_screen())

    def first_screen(self):
        if self.view == "map":
            from hillclimb.tui.similarity_mapview import MapScreen, RunMapScreen

            if self.run is not None:
                return RunMapScreen(self.config, *self.run, metric=self.metric)
            return MapScreen(self.config, self.search, metric=self.metric)
        if self.run is not None:
            run_id, problem_key = self.run
            return RunSimilarityScreen(self.config, run_id, problem_key, reference=self.reference)
        return SimilarityScreen(self.config, self.search, reference=self.reference)
