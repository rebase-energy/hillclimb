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
from hillclimb.tui.theme import HILLCLIMB_CSS, apply_theme, themed_plot, boxed_plot
from hillclimb.tui.watch import STATE_STYLE, LiveScreen

from plotui.textual import PlotWidget

# Rank-bin ramp, worst → best: cold purple through hot orange to pale
# yellow — reads as temperature on the dark theme and stays clear of both
# the cyan chrome and the champion's gold.
# hillclimb.sh's score ramp (indigo → cyan → green → gold → orange), sampled
# at the N_BINS rank bins — the terminal figure reads like the one on the site
_SITE_RAMP = [(59, 63, 153), (43, 123, 186), (34, 184, 201), (46, 230, 230), (155, 225, 93), (253, 195, 40), (255, 122, 44)]


def _ramp(t: float) -> tuple[int, int, int]:
    x = t * (len(_SITE_RAMP) - 1)
    lo = min(int(x), len(_SITE_RAMP) - 2)
    a, b, f = _SITE_RAMP[lo], _SITE_RAMP[lo + 1], x - lo
    return tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3))  # type: ignore[return-value]


SCORE_RGB = [_ramp(i / (N_BINS - 1)) for i in range(N_BINS)]
BEST_RGB = (255, 200, 40)       # the champion — same gold as tree/surface
REFERENCE_RGB = (235, 238, 240)  # the origin, an open diamond in its corner
UNSCORED_RGB = (228, 55, 48)    # never scored: failed, a small red dot
LINEAGE_RGB = (255, 255, 255)   # the best's lineage, like the tree figure's
# marks are small, as on the site: the cloud's shape is the picture
NODE_SIZE = 3.0
BEST_SIZE = 4.6
UNSCORED_SIZE = 2.0
AXIS_NAMES = ("behaviour", "code", "lineage")  # x, y, z of the reference cube


def fate_shape(fate: str, scored: bool = True) -> str:
    """The site's marks: built on = a filled disc, left = an open circle,
    failed = a small dot (pending keeps the tree's triangle)."""
    if not scored or fate in ("failed", "pruned"):
        return "dot"
    return {"expanded": "disc", "pending": "triangle"}.get(fate, "circle")


def legend_markup(
    scope: str = "search", origin: str = "origin, the corner",
    line: tuple[tuple[int, int, int], str] = (LINEAGE_RGB, "the best's lineage"),
) -> str:
    """The row under the plot, the site's legend in the terminal's glyphs.
    `line` is the view's path: its colour and what it is."""
    def rgb(c):
        return f"rgb({c[0]},{c[1]},{c[2]})"

    cyan = rgb(SCORE_RGB[3])
    items = [
        f"[{rgb(REFERENCE_RGB)}]◇[/] {origin}",
        f"[{rgb(BEST_RGB)}]◆[/] best",
        f"[{cyan}]●[/] built on",
        f"[{cyan}]○[/] left",
        f"[{rgb(UNSCORED_RGB)}]·[/] failed",
        f"[{rgb(line[0])}]━[/] {line[1]}",
    ]
    if scope != "run":
        items.append("".join(f"[{rgb(c)}]▬[/]" for c in SCORE_RGB) + " score, low → high")
    return "   ".join(items)


# --- the legend in the plot ---------------------------------------------------
# The same two legends as the tree figure, so the views read as one tool: the
# marks in plotui's own legend box at the top left (rows this module declares,
# drawn in the image as miniatures of the marks), and the score ramp as a
# column of terminal cells at the top right. `legend_markup` is the row under
# the plot a plotui without those rows falls back to.

# plotui legend rows: (label, swatch, colour, border, visible)
LegendRow = tuple[str, str, tuple[int, int, int], tuple[int, int, int] | None, bool]


def legend_rows(
    origin: str = "origin, the corner",
    line: tuple[tuple[int, int, int], str] = (LINEAGE_RGB, "the best's lineage"),
) -> list[LegendRow]:
    """The marks as plotui legend rows, each with the swatch the plot draws
    that mark with. `line` is the view's path: its colour and what it is."""
    cyan = SCORE_RGB[3]
    return [
        (origin, "diamond-open", REFERENCE_RGB, None, True),
        ("best", "diamond", BEST_RGB, None, True),
        ("built on", "disc", cyan, None, True),
        ("left", "ring", cyan, None, True),
        ("failed", "dot", UNSCORED_RGB, None, True),
        (line[1], "line", line[0], None, True),
    ]


def apply_legend(plot, rows: list[LegendRow]) -> bool:
    """Put `rows` on `plot` as its top-left legend, like the tree's. False
    on a plotui that cannot draw them (no host rows, or no diamond and dot
    swatches): the caller keeps the text row under the plot instead."""
    try:
        plot.set_legend_entries(list(rows))
        plot.set_legend_corner("top-left")
    except (AttributeError, ValueError):
        if hasattr(type(plot), "legend_visible"):
            plot.legend_visible = False
        return False
    plot.legend_visible = True
    return True


def plot_hosts_legend() -> bool:
    """Can this plotui draw the legend in the plot?"""
    return apply_legend(themed_plot(), legend_rows())


RAMP_ROW = 1     # the ramp's caption row — the tree's, so the two figures line up
RAMP_MARGIN = 1


def ramp_spans(cols: int) -> list[tuple[int, int, str, str]]:
    """`(row, col, text, style)` spans of the score ramp at the top right:
    one cell per colour bin, best at the top, its two ends named. Colour is
    the score's rank, so the ends are `high` and `low`, not values."""
    bar_col = cols - RAMP_MARGIN - 1
    labels = {0: "high", len(SCORE_RGB) - 1: "low"}
    width = max(len(v) for v in labels.values())
    spans = [(RAMP_ROW, max(0, bar_col + 1 - len("score")), "score", "bold white")]
    for i, (r, g, b) in enumerate(reversed(SCORE_RGB)):
        row = RAMP_ROW + 1 + i
        spans.append((row, bar_col, "█", f"rgb({r},{g},{b})"))
        if i in labels:
            spans.append((row, bar_col - 1 - width, labels[i].rjust(width), "white"))
    return spans


# The site's composition: the origin at the bottom left, lineage rising from
# it, behaviour and code running off to the right; seen slightly from above
# (negative pitch) and zoomed out so the whole cube and its axis names fit.
START_CAMERA = (1.2, -0.3, 0.62, 0.0, 0.0)
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
    """The reference cube as hillclimb.sh draws it: small marks whose shape
    is the candidate's fate and whose colour its score, the origin an open
    diamond in its corner, the best a gold one, the best's lineage a white
    line from the corner out, and the three axes named."""
    plot = boxed_plot()
    if hasattr(plot, "set_axis_titles3d"):
        plot.set_axis_titles3d(*AXIS_NAMES)
    colours, sizes, shapes = [], [], []
    for node in view.nodes:
        scored = node.bin is not None
        if node.id in view.reference_ids:
            style = (REFERENCE_RGB, BEST_SIZE, "diamond-open")
        elif node.best:
            style = (BEST_RGB, BEST_SIZE, "diamond")
        elif view.scope == "run":
            style = (experiment_colour(view, node.experiment), rank_size(node.bin), fate_shape(node.fate, scored))
        elif not scored:
            style = (UNSCORED_RGB, UNSCORED_SIZE, "dot")
        else:
            style = (SCORE_RGB[node.bin], NODE_SIZE, fate_shape(node.fate))
        colours.append(style[0])
        sizes.append(style[1])
        shapes.append(style[2])
    if view.nodes:
        # far to near is the renderer's business; the origin and then the
        # best go last, so at equal depth no ordinary mark covers them
        order = sorted(range(len(view.nodes)), key=lambda i: (view.nodes[i].best, view.nodes[i].id in view.reference_ids))
        nodes = [view.nodes[i] for i in order]
        plot.add_graph3d(
            [n.x for n in nodes], [n.y for n in nodes], [n.z for n in nodes], edges=[],
            node_colors=[colours[i] for i in order], size=NODE_SIZE,
            node_sizes=[sizes[i] for i in order], node_shapes=[shapes[i] for i in order],
        )
    if view.scope == "search" and view.reference != "champion":
        # the best's lineage, from the corner out: its accepted ancestors in
        # the order they left the origin (the lineage axis is exactly that)
        path = sorted((n for n in view.nodes if n.on_path and n.id not in view.reference_ids), key=lambda n: n.z)
        if path:
            xs, ys, zs = [0.0, *(n.x for n in path)], [0.0, *(n.y for n in path)], [0.0, *(n.z for n in path)]
            plot.add_line3d(xs, ys, zs, color=LINEAGE_RGB, width=1.6)
    plot.set_bounds(*BOUNDS)
    apply_legend(plot, legend_rows())
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


PLOT_LEGEND = plot_hosts_legend()


class SimilarityPlotWidget(PlotWidget):
    """Free-orbit scatter that survives data refreshes: the camera the user
    set carries across rebuilds; only the first plot gets START_CAMERA."""

    def __init__(self, **kwargs):
        super().__init__(themed_plot(), **kwargs)
        self._has_view = False
        self._ramp = False  # the score ramp is drawn (a search's cube; a run's is coloured by experiment)

    def set_view(self, view: SimilarityView) -> None:
        camera = self._plot.camera_state() if self._has_view else START_CAMERA
        plot = build_similarity_plot(view)
        plot.set_camera_state(*camera)
        self._plot = plot
        self._has_view = True
        self._ramp = view.scope != "run"
        self._sync_ramp()
        self.invalidate()

    def _sync_ramp(self) -> None:
        from rich.style import Style

        spans = ramp_spans(self.size.width) if self._ramp and self._has_view and PLOT_LEGEND else []
        self.set_overlay([(r, c, t, Style.parse(st)) for r, c, t, st in spans])

    def on_resize(self) -> None:
        self._sync_ramp()

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
    SimilarityBase #similarity-legend {
        height: auto; padding: 0 2; text-align: center; color: $text-muted; background: #0e1113;
    }
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
        if not PLOT_LEGEND:  # an older plotui: the legend is a row under the plot
            yield Label(legend_markup(), id="similarity-legend")
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
        # (run_id, problem_key) -> run scope. Never `self.run`: that would
        # shadow App.run() and leave `SimilarityApp(...).run()` calling None
        self.run_scope = run
        self.view = view  # "reference" (the cube) | "map"
        self.metric = metric  # the map's opening metric

    def on_mount(self) -> None:
        apply_theme(self)
        self._init_timezone()
        self.push_screen(self.first_screen())

    def first_screen(self):
        if self.view == "map":
            from hillclimb.tui.similarity_mapview import MapScreen, RunMapScreen

            if self.run_scope is not None:
                return RunMapScreen(self.config, *self.run_scope, metric=self.metric)
            return MapScreen(self.config, self.search, metric=self.metric)
        if self.run_scope is not None:
            run_id, problem_key = self.run_scope
            return RunSimilarityScreen(self.config, run_id, problem_key, reference=self.reference)
        return SimilarityScreen(self.config, self.search, reference=self.reference)
