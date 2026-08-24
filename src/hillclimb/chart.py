"""`hillclimb chart` — the live hillclimb curve.

One staircase per problem: the best validation score so far across every
search of it, against minutes since the first of them started, with every
scored candidate as a dot on the same axes — bright where it set a new best,
dim where it missed. The demo's three parallel searches are one climb, not
three; a search only gets its own line in an experiment, where the arms are
the comparison (build_plot). Same figure as the website's, in the same
colours. Strictly a viewer like watch.py: everything is read through the configured store
(store.py — the hillclimb folder by default, or the SQLite index), and the
pure data functions at the top stay testable without Textual.

`--detail` anchors on one search and overlays its exploration tree on the
curve: every scored candidate as a mark at (minute it landed, its score),
parent→child edges between them, the accepted lineage bold — the climb and
the attempts it took, on one pair of axes (tree.py supplies the lineage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from hillclimb.candidate import Candidate
from hillclimb.config import Config
from hillclimb.direction import better
from hillclimb.run import SearchMeta, load_search_meta, search_ref
from hillclimb.journal import Journal
from hillclimb.store import DataStore, FileDataStore, SearchRecord, key_for, open_store, resolve_search

# Searches drawn at once; older ones of the same problem fall off the chart
# rather than turning it into a haystack.
MAX_CURVES = 8


# plotui's line palette, mirrored so curves of one experiment arm can share
# a colour (repeats) while arms differ — the chart's "colour by arm"
ARM_PALETTE = (
    (57, 135, 229), (25, 158, 112), (201, 133, 0), (0, 131, 0),
    (144, 133, 233), (230, 103, 103), (213, 81, 129), (217, 89, 38),
)


@dataclass
class Curve:
    label: str
    state: str
    xs: list[float] = field(default_factory=list)  # minutes since the search started
    ys: list[float] = field(default_factory=list)  # best-so-far val score
    arm: str | None = None  # experiment arm, when the search is one

    @property
    def best(self) -> float | None:
        return self.ys[-1] if self.ys else None


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def curve_from_candidates(
    candidates: list[Candidate],
    *,
    label: str,
    state: str,
    higher_is_better: bool = True,
    started_at: str | None = None,
) -> Curve:
    """Best-so-far curve for one search. A point per scored, unpruned
    candidate at the minute it finished; the y value only ever moves in the
    metric's good direction, so the line is the staircase the search climbed."""
    higher = higher_is_better
    start = _parse_ts(started_at)
    curve = Curve(label=label, state=state)
    scored = []
    for cand in candidates:
        if cand.pruned or cand.val_score is None:
            continue
        when = _parse_ts(cand.finished_at) or _parse_ts(cand.created_at)
        if when is None:
            continue
        scored.append((when, cand.val_score))
    scored.sort(key=lambda item: item[0])
    if not scored:
        return curve
    origin = start or scored[0][0]
    best: float | None = None
    for when, score in scored:
        if best is None or better(score, best, higher):
            best = score
        curve.xs.append(max(0.0, (when - origin).total_seconds() / 60.0))
        curve.ys.append(best)
    return curve


def _record_curve(store: DataStore, record: SearchRecord, label: str) -> Curve:
    curve = curve_from_candidates(
        list(Journal(store.journal(record.key)).candidates.values()),
        label=label,
        state=record.state,
        higher_is_better=bool(record.meta.higher_is_better),
        started_at=record.meta.started_at,
    )
    curve.arm = record.meta.arm if record.meta.experiment else None
    return curve


def curve_label(record: SearchRecord, per_run: dict[str, int]) -> str:
    """Run name (what the user chose), plus the search id when the run holds
    several searches on the problem; an experiment search is its arm and
    repeat instead — the comparison the chart is then drawing."""
    meta = record.meta
    if meta.experiment and meta.arm:
        return f"{meta.arm} r{meta.repeat}" if meta.repeat else meta.arm
    label = record.run_name
    if per_run.get(record.run_id, 0) > 1:
        label += f"/{record.search_id}"
    return label


def climb_curve(search_dir: Path, label: str | None = None, meta: SearchMeta | None = None) -> Curve:
    """Curve for one search dir, read from the folder (the watch TUI's
    jump-to-search path). `meta` is accepted for callers that already loaded
    search.yaml."""
    store = FileDataStore(search_dir.parents[2])
    record = store.search(key_for(search_dir))
    if record is None:
        meta = meta or load_search_meta(search_dir)
        if meta is None:
            return Curve(label=label or search_ref(search_dir), state="unknown")
        record = SearchRecord(meta=meta, run_name=meta.run_id, state="unknown", search_dir=search_dir)
    return _record_curve(store, record, label or search_ref(search_dir))


def climb_curves(store: DataStore | Path, problem_key: str, limit: int = MAX_CURVES) -> list[Curve]:
    """Curves for every search of `problem_key` across runs, oldest first so
    the palette assigns colors in start order (the newest gets the last one).
    Labelled by run name, which is what the user chose — plus the search id
    when a run holds several searches on the problem. Accepts a runs dir as
    shorthand for its FileDataStore."""
    if isinstance(store, Path):
        store = FileDataStore(store)
    records = store.searches(problem_key=problem_key)  # already oldest first
    per_run: dict[str, int] = {}
    for record in records:
        per_run[record.run_id] = per_run.get(record.run_id, 0) + 1
    return [_record_curve(store, record, curve_label(record, per_run)) for record in records[-limit:]]


@dataclass(frozen=True)
class ClimbEvent:
    x: float            # minutes since the climb's origin
    y: float            # val score
    best: bool          # set a new best across every search, when it landed
    search: str         # label of the search it came from
    operator: str


@dataclass
class Climb:
    """Every scored candidate of a problem, across its searches, in landing
    order; the staircase is the `best` ones."""

    events: list[ClimbEvent] = field(default_factory=list)
    extent: float = 0.0  # minutes — the staircase runs flat to here
    searches: int = 0

    @property
    def best(self) -> float | None:
        hits = [e for e in self.events if e.best]
        return hits[-1].y if hits else None

    @property
    def hits(self) -> int:
        return sum(1 for e in self.events if e.best)

    def staircase(self) -> tuple[list[float], list[float]]:
        """The best-so-far line as step points, flat to `extent`."""
        hits = [e for e in self.events if e.best]
        return step_points([e.x for e in hits], [e.y for e in hits], self.extent)


def step_points(xs: list[float], ys: list[float], extent: float | None = None) -> tuple[list[float], list[float]]:
    """Expand (x, y) samples into the points of a step plot: hold each y until
    the next x, rise there, and run flat to `extent` (or the last x). A
    best-so-far curve is a staircase by nature — a line drawn straight
    between improvements would claim a score that was never held."""
    if not xs:
        return [], []
    sx, sy = [xs[0]], [ys[0]]
    for x, y in zip(xs[1:], ys[1:]):
        sx.extend((x, x))
        sy.extend((sy[-1], y))
    end = max(extent if extent is not None else xs[-1], xs[-1])
    if end > sx[-1]:
        sx.append(end)
        sy.append(sy[-1])
    return sx, sy


def climb_from_searches(
    searches: list[tuple[str, list[Candidate], str | None]],
    *,
    higher_is_better: bool = True,
) -> Climb:
    """Fold every search's scored, unpruned candidates into one climb.
    `searches` is (label, candidates, started_at) per search; the origin is
    the earliest start (else the first landing), and `best` is judged
    against everything that landed before, whichever search it came from."""
    higher = higher_is_better
    starts = [_parse_ts(started) for _, _, started in searches]
    starts = [t for t in starts if t is not None]
    landed: list[tuple[datetime, float, str, str]] = []
    for label, candidates, _ in searches:
        for cand in candidates:
            if cand.pruned or cand.val_score is None:
                continue
            when = _parse_ts(cand.finished_at) or _parse_ts(cand.created_at)
            if when is None:
                continue
            landed.append((when, cand.val_score, label, cand.operator or ""))
    landed.sort(key=lambda item: item[0])
    climb = Climb(searches=len(searches))
    if not landed:
        return climb
    origin = min(starts) if starts else landed[0][0]
    best: float | None = None
    for when, score, label, operator in landed:
        improved = best is None or better(score, best, higher)
        if improved:
            best = score
        x = max(0.0, (when - origin).total_seconds() / 60.0)
        climb.events.append(ClimbEvent(x, score, improved, label, operator))
    climb.extent = climb.events[-1].x
    return climb


def climb_for_problem(store: DataStore | Path, problem_key: str, limit: int = MAX_CURVES) -> Climb:
    """The climb across every search of `problem_key` (the newest `limit`)."""
    if isinstance(store, Path):
        store = FileDataStore(store)
    records = store.searches(problem_key=problem_key)
    per_run: dict[str, int] = {}
    for record in records:
        per_run[record.run_id] = per_run.get(record.run_id, 0) + 1
    return climb_from_searches(
        [
            (
                curve_label(record, per_run),
                list(Journal(store.journal(record.key)).candidates.values()),
                record.meta.started_at,
            )
            for record in records[-limit:]
        ],
        higher_is_better=bool(records[-1].meta.higher_is_better) if records else True,
    )


@dataclass(frozen=True)
class DetailMark:
    id: str
    x: float            # minutes since the search started
    y: float            # val score
    operator: str
    fate: str
    on_path: bool       # part of the accepted lineage


@dataclass(frozen=True)
class DetailEdge:
    x0: float
    y0: float
    x1: float
    y1: float
    operator: str       # the child's
    on_path: bool


@dataclass
class DetailLayout:
    curve: Curve
    marks: list[DetailMark] = field(default_factory=list)
    edges: list[DetailEdge] = field(default_factory=list)
    unscored: int = 0   # failed/pending/pruned candidates — not drawn, reported


def detail_layout(
    candidates: list[Candidate],
    *,
    label: str,
    state: str,
    higher_is_better: bool = True,
    started_at: str | None = None,
) -> DetailLayout:
    """The curve plus the tree behind it. A mark per scored, unpruned
    candidate; an edge from its parent when the parent is scored too (a
    failed parent leaves its children rootless rather than inventing a y)."""
    from hillclimb.tree import build_tree, minutes_since

    curve = curve_from_candidates(
        candidates, label=label, state=state, higher_is_better=higher_is_better, started_at=started_at
    )
    layout = DetailLayout(curve=curve)
    tree = build_tree(candidates, higher_is_better)
    scored = {c.candidate_id: c for c in candidates if c.val_score is not None and not c.pruned}
    if not scored:
        layout.unscored = len(candidates)
        return layout
    origin = started_at or min(c.finished_at or c.created_at for c in scored.values())
    on_path = set(tree.accepted)
    at: dict[str, tuple[float, float]] = {}
    for node in tree.nodes:
        cand = scored.get(node.id)
        if cand is None or node.score is None:
            layout.unscored += 1
            continue
        x = minutes_since(cand.finished_at or cand.created_at, origin)
        if x is None:
            continue
        at[node.id] = (x, node.score)
        layout.marks.append(DetailMark(node.id, x, node.score, node.operator, node.fate, node.id in on_path))
    by_id = {n.id: n for n in tree.nodes}
    for edge in tree.edges:
        if edge.kind != "parent" or edge.src not in at or edge.dst not in at:
            continue
        (x0, y0), (x1, y1) = at[edge.src], at[edge.dst]
        layout.edges.append(DetailEdge(x0, y0, x1, y1, by_id[edge.dst].operator, edge.on_path))
    return layout


def record_detail(store: DataStore, record: SearchRecord, label: str) -> DetailLayout:
    return detail_layout(
        list(Journal(store.journal(record.key)).candidates.values()),
        label=label,
        state=record.state,
        higher_is_better=bool(record.meta.higher_is_better),
        started_at=record.meta.started_at,
    )


def chart_problem(config: Config, search: str | None = None) -> SearchMeta | None:
    """The search the chart is anchored on — the given ref's, else the
    latest — as its metadata (problem_key, metric, direction); None when
    there are no searches yet."""
    store = open_store(config)
    try:
        return resolve_search(store, search).meta
    except LookupError:
        return None
    finally:
        store.close()


# --- Textual app ---

from plotui import Plot  # noqa: E402
from plotui.textual import PlotWidget  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import Vertical  # noqa: E402
from textual.widgets import Footer, Label  # noqa: E402

from hillclimb.header import HillclimbHeader, TimezoneMixin  # noqa: E402
from hillclimb.keys import KEYS_BINDING, QUIT_BINDINGS  # noqa: E402

from hillclimb.theme import CYAN, HILLCLIMB_CSS, PLOT_BG, apply_theme, themed_plot  # noqa: E402
from hillclimb.watch import STATE_STYLE, LiveScreen  # noqa: E402


def curve_colors(curves: list[Curve]) -> list[tuple[int, int, int] | None]:
    """A colour per curve: curves of the same experiment arm share one, so an
    arm's repeats read as one family against the others; curves without an
    arm (None) take plotui's next palette slot as before."""
    arms = list(dict.fromkeys(c.arm for c in curves if c.arm))
    return [ARM_PALETTE[arms.index(c.arm) % len(ARM_PALETTE)] if c.arm else None for c in curves]


def build_plot(curves: list[Curve]) -> Plot:
    """One step line per curve — the experiment view, where each arm is a
    series of its own, and the base of the detail overlay."""
    plot = themed_plot()
    for curve, color in zip(curves, curve_colors(curves)):
        if not curve.xs:
            continue
        # a single point draws nothing as a line; the flat stub makes the
        # first candidate visible the moment it lands
        xs, ys = step_points(curve.xs, curve.ys, curve.xs[-1] + (0.1 if len(curve.xs) == 1 else 0.0))
        plot.add_line(xs, ys, color=color, name=curve.label)
    return plot


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


# A miss is the same hue as the staircase, receded towards the background —
# the site's `.miss { fill-opacity: .45 }` — so the climb stays the figure
# and the attempts stay the ground.
MISS_RGB = _mix(PLOT_BG, CYAN, 0.45)


def build_climb_plot(climb: Climb) -> Plot:
    """The website's figure: the staircase in cyan, a bright dot where a
    candidate set a new best, a dim one where it scored but did not."""
    plot = themed_plot()
    misses = [e for e in climb.events if not e.best]
    if misses:
        plot.add_scatter([e.x for e in misses], [e.y for e in misses], color=MISS_RGB, size=2.4, name="attempt")
    xs, ys = climb.staircase()
    if len(xs) == 1:
        xs, ys = step_points(xs, ys, xs[0] + 0.1)
    if xs:
        plot.add_line(xs, ys, color=CYAN, width=2.0, name="best so far")
    hits = [e for e in climb.events if e.best]
    if hits:
        plot.add_scatter([e.x for e in hits], [e.y for e in hits], color=CYAN, size=3.0, name="new best")
    return plot


def build_detail_plot(layout: DetailLayout) -> Plot:
    """The curve as in build_plot, then the tree: one thin line per edge
    (dim, the child's operator colour; bold on the accepted lineage) and a
    scatter per operator so the legend names them; accepted candidates get
    a larger mark on top."""
    from hillclimb.treeview import OPERATOR_RGB, dim_rgb

    plot = build_plot([layout.curve])
    for edge in layout.edges:
        rgb = OPERATOR_RGB.get(edge.operator, (160, 160, 160))
        plot.add_line(
            [edge.x0, edge.x1], [edge.y0, edge.y1],
            color=rgb if edge.on_path else dim_rgb(rgb, 0.45),
            width=2.0 if edge.on_path else 1.0,
        )
    by_operator: dict[str, list[DetailMark]] = {}
    for mark in layout.marks:
        by_operator.setdefault(mark.operator, []).append(mark)
    for operator, marks in by_operator.items():
        plot.add_scatter(
            [m.x for m in marks], [m.y for m in marks],
            color=OPERATOR_RGB.get(operator, (160, 160, 160)), size=2.5, name=operator,
        )
    accepted = [m for m in layout.marks if m.on_path]
    if accepted:
        plot.add_scatter(
            [m.x for m in accepted], [m.y for m in accepted],
            color=(255, 255, 255), size=4.0, name="accepted",
        )
    return plot


class ChartScreen(LiveScreen):
    BINDINGS = [
        Binding("d", "toggle_detail", "detail", tooltip="overlay the exploration tree"),
        Binding("r", "refresh", "refresh", show=False),
        KEYS_BINDING,
        *QUIT_BINDINGS,
    ]

    DEFAULT_CSS = """
    ChartScreen #chartline { height: 1; padding: 0 1; background: $surface; }
    ChartScreen #chart-stage { width: 1fr; height: 1fr; background: #0e1113; }
    """

    def __init__(self, config: Config, search: str | None = None, detail: bool = False):
        super().__init__()
        self.config = config
        self.search = search
        self.detail = detail  # one search, with its exploration tree on the curve
        self._anchor: SearchMeta | None = None  # resolved once; `r` re-resolves
        self._key: tuple | None = None
        self._store: DataStore | None = None  # opened on first refresh, kept for the session

    def compose(self) -> ComposeResult:
        yield HillclimbHeader()
        yield Label(id="chartline")
        yield Vertical(id="chart-stage")
        yield Footer()

    def on_mount(self) -> None:
        self.start_live()

    def action_refresh(self) -> None:
        self._anchor = None
        self._key = None
        self.refresh_data()

    def action_toggle_detail(self) -> None:
        self.detail = not self.detail
        self._key = None
        self.refresh_data()

    def refresh_data(self) -> None:
        if self._anchor is None:
            self._anchor = chart_problem(self.config, self.search)
        anchor = self._anchor
        if anchor is None:
            self.query_one("#chartline", Label).update(
                f"no searches in {self.config.paths.runs_dir} yet — start one with `hillclimb run`"
            )
            return
        if self._store is None:
            self._store = open_store(self.config)
        direction = "higher" if anchor.higher_is_better else "lower"
        parts = [f"[bold]{anchor.problem_key}[/]  {anchor.metric} ({direction} is better)"]
        layout: DetailLayout | None = None
        if self.detail:
            record = self._store.search((anchor.run_id, anchor.search_id))
            if record is None:
                layout = DetailLayout(curve=Curve(label=f"{anchor.run_id}/{anchor.search_id}", state="unknown"))
            else:
                layout = record_detail(self._store, record, f"{record.run_name}/{record.search_id}")
            curves = [layout.curve]
            parts.append(
                f"detail: {len(layout.marks)} scored · {len(layout.edges)} edges"
                + (f" · {layout.unscored} unscored" if layout.unscored else "")
            )
        else:
            curves = climb_curves(self._store, anchor.problem_key)
        climb: Climb | None = None
        if layout is not None or any(c.arm for c in curves):
            # one line per search: the detail overlay, or an experiment
            # where the arms are the comparison
            for curve in curves:
                style = STATE_STYLE.get(curve.state, "")
                best = f"{curve.best:.5g}" if curve.best is not None else "-"
                parts.append(f"{curve.label} [{style}]{curve.state}[/] best={best}")
        else:
            climb = climb_for_problem(self._store, anchor.problem_key)
            best = f"{climb.best:.5g}" if climb.best is not None else "-"
            n = climb.searches
            running = sum(1 for c in curves if c.state == "running")
            parts.append(
                f"best={best} · {n} search{'' if n == 1 else 'es'}"
                + (f" ([{STATE_STYLE.get('running', '')}]{running} running[/])" if running else "")
                + f" · {climb.hits} of {len(climb.events)} candidates improved"
            )
        self.query_one("#chartline", Label).update("  ·  ".join(parts))
        key: tuple = (self.detail, tuple((c.label, tuple(c.xs), tuple(c.ys)) for c in curves))
        if layout is not None:
            key += (tuple((m.id, m.x, m.y) for m in layout.marks),)
        if climb is not None:
            key += (tuple((e.x, e.y, e.best) for e in climb.events),)
        if key == self._key:
            return
        self._key = key
        stage = self.query_one("#chart-stage", Vertical)
        if any(c.xs for c in curves):
            if layout is not None:
                plot = build_detail_plot(layout)
            elif climb is not None:
                plot = build_climb_plot(climb)
            else:
                plot = build_plot(curves)
            canvas = stage.query(PlotWidget)
            if canvas:
                # swap the plot in place: widget removal is asynchronous, so
                # remounting under the same id would collide with the old one
                canvas.first()._plot = plot
                canvas.first().invalidate()
            else:
                stage.remove_children()
                stage.mount(PlotWidget(plot, id="chart-canvas"))
        elif not stage.query("#chart-empty"):
            stage.remove_children()
            stage.mount(Label("waiting for the first scored candidate…", id="chart-empty"))


class ChartApp(TimezoneMixin, App):
    """Standalone shell for `hillclimb chart`."""

    BINDINGS = [
        Binding("t", "choose_timezone", "time zone", show=False),
        Binding("ctrl+c", "quit", "quit", show=False, priority=True),
    ]
    CSS = HILLCLIMB_CSS

    def __init__(self, config: Config | None = None, search: str | None = None, detail: bool = False):
        super().__init__()
        self.config = config or Config.load()
        self.search = search
        self.detail = detail

    def on_mount(self) -> None:
        apply_theme(self)
        self._init_timezone()
        self.push_screen(ChartScreen(self.config, self.search, detail=self.detail))
