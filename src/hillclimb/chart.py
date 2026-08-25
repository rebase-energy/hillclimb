"""`hillclimb chart` — the live hillclimb curve.

One staircase per problem: the best validation score so far across every
search of it, against the number of tested candidate solutions, with every
scored candidate as a dot on the same axes — bright where it set a new best,
dim where it missed. The demo's three parallel searches are one climb, not
three; a search only gets its own line in an experiment, where the arms are
the comparison (build_plot). Same figure as the website's, in the same
colours. Strictly a viewer like watch.py: everything is read through the configured store
(store.py — the hillclimb folder by default, or the SQLite index), and the
pure data functions at the top stay testable without Textual.

`--detail` anchors on one search and overlays its exploration tree on the
curve: every scored candidate as a mark at (evaluation number, its score),
parent→child edges between them, the accepted lineage bold — the climb and
the attempts it took, on one pair of axes (tree.py supplies the lineage).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
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

# Reference lines sit behind the climb. The first is deliberately neutral for
# the usual "baseline" floor; additional named comparisons cycle through hues
# distinct from the chart's cyan best-so-far line.
CHART_BASELINE_PALETTE = (
    (144, 153, 160),
    (234, 179, 8),
    (168, 85, 247),
    (249, 115, 22),
    (59, 130, 246),
    (236, 72, 153),
    (34, 197, 94),
)

# Kitty reserves z values below INT32_MIN / 2 for images that must sit below
# cells with explicit backgrounds. Annotation tags use an explicit black
# background, so this makes the whole tag occlude the plot instead of letting
# the staircase show through the gaps inside and between glyphs.
_KITTY_BELOW_CELL_BACKGROUND_Z = -1_073_741_825


@dataclass
class Curve:
    label: str
    state: str
    xs: list[float] = field(default_factory=list)  # 1-based scored-candidate count
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
    candidate in finish order; the 1-based x value is how many candidates
    have been tested. The y value only ever moves in the metric's good
    direction, so the line is the staircase the search climbed.

    `started_at` remains accepted for callers reading older search metadata;
    candidate count no longer depends on the search's wall-clock origin.
    """
    higher = higher_is_better
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
    best: float | None = None
    for experiment, (_when, score) in enumerate(scored, start=1):
        if best is None or better(score, best, higher):
            best = score
        curve.xs.append(float(experiment))
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
    x: float            # 1-based scored-candidate count across the climb
    y: float            # val score
    best: bool          # set a new best across every search, when it landed
    search: str         # label of the search it came from
    operator: str
    summary: str = ""   # what changed, for new-best annotations


@dataclass
class Climb:
    """Every scored candidate of a problem, across its searches, in landing
    order; the staircase is the `best` ones."""

    events: list[ClimbEvent] = field(default_factory=list)
    extent: float = 0.0  # tested candidates — the staircase runs flat to here
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
    `searches` is (label, candidates, started_at) per search; `started_at` is
    retained in that shape for callers but x is the 1-based candidate count.
    `best` is judged against everything that landed before, whichever search
    it came from."""
    higher = higher_is_better
    landed: list[tuple[datetime, float, str, str, str]] = []
    for label, candidates, _ in searches:
        for cand in candidates:
            if cand.pruned or cand.val_score is None:
                continue
            when = _parse_ts(cand.finished_at) or _parse_ts(cand.created_at)
            if when is None:
                continue
            landed.append((when, cand.val_score, label, cand.operator or "", cand.summary or ""))
    landed.sort(key=lambda item: item[0])
    climb = Climb(searches=len(searches))
    if not landed:
        return climb
    best: float | None = None
    for experiment, (_when, score, label, operator, summary) in enumerate(landed, start=1):
        improved = best is None or better(score, best, higher)
        if improved:
            best = score
        climb.events.append(ClimbEvent(float(experiment), score, improved, label, operator, summary))
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
    x: float            # 1-based scored-candidate count
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
    from hillclimb.tree import build_tree

    curve = curve_from_candidates(
        candidates, label=label, state=state, higher_is_better=higher_is_better, started_at=started_at
    )
    layout = DetailLayout(curve=curve)
    tree = build_tree(candidates, higher_is_better)
    scored = {c.candidate_id: c for c in candidates if c.val_score is not None and not c.pruned}
    if not scored:
        layout.unscored = len(candidates)
        return layout
    landed = []
    for cand in scored.values():
        when = _parse_ts(cand.finished_at) or _parse_ts(cand.created_at)
        if when is not None:
            landed.append((when, cand.candidate_id))
    landed.sort(key=lambda item: item[0])
    experiment_by_id = {
        candidate_id: float(index)
        for index, (_when, candidate_id) in enumerate(landed, start=1)
    }
    on_path = set(tree.accepted)
    at: dict[str, tuple[float, float]] = {}
    for node in tree.nodes:
        cand = scored.get(node.id)
        if cand is None or node.score is None:
            layout.unscored += 1
            continue
        x = experiment_by_id.get(node.id)
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


def chart_baselines(config: Config, meta: SearchMeta) -> dict[str, float]:
    """Current problem-config reference lines, falling back to the snapshot
    in search metadata when the original local problem is unavailable.

    Reloading local problem.yaml makes chart-only edits take effect for old
    searches too. Provider problems have no user-owned problem.yaml, so their
    persisted copy keeps charts portable without rematerializing the provider
    on every live refresh.
    """
    from hillclimb.problem import load_problem

    if "://" in meta.problem:
        return dict(meta.chart_baselines)
    try:
        return dict(load_problem(meta.problem, config).chart_baselines)
    except (ImportError, KeyError, OSError, ValueError):
        return dict(meta.chart_baselines)


# --- Textual app ---

from plotui import Plot  # noqa: E402
from plotui.textual import OverlaySpan, PlotWidget  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import Vertical  # noqa: E402
from textual.widgets import Footer, Label  # noqa: E402

from hillclimb.header import HillclimbHeader, TimezoneMixin  # noqa: E402
from hillclimb.keys import KEYS_BINDING, QUIT_BINDINGS  # noqa: E402

from hillclimb.theme import CYAN, HILLCLIMB_CSS, PLOT_BG, apply_theme, themed_plot  # noqa: E402
from hillclimb.watch import STATE_STYLE, LiveScreen  # noqa: E402


class ChartPlotWidget(PlotWidget):
    """Plot widget with terminal-native improvement labels and safe compositing.

    The chart is a static figure, like the website's: zoom and pan are
    disabled (a stray scroll used to shrink the staircase to a speck), so the
    camera always frames the whole climb.

    Kitty uploads ride on a zero-width Rich control segment. Textual's
    monochrome filter expects composited segments to carry a Style, so give
    only otherwise-unstyled segments a neutral one. This is invisible in a
    real colour terminal and keeps NO_COLOR/headless rendering valid.
    """

    def __init__(
        self,
        plot: Plot,
        *,
        annotations: list[ImprovementAnnotation] | None = None,
        annotation_bounds: tuple[float, float, float, float] | None = None,
        **kwargs,
    ):
        self._annotations = list(annotations or [])
        self._annotation_bounds = annotation_bounds
        super().__init__(plot, **kwargs)

    def set_annotations(
        self,
        annotations: list[ImprovementAnnotation],
        bounds: tuple[float, float, float, float] | None,
    ) -> None:
        self._annotations = list(annotations)
        self._annotation_bounds = bounds
        self._sync_annotations()

    def _sync_annotations(self) -> None:
        spans = []
        if self._annotation_bounds is not None:
            spans = annotation_spans(
                self._annotations,
                self._annotation_bounds,
                self.size.width,
                self.size.height,
                camera_state=self._plot.camera_state(),
                cell_px=(self._cell_w, self._cell_h),
            )
        self.set_overlay(spans)

    def on_resize(self) -> None:
        self._sync_annotations()

    def apply_pan(self, dx: float, dy: float) -> None:
        pass  # static figure — the whole climb stays framed

    def apply_zoom(self, factor: float) -> None:
        pass  # static figure — a stray scroll must not shrink the chart

    def apply_reset(self) -> None:
        super().apply_reset()
        self._sync_annotations()

    def _ensure_frame(self) -> None:
        super()._ensure_frame()
        if (
            self._mode == "placeholder"
            and self._transmit
            and "a=T,U=1,z=" not in self._transmit
        ):
            # A regular negative z-index only puts the image below glyphs;
            # this lower protocol layer also puts it below their backgrounds.
            self._transmit = self._transmit.replace(
                "a=T,U=1,",
                f"a=T,U=1,z={_KITTY_BELOW_CELL_BACKGROUND_Z},",
                1,
            )
        elif self._mode == "direct" and self._transmit:
            # iTerm2 uses direct placement. plotui already puts that image at
            # z=-1 (below glyphs); lower it past Kitty's background threshold
            # as well so the black annotation tag masks the graph completely.
            self._transmit = self._transmit.replace(
                "z=-1,",
                f"z={_KITTY_BELOW_CELL_BACKGROUND_Z},",
                1,
            )

    def render_line(self, y: int):
        from rich.segment import Segment
        from rich.style import Style
        from textual.strip import Strip

        strip = super().render_line(y)
        return Strip(
            [
                segment if segment.style is not None
                else Segment(segment.text, Style(), segment.control)
                for segment in strip
            ],
            strip.cell_length,
        )


def curve_colors(curves: list[Curve]) -> list[tuple[int, int, int] | None]:
    """A colour per curve: curves of the same experiment arm share one, so an
    arm's repeats read as one family against the others; curves without an
    arm (None) take plotui's next palette slot as before."""
    arms = list(dict.fromkeys(c.arm for c in curves if c.arm))
    return [ARM_PALETTE[arms.index(c.arm) % len(ARM_PALETTE)] if c.arm else None for c in curves]


def _add_chart_baselines(
    plot: Plot,
    baselines: Mapping[str, float],
    extent: float,
    *,
    show_legend: bool = True,
    hidden: frozenset[str] | set[str] = frozenset(),
) -> None:
    """Add arbitrary named horizontal score references behind the data.

    `hidden` entries are skipped but keep their palette slot, so toggling one
    off never recolours the others out from under the legend."""
    right = max(1.0, extent)
    for index, (label, value) in enumerate(baselines.items()):
        if label in hidden:
            continue
        plot.add_line(
            [0.0, right],
            [value, value],
            color=CHART_BASELINE_PALETTE[index % len(CHART_BASELINE_PALETTE)],
            width=1.0,
            name=label if show_legend else None,
        )


def build_plot(
    curves: list[Curve],
    baselines: Mapping[str, float] | None = None,
    *,
    show_legend: bool = True,
    hidden: frozenset[str] | set[str] = frozenset(),
) -> Plot:
    """One step line per curve — the experiment view, where each arm is a
    series of its own, and the base of the detail overlay. `hidden` names
    legend entries toggled off: their traces are left out of the plot."""
    plot = themed_plot()
    extent = max((max(c.xs, default=0.0) for c in curves), default=0.0)
    _add_chart_baselines(plot, baselines or {}, extent, show_legend=show_legend, hidden=hidden)
    trace_index = len(baselines or {})
    for curve, color in zip(curves, curve_colors(curves)):
        if not curve.xs:
            continue
        # Pin the colour plot_legend assigns this slot: skipping a hidden
        # trace must not let plotui's next-palette-slot drift under the rest.
        rgb = color or ARM_PALETTE[trace_index % len(ARM_PALETTE)]
        trace_index += 1
        if curve.label in hidden:
            continue
        if len(curve.xs) == 1:
            # Keep the domain in whole candidate counts; a one-point line is
            # invisible, so render that first evaluation as a dot.
            plot.add_scatter(
                curve.xs, curve.ys, color=rgb, size=3.0,
                name=curve.label if show_legend else None,
            )
        else:
            xs, ys = step_points(curve.xs, curve.ys)
            plot.add_line(xs, ys, color=rgb, name=curve.label if show_legend else None)
    return plot


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


# A miss is the same hue as the staircase, receded towards the background —
# the site's `.miss { fill-opacity: .45 }` — so the climb stays the figure
# and the attempts stay the ground.
MISS_RGB = _mix(PLOT_BG, CYAN, 0.45)


def build_climb_plot(
    climb: Climb,
    baselines: Mapping[str, float] | None = None,
    *,
    show_legend: bool = True,
    hidden: frozenset[str] | set[str] = frozenset(),
) -> Plot:
    """The website's figure: the staircase in cyan, a bright dot where a
    candidate set a new best, a dim one where it scored but did not.
    `hidden` names legend entries toggled off — their traces are left out."""
    plot = themed_plot()
    _add_chart_baselines(plot, baselines or {}, climb.extent, show_legend=show_legend, hidden=hidden)
    misses = [e for e in climb.events if not e.best]
    if misses and "attempt" not in hidden:
        plot.add_scatter(
            [e.x for e in misses], [e.y for e in misses], color=MISS_RGB, size=2.4,
            name="attempt" if show_legend else None,
        )
    xs, ys = climb.staircase()
    if len(xs) > 1 and "best so far" not in hidden:
        plot.add_line(
            xs, ys, color=CYAN, width=2.0,
            name="best so far" if show_legend else None,
        )
    hits = [e for e in climb.events if e.best]
    if hits and "new best" not in hidden:
        plot.add_scatter(
            [e.x for e in hits], [e.y for e in hits], color=CYAN, size=3.0,
            name="new best" if show_legend else None,
        )
    return plot


def build_detail_plot(
    layout: DetailLayout,
    baselines: Mapping[str, float] | None = None,
    *,
    show_legend: bool = True,
    hidden: frozenset[str] | set[str] = frozenset(),
) -> Plot:
    """The curve as in build_plot, then the tree: one thin line per edge
    (dim, the child's operator colour; bold on the accepted lineage) and a
    scatter per operator so the legend names them; accepted candidates get
    a larger mark on top. `hidden` names legend entries toggled off — an
    operator takes its marks and its edges with it."""
    from hillclimb.treeview import OPERATOR_RGB, dim_rgb

    plot = build_plot([layout.curve], baselines, show_legend=show_legend, hidden=hidden)
    for edge in layout.edges:
        if edge.operator in hidden:
            continue
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
        if operator in hidden:
            continue
        plot.add_scatter(
            [m.x for m in marks], [m.y for m in marks],
            color=OPERATOR_RGB.get(operator, (160, 160, 160)), size=2.5,
            name=operator if show_legend else None,
        )
    accepted = [m for m in layout.marks if m.on_path]
    if accepted and "accepted" not in hidden:
        plot.add_scatter(
            [m.x for m in accepted], [m.y for m in accepted],
            color=(255, 255, 255), size=4.0,
            name="accepted" if show_legend else None,
        )
    return plot


# (label, colour, glyph) — the glyph mirrors the trace's mark, the way the
# knowledge graph's legend echoes each node type's marker: "─" for a line
# trace, "●" for a scatter. legend_text tolerates the old two-field shape.
LegendEntry = tuple[str, tuple[int, int, int], str]


def _baseline_legend(baselines: Mapping[str, float]) -> list[LegendEntry]:
    return [
        (label, CHART_BASELINE_PALETTE[index % len(CHART_BASELINE_PALETTE)], "─")
        for index, label in enumerate(baselines)
    ]


def plot_legend(curves: list[Curve], baselines: Mapping[str, float]) -> list[LegendEntry]:
    """Legend for an experiment/detail base plot, in trace order."""
    entries = _baseline_legend(baselines)
    trace_index = len(baselines)
    for curve, color in zip(curves, curve_colors(curves)):
        if not curve.xs:
            continue
        # ARM_PALETTE mirrors plotui's default trace palette. An uncoloured
        # trace takes the slot determined by everything already added.
        entries.append((curve.label, color or ARM_PALETTE[trace_index % len(ARM_PALETTE)], "─"))
        trace_index += 1
    return entries


def climb_legend(climb: Climb, baselines: Mapping[str, float]) -> list[LegendEntry]:
    entries = _baseline_legend(baselines)
    if any(not event.best for event in climb.events):
        entries.append(("attempt", MISS_RGB, "●"))
    if len(climb.staircase()[0]) > 1:
        entries.append(("best so far", CYAN, "─"))
    if any(event.best for event in climb.events):
        entries.append(("new best", CYAN, "●"))
    return entries


def detail_legend(layout: DetailLayout, baselines: Mapping[str, float]) -> list[LegendEntry]:
    from hillclimb.treeview import OPERATOR_RGB

    entries = plot_legend([layout.curve], baselines)
    for operator in dict.fromkeys(mark.operator for mark in layout.marks):
        entries.append((operator, OPERATOR_RGB.get(operator, (160, 160, 160)), "●"))
    if any(mark.on_path for mark in layout.marks):
        entries.append(("accepted", (255, 255, 255), "●"))
    return entries


@dataclass(frozen=True)
class ImprovementAnnotation:
    x: float
    y: float
    text: str


def brief_improvement(summary: str, operator: str, max_chars: int = 42) -> str:
    """Turn an agent's result summary into one chart-sized improvement label."""
    if operator == "baseline":
        return "baseline"
    text = " ".join(summary.replace("`", "").replace("**", "").split()).strip()
    if not text:
        return operator or "improvement"

    # Agent summaries often lead with a useful named technique before a
    # colon, followed by the full rationale. Prefer that natural title.
    prefix, separator, _rest = text.partition(":")
    if separator and 1 < len(prefix.split()) <= 7 and len(prefix) <= max_chars:
        return prefix.strip().rstrip(".,;")

    text = re.sub(r"^(?:this candidate|the candidate)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:uses?|i)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:a|an)\s+", "", text, flags=re.IGNORECASE)
    text = text.split(".", 1)[0].strip()
    words = text.split()
    shortened = " ".join(words[:7])
    clipped = shortened[:max_chars].rstrip(" ,.;:-")
    if len(shortened) > max_chars:
        clipped = clipped.rsplit(" ", 1)[0] or clipped
    if len(words) > 7 or len(shortened) > max_chars:
        clipped += "…"
    return clipped or operator or "improvement"


def improvement_annotations(climb: Climb) -> list[ImprovementAnnotation]:
    return [
        ImprovementAnnotation(
            event.x,
            event.y,
            brief_improvement(event.summary, event.operator),
        )
        for event in climb.events
        if event.best
    ]


def climb_plot_bounds(
    climb: Climb,
    baselines: Mapping[str, float],
) -> tuple[float, float, float, float]:
    """The same 5%-padded data bounds plotui derives from chart traces."""
    xs = [event.x for event in climb.events]
    ys = [event.y for event in climb.events]
    if baselines:
        right = max(1.0, climb.extent)
        xs.extend((0.0, right))
        ys.extend(float(value) for value in baselines.values())
    if not xs or not ys:
        return (-1.0, 1.0, -1.0, 1.0)

    def padded(lo: float, hi: float) -> tuple[float, float]:
        span = hi - lo
        pad = span * 0.05 if span > 0 else 1.0
        return lo - pad, hi + pad

    xlo, xhi = padded(min(xs), max(xs))
    ylo, yhi = padded(min(ys), max(ys))
    return xlo, xhi, ylo, yhi


def annotation_spans(
    annotations: list[ImprovementAnnotation],
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    camera_state: tuple[float, float, float, float, float] = (0.0, 0.0, 1.0, 0.0, 0.0),
    cell_px: tuple[int, int] = (12, 24),
) -> list[OverlaySpan]:
    """Place short new-best labels near their dots, using collision lanes."""
    if not annotations or width < 20 or height < 8:
        return []
    from rich.style import Style

    xlo, xhi, ylo, yhi = bounds
    if xhi <= xlo or yhi <= ylo:
        return []
    left = max(7, min(14, width // 16))
    right = max(left + 4, width - 2)
    top, bottom = 1, max(5, height - 3)
    _yaw, _pitch, zoom, pan_x, pan_y = camera_state
    cx, cy = (left + right) / 2, (top + bottom) / 2

    occupied: dict[int, list[tuple[int, int]]] = {}
    spans: list[OverlaySpan] = []
    # Textual applies this overlay after plotui has rasterized the chart, so
    # these opaque tags always sit above (and hide) the staircase beneath.
    style = Style.parse(
        f"italic rgb({CYAN[0]},{CYAN[1]},{CYAN[2]}) on #000000"
    )
    for annotation in sorted(annotations, key=lambda item: item.x):
        base_col = left + (annotation.x - xlo) / (xhi - xlo) * (right - left)
        base_row = top + (yhi - annotation.y) / (yhi - ylo) * (bottom - top)
        point_col = round(cx + (base_col - cx) * zoom + pan_x / max(1, cell_px[0]))
        point_row = round(cy + (base_row - cy) * zoom + pan_y / max(1, cell_px[1]))
        if not (left <= point_col <= right and top <= point_row <= bottom):
            continue

        max_label = max(12, min(44, width // 3))
        label = annotation.text
        if len(label) > max_label:
            label = label[: max_label - 1].rstrip() + "…"
        text = f" ╱ {label} "
        start = point_col + 1
        if start + len(text) > right:
            start = max(left, point_col - len(text) - 1)

        placed = False
        # Above first (like the reference chart), then below; successive lanes
        # keep nearby improvements readable without hiding the staircase.
        offsets = [
            offset
            for distance in range(1, height)
            for offset in (-distance, distance)
        ]
        for offset in offsets:
            row = point_row + offset
            end = start + len(text)
            if row < top or row > bottom:
                continue
            if any(not (end + 1 < lo or start > hi + 1) for lo, hi in occupied.get(row, [])):
                continue
            occupied.setdefault(row, []).append((start, end))
            spans.append((row, start, text, style))
            placed = True
            break
        if not placed:
            # More improvements than available lanes is intrinsically dense;
            # keep the annotation visible rather than silently omitting it.
            row = min(bottom, max(top, point_row - 1))
            spans.append((row, start, text, style))
    return spans


def legend_text(
    entries: list[LegendEntry],
    width: int | None = None,
    *,
    hidden: frozenset[str] | set[str] = frozenset(),
    interactive: bool = False,
) -> "Text":
    """A terminal-crisp legend, wrapped only between complete entries.

    Entries render the way the knowledge graph's legend does: a dim hotkey
    number, then the trace's glyph and name in the trace's own colour; an
    entry in `hidden` loses its glyph and goes dim and struck through, so the
    legend itself shows what the chart is not drawing. `interactive` turns
    every entry into a click target that toggles its series
    (`screen.toggle_series`) and adds the 1-9 hotkey prefix."""
    from rich.cells import cell_len
    from rich.style import Style
    from rich.text import Text

    text = Text(no_wrap=width is not None, overflow="crop" if width is not None else None)
    line_width = 0
    for index, (label, (red, green, blue), *rest) in enumerate(entries):
        glyph = rest[0] if rest else "●"
        off = label in hidden
        meta = {"@click": f"screen.toggle_series({label!r})"} if interactive else None
        item = Text()
        if interactive and index < 9:
            item.append(f"{index + 1} ", style=Style.parse("dim") + Style(meta=meta))
        if off:
            item.append(f"  {label}", style=Style.parse("dim strike") + Style(meta=meta))
        else:
            item.append(f"{glyph} {label}", style=Style.parse(f"rgb({red},{green},{blue})") + Style(meta=meta))
        item_width = cell_len(item.plain)
        separator_width = 3 if line_width else 0
        if width is not None and line_width and line_width + separator_width + item_width > width:
            text.append("\n")
            line_width = 0
            separator_width = 0
        if separator_width:
            text.append(" " * separator_width)
        text.append(item)
        line_width += separator_width + item_width
    return text


class ChartLegend(Label):
    """Responsive legend whose entries move as units between rows. Every
    entry is a click target: clicking toggles that series on the chart
    (`ChartScreen.action_toggle_series`), and a toggled-off entry stays in
    the legend — receded and struck through — as the way back."""

    def __init__(self, *, id: str):
        super().__init__(id=id)
        # Textual restyles any @click text as a hyperlink (underline, theme
        # link colour) — which would flatten the per-trace colours to one.
        # Clicks still dispatch without the link dress-up.
        self.auto_links = False
        self._entries: list[LegendEntry] = []
        self._hidden: frozenset[str] = frozenset()
        self._layout_width = -1

    def set_entries(self, entries: list[LegendEntry], hidden: set[str] | frozenset[str] = frozenset()) -> None:
        self._entries = list(entries)
        self._hidden = frozenset(hidden)
        self._layout_width = -1
        self._sync_entries()

    def _sync_entries(self, fallback_width: int | None = None) -> None:
        width = self.content_region.width
        if width <= 0:
            width = max(1, (fallback_width or self.size.width) - 4)
        if width == self._layout_width:
            return
        self._layout_width = width
        self.update(legend_text(self._entries, width, hidden=self._hidden, interactive=True))

    def on_resize(self, event) -> None:
        self._sync_entries(event.size.width)


class ChartScreen(LiveScreen):
    BINDINGS = [
        Binding("d", "toggle_detail", "detail", tooltip="overlay the exploration tree"),
        Binding(
            "t", "toggle_annotations", "text",
            tooltip="show or hide improvement text", priority=True,
        ),
        Binding("r", "refresh", "refresh", show=False),
        # One binding per legend slot, like the knowledge graph's 1-8; only
        # the first is described, so the `?` panel shows a single "1-9" row.
        Binding("1", "toggle_entry(0)", "hide/show a series", show=False, key_display="1-9"),
        *(Binding(str(i + 1), f"toggle_entry({i})", show=False) for i in range(1, 9)),
        KEYS_BINDING,
        *QUIT_BINDINGS,
    ]

    DEFAULT_CSS = """
    ChartScreen #chartline { height: 1; padding: 0 1; background: $surface; }
    ChartScreen #chart-body { width: 1fr; height: 1fr; background: #0e1113; }
    ChartScreen #chart-legend {
        width: 1fr; height: auto; min-height: 1; padding: 0 2;
        text-align: center; background: #0e1113;
    }
    ChartScreen #chart-stage { width: 1fr; height: 1fr; background: #0e1113; }
    ChartScreen #chart-x-label {
        width: 1fr; height: 1; text-align: center; color: $secondary; background: #0e1113;
    }
    """

    def __init__(self, config: Config, search: str | None = None, detail: bool = False):
        super().__init__()
        self.config = config
        self.search = search
        self.detail = detail  # one search, with its exploration tree on the curve
        self.show_annotations = False  # `t` reveals the improvement labels
        self.hidden_series: set[str] = set()  # legend entries toggled off by click
        self._legend_entries: list[LegendEntry] = []  # what 1-9 index into
        self._anchor: SearchMeta | None = None  # resolved once; `r` re-resolves
        self._key: tuple | None = None
        self._store: DataStore | None = None  # opened on first refresh, kept for the session

    def compose(self) -> ComposeResult:
        yield HillclimbHeader()
        yield Label(id="chartline")
        with Vertical(id="chart-body"):
            yield Vertical(id="chart-stage")
            yield Label("candidates", id="chart-x-label")
            yield ChartLegend(id="chart-legend")
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

    def action_toggle_annotations(self) -> None:
        self.show_annotations = not self.show_annotations
        self._key = None
        self.refresh_data()

    def action_toggle_series(self, label: str) -> None:
        """Show or hide one series — reached by clicking its legend entry."""
        self.hidden_series.symmetric_difference_update({label})
        self._key = None
        self.refresh_data()

    def action_toggle_entry(self, index: int) -> None:
        """The keyboard route to the same toggle: 1-9 by legend position."""
        if 0 <= index < len(self._legend_entries):
            self.action_toggle_series(self._legend_entries[index][0])

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
        # The website's climb head, word for word, so both renderings of the
        # figure introduce it the same way.
        direction = "higher" if anchor.higher_is_better else "lower"
        parts = [f"[bold]{anchor.problem_key}[/] — best {anchor.metric} so far ({direction} is better)"]
        baselines = chart_baselines(self.config, anchor)
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
        key: tuple = (
            self.detail,
            self.show_annotations,
            tuple(sorted(self.hidden_series)),
            tuple((c.label, tuple(c.xs), tuple(c.ys)) for c in curves),
        )
        key += (tuple(baselines.items()),)
        if layout is not None:
            key += (tuple((m.id, m.x, m.y) for m in layout.marks),)
        if climb is not None:
            key += (tuple((e.x, e.y, e.best, e.summary) for e in climb.events),)
        if key == self._key:
            return
        self._key = key
        stage = self.query_one("#chart-stage", Vertical)
        legend = self.query_one("#chart-legend", ChartLegend)
        if any(c.xs for c in curves):
            hidden = self.hidden_series
            annotations: list[ImprovementAnnotation] = []
            annotation_bounds = None
            if layout is not None:
                plot = build_detail_plot(layout, baselines, show_legend=False, hidden=hidden)
                entries = detail_legend(layout, baselines)
            elif climb is not None:
                plot = build_climb_plot(climb, baselines, show_legend=False, hidden=hidden)
                entries = climb_legend(climb, baselines)
                # The labels annotate the new-best dots; they hide with them.
                if self.show_annotations and "new best" not in hidden:
                    annotations = improvement_annotations(climb)
                    annotation_bounds = climb_plot_bounds(climb, baselines)
            else:
                plot = build_plot(curves, baselines, show_legend=False, hidden=hidden)
                entries = plot_legend(curves, baselines)
            self._legend_entries = entries
            legend.set_entries(entries, hidden)
            canvas = stage.query(PlotWidget)
            if canvas:
                # swap the plot in place: widget removal is asynchronous, so
                # remounting under the same id would collide with the old one
                widget = canvas.first()
                widget._plot = plot
                if isinstance(widget, ChartPlotWidget):
                    widget.set_annotations(annotations, annotation_bounds)
                widget.invalidate()
            else:
                stage.remove_children()
                stage.mount(ChartPlotWidget(
                    plot,
                    annotations=annotations,
                    annotation_bounds=annotation_bounds,
                    id="chart-canvas",
                ))
        elif not stage.query("#chart-empty"):
            self._legend_entries = []
            legend.set_entries([])
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
