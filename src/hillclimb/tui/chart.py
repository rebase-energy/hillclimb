"""`hillclimb chart` — the live hillclimb curve.

One staircase per problem: the best validation score so far across every
search of it, against the number of tested candidate solutions, with every
scored candidate as a dot on the same axes — bright where it set a new best,
dim where it missed. Three parallel searches on a problem are one climb, not
three; a search only gets its own line in a study, where the experiments are
the comparison (build_plot) — and there the chart stays inside the anchor's
run, so two runs of one experiment on the same problem never overlay each
other's "greedy r1". Same figure as the website's, in the same colours. Strictly a viewer like watch.py: everything is read through the configured store
(store.py — the hillclimb folder by default, or the SQLite index), and the
pure data functions at the top stay testable without Textual.

The x axis counts the candidates the climber tested, from 1. The harness's
own floor — the problem's baseline (and a `--seed-from` seed), scored before
any climber spent anything — sits at x = 0: it is the "naive" case the climb
is measured against, so the axis and the reference lines start there when
the floor was scored, and at 1 when it was not (an unscored placeholder, or
no baseline at all).

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

from hillclimb.harness.budget import FLOOR_OPERATORS
from hillclimb.harness.candidate import Candidate
from hillclimb.config import Config
from hillclimb.harness.direction import better
from hillclimb.harness.run import SearchMeta, load_search_meta, search_ref
from hillclimb.harness.journal import Journal
from hillclimb.harness.store import DataStore, FileDataStore, SearchRecord, key_for, open_store, resolve_search

# Searches drawn at once; older ones of the same problem fall off the chart
# rather than turning it into a haystack. Sized so a three-experiment study
# with three repeats (nine curves) fits with room to spare.
MAX_CURVES = 12


# plotui's line palette, mirrored so curves of one experiment can share
# a colour (repeats) while experiments differ — the chart's "colour by experiment"
EXPERIMENT_PALETTE = (
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
    xs: list[float] = field(default_factory=list)  # tested-candidate count; the floor at 0
    ys: list[float] = field(default_factory=list)  # best-so-far score (val, or its holdout)
    experiment: str | None = None  # the study experiment, when the search is one

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


def _landing_time(cand: Candidate) -> datetime | None:
    return _parse_ts(cand.finished_at) or _parse_ts(cand.created_at)


def score_time(cand: Candidate) -> datetime | None:
    """When the candidate's current score was measured: the best trial's
    finish when a later tune trial won (a tune gain must not be drawn back
    at the candidate's original landing), else the landing time."""
    best = cand.best_trial
    if best is not None and best.index > 0:
        when = _parse_ts(best.finished_at or "")
        if when is not None:
            return when
    return _landing_time(cand)


def is_floor(cand: Candidate) -> bool:
    """The harness's own floor (baseline/seed): scored before the climber
    spent anything, so it takes x = 0 rather than a tested-candidate slot."""
    return cand.operator in FLOOR_OPERATORS


def _slot_numbers(floors: list[bool]) -> list[float]:
    """x per landed candidate, in landing order: floors at 0, everything
    else counted from 1."""
    xs: list[float] = []
    tested = 0
    for floor in floors:
        if floor:
            xs.append(0.0)
        else:
            tested += 1
            xs.append(float(tested))
    return xs


def _counts_as_scored(cand: Candidate) -> bool:
    """The one predicate for "this candidate occupies an x slot on the climb".
    `climb_from_searches` and `cost_series` must agree on it exactly, or the
    cost overlay's x values drift off the staircase's."""
    return not cand.pruned and cand.val_score is not None and _landing_time(cand) is not None


def curve_from_candidates(
    candidates: list[Candidate],
    *,
    label: str,
    state: str,
    higher_is_better: bool = True,
    started_at: str | None = None,
    split: str = "val",
) -> Curve:
    """Best-so-far curve for one search. A point per scored, unpruned
    candidate in finish order; the x value is how many candidates have been
    tested — the floor (baseline/seed) sits at 0, the first tested at 1.

    `split="val"` (default) plots the score the search climbs on, so the line
    only ever moves in the metric's good direction. `split="holdout"` keeps
    incumbency judged on validation — the search never selects on holdout —
    but plots each incumbent's holdout score: the line may move the wrong way
    where a validation gain did not transfer, which is the point of the view.
    Incumbents without a holdout score keep the previous plotted value.

    `started_at` remains accepted for callers reading older search metadata;
    candidate count no longer depends on the search's wall-clock origin.
    """
    higher = higher_is_better
    curve = Curve(label=label, state=state)
    scored = []
    for cand in candidates:
        if cand.pruned or cand.val_score is None:
            continue
        when = score_time(cand)
        if when is None:
            continue
        scored.append((when, is_floor(cand), cand.val_score, cand.holdout_score))
    scored.sort(key=lambda item: item[0])
    if not scored:
        return curve
    best: float | None = None
    plotted: float | None = None
    slots = _slot_numbers([floor for _when, floor, _val, _holdout in scored])
    for experiment, (_when, _floor, val, holdout) in zip(slots, scored):
        if best is None or better(val, best, higher):
            best = val
            if split == "holdout":
                plotted = holdout if holdout is not None else plotted
            else:
                plotted = val
        if plotted is None:
            continue
        curve.xs.append(experiment)
        curve.ys.append(plotted)
    return curve


def _record_curve(store: DataStore, record: SearchRecord, label: str, split: str = "val") -> Curve:
    curve = curve_from_candidates(
        list(Journal(store.journal(record.key)).candidates.values()),
        label=label,
        state=record.state,
        higher_is_better=bool(record.meta.higher_is_better),
        started_at=record.meta.started_at,
        split=split,
    )
    curve.experiment = record.meta.experiment if record.meta.study else None
    return curve


def curve_label(record: SearchRecord, per_run: dict[str, int]) -> str:
    """Run name (what the user chose), plus the search id when the run holds
    several searches on the problem; a study's search is its experiment and
    repeat instead — the comparison the chart is then drawing."""
    meta = record.meta
    if meta.study and meta.experiment:
        return f"{meta.experiment} r{meta.repeat}" if meta.repeat else meta.experiment
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


def chart_run_scope(anchor: SearchMeta) -> str | None:
    """The run the chart confines itself to: a study's search's own run
    (its experiments are the comparison, and another run of the same study
    would repeat every `experiment rN` label), else None — a plain problem's climb
    is one staircase across every run that worked it."""
    return anchor.run_id if anchor.study else None


def climb_curves(
    store: DataStore | Path,
    problem_key: str,
    limit: int = MAX_CURVES,
    split: str = "val",
    run_id: str | None = None,
) -> list[Curve]:
    """Curves for every search of `problem_key` across runs (or inside
    `run_id`), oldest first so the palette assigns colors in start order
    (the newest gets the last one). Labelled by run name, which is what the
    user chose — plus the search id when a run holds several searches on
    the problem. Accepts a runs dir as shorthand for its FileDataStore."""
    if isinstance(store, Path):
        store = FileDataStore(store)
    records = store.searches(problem_key=problem_key, run_id=run_id)  # already oldest first
    per_run: dict[str, int] = {}
    for record in records:
        per_run[record.run_id] = per_run.get(record.run_id, 0) + 1
    return [
        _record_curve(store, record, curve_label(record, per_run), split)
        for record in records[-limit:]
    ]


@dataclass(frozen=True)
class ClimbEvent:
    x: float            # 1-based scored-candidate count across the climb
    y: float            # plotted score (val, or the candidate's holdout in the holdout view)
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
    def origin(self) -> float:
        """Where the axis starts: 0 when a scored floor is on the chart, else 1."""
        return curves_origin([e.x for e in self.events])

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


def curves_origin(xs: list[float]) -> float:
    """The left edge of the candidate axis for a set of plotted x values: 0
    when a floor (x = 0) is among them, else 1 — the chart starts at the
    first tested candidate unless there is a naive case to improve on."""
    return 0.0 if any(x == 0.0 for x in xs) else 1.0


def x_extent(origin: float, extent: float) -> tuple[float, float]:
    """The candidate axis as drawn: flush with `origin` on the left — the
    chart starts AT the floor or the first candidate, with no room to the
    left of it — and plotui's usual 5% of the span past the last slot on the
    right, so the newest mark never sits on the frame."""
    right = max(origin + 1.0, extent)
    return origin, right + (right - origin) * 0.05


def y_axis_title(metric: str, higher_is_better: bool, *, holdout: bool = False) -> str:
    """What the y axis measures, as the problem names it, with the direction
    that makes a step on it an improvement: `normalized-min-triangle-area
    (higher is better)`; the holdout view says so up front."""
    direction = "higher" if higher_is_better else "lower"
    return f"{'holdout ' if holdout else ''}{metric} ({direction} is better)"


def _pin_x_extent(
    plot: Plot,
    origin: float,
    extent: float,
    *,
    higher_is_better: bool = True,
    y_title: str | None = None,
) -> None:
    """Apply `x_extent` to a plot that supports an explicit x range; an older
    plotui keeps its padded autoscale. The hover readout names the x
    coordinate `candidate` and ranks its rows best-first in the metric's
    direction — hovering reads as the leaderboard of the climb against its
    references (an older plotui says `x` and keeps trace order) — and splits
    the rows by axis, so the cost overlay's tokens and minutes sit under a
    rule of their own instead of trailing the scores as if they were more of
    them. `y_title` (see `y_axis_title`) names the score on the axis
    itself."""
    if hasattr(plot, "set_x_range"):
        plot.set_x_range(x_extent(origin, extent))
    if hasattr(plot, "set_readout_x_label"):
        plot.set_readout_x_label("candidate")
    if hasattr(plot, "set_readout_order"):
        plot.set_readout_order("descending" if higher_is_better else "ascending")
    if hasattr(plot, "set_readout_split_axes"):
        plot.set_readout_split_axes(True)
    if y_title and hasattr(plot, "set_y_title"):
        plot.set_y_title(y_title)


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
    split: str = "val",
) -> Climb:
    """Fold every search's scored, unpruned candidates into one climb.
    `searches` is (label, candidates, started_at) per search; `started_at` is
    retained in that shape for callers but x is the tested-candidate count
    (the floor — every search's baseline/seed — at 0, the first tested at 1).
    `best` is judged against everything that landed before, whichever search
    it came from — always on the validation score, the signal the searches
    climb on. `split="holdout"` plots each event at its candidate's holdout
    score instead (candidates without one keep their x slot but are not
    drawn), so the staircase reads as the incumbent's held-out result and may
    move the wrong way where a validation gain did not transfer."""
    higher = higher_is_better
    landed: list[tuple[datetime, bool, float, float | None, str, str, str]] = []
    for label, candidates, _ in searches:
        for cand in candidates:
            if not _counts_as_scored(cand):
                continue
            landed.append((
                _landing_time(cand), is_floor(cand), cand.val_score, cand.holdout_score,
                label, cand.operator or "", cand.summary or "",
            ))
    landed.sort(key=lambda item: item[0])
    climb = Climb(searches=len(searches))
    if not landed:
        return climb
    best: float | None = None
    slots = _slot_numbers([item[1] for item in landed])
    for experiment, (_when, _floor, val, holdout, label, operator, summary) in zip(slots, landed):
        improved = best is None or better(val, best, higher)
        if improved:
            best = val
        y = holdout if split == "holdout" else val
        if y is None:
            continue
        climb.events.append(ClimbEvent(experiment, y, improved, label, operator, summary))
    climb.extent = max(slots)
    return climb


def climb_for_problem(
    store: DataStore | Path,
    problem_key: str,
    limit: int = MAX_CURVES,
    split: str = "val",
    run_id: str | None = None,
) -> Climb:
    """The climb across every search of `problem_key` (the newest `limit`),
    across runs or inside `run_id`."""
    if isinstance(store, Path):
        store = FileDataStore(store)
    records = store.searches(problem_key=problem_key, run_id=run_id)
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
        split=split,
    )


@dataclass(frozen=True)
class CostSeries:
    """What the climb cost, cumulatively, sampled at the climb's own x slots:
    agent tokens and verifier CPU-minutes (holdout runs included)."""

    xs: list[float] = field(default_factory=list)       # ClimbEvent.x slots
    tokens: list[float] = field(default_factory=list)   # cumulative agent tokens
    cpu_min: list[float] = field(default_factory=list)  # cumulative CPU-minutes
    wall_min: list[float] = field(default_factory=list) # wall-clock minutes since the climb began
    total_tokens: float = 0.0
    total_cpu_min: float = 0.0
    total_wall_min: float = 0.0
    evaluations: int = 0                                # verifier trials the climber spent


def _candidate_cost(cand: Candidate) -> tuple[float, float]:
    """(tokens, cpu seconds) one candidate burned: the agent call's own CPU
    (the agent process and every tool it ran; None on journals predating
    the field) plus every verifier trial's. Verifier CPU falls back to trial
    wall-clock where cpu_s predates the journal field — verifier envs are
    single-threaded, so wall ≈ cpu there. Holdout CPU has no such fallback:
    old journals never measured it, so it is simply absent."""
    tokens = float(cand.agent.total_tokens or 0)
    cpu = (cand.agent.cpu_s or 0.0) + sum(
        sum((r.cpu_s if r.cpu_s is not None else r.duration_s or 0.0) for r in t.replicates)
        + (
            (t.unit_tests.cpu_s if t.unit_tests.cpu_s is not None else t.unit_tests.duration_s)
            if t.unit_tests is not None else 0.0
        )
        + (t.holdout_cpu_s or 0.0)
        for t in cand.trials
    )
    return tokens, cpu


def _candidate_evaluations(cand: Candidate) -> int:
    """Verifier trials this candidate cost the climber — the budget's own
    rule (`budget.journal_spend`): every trial, except the floor's first."""
    trials = len(cand.trials)
    return max(0, trials - 1) if is_floor(cand) else trials


def cost_series(searches: list[tuple[str, list[Candidate], str | None]]) -> CostSeries:
    """Cumulative cost across the same searches the climb is built from.

    Every candidate is walked — buggy, pruned, agent-failed included: they
    burned tokens and CPU even though they never became climb events. Only a
    candidate that `climb_from_searches` counts (`_counts_as_scored`) advances
    x and emits the running totals, so `xs` lands 1:1 on the tested slots of
    `ClimbEvent.x`; a failure's cost surfaces at the next scored slot, and so
    does the floor's (the baseline/seed sits at x = 0 on the climb and is not
    the climber's spend). Cost trailing the last scored candidate lands as
    one final point at the same x (a vertical step to the true total).
    Timestampless candidates sort first, attaching their cost to slot 1.

    Wall clock is each slot's landing time since the climb began — the
    earliest `started_at` of the searches, else the first landing — so with
    several searches folded it reads as one clock, and the gap to the CPU
    line is the parallelism and the waiting. Evaluations follow the budget's
    rule (`_candidate_evaluations`) and are a total, not a line."""
    landed: list[tuple[tuple[bool, datetime], Candidate]] = []
    starts: list[datetime] = []
    for _label, candidates, started_at in searches:
        started = _parse_ts(started_at)
        if started is not None:
            starts.append(started)
        for cand in candidates:
            when = _landing_time(cand)
            # (has-timestamp, time) sorts the timestampless first without
            # ever comparing a naive datetime.min against aware timestamps
            landed.append(((when is not None, when or datetime.min), cand))
    landed.sort(key=lambda item: item[0])
    began = min(starts) if starts else next(
        (when for (stamped, when), _cand in landed if stamped), None
    )
    xs: list[float] = []
    token_points: list[float] = []
    cpu_points: list[float] = []
    wall_points: list[float] = []
    tokens = cpu = wall = 0.0
    evaluations = 0
    slot = 0
    for (stamped, when), cand in landed:
        cand_tokens, cand_cpu = _candidate_cost(cand)
        tokens += cand_tokens
        cpu += cand_cpu
        evaluations += _candidate_evaluations(cand)
        if stamped and began is not None:
            wall = max(wall, (when - began).total_seconds() / 60.0)
        if _counts_as_scored(cand) and not is_floor(cand):
            slot += 1
            xs.append(float(slot))
            token_points.append(tokens)
            cpu_points.append(cpu / 60.0)
            wall_points.append(wall)
    if xs and (tokens > token_points[-1] or cpu / 60.0 > cpu_points[-1]):
        xs.append(float(slot))
        token_points.append(tokens)
        cpu_points.append(cpu / 60.0)
        wall_points.append(wall)
    return CostSeries(
        xs=xs,
        tokens=token_points,
        cpu_min=cpu_points,
        wall_min=wall_points,
        total_tokens=tokens,
        total_cpu_min=cpu / 60.0,
        total_wall_min=wall,
        evaluations=evaluations,
    )


def cost_for_problem(
    store: DataStore | Path, problem_key: str, limit: int = MAX_CURVES, run_id: str | None = None
) -> CostSeries:
    """The cost of every search of `problem_key` — the same records window
    `climb_for_problem` folds (same `run_id` scope), so the two views share
    x slots."""
    if isinstance(store, Path):
        store = FileDataStore(store)
    records = store.searches(problem_key=problem_key, run_id=run_id)
    return cost_series(
        [
            ("", list(Journal(store.journal(record.key)).candidates.values()), record.meta.started_at)
            for record in records[-limit:]
        ]
    )


@dataclass(frozen=True)
class DetailMark:
    id: str
    x: float            # tested-candidate count; the floor at 0
    y: float            # plotted score (val, or holdout in the holdout view)
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
    split: str = "val",
) -> DetailLayout:
    """The curve plus the tree behind it. A mark per scored, unpruned
    candidate; an edge from its parent when the parent is scored too (a
    failed parent leaves its children rootless rather than inventing a y).
    `split="holdout"` places marks at holdout scores; candidates without one
    join the unscored count."""
    from hillclimb.tui.tree import build_tree

    curve = curve_from_candidates(
        candidates, label=label, state=state, higher_is_better=higher_is_better,
        started_at=started_at, split=split,
    )
    layout = DetailLayout(curve=curve)
    tree = build_tree(candidates, higher_is_better)
    scored = {c.candidate_id: c for c in candidates if c.val_score is not None and not c.pruned}
    if not scored:
        layout.unscored = len(candidates)
        return layout
    landed = []
    for cand in scored.values():
        when = score_time(cand)
        if when is not None:
            landed.append((when, is_floor(cand), cand.candidate_id))
    landed.sort(key=lambda item: item[0])
    slots = _slot_numbers([floor for _when, floor, _id in landed])
    experiment_by_id = {
        candidate_id: slot
        for slot, (_when, _floor, candidate_id) in zip(slots, landed)
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
        y = cand.holdout_score if split == "holdout" else node.score
        if y is None:
            layout.unscored += 1
            continue
        at[node.id] = (x, y)
        layout.marks.append(DetailMark(node.id, x, y, node.operator, node.fate, node.id in on_path))
    by_id = {n.id: n for n in tree.nodes}
    for edge in tree.edges:
        if edge.kind != "parent" or edge.src not in at or edge.dst not in at:
            continue
        (x0, y0), (x1, y1) = at[edge.src], at[edge.dst]
        layout.edges.append(DetailEdge(x0, y0, x1, y1, by_id[edge.dst].operator, edge.on_path))
    return layout


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


@dataclass(frozen=True)
class ChartRow:
    """One chart the folder can show: a problem worked in a run. `anchor` is
    the ref the chart opens on (the newest search of that pair). An
    study row's chart stays inside that run; a plain problem's chart
    still folds every search of the problem across runs (chart_run_scope)."""

    run_id: str
    run_name: str
    problem_key: str
    anchor: str
    searches: int
    running: int
    experiments: tuple[str, ...]
    state: str
    best: float | None
    activity_at: str


def chart_index(store: DataStore) -> list[ChartRow]:
    """Every (run, problem) pair with a search, newest activity first — what
    a bare `hillclimb chart` lists when the folder holds more than one."""
    groups: dict[tuple[str, str], list[SearchRecord]] = {}
    for record in store.searches():  # oldest first
        groups.setdefault((record.run_id, record.meta.problem_key), []).append(record)
    rows: list[ChartRow] = []
    for (run_id, problem_key), records in groups.items():
        newest = max(records, key=lambda r: (r.activity_at, r.ref))
        states = [r.state for r in records]
        experiments: list[str] = []
        best: float | None = None
        for record in records:
            if record.meta.experiment and record.meta.experiment not in experiments:
                experiments.append(record.meta.experiment)
            status = store.read_status(record.key)
            score = status.best.val_score if status and status.best else None
            if score is not None and (best is None or better(score, best, newest.meta.higher_is_better)):
                best = score
        rows.append(ChartRow(
            run_id=run_id,
            run_name=newest.run_name,
            problem_key=problem_key,
            anchor=newest.ref,
            searches=len(records),
            running=states.count("running"),
            experiments=tuple(experiments),
            state=_state_summary(states),
            best=best,
            activity_at=newest.activity_at,
        ))
    return sorted(rows, key=lambda row: (row.activity_at, row.anchor), reverse=True)


# provider reference lines resolved once per target — a live chart refreshes
# every few seconds and must not re-import the provider each tick
_provider_baselines: dict[str, dict[str, float]] = {}


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
        if meta.chart_baselines:
            return dict(meta.chart_baselines)
        # Searches that predate the provider snapshot: ask the provider for
        # its reference lines lazily (cheap — no data materialization).
        if meta.problem not in _provider_baselines:
            from hillclimb.problem import provider_chart_baselines

            try:
                _provider_baselines[meta.problem] = provider_chart_baselines(meta.problem)
            except Exception:  # noqa: BLE001 — a chart must render without the extra installed
                _provider_baselines[meta.problem] = {}
        return dict(_provider_baselines[meta.problem])
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
from textual.widgets import DataTable, Footer, Label  # noqa: E402

from hillclimb.tui.header import HillclimbHeader, TimezoneMixin  # noqa: E402
from hillclimb.tui.keys import KEYS_BINDING, QUIT_BINDINGS, back_binding  # noqa: E402

from hillclimb.tui.theme import CYAN, HILLCLIMB_CSS, PLOT_BG, apply_theme, themed_plot  # noqa: E402
from hillclimb.tui.watch import (  # noqa: E402
    STATE_STYLE, LiveScreen, _fmt, _fmt_tokens, _restore_table, _snapshot_table, _state_summary,
)


class ChartPlotWidget(PlotWidget):
    """Plot widget with terminal-native improvement labels and safe compositing.

    The chart is a static figure, like the website's: zoom and pan are
    disabled (a stray scroll used to shrink the staircase to a speck), so the
    camera always frames the whole climb.

    Kitty uploads ride on a zero-width Rich control segment. Textual's
    monochrome filter expects composited segments to carry a Style, so give
    only otherwise-unstyled segments a neutral one. This is invisible in a
    real colour terminal and keeps NO_COLOR/headless rendering valid.

    The cells under the image carry no background of their own: the image
    is placed below cell backgrounds (in both modes) so the annotation tags
    can mask it, and a painted cell would hide the plot under it. iTerm2
    extends each row's LAST cell background across the right window margin,
    so a row of default-background cells paints that margin in the
    profile's colour — a lighter stripe beside the opaque PLOT_BG canvas.
    In direct mode (iTerm2) the last cell of each row therefore gets
    PLOT_BG as an explicit background: it hides one column of blank canvas
    at the plot's right edge and makes the margin match.
    """

    def __init__(
        self,
        plot: Plot,
        *,
        annotations: list[ImprovementAnnotation] | None = None,
        annotation_bounds: tuple[float, float, float, float] | None = None,
        annotation_right_margin: int = 2,
        **kwargs,
    ):
        self._annotations = list(annotations or [])
        self._annotation_bounds = annotation_bounds
        self._annotation_right_margin = annotation_right_margin
        super().__init__(plot, **kwargs)

    def set_annotations(
        self,
        annotations: list[ImprovementAnnotation],
        bounds: tuple[float, float, float, float] | None,
        right_margin: int = 2,
    ) -> None:
        self._annotations = list(annotations)
        self._annotation_bounds = bounds
        self._annotation_right_margin = right_margin
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
                right_margin=self._annotation_right_margin,
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
        segments = [
            segment if segment.style is not None
            else Segment(segment.text, Style(), segment.control)
            for segment in strip
        ]
        if self._mode == "direct":
            from rich.color import Color

            edge = Style(bgcolor=Color.from_rgb(*PLOT_BG))
            for index in range(len(segments) - 1, -1, -1):
                segment = segments[index]
                if segment.control or not segment.text:
                    continue
                if segment.text.endswith(" ") and segment.style == Style():
                    # split the row's last blank cell off with the edge style
                    segments[index : index + 1] = [
                        Segment(segment.text[:-1], segment.style),
                        Segment(" ", edge),
                    ]
                break
        return Strip(segments, strip.cell_length)


def curve_colors(curves: list[Curve]) -> list[tuple[int, int, int] | None]:
    """A colour per curve: curves of the same experiment share one, so an
    experiment's repeats read as one family against the others; curves
    without an experiment (None) take plotui's next palette slot as before."""
    experiments = list(dict.fromkeys(c.experiment for c in curves if c.experiment))
    return [EXPERIMENT_PALETTE[experiments.index(c.experiment) % len(EXPERIMENT_PALETTE)] if c.experiment else None for c in curves]


def _hide_plot_legend(plot: Plot, show_legend: bool) -> None:
    """The ChartScreen draws its legend in a Textual band under the plot, so
    the in-canvas box is switched off — but the traces keep their names, so
    the hover readout says `greedy r1  0.0115` rather than `series 3`. Older
    plotui builds without the switch fall back to unnamed traces."""
    if show_legend:
        return
    if hasattr(type(plot), "legend_visible"):
        plot.legend_visible = False


def _add_chart_baselines(
    plot: Plot,
    baselines: Mapping[str, float],
    extent: float,
    *,
    origin: float = 1.0,
    show_legend: bool = True,
    hidden: frozenset[str] | set[str] = frozenset(),
) -> None:
    """Add arbitrary named horizontal score references behind the data,
    from `origin` (the axis's left edge: 0 with a scored floor, else 1) to
    the climb's extent — sampled at every candidate slot, not just the two
    ends, so the hover readout lists each reference beside whichever
    candidate the crosshair is on.

    `hidden` entries are skipped but keep their palette slot, so toggling one
    off never recolours the others out from under the legend."""
    right = max(origin + 1.0, extent)
    xs = [origin + step for step in range(int(right - origin) + 1)]
    # dashed, so a reference reads as a reference and not as a series
    # (hillclimb.sh draws them the same way); solid on an older plotui
    dash = {"dash": BENCHMARK_DASH} if _plot_supports("dash") else {}
    for index, (label, value) in enumerate(baselines.items()):
        if label in hidden:
            continue
        plot.add_line(
            xs,
            [value] * len(xs),
            color=CHART_BASELINE_PALETTE[index % len(CHART_BASELINE_PALETTE)],
            width=1.0,
            name=label,
            **dash,
        )


def build_plot(
    curves: list[Curve],
    baselines: Mapping[str, float] | None = None,
    *,
    show_legend: bool = True,
    hidden: frozenset[str] | set[str] = frozenset(),
    higher_is_better: bool = True,
    y_title: str | None = None,
) -> Plot:
    """One step line per curve — the study view, where each experiment is a
    series of its own, and the base of the detail overlay. `hidden` names
    legend entries toggled off: their traces are left out of the plot."""
    plot = themed_plot()
    _hide_plot_legend(plot, show_legend)
    extent = max((max(c.xs, default=0.0) for c in curves), default=0.0)
    origin = curves_origin([x for c in curves for x in c.xs[:1]])
    _pin_x_extent(plot, origin, extent, higher_is_better=higher_is_better, y_title=y_title)
    _add_chart_baselines(
        plot, baselines or {}, extent, origin=origin, show_legend=show_legend, hidden=hidden,
    )
    trace_index = len(baselines or {})
    for curve, color in zip(curves, curve_colors(curves)):
        if not curve.xs:
            continue
        # Pin the colour plot_legend assigns this slot: skipping a hidden
        # trace must not let plotui's next-palette-slot drift under the rest.
        rgb = color or EXPERIMENT_PALETTE[trace_index % len(EXPERIMENT_PALETTE)]
        trace_index += 1
        if curve.label in hidden:
            continue
        if len(curve.xs) == 1:
            # Keep the domain in whole candidate counts; a one-point line is
            # invisible, so render that first evaluation as a dot.
            plot.add_scatter(
                curve.xs, curve.ys, color=rgb, size=3.0,
                name=curve.label,
            )
        else:
            xs, ys = step_points(curve.xs, curve.ys)
            plot.add_line(xs, ys, color=rgb, name=curve.label)
    return plot


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


# A miss is the same hue as the staircase, receded towards the background —
# the site's `.miss { fill-opacity: .45 }` — so the climb stays the figure
# and the attempts stay the ground.
MISS_RGB = _mix(PLOT_BG, CYAN, 0.45)

# The cost overlay: cumulative agent tokens (y2, chartreuse) and cumulative
# verifier CPU-minutes (y3, coral) — hues that CHART_BASELINE_PALETTE does
# not use, so a cost line is never mistaken for a reference line (the old
# amber and violet were the palette's own yellow and purple), and distinct
# from the cyan staircase and the neutral baseline gray. plotui tints each
# right axis's tick labels to its series colour, so these also label the
# columns.
COST_TOKENS_RGB = (174, 204, 64)
COST_CPU_RGB = (232, 118, 104)
COST_WALL_RGB = (247, 186, 176)   # the cpu coral, paled: the two minutes lines are a pair
COST_GROUP = "Cost"  # the legend heading the overlay's series sit under
COST_TOKENS_LABEL = "agent tokens"
COST_CPU_LABEL = "cpu time"
COST_WALL_LABEL = "wall clock"


def _plot_supports_axis() -> bool:
    """Whether the installed plotui has right-hand axes (0.3.0+). Checked the
    way themed_plot feature-guards set_chrome, so the chart still renders —
    minus the overlay — against an older wheel."""
    return _plot_supports("axis")


def _plot_supports(parameter: str) -> bool:
    """Whether the installed plotui's `add_line` takes `parameter`."""
    import inspect

    try:
        return parameter in inspect.signature(Plot.add_line).parameters
    except (TypeError, ValueError):  # builtins without introspectable signatures
        return False


def add_cost_overlay(
    plot: Plot,
    cost: CostSeries | None,
    *,
    show_legend: bool = True,
    hidden: frozenset[str] | set[str] = frozenset(),
) -> None:
    """Draw the cumulative cost lines on their own right-hand axes: tokens on
    y2 — in millions, with `M` on its tick labels and readout values, so
    `2.2e7` reads as `22M` — and the two clocks on y3, CPU time and wall
    clock, which share a unit (`min`) and so an axis: the gap between them
    is the parallelism and the waiting. Straight lines, not steps — cost
    accrues continuously between scoring instants and only ever rises. A
    hidden or all-zero series is not added, and plotui then reserves no
    column for an axis nothing is on. No-ops on plotui builds without
    `axis=` support."""
    if cost is None or not cost.xs or not _plot_supports_axis():
        return
    if COST_TOKENS_LABEL not in hidden and cost.total_tokens > 0:
        plot.add_line(
            cost.xs, [tokens / 1e6 for tokens in cost.tokens], color=COST_TOKENS_RGB, width=1.0,
            name=COST_TOKENS_LABEL, axis="y2",
        )
        if hasattr(plot, "set_axis_unit"):
            plot.set_axis_unit("y2", "M")
    minutes = False
    if COST_CPU_LABEL not in hidden and cost.total_cpu_min > 0:
        plot.add_line(
            cost.xs, cost.cpu_min, color=COST_CPU_RGB, width=1.0,
            name=COST_CPU_LABEL, axis="y3",
        )
        minutes = True
    if COST_WALL_LABEL not in hidden and cost.total_wall_min > 0:
        plot.add_line(
            cost.xs, cost.wall_min, color=COST_WALL_RGB, width=1.0,
            name=COST_WALL_LABEL, axis="y3",
        )
        minutes = True
    if minutes and hasattr(plot, "set_axis_unit"):
        plot.set_axis_unit("y3", " min")


def _cost_legend(cost: CostSeries | None) -> list[LegendEntry]:
    """The overlay's series as the `Cost` group: `legend_text` opens them on
    a row of their own under that heading, after the benchmarks, so what the
    climb spent never reads as one more score series."""
    if cost is None or not cost.xs or not _plot_supports_axis():
        return []
    entries: list[LegendEntry] = []
    if cost.total_tokens > 0:
        entries.append((COST_TOKENS_LABEL, COST_TOKENS_RGB, "─", COST_GROUP))
    if cost.total_cpu_min > 0:
        entries.append((COST_CPU_LABEL, COST_CPU_RGB, "─", COST_GROUP))
    if cost.total_wall_min > 0:
        entries.append((COST_WALL_LABEL, COST_WALL_RGB, "─", COST_GROUP))
    return entries


def build_climb_plot(
    climb: Climb,
    baselines: Mapping[str, float] | None = None,
    *,
    show_legend: bool = True,
    hidden: frozenset[str] | set[str] = frozenset(),
    cost: CostSeries | None = None,
    higher_is_better: bool = True,
    y_title: str | None = None,
) -> Plot:
    """The website's figure: the staircase in cyan, a bright dot where a
    candidate set a new best, a dim one where it scored but did not.
    `hidden` names legend entries toggled off — their traces are left out.
    `cost` overlays the cumulative token/CPU lines on right-hand axes."""
    plot = themed_plot()
    _hide_plot_legend(plot, show_legend)
    _pin_x_extent(
        plot, climb.origin, climb.extent, higher_is_better=higher_is_better, y_title=y_title,
    )
    _add_chart_baselines(
        plot, baselines or {}, climb.extent, origin=climb.origin,
        show_legend=show_legend, hidden=hidden,
    )
    misses = [e for e in climb.events if not e.best]
    if misses and "attempt" not in hidden:
        plot.add_scatter(
            [e.x for e in misses], [e.y for e in misses], color=MISS_RGB, size=2.4,
            name="attempt",
        )
    xs, ys = climb.staircase()
    if len(xs) > 1 and "best so far" not in hidden:
        plot.add_line(
            xs, ys, color=CYAN, width=2.0,
            name="best so far",
        )
    hits = [e for e in climb.events if e.best]
    if hits and "new best" not in hidden:
        plot.add_scatter(
            [e.x for e in hits], [e.y for e in hits], color=CYAN, size=3.0,
            name="new best",
        )
    add_cost_overlay(plot, cost, show_legend=show_legend, hidden=hidden)
    return plot


def build_detail_plot(
    layout: DetailLayout,
    baselines: Mapping[str, float] | None = None,
    *,
    show_legend: bool = True,
    hidden: frozenset[str] | set[str] = frozenset(),
    cost: CostSeries | None = None,
    higher_is_better: bool = True,
    y_title: str | None = None,
) -> Plot:
    """The curve as in build_plot, then the tree: one thin line per edge
    (dim, the child's operator colour; bold on the accepted lineage) and a
    scatter per operator so the legend names them; accepted candidates get
    a larger mark on top. `hidden` names legend entries toggled off — an
    operator takes its marks and its edges with it."""
    from hillclimb.tui.treeview import OPERATOR_RGB, dim_rgb

    plot = build_plot(
        [layout.curve], baselines, show_legend=show_legend, hidden=hidden,
        higher_is_better=higher_is_better, y_title=y_title,
    )
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
            name=operator,
        )
    accepted = [m for m in layout.marks if m.on_path]
    if accepted and "accepted" not in hidden:
        plot.add_scatter(
            [m.x for m in accepted], [m.y for m in accepted],
            color=(255, 255, 255), size=4.0,
            name="accepted",
        )
    add_cost_overlay(plot, cost, show_legend=show_legend, hidden=hidden)
    return plot


# (label, colour, glyph) — the glyph mirrors the trace's mark, the way the
# knowledge graph's legend echoes each node type's marker: "─" for a line
# trace, "●" for a scatter. legend_text tolerates the old two-field shape.
# (label, colour, glyph) — or with a fourth element, the heading of the
# group the entry belongs to: `legend_text` starts that group on a row of
# its own under the heading, so the reference lines read as one block
# apart from the search's own series.
LegendEntry = tuple[str, tuple[int, int, int], str] | tuple[str, tuple[int, int, int], str, str]

BENCHMARKS_GROUP = "Benchmarks"
# benchmark lines: 4px drawn, 3px skipped; the legend echoes it with "╌"
BENCHMARK_DASH = (4.0, 3.0)


def _baseline_legend(baselines: Mapping[str, float]) -> list[LegendEntry]:
    return [
        (label, CHART_BASELINE_PALETTE[index % len(CHART_BASELINE_PALETTE)], "╌", BENCHMARKS_GROUP)
        for index, label in enumerate(baselines)
    ]


def _with_benchmarks(
    own: list[LegendEntry], baselines: Mapping[str, float], cost: CostSeries | None = None
) -> list[LegendEntry]:
    """The search's own series first, the published references after them
    as their own group, and the cost overlay's series last as theirs — what
    was climbed, what it is measured against, then what it cost. Each group
    is a row of the legend band."""
    return [*own, *_baseline_legend(baselines), *_cost_legend(cost)]


def _curve_legend(curves: list[Curve], baselines: Mapping[str, float]) -> list[LegendEntry]:
    entries: list[LegendEntry] = []
    # the reference lines are added to the plot first, so plotui's palette
    # slot for an uncoloured trace counts on from them
    trace_index = len(baselines)
    for curve, color in zip(curves, curve_colors(curves)):
        if not curve.xs:
            continue
        # EXPERIMENT_PALETTE mirrors plotui's default trace palette. An uncoloured
        # trace takes the slot determined by everything already added.
        entries.append((curve.label, color or EXPERIMENT_PALETTE[trace_index % len(EXPERIMENT_PALETTE)], "─"))
        trace_index += 1
    return entries


def plot_legend(curves: list[Curve], baselines: Mapping[str, float]) -> list[LegendEntry]:
    """Legend for a study/detail base plot: the curves, then the
    references."""
    return _with_benchmarks(_curve_legend(curves, baselines), baselines)


def climb_legend(
    climb: Climb, baselines: Mapping[str, float], cost: CostSeries | None = None
) -> list[LegendEntry]:
    entries: list[LegendEntry] = []
    if any(not event.best for event in climb.events):
        entries.append(("attempt", MISS_RGB, "●"))
    if len(climb.staircase()[0]) > 1:
        entries.append(("best so far", CYAN, "─"))
    if any(event.best for event in climb.events):
        entries.append(("new best", CYAN, "●"))
    return _with_benchmarks(entries, baselines, cost)


def detail_legend(
    layout: DetailLayout, baselines: Mapping[str, float], cost: CostSeries | None = None
) -> list[LegendEntry]:
    from hillclimb.tui.treeview import OPERATOR_RGB

    entries = _curve_legend([layout.curve], baselines)
    for operator in dict.fromkeys(mark.operator for mark in layout.marks):
        entries.append((operator, OPERATOR_RGB.get(operator, (160, 160, 160)), "●"))
    if any(mark.on_path for mark in layout.marks):
        entries.append(("accepted", (255, 255, 255), "●"))
    return _with_benchmarks(entries, baselines, cost)


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
    """The data bounds as plotui draws them: x pinned by `x_extent` (flush
    with the origin, 5% past the last slot), y the 5%-padded autoscale."""
    ys = [event.y for event in climb.events]
    if baselines:
        ys.extend(float(value) for value in baselines.values())
    if not climb.events and not ys:
        return (-1.0, 1.0, -1.0, 1.0)
    xlo, xhi = x_extent(climb.origin, climb.extent)
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    return xlo, xhi, lo - pad, hi + pad


def annotation_spans(
    annotations: list[ImprovementAnnotation],
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    camera_state: tuple[float, float, float, float, float] = (0.0, 0.0, 1.0, 0.0, 0.0),
    cell_px: tuple[int, int] = (12, 24),
    right_margin: int = 2,
) -> list[OverlaySpan]:
    """Place short new-best labels near their dots, using collision lanes.
    `right_margin` approximates the cells the plot's right gutter occupies —
    the cost overlay's tick-label columns shift the true plot rect left, and
    without this the labels drift off their dots. Approximate until plotui
    exposes its computed plot rect."""
    if not annotations or width < 20 or height < 8:
        return []
    from rich.style import Style

    xlo, xhi, ylo, yhi = bounds
    if xhi <= xlo or yhi <= ylo:
        return []
    left = max(7, min(14, width // 16))
    right = max(left + 4, width - right_margin)
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
    (`screen.toggle_series`) and adds the 1-9 hotkey prefix. An entry with a
    group heading (its fourth element) opens that group on a row of its own,
    the heading first in dim ink; the hotkey numbering runs on across it."""
    from rich.cells import cell_len
    from rich.style import Style
    from rich.text import Text

    text = Text(no_wrap=width is not None, overflow="crop" if width is not None else None)
    line_width = 0
    group: str | None = None
    after_heading = False
    for index, (label, (red, green, blue), *rest) in enumerate(entries):
        glyph = rest[0] if rest else "●"
        entry_group = rest[1] if len(rest) > 1 else None
        if entry_group != group:
            group = entry_group
            if group is not None:
                if line_width:
                    text.append("\n")
                heading = f"{group}: "
                text.append(heading, style=Style.parse("dim"))
                line_width = cell_len(heading)
                after_heading = True
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
        separator_width = 0 if after_heading or not line_width else 3
        after_heading = False
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
    # The legend entries are click targets; a double click on one must
    # toggle its series, not paint Textual's text selection across the
    # legend band. The terminal's own modifier still selects text.
    ALLOW_SELECT = False

    BINDINGS = [
        Binding("d", "toggle_detail", "detail", tooltip="overlay the exploration tree"),
        # shown only for a problem that scores a holdout — see check_action
        Binding(
            "h", "toggle_holdout", "holdout",
            tooltip="toggle holdout vs validation scores",
        ),
        Binding(
            "t", "toggle_annotations", "text",
            tooltip="show or hide improvement text", priority=True,
        ),
        Binding(
            "c", "toggle_cost", "cost",
            tooltip="overlay cumulative tokens, cpu-minutes and wall-clock minutes",
        ),
        # shown only when the folder holds a second problem — see check_action
        Binding(
            "p", "next_problem", "switch problem",
            tooltip="switch to the next problem in this folder",
        ),
        Binding("r", "refresh", "refresh", show=False),
        # only live when a list (the chart picker, the watch tables) pushed
        # this screen — see check_action
        back_binding("back"),
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

    def __init__(
        self,
        config: Config,
        search: str | None = None,
        detail: bool = False,
        holdout: bool | None = None,
        cost: bool = False,
    ):
        super().__init__()
        self.config = config
        self.search = search
        self.detail = detail  # one search, with its exploration tree on the curve
        # plot holdout scores (incumbency stays judged on val). None = decide
        # from the anchor: a problem that scores a holdout defaults to the
        # holdout view — the reference baselines are holdout numbers, and
        # pears must compare with pears
        self.holdout = holdout
        self.show_cost = cost  # `c` overlays cumulative tokens/cpu on y2/y3
        self.show_annotations = False  # `t` reveals the improvement labels
        self.hidden_series: set[str] = set()  # legend entries toggled off by click
        self._legend_entries: list[LegendEntry] = []  # what 1-9 index into
        self._anchor: SearchMeta | None = None  # resolved once; `r` re-resolves
        self._problem_keys: list[str] = []  # distinct problems in the folder, first-seen order
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

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action == "back":
            # the app's default screen sits at the bottom of the stack; a
            # standalone chart is the only screen above it and has nowhere
            # to go back to — hide the key rather than show a dead one
            return True if len(self.app.screen_stack) > 2 else None
        # A key that could only say "no" is left out of the footer: nothing
        # to switch to, or a problem that never scored a holdout. Both are
        # re-decided on every refresh (`refresh_bindings` in refresh_data).
        if action == "next_problem":
            return len(self._problem_keys) > 1
        if action == "toggle_holdout":
            return self._anchor is not None and bool(self._anchor.holdout_enabled)
        return True

    def _list_problems(self) -> list[str]:
        """The distinct problem keys worked in this folder, in first-seen
        order — what `p` cycles through."""
        assert self._store is not None
        keys: list[str] = []
        for record in self._store.searches():  # oldest first
            if record.meta.problem_key not in keys:
                keys.append(record.meta.problem_key)
        return keys

    def action_back(self) -> None:
        if len(self.app.screen_stack) > 2:
            self.app.pop_screen()

    def action_refresh(self) -> None:
        self._anchor = None
        self._key = None
        self.refresh_data()

    def action_toggle_detail(self) -> None:
        self.detail = not self.detail
        self._key = None
        self.refresh_data()

    def action_toggle_holdout(self) -> None:
        self.holdout = not bool(self.holdout)
        self._key = None
        self.refresh_data()

    def action_toggle_annotations(self) -> None:
        self.show_annotations = not self.show_annotations
        self._key = None
        self.refresh_data()

    def action_toggle_cost(self) -> None:
        if not _plot_supports_axis():
            self.notify("the cost overlay needs plotui 0.3+ (secondary axes)")
            return
        self.show_cost = not self.show_cost
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

    def action_next_problem(self) -> None:
        """Re-anchor the chart on the next problem worked in this folder,
        cycling through the distinct problem keys in first-seen order."""
        if self._store is None:
            self._store = open_store(self.config)
        if self._anchor is None:
            self._anchor = chart_problem(self.config, self.search)
        anchor = self._anchor
        keys = self._problem_keys = self._list_problems()
        if anchor is None or len(keys) < 2:
            self.notify("only one problem in this folder")
            return
        current = anchor.problem_key
        target = keys[(keys.index(current) + 1) % len(keys)] if current in keys else keys[0]
        of_target = [r for r in self._store.searches() if r.meta.problem_key == target]
        newest = max(of_target, key=lambda r: (r.activity_at, r.ref))
        self._anchor = newest.meta
        # `r` re-resolves the anchor from self.search — pin it to the chosen
        # problem so a refresh does not snap back to the folder's latest
        self.search = newest.ref
        self.hidden_series.clear()  # legend toggles are per-problem
        self.holdout = None  # re-decide the fair split for the new problem
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
        # the footer offers `p` and `h` only where they can do something
        self._problem_keys = self._list_problems()
        self.refresh_bindings()
        if self.holdout is None:
            # fair by default: a holdout-scored problem opens on the holdout
            # view, the split its reference baselines live on (`h` toggles)
            self.holdout = bool(anchor.holdout_enabled)
        # The problem alone heads the line: the metric sentence the website
        # opens its chart with ("best <metric> so far (higher is better)")
        # pushed the live numbers off a normal-width terminal. The y axis
        # carries the metric and its direction instead (`y_axis_title`), and
        # says "holdout" when that is the score plotted.
        split = "holdout" if self.holdout else "val"
        y_title = y_axis_title(anchor.metric, bool(anchor.higher_is_better), holdout=self.holdout)
        parts = [f"[bold]{anchor.problem_key}[/]"]
        baselines = chart_baselines(self.config, anchor)
        layout: DetailLayout | None = None
        cost: CostSeries | None = None
        if self.detail:
            record = self._store.search((anchor.run_id, anchor.search_id))
            if record is None:
                layout = DetailLayout(curve=Curve(label=f"{anchor.run_id}/{anchor.search_id}", state="unknown"))
            else:
                label = f"{record.run_name}/{record.search_id}"
                # one journal read feeds the tree layout and the cost fold
                cands = list(Journal(self._store.journal(record.key)).candidates.values())
                layout = detail_layout(
                    cands, label=label, state=record.state,
                    higher_is_better=bool(record.meta.higher_is_better),
                    started_at=record.meta.started_at, split=split,
                )
                cost = cost_series([(label, cands, record.meta.started_at)])
            curves = [layout.curve]
            parts.append(
                f"detail: {len(layout.marks)} scored · {len(layout.edges)} edges"
                + (f" · {layout.unscored} unscored" if layout.unscored else "")
            )
        else:
            scope = chart_run_scope(anchor)
            curves = climb_curves(self._store, anchor.problem_key, split=split, run_id=scope)
            cost = cost_for_problem(self._store, anchor.problem_key, run_id=scope)
        climb: Climb | None = None
        if layout is not None or any(c.experiment for c in curves):
            # one line per search: the detail overlay, or a study
            # where the experiments are the comparison
            for curve in curves:
                style = STATE_STYLE.get(curve.state, "")
                best = f"{curve.best:.5g}" if curve.best is not None else "-"
                parts.append(f"{curve.label} [{style}]{curve.state}[/] best={best}")
        else:
            climb = climb_for_problem(
                self._store, anchor.problem_key, split=split, run_id=chart_run_scope(anchor)
            )
            best = f"{climb.best:.5g}" if climb.best is not None else "-"
            n = climb.searches
            running = sum(1 for c in curves if c.state == "running")
            parts.append(
                f"best={best} · {n} search{'' if n == 1 else 'es'}"
                + (f" ([{STATE_STYLE.get('running', '')}]{running} running[/])" if running else "")
                + f" · {climb.hits} of {len(climb.events)} candidates improved"
            )
        if cost is not None and (cost.total_tokens > 0 or cost.total_cpu_min > 0):
            # what the climb has cost so far, always on view
            parts.append(
                f"{_fmt_tokens(int(cost.total_tokens))} tok · {cost.total_cpu_min:.3g} cpu-min"
                + (f" · {cost.total_wall_min:.3g} wall-min" if cost.total_wall_min > 0 else "")
                + f" · {cost.evaluations} eval{'' if cost.evaluations == 1 else 's'}"
            )
        self.query_one("#chartline", Label).update("  ·  ".join(parts))
        key: tuple = (
            self.detail,
            self.holdout,
            self.show_annotations,
            self.show_cost,
            tuple(sorted(self.hidden_series)),
            tuple((c.label, tuple(c.xs), tuple(c.ys)) for c in curves),
        )
        key += (tuple(baselines.items()),)
        if cost is not None:
            key += (tuple(cost.xs), tuple(cost.tokens), tuple(cost.cpu_min))
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
            overlay = cost if self.show_cost else None
            annotations: list[ImprovementAnnotation] = []
            annotation_bounds = None
            if layout is not None:
                plot = build_detail_plot(
                    layout, baselines, show_legend=False, hidden=hidden, cost=overlay,
                    higher_is_better=bool(anchor.higher_is_better), y_title=y_title,
                )
                entries = detail_legend(layout, baselines, overlay)
            elif climb is not None:
                plot = build_climb_plot(
                    climb, baselines, show_legend=False, hidden=hidden, cost=overlay,
                    higher_is_better=bool(anchor.higher_is_better), y_title=y_title,
                )
                entries = climb_legend(climb, baselines, overlay)
                # The labels annotate the new-best dots; they hide with them.
                if self.show_annotations and "new best" not in hidden:
                    annotations = improvement_annotations(climb)
                    annotation_bounds = climb_plot_bounds(climb, baselines)
            else:
                plot = build_plot(
                    curves, baselines, show_legend=False, hidden=hidden,
                    higher_is_better=bool(anchor.higher_is_better), y_title=y_title,
                )
                entries = plot_legend(curves, baselines)
            # each visible cost series adds a tick-label column (~7 cells) on
            # the right, shifting the true plot rect the annotations map into
            visible_cost = sum(1 for entry in _cost_legend(overlay) if entry[0] not in hidden)
            annotation_margin = 2 + 7 * visible_cost
            self._legend_entries = entries
            legend.set_entries(entries, hidden)
            canvas = stage.query(PlotWidget)
            if canvas:
                # swap the plot in place: widget removal is asynchronous, so
                # remounting under the same id would collide with the old one
                widget = canvas.first()
                widget._plot = plot
                if isinstance(widget, ChartPlotWidget):
                    widget.set_annotations(annotations, annotation_bounds, annotation_margin)
                widget.invalidate()
            else:
                stage.remove_children()
                stage.mount(ChartPlotWidget(
                    plot,
                    annotations=annotations,
                    annotation_bounds=annotation_bounds,
                    annotation_right_margin=annotation_margin,
                    id="chart-canvas",
                ))
        elif not stage.query("#chart-empty"):
            self._legend_entries = []
            legend.set_entries([])
            stage.remove_children()
            stage.mount(Label("waiting for the first scored candidate…", id="chart-empty"))


class ChartPickerScreen(LiveScreen):
    """The charts this folder can show, one row per problem worked in a
    run: enter opens the chart anchored on that row, esc in the chart comes
    back here. Refreshes like the watch tables so a running study's
    rows keep moving."""

    BINDINGS = [
        Binding("enter", "open_chart", "chart", priority=True),
        KEYS_BINDING,
        *QUIT_BINDINGS,
    ]

    def __init__(
        self,
        config: Config,
        detail: bool = False,
        holdout: bool | None = None,
        cost: bool = False,
    ):
        super().__init__()
        self.config = config
        self.detail = detail
        self.holdout = holdout
        self.cost = cost
        self.store = open_store(config)

    def compose(self) -> ComposeResult:
        yield HillclimbHeader()
        yield DataTable(id="charts", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#charts", DataTable)
        table.add_columns("run", "problem", "experiments", "searches", "state", "best val", "last activity")
        self.start_live()

    def refresh_data(self) -> None:
        from rich.text import Text

        table = self.query_one("#charts", DataTable)
        snapshot = _snapshot_table(table)
        table.clear()
        for row in chart_index(self.store):
            searches = str(row.searches) + (f" ({row.running} running)" if row.running else "")
            table.add_row(
                row.run_name,
                row.problem_key,
                ", ".join(row.experiments) or "-",
                searches,
                Text(row.state, style=STATE_STYLE.get(row.state, "")),
                _fmt(row.best),
                row.activity_at[:19].replace("T", " ") if row.activity_at else "-",
                key=row.anchor,  # unique per row: the search ref enter opens
            )
        _restore_table(table, snapshot)

    def action_open_chart(self) -> None:
        table = self.query_one("#charts", DataTable)
        if not table.row_count:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        self.app.push_screen(
            ChartScreen(
                self.config, row_key.value,
                detail=self.detail, holdout=self.holdout, cost=self.cost,
            )
        )


class ChartApp(TimezoneMixin, App):
    """Standalone shell for `hillclimb chart`: a bare invocation on a folder
    with several charts to show opens the picker, otherwise the chart."""

    BINDINGS = [
        Binding("t", "choose_timezone", "time zone", show=False),
        Binding("ctrl+c", "quit", "quit", show=False, priority=True),
    ]
    CSS = HILLCLIMB_CSS

    def __init__(
        self,
        config: Config | None = None,
        search: str | None = None,
        detail: bool = False,
        holdout: bool | None = None,  # None = holdout view when the problem scores one
        cost: bool = False,
    ):
        super().__init__()
        self.config = config or Config.load()
        self.search = search
        self.detail = detail
        self.holdout = holdout
        self.cost = cost

    def on_mount(self) -> None:
        apply_theme(self)
        self._init_timezone()
        if self.search is None and self._has_several_charts():
            self.push_screen(
                ChartPickerScreen(
                    self.config, detail=self.detail, holdout=self.holdout, cost=self.cost,
                )
            )
            return
        self.push_screen(
            ChartScreen(
                self.config, self.search,
                detail=self.detail, holdout=self.holdout, cost=self.cost,
            )
        )

    def _has_several_charts(self) -> bool:
        store = open_store(self.config)
        try:
            return len(chart_index(store)) > 1
        finally:
            store.close()
