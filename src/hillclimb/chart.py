"""`hillclimb chart` — the live hillclimb curve.

Best-so-far validation score against minutes into the search, one line per
search of the same problem, so successive searches (the demo runs several)
sit on one pair of axes and the climb is visible at a glance. Strictly a
viewer like watch.py: everything is read from journal.jsonl and search.yaml,
and the pure data functions at the top stay testable without Textual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from hillclimb.config import Config
from hillclimb.journal import Journal
from hillclimb.run import (
    iter_run_dirs,
    iter_search_dirs,
    latest_search_dir,
    load_run_meta,
    load_search_meta,
)
from hillclimb.status import effective_state

# Searches drawn at once; older ones of the same problem fall off the chart
# rather than turning it into a haystack.
MAX_CURVES = 8


@dataclass
class Curve:
    label: str
    search_dir: Path
    state: str
    xs: list[float] = field(default_factory=list)  # minutes since the search started
    ys: list[float] = field(default_factory=list)  # best-so-far val score

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


def climb_curve(search_dir: Path, label: str | None = None) -> Curve:
    """Best-so-far curve for one search. A point per scored, unpruned
    candidate at the minute it finished; the y value only ever moves in the
    metric's good direction, so the line is the staircase the search climbed."""
    meta = load_search_meta(search_dir)
    higher = bool(meta.higher_is_better) if meta else True
    start = _parse_ts(meta.started_at) if meta else None
    journal = Journal(search_dir / "journal.jsonl")
    curve = Curve(
        label=label or f"{search_dir.parents[1].name}/{search_dir.name}",
        search_dir=search_dir,
        state=effective_state(search_dir),
    )
    scored = []
    for cand in journal.candidates.values():
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
        if best is None or (score > best if higher else score < best):
            best = score
        curve.xs.append(max(0.0, (when - origin).total_seconds() / 60.0))
        curve.ys.append(best)
    return curve


def climb_curves(runs_dir: Path, problem_id: str, limit: int = MAX_CURVES) -> list[Curve]:
    """Curves for every search of `problem_id` across runs, oldest first so
    the palette assigns colors in start order (the newest gets the last one).
    Labelled by run name, which is what the user chose."""
    found: list[tuple[str, Path, str]] = []
    for run_dir in iter_run_dirs(runs_dir):
        run_meta = load_run_meta(run_dir)
        for search_dir in iter_search_dirs(run_dir):
            meta = load_search_meta(search_dir)
            if meta is None or meta.problem_id != problem_id:
                continue
            label = run_meta.name if run_meta and run_meta.name else run_dir.name
            found.append((meta.started_at, search_dir, label))
    found.sort(key=lambda item: item[0])
    found = found[-limit:]
    return [climb_curve(search_dir, label) for _, search_dir, label in found]


def chart_problem(config: Config, search: str | None = None) -> tuple[str, Path | None]:
    """(problem_id, anchor search dir) for the chart: the given search's
    problem, else the latest search's."""
    from hillclimb.cli import resolve_search_dir

    runs_dir = config.paths.runs_dir
    search_dir = resolve_search_dir(config, search) if search else latest_search_dir(runs_dir)
    if search_dir is None:
        return "", None
    meta = load_search_meta(search_dir)
    return (meta.problem_id if meta else search_dir.name), search_dir


# --- Textual app ---

from plotui import Plot  # noqa: E402
from plotui.textual import PlotWidget  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import Vertical  # noqa: E402
from textual.screen import Screen  # noqa: E402
from textual.widgets import Footer, Header, Label  # noqa: E402

from hillclimb.theme import HILLCLIMB_CSS, apply_theme  # noqa: E402
from hillclimb.watch import REFRESH_S, STATE_STYLE  # noqa: E402


def build_plot(curves: list[Curve]) -> Plot:
    plot = Plot()
    for curve in curves:
        if not curve.xs:
            continue
        xs = list(curve.xs)
        ys = list(curve.ys)
        if len(xs) == 1:
            # a single point draws nothing as a line; give it a flat stub so
            # the first candidate is visible the moment it lands
            xs.append(xs[0] + 0.1)
            ys.append(ys[0])
        plot.add_line(xs, ys, name=curve.label)
    return plot


class ChartScreen(Screen):
    BINDINGS = [
        Binding("q", "app.quit", "quit"),
        Binding("r", "refresh", "refresh"),
    ]

    DEFAULT_CSS = """
    ChartScreen #chartline { height: 1; padding: 0 1; background: $surface; }
    ChartScreen #chart-stage { width: 1fr; height: 1fr; }
    """

    def __init__(self, config: Config, search: str | None = None):
        super().__init__()
        self.config = config
        self.search = search
        self._problem_id = ""
        self._key: tuple | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label(id="chartline")
        yield Vertical(id="chart-stage")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_data()
        self.set_interval(REFRESH_S, self.refresh_data)

    def action_refresh(self) -> None:
        self._key = None
        self.refresh_data()

    def refresh_data(self) -> None:
        problem_id, anchor = chart_problem(self.config, self.search)
        if anchor is None:
            self.query_one("#chartline", Label).update(
                f"no searches in {self.config.paths.runs_dir} yet — start one with `hillclimb run`"
            )
            return
        self._problem_id = problem_id
        curves = climb_curves(self.config.paths.runs_dir, problem_id)
        meta = load_search_meta(anchor)
        metric = meta.metric if meta else "score"
        direction = "higher" if meta is None or meta.higher_is_better else "lower"
        parts = [f"[bold]{problem_id}[/]  {metric} ({direction} is better)"]
        for curve in curves:
            style = STATE_STYLE.get(curve.state, "")
            best = f"{curve.best:.5g}" if curve.best is not None else "-"
            parts.append(f"{curve.label} [{style}]{curve.state}[/] best={best}")
        self.query_one("#chartline", Label).update("  ·  ".join(parts))
        key = tuple((c.label, tuple(c.xs), tuple(c.ys)) for c in curves)
        if key == self._key:
            return
        self._key = key
        stage = self.query_one("#chart-stage", Vertical)
        for child in list(stage.children):
            child.remove()
        if any(c.xs for c in curves):
            stage.mount(PlotWidget(build_plot(curves), id="chart-canvas"))
        else:
            stage.mount(Label("waiting for the first scored candidate…", id="chart-empty"))


class ChartApp(App):
    """Standalone shell for `hillclimb chart`."""

    CSS = HILLCLIMB_CSS
    TITLE = "hillclimb chart"

    def __init__(self, config: Config | None = None, search: str | None = None):
        super().__init__()
        self.config = config or Config.load()
        self.search = search

    def on_mount(self) -> None:
        apply_theme(self)
        self.push_screen(ChartScreen(self.config, self.search))
