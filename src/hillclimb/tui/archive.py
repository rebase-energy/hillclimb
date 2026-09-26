"""`hillclimb archive` — the archive tree beside the progress chart, the
Darwin Gödel Machine paper's two-panel figure on hillclimb's journal. This
is the pure layer for the right-hand panel; `tree2.py` draws the left one,
`archiveview.py` is the Textual shell that scrubs both together.

The two panels share one x: the **candidate number** — the number inside
each circle of the tree (`tree2.node_number`: `c017` → 17, the order the
search created candidates in), so a dot on the chart and a circle in the tree
name the same candidate. Every step along x is one candidate. The chart is `chart`'s climb re-indexed onto that axis:
every scored candidate a dot at (number, score), the best-so-far
staircase over them in number order, a brighter dot where a candidate
set a new best, and the **lineage** of the final best — its parent chain,
the path `tree2` draws bold on the left — as a thick line through the
ancestors that scored, ending at the best. Scrubbing back through time
shows the chart as it stood at that tick (the scrubbed tree is the input)
with a cursor at the candidate that landed there; the frame (the live tree)
pins both axes so the picture never rescales while stepping.

Number order, not landing order: with parallel operators a later-numbered
candidate may land first. The staircase here climbs by number — what the
paper plots — and reaches the same final best as `tree.accepted` (the best
of a set is the best whichever way it is walked); only the intermediate
steps can differ from `chart`'s.
"""

from __future__ import annotations

from dataclasses import dataclass

from hillclimb.tui.chart import CYAN, MISS_RGB, LegendEntry, step_points
from hillclimb.harness.direction import better
from hillclimb.tui.tree import SearchTree, TreeNode
from hillclimb.tui.tree2 import LINEAGE_RGB, LINEAGE_WIDTH, best_lineage, node_number, score_range

RGB = tuple[int, int, int]

CURSOR_RGB = (120, 126, 132)   # the scrub cursor: the tree's "pending" grey
SELECTED_RGB = (255, 200, 40)  # the selected candidate's ring on the chart: gold, a hue no series uses
PAD = 0.05                     # axis padding, the same fraction plotui pads chart traces by
SERIES = ("attempt", "best so far", "new best", "lineage")  # legend order


@dataclass(frozen=True)
class ProgressPoint:
    """One scored candidate on the progress chart."""

    id: str
    number: int         # the candidate number: the tree's circle number, the chart's x
    score: float
    best: bool          # set a new best-so-far, walking the numbers in order
    on_lineage: bool    # an ancestor of the final best (or the best itself)


def candidate_numbers(tree: SearchTree) -> dict[str, int]:
    """Node id → candidate number. The number in the id (`c017` → 17) when every
    node carries one — then the chart's x is the tree's circle number — else
    the node's place in creation order, which is what the number means."""
    numbers = {}
    for node in tree.nodes:
        text = node_number(node.id)
        if not text.isdigit():
            return {node.id: i for i, node in enumerate(tree.nodes)}
        numbers[node.id] = int(text)
    return numbers


def counts_as_scored(node: TreeNode) -> bool:
    """Whether a node is a dot on the chart: it scored and was not cut —
    `chart`'s rule, on tree nodes."""
    return node.score is not None and not node.pruned and node.fate != "pending"


def progress_points(tree: SearchTree, higher_is_better: bool = True) -> list[ProgressPoint]:
    """Every scored node in number order, flagged `best` where it beat
    everything before it (in that order) and `on_lineage` where the final
    best descends from it."""
    numbers = candidate_numbers(tree)
    lineage = set(best_lineage(tree))
    scored = sorted((n for n in tree.nodes if counts_as_scored(n)), key=lambda n: (numbers[n.id], n.id))
    points: list[ProgressPoint] = []
    best: float | None = None
    for node in scored:
        improved = best is None or better(node.score, best, higher_is_better)
        if improved:
            best = node.score
        points.append(ProgressPoint(node.id, numbers[node.id], node.score, improved, node.id in lineage))
    return points


def progress_extent(tree: SearchTree) -> int:
    """The last candidate number the tree knows of, scored or not — the staircase
    runs flat to here and the x axis ends here."""
    return max(candidate_numbers(tree).values(), default=0)


def staircase(points: list[ProgressPoint], extent: int | None = None) -> tuple[list[float], list[float]]:
    """The best-so-far line as step points, flat to `extent`."""
    hits = [p for p in points if p.best]
    return step_points([float(p.number) for p in hits], [p.score for p in hits], extent)


def lineage_points(points: list[ProgressPoint]) -> list[ProgressPoint]:
    """The ancestors of the final best that scored, root first — the thick
    line. Ancestors without a score (a debugged buggy parent) are skipped:
    the line joins the scores either side of them."""
    return [p for p in points if p.on_lineage]


def cursor_number(tree: SearchTree, until: str | None) -> int | None:
    """Where the scrub cursor sits: the number of the candidate whose
    result landed at tick `until` (the latest when several share the stamp);
    None when live or nothing landed then."""
    if until is None:
        return None
    numbers = candidate_numbers(tree)
    landed = [numbers[n.id] for n in tree.nodes if n.finished_at == until]
    return max(landed) if landed else None


def progress_bounds(
    frame: SearchTree,
) -> tuple[tuple[float, float], tuple[float, float] | None]:
    """`((x_lo, x_hi), (y_lo, y_hi) | None)`: the axes of the picture, from the
    frame tree, padded like plotui pads its own — x from candidate 0 to the
    last created candidate, y over the scored range."""
    extent = float(progress_extent(frame))
    x_pad = max(1.0, extent * PAD)
    span = score_range(frame)
    if span is None:
        return (0.0 - x_pad, extent + x_pad), None
    lo, hi = span
    y_pad = (hi - lo) * PAD if hi > lo else max(abs(hi) * PAD, 1e-9)
    return (0.0 - x_pad, extent + x_pad), (lo - y_pad, hi + y_pad)


def build_progress_plot(
    tree: SearchTree,
    higher_is_better: bool = True,
    *,
    frame: SearchTree | None = None,
    cursor: int | None = None,
    selected: str | None = None,
    hidden: frozenset[str] | set[str] = frozenset(),
    metric: str = "score",
):
    """SearchTree → a fresh plotui Plot of the progress panel. `frame` (the
    live tree while scrubbing) pins both axes; `cursor` draws the scrub
    position as a vertical guide; `selected` rings that candidate's dot;
    `hidden` names legend series left out."""
    from hillclimb.tui.theme import themed_plot

    plot = themed_plot()
    if hasattr(type(plot), "legend_visible"):
        plot.legend_visible = False  # the legend is a text band under the plot; names feed the hover readout
    for setter, text in (("set_x_title", "candidate"), ("set_y_title", metric)):
        if hasattr(plot, setter):
            getattr(plot, setter)(text)
    (x_lo, x_hi), y_range = progress_bounds(frame if frame is not None else tree)
    if hasattr(plot, "set_x_range"):
        plot.set_x_range((x_lo, x_hi))
    if y_range is not None and hasattr(plot, "set_y_range"):
        plot.set_y_range(y_range)

    points = progress_points(tree, higher_is_better)
    # the staircase runs flat to the last candidate *this* tree knows of: a
    # scrubbed view stops at its tick, only the axes come from the frame
    extent = progress_extent(tree)
    misses = [p for p in points if not p.best and not p.on_lineage]
    if misses and "attempt" not in hidden:
        plot.add_scatter(
            [float(p.number) for p in misses], [p.score for p in misses],
            color=MISS_RGB, size=2.4, name="attempt",
        )
    xs, ys = staircase(points, extent)
    if len(xs) > 1 and "best so far" not in hidden:
        plot.add_line(xs, ys, color=CYAN, width=2.0, name="best so far")
    hits = [p for p in points if p.best and not p.on_lineage]
    if hits and "new best" not in hidden:
        plot.add_scatter(
            [float(p.number) for p in hits], [p.score for p in hits],
            color=CYAN, size=3.0, name="new best",
        )
    chain = lineage_points(points)
    if chain and "lineage" not in hidden:
        if len(chain) > 1:
            plot.add_line(
                [float(p.number) for p in chain], [p.score for p in chain],
                color=LINEAGE_RGB, width=LINEAGE_WIDTH, name="lineage",
            )
        plot.add_scatter(
            [float(p.number) for p in chain], [p.score for p in chain],
            color=LINEAGE_RGB, size=3.2, name="lineage",
        )
    if cursor is not None and y_range is not None:
        plot.add_line(
            [float(cursor), float(cursor)], [y_range[0], y_range[1]],
            color=CURSOR_RGB, width=1.0, name="now",
        )
    if selected is not None:
        mark = next((p for p in points if p.id == selected), None)
        if mark is not None:
            plot.add_scatter(
                [float(mark.number)], [mark.score], color=SELECTED_RGB, size=4.5, name=selected,
            )
    return plot


def progress_legend(points: list[ProgressPoint] | None = None) -> list[LegendEntry]:
    """The legend's entries, in `SERIES` order, with the lineage's length
    beside its name when there is one."""
    chain = lineage_points(points) if points else []
    lineage = f"lineage ({len(chain)})" if chain else "lineage"
    return [
        ("attempt", MISS_RGB, "●"),
        ("best so far", CYAN, "─"),
        ("new best", CYAN, "●"),
        (lineage, LINEAGE_RGB, "━"),
    ]


def legend_series(label: str) -> str:
    """The series a legend entry toggles: `lineage (4)` → `lineage`."""
    return label.split(" (")[0]


# --- the legend overlay ------------------------------------------------------
# Drawn in terminal cells over the plot, like tree2's ring ladder, in the
# corner the climb leaves empty: a higher-is-better staircase rises to the
# top right with its attempts below it, so the top left is clear; a
# lower-is-better one falls to the bottom right with its attempts above, so
# the bottom left is. LEGEND_COL clears the y-axis tick labels; the bottom
# placement clears the x-axis labels and title.

LEGEND_COL = 9
LEGEND_TOP_ROW = 1
LEGEND_BOTTOM_MARGIN = 4   # rows under the legend: x ticks, x title, frame
LEGEND_WIDTH = 2 + 2 + max(len(s) for s in SERIES) + 5  # "1 " + glyph + " " + name + " (nn)"


def legend_rows(higher_is_better: bool, n_lines: int, rows: int) -> int:
    """The row the first legend line goes on: the top for a rising climb,
    the bottom for a falling one (never above the top)."""
    if higher_is_better:
        return LEGEND_TOP_ROW
    return max(LEGEND_TOP_ROW, rows - LEGEND_BOTTOM_MARGIN - n_lines)


def legend_spans(
    points: list[ProgressPoint] | None,
    higher_is_better: bool = True,
    *,
    hidden: frozenset[str] | set[str] = frozenset(),
    rows: int = 24,
    first_key: int = 1,
) -> list[tuple[int, int, str, str]]:
    """`(row, col, text, style)` spans: one line per series — hotkey (from
    `first_key`: the archive view numbers the tree's legend first), glyph
    and name in the series' colour, dim and struck through when hidden."""
    entries = progress_legend(points)
    first = legend_rows(higher_is_better, len(entries), rows)
    spans: list[tuple[int, int, str, str]] = []
    for index, (label, (r, g, b), glyph) in enumerate(entries):
        row = first + index
        spans.append((row, LEGEND_COL, f"{(first_key + index) % 10} ", "dim"))
        if legend_series(label) in hidden:
            spans.append((row, LEGEND_COL + 2, f"  {label}", "dim strike"))
        else:
            spans.append((row, LEGEND_COL + 2, f"{glyph} {label}", f"rgb({r},{g},{b})"))
    return spans


def legend_entry_at(col: int, row: int, higher_is_better: bool = True, *, rows: int = 24) -> str | None:
    """The series under canvas cell `(col, row)`, or None off the legend."""
    index = row - legend_rows(higher_is_better, len(SERIES), rows)
    if 0 <= index < len(SERIES) and LEGEND_COL <= col < LEGEND_COL + LEGEND_WIDTH:
        return SERIES[index]
    return None
