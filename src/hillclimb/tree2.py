"""`hillclimb tree2` — the archive tree, drawn the way the Darwin Gödel
Machine paper draws its archive — pure data + the plotui adapter, no Textual.

Same layout as `tree` (`tree.build_tree`: roots on top, one row per operator
step, children left to right in creation order) with a different encoding
on every node, all of it read off the journal at render time:

* the **number inside** the circle is the iteration — the candidate's
  sequence number (`c017` → `17`), the order the search created it in;
* the **fill** is the candidate's score on a viridis ramp over the search's
  own scored range, oriented so the best score is always the bright end
  (lower-is-better searches flip the ramp); unscored nodes are near black;
* the **ring** is how far the candidate got — the paper's "10 / 60 / 200
  tasks" ladder, which in hillclimb is the fate ladder: red = no working
  solution (buggy, abandoned, parked, pruned), yellow = scored but never
  built on, green = selected as a parent (children were built on it, or it
  is the current best, where the next improve hangs);
* the **lineage** of the final best — its parent chain, not the accepted
  staircase — is drawn as one thick line under the grey edges;
* the best itself is a **star** with a gold ring.

One plotui Graph3d trace carries all of it: fill = node colour, ring =
`set_graph_borders`, number = `set_graph_labels` (drawn by plotui inside the
mark, only where it fits). Circles never overlap: `fit_radius` sizes every
mark from the closest pair of projected centres at the current zoom, so the
fit view is a field of small unnumbered circles and the numbers appear as
the tree is zoomed into. `tree2view.py` is the Textual shell: it subclasses
the `tree` widget and swaps the plot builder and legend through the hooks
it exposes, re-sizing the marks on every zoom.
"""

from __future__ import annotations

import re

from hillclimb.graphview import VNode
from hillclimb.tree import SearchTree, TreeNode
from hillclimb.treeview import COL_W, ROW_H

RGB = tuple[int, int, int]

# plotui's own viridis stops (plotui-core `Colormap::sample`), so the ramp
# the legend paints in terminal cells is the ramp the nodes are filled with
VIRIDIS: tuple[RGB, ...] = ((68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37))

# Ring colours: the fate ladder, legend order. The paper's red / yellow / green.
STAGES = ("expanded", "scored", "failed")
STAGE_RGB: dict[str, RGB] = {
    "expanded": (58, 178, 104),
    "scored": (243, 190, 52),
    "failed": (228, 55, 48),
    "pending": (120, 126, 132),
}
STAGE_HELP = {
    "expanded": "built on (children, or the current best)",
    "scored": "scored, never built on",
    "failed": "no working solution (buggy / abandoned / pruned)",
}
BEST_RING_RGB = (255, 200, 40)      # the best's ring: gold
LINEAGE_RGB = (235, 238, 240)       # the thick path to the best: the paper's bold black, on a dark ground
LINEAGE_WIDTH = 3.5
EDGE_RGB = (78, 86, 94)             # every other parent edge: thin grey
ENSEMBLE_EDGE_RGB = (110, 80, 130)  # extra ensemble inputs, faintly purple
UNSCORED_RGB = (30, 30, 38)         # fill for nodes with no score: off the ramp, near black
BEST_SIZE_SCALE = 1.4               # the star is drawn larger than the discs

# Mark sizing: a node's radius comes from the closest pair of projected
# centres so no two circles (ring included — plotui draws the border outside
# the mark, BORDER of the radius wide) ever touch. The budget is in plotui
# units — the pixels a 500 px frame draws them at; a wider (high-DPI) frame
# multiplies them by `Plot.mark_scale`, and so do its text sizes, so the cap
# is the same fraction of the screen, and the same legibility, everywhere.
BORDER = 0.2                        # plotui-core BORDER_FRACTION
STAR_REACH = 1.3                    # a plotui star's points reach this far past its nominal radius
GAP_PX = 1.5                        # clear units between neighbouring rings
R_MIN_PX = 2.0                      # never smaller than this, even when a row is packed
R_MAX_PX = 24.0                     # never larger, however far the tree is zoomed in
DEFAULT_RADIUS = 4.5                # before the first projection

# plotui fits the tree's bounding box into the view with one scale, so a
# wide, shallow archive (80 iterations, 8 deep) would be a flat strip at
# `tree`'s row height. Rows stretch with the width until the picture is
# about ASPECT wide for 1 tall — the paper's figure — never below ROW_H.
ASPECT = 1.6

_CANDIDATE_ID = re.compile(r"^c0*(\d+)$")


# --- colour -----------------------------------------------------------------

def viridis(t: float) -> RGB:
    """The ramp at `t` in [0, 1] (clamped): linear between plotui's stops."""
    t = min(1.0, max(0.0, float(t)))
    pos = t * (len(VIRIDIS) - 1)
    i = min(int(pos), len(VIRIDIS) - 2)
    frac = pos - i
    a, b = VIRIDIS[i], VIRIDIS[i + 1]
    return tuple(round(a[k] + (b[k] - a[k]) * frac) for k in range(3))  # type: ignore[return-value]


def score_range(tree: SearchTree) -> tuple[float, float] | None:
    """`(lo, hi)` over the tree's scored nodes; None when nothing is scored."""
    scores = [n.score for n in tree.nodes if n.score is not None]
    if not scores:
        return None
    return min(scores), max(scores)


def score_t(score: float, lo: float, hi: float, higher_is_better: bool = True) -> float:
    """Where `score` sits on the ramp: 1 is the best end whichever way the
    metric runs; a search whose scores are all equal sits at the top."""
    if hi <= lo:
        return 1.0
    t = (score - lo) / (hi - lo)
    return t if higher_is_better else 1.0 - t


def fill_rgb(node: TreeNode, span: tuple[float, float] | None, higher_is_better: bool = True) -> RGB:
    if node.score is None or span is None:
        return UNSCORED_RGB
    return viridis(score_t(node.score, span[0], span[1], higher_is_better))


# --- geometry ---------------------------------------------------------------

def row_height(tree: SearchTree | None) -> float:
    """World units per depth row for `tree`: `tree`'s ROW_H, stretched so
    the whole picture is at most ASPECT times wider than tall."""
    if tree is None or not tree.nodes:
        return ROW_H
    xs = [n.x for n in tree.nodes]
    width = (max(xs) - min(xs)) * COL_W
    rows = max(tree.depth - 1, 1)
    return max(ROW_H, width / ASPECT / rows)


def world_xy(node: TreeNode, row_h: float = ROW_H) -> tuple[float, float]:
    """Tree column/depth → the screen plane (+y up, so depth goes down)."""
    return node.x * COL_W, -node.depth * row_h


def tree_extent(tree: SearchTree, row_h: float) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not tree.nodes:
        return None
    xs, ys = zip(*(world_xy(n, row_h) for n in tree.nodes))
    return (min(xs), min(ys)), (max(xs), max(ys))


def closest_pair_px(points: list[tuple[float, float]]) -> float | None:
    """The smallest distance between two of `points` (pixels), or None with
    fewer than two. A tidy tree is rows of nodes, so the closest pair is
    either two neighbours in a row or two rows: a sort per row plus the row
    gaps, not the quadratic scan."""
    if len(points) < 2:
        return None
    rows: dict[int, list[float]] = {}
    for x, y in points:
        rows.setdefault(round(y * 4), []).append(x)  # quarter-pixel buckets: one row, one bucket
    best = float("inf")
    for xs in rows.values():
        xs.sort()
        for a, b in zip(xs, xs[1:]):
            best = min(best, b - a)
    ys = sorted(k / 4 for k in rows)
    for a, b in zip(ys, ys[1:]):
        best = min(best, b - a)
    return best


def nearest_px(points: list[tuple[float, float]], index: int) -> float | None:
    """The distance from `points[index]` to its nearest other point."""
    if len(points) < 2:
        return None
    x0, y0 = points[index]
    return min(
        ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 for i, (x, y) in enumerate(points) if i != index
    )


def radius_px(closest: float | None) -> float:
    """The disc radius that keeps rings `GAP_PX` apart at a closest pair
    `closest` apart (both in plotui units), clamped to [R_MIN_PX, R_MAX_PX];
    the maximum when there is no pair to keep apart."""
    if closest is None:
        return R_MAX_PX
    r = (closest / 2.0 - GAP_PX) / (1.0 + BORDER)
    return min(R_MAX_PX, max(R_MIN_PX, r))


def star_reach(radius: float) -> float:
    """How far a star of nominal `radius` extends, border included."""
    return radius * STAR_REACH * (1.0 + BORDER)


def star_scale(radius: float, nearest: float | None) -> float:
    """How much larger than the discs the star may be: BEST_SIZE_SCALE,
    unless its own nearest neighbour (a disc, `radius` wide) is closer than
    that allows — then just what fits, never below the discs' size."""
    if nearest is None:
        return BEST_SIZE_SCALE
    room = nearest - radius * (1.0 + BORDER) - GAP_PX  # what is left for the star's reach
    return min(BEST_SIZE_SCALE, max(1.0, room / star_reach(radius)))


def fit_radius(plot, px_w: int, px_h: int, best_index: int | None = None) -> tuple[float, float]:
    """`(radius, star_scale)` to rebuild `plot`'s tree with (plotui units)
    so its marks never overlap at the camera it holds now, from the nodes'
    projection: the closest pair in framebuffer pixels, brought back to
    plotui units by the frame's mark scale, sets the discs; the star at
    `best_index` gets its own budget from its own nearest neighbour."""
    from plotui import Plot

    projected = [(p[0], p[1]) for p in plot.project_nodes(px_w, px_h)]
    scale = Plot.mark_scale(px_w)
    closest = closest_pair_px(projected)
    radius = radius_px(None if closest is None else closest / scale)
    nearest = nearest_px(projected, best_index) if best_index is not None else None
    return radius, star_scale(radius, None if nearest is None else nearest / scale)


# --- per-node encoding ------------------------------------------------------

def node_number(node_id: str) -> str:
    """The iteration a candidate id stands for: `c017` → `17`. Ids that are
    not of that form are shown as they are."""
    match = _CANDIDATE_ID.match(node_id)
    return match.group(1) if match else node_id


def stage(node: TreeNode) -> str:
    """The ring: which rung of the fate ladder the candidate reached."""
    if node.fate == "pending":
        return "pending"
    if node.fate in ("failed", "pruned"):
        return "failed"
    if node.fate in ("expanded", "best"):
        return "expanded"
    return "scored"


def ring_rgb(node: TreeNode) -> RGB:
    return BEST_RING_RGB if node.fate == "best" else STAGE_RGB[stage(node)]


def node_shape(node: TreeNode) -> str:
    return "star" if node.fate == "best" else "disc"


def node_radius(node: TreeNode, radius: float, star: float = BEST_SIZE_SCALE) -> float:
    return radius * star if node.fate == "best" else radius


def node_label(node: TreeNode) -> str:
    return node_number(node.id)


def best_lineage(tree: SearchTree) -> tuple[str, ...]:
    """The final best's ancestry, root first — the path the paper draws bold.
    Unlike `tree.accepted` (the best-so-far staircase) this follows parent
    links, so it is the code's actual descent."""
    best = tree.best_id
    if best is None:
        return ()
    by_id = {n.id: n for n in tree.nodes}
    chain: list[str] = []
    seen: set[str] = set()
    cursor: str | None = best
    while cursor is not None and cursor in by_id and cursor not in seen:
        chain.append(cursor)
        seen.add(cursor)
        cursor = by_id[cursor].parent_id
    chain.reverse()
    return tuple(chain)


def stage_counts(tree: SearchTree) -> dict[str, int]:
    counts = {s: 0 for s in STAGES}
    for node in tree.nodes:
        s = stage(node)
        if s in counts:
            counts[s] += 1
    return counts


# --- the plot ---------------------------------------------------------------

def build_tree2_plot(
    tree: SearchTree, higher_is_better: bool = True, *,
    selected: str | None = None, frame: SearchTree | None = None, radius: float = DEFAULT_RADIUS,
    star: float = BEST_SIZE_SCALE,
):
    """SearchTree → a fresh plotui Plot: the best's lineage as a thick line
    (underneath), then one Graph3d trace — score fills, stage rings, the
    numbers inside, the best a star — with marks of `radius` (plotui units)
    and the star `star` times that (see `fit_radius`); face-on camera at
    zoom 1; plus the flat-index →
    node-id list. `frame` pins the view to another tree's extent and row
    height, as `build_tree_plot` does with its extent."""
    from hillclimb.theme import themed_plot

    ids = [n.id for n in tree.nodes]
    index_of = {node_id: i for i, node_id in enumerate(ids)}
    row_h = row_height(frame if frame is not None else tree)
    plot = themed_plot()
    plot.set_show_box(False)
    if hasattr(type(plot), "legend_visible"):
        plot.legend_visible = False  # the legend is a text overlay (`legend_spans`); traces keep names for hover
    plot.set_camera_state(0.0, 0.0, 1.0, 0.0, 0.0)
    extent = tree_extent(frame, row_h) if frame is not None else None
    if extent is not None:
        (x0, y0), (x1, y1) = extent
        plot.set_bounds((x0, 0.0, y0), (x1, 0.0, y1))
    if not tree.nodes:
        return plot, ids
    span = score_range(tree)
    xs, ys = (list(v) for v in zip(*(world_xy(n, row_h) for n in tree.nodes)))
    zeros = [0.0] * len(ids)
    # the lineage goes down first so the nodes sit on top of it; a line
    # trace has no pickable nodes, so the discs still start at flat index 0.
    # Its parent edges are left out of the grey set: drawn in the same plane
    # they would cut a grey seam through the thick line
    chain = [i for i in best_lineage(tree) if i in index_of]
    lineage = [index_of[i] for i in chain]
    on_lineage = set(zip(chain, chain[1:]))
    edges = [
        e for e in tree.edges
        if e.src in index_of and e.dst in index_of and (e.src, e.dst) not in on_lineage
    ]
    if len(lineage) > 1:
        plot.add_line3d(
            [xs[i] for i in lineage], [0.0] * len(lineage), [ys[i] for i in lineage],
            color=LINEAGE_RGB, width=LINEAGE_WIDTH, name="lineage",
        )
    handle = plot.add_graph3d(
        xs, zeros, ys,
        edges=[(index_of[e.src], index_of[e.dst]) for e in edges],
        node_colors=[fill_rgb(n, span, higher_is_better) for n in tree.nodes],
        size=radius,
        node_sizes=[node_radius(n, radius, star) for n in tree.nodes],
        edge_colors=[ENSEMBLE_EDGE_RGB if e.kind == "ensemble-input" else EDGE_RGB for e in edges],
        node_shapes=[node_shape(n) for n in tree.nodes],
        name="candidates",
    )
    plot.set_graph_borders(handle, [ring_rgb(n) for n in tree.nodes])
    plot.set_graph_labels(handle, [node_label(n) for n in tree.nodes])
    if selected in index_of:
        plot.set_selected(index_of[selected])
    return plot, ids


def label_nodes(tree: SearchTree, frame: SearchTree | None = None) -> list[VNode]:
    """The numbers as graphview VNodes at the plot's row height, for the
    overlay that names the hovered/selected node beside its mark when the
    mark is too small to carry the number itself."""
    row_h = row_height(frame if frame is not None else tree)
    lineage = set(best_lineage(tree))
    out = []
    for node in tree.nodes:
        x, y = world_xy(node, row_h)
        out.append(VNode(
            id=node.id, type="candidate", label=node_label(node), x=x, y=y, z=0.0,
            count=3 if node.fate == "best" else 2 if node.id in lineage else 1,
        ))
    return out


# --- the legend -------------------------------------------------------------
# Two overlays, both terminal text so they cost the canvas nothing: the ring
# ladder at the top left, and the score ramp — a column of block cells in
# viridis with the best and worst score beside it — at the top right.

LEGEND_ROW, LEGEND_COL = 1, 1
RAMP_ROWS = 8
RAMP_MARGIN = 1


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def legend_spans(
    tree: SearchTree | None, metric: str = "score", higher_is_better: bool = True, cols: int = 80,
) -> list[tuple[int, int, str, str]]:
    """`(row, col, text, style)` spans: the ring ladder with counts, the
    best's star, and the ramp with its two end values."""
    spans: list[tuple[int, int, str, str]] = []
    counts = stage_counts(tree) if tree is not None else {}
    for index, entry in enumerate(STAGES):
        r, g, b = STAGE_RGB[entry]
        count = f" {counts[entry]}" if counts.get(entry) else ""
        spans.append((LEGEND_ROW + index, LEGEND_COL, "◯", f"rgb({r},{g},{b})"))
        spans.append((LEGEND_ROW + index, LEGEND_COL + 2, f"{entry}{count}", "white"))
    r, g, b = BEST_RING_RGB
    spans.append((LEGEND_ROW + len(STAGES), LEGEND_COL, "★", f"rgb({r},{g},{b})"))
    spans.append((LEGEND_ROW + len(STAGES), LEGEND_COL + 2, "best", "white"))

    span = score_range(tree) if tree is not None else None
    if span is None:
        return spans
    lo, hi = span
    top, bottom = (hi, lo) if higher_is_better else (lo, hi)
    labels = {0: _fmt(top), RAMP_ROWS - 1: _fmt(bottom)}
    width = max(len(v) for v in labels.values())
    bar_col = cols - RAMP_MARGIN - 1
    label_col = bar_col - 1 - width
    caption = metric[: bar_col + 1 - RAMP_MARGIN]
    spans.append((LEGEND_ROW, max(0, bar_col + 1 - len(caption)), caption, "bold white"))
    for i in range(RAMP_ROWS):
        row = LEGEND_ROW + 1 + i
        r, g, b = viridis(1.0 - i / (RAMP_ROWS - 1))
        spans.append((row, bar_col, "█", f"rgb({r},{g},{b})"))
        if i in labels:
            spans.append((row, label_col, labels[i].rjust(width), "white"))
    return spans
