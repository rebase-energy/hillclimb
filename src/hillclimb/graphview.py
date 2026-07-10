"""Interactive knowledge-graph screen for the watch TUI.

Strictly a *viewer* over `knowledge/graph.json` (built by graph.py): zoom,
pan, click nodes, scrub through time (one tick per finished search), filter
and color by concept. The pure functions at the top carry all geometry,
rasterization, and hit-testing so they stay testable without driving Textual;
the widgets below are thin shells in the style of watch.py.

Geometry lives in DOT SPACE: each terminal cell is 2x4 dots, and a dot is
roughly square on screen — so distances, camera math, and hit tests done in
dots sidestep the 1:2 cell aspect distortion entirely. Edges render as thin
continuous strokes (─ │ ╱ ╲ picked per cell by line direction), nodes as
single-cell glyphs (a node owns its cell's style — one Rich style per cell),
labels as plain text with a greedy collision mask.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path

from rich.segment import Segment
from rich.style import Style
from rich.text import Text

from hillclimb.config import Config
from hillclimb.graph import GraphNode, KnowledgeGraph, graph_at

# --- pure data layer ---

# Edges rasterize as thin box-drawing strokes, one character per cell picked
# by the line's local direction — continuous connecting lines, not scattered
# dots (braille) or half-cell slabs (quadrant blocks). Geometry stays in 2x4
# dot space (square dots); rasterization walks the line at cell resolution.
LINE_H, LINE_V, LINE_UP, LINE_DOWN = "─", "│", "╱", "╲"

SCALE_MIN = 1.0
SCALE_MAX = 20000.0
ZOOM_FACTOR = 1.25
LABEL_SCALE = 55.0     # world coords are spring-normalized to ~[-1,1]
COLLAPSE_ENTER = 25.0  # below this scale entities fold into concept supernodes
COLLAPSE_EXIT = COLLAPSE_ENTER * 1.4
LABEL_MAX = 18
HIT_RADIUS_DOTS = 6.0

NODE_GLYPHS = {
    "problem": "○", "family": "◎", "search": "▲", "operator": "•",
    "concept": "◉", "claim": "◇", "technique": "●", "library": "■",
    "model_family": "◆", "feature": "◆", "practice": "◆", "supernode": "◉",
}
NODE_COLORS = {
    "problem": "yellow", "family": "bold yellow", "search": "blue",
    "operator": "bright_black", "concept": "magenta", "claim": "green",
    "technique": "cyan", "library": "bold cyan", "model_family": "cyan",
    "feature": "cyan", "practice": "cyan", "supernode": "bold magenta",
}
EDGE_COLORS = {
    "ran_on": "dim blue", "belongs_to": "dim yellow", "used": "dim cyan",
    "about": "dim green", "derived_from": "dim green", "applies_to": "dim green",
    "supersedes": "dim red", "has_concept": "dim magenta", "is_a": "dim magenta",
}
# fixed 8-color pool for concept coloring — theme-safe, no RGB gradients
CONCEPT_COLOR_POOL = (
    "cyan", "magenta", "green", "yellow", "blue", "red", "bright_cyan", "bright_magenta",
)


@dataclass(frozen=True)
class Camera:
    cx: float
    cy: float
    scale: float  # braille dots per world unit


@dataclass(frozen=True)
class WorldBounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float


@dataclass(frozen=True)
class VNode:
    """A drawable node: original graph node(s) resolved to a world position.
    count > 1 marks a concept supernode."""

    id: str
    type: str
    label: str
    x: float
    y: float
    concepts: tuple[str, ...] = ()
    first_seen: str = ""
    count: int = 1
    members: tuple[str, ...] = ()


@dataclass(frozen=True)
class VEdge:
    src: str
    dst: str
    type: str
    weight: float = 1.0


@dataclass(frozen=True)
class VisibleGraph:
    nodes: tuple[VNode, ...]
    edges: tuple[VEdge, ...]


@dataclass(frozen=True)
class PlacedNode:
    node_id: str
    dot_x: float
    dot_y: float


def fallback_pos(node_id: str) -> tuple[float, float]:
    """Deterministic radial placement for nodes the index left unplaced
    (engine-only installs without networkx)."""
    digest = int(hashlib.sha1(node_id.encode()).hexdigest()[:12], 16)
    angle = (digest % 3600) / 3600 * 2 * math.pi
    radius = 0.4 + ((digest // 3600) % 600) / 1000
    return (radius * math.cos(angle), radius * math.sin(angle))


def node_pos(node: GraphNode) -> tuple[float, float]:
    return node.pos if node.pos is not None else fallback_pos(node.id)


def filter_concepts(graph: KnowledgeGraph, enabled: frozenset[str] | None) -> KnowledgeGraph:
    """Concept filter: hides concept-bearing nodes (entities, claims,
    problems) whose concepts miss the enabled set, and the concept nodes
    themselves when deselected. Nodes without concepts (searches, families,
    operators) always stay — they are the graph's skeleton."""
    if enabled is None:
        return graph
    nodes = [
        n for n in graph.nodes
        if (n.type == "concept" and n.label in enabled)
        or (n.type != "concept" and (not n.concepts or set(n.concepts) & enabled))
    ]
    ids = {n.id for n in nodes}
    edges = [e for e in graph.edges if e.src in ids and e.dst in ids]
    return KnowledgeGraph(
        schema_version=graph.schema_version, built_at=graph.built_at,
        events=graph.events, nodes=nodes, edges=edges,
    )


def lod_collapsed(scale: float, was_collapsed: bool) -> bool:
    """Hysteresis so the boundary doesn't flicker while zooming."""
    if was_collapsed:
        return scale < COLLAPSE_EXIT
    return scale < COLLAPSE_ENTER


def apply_lod(graph: KnowledgeGraph, collapsed: bool) -> VisibleGraph:
    """Semantic zoom: collapsed folds every concept-bearing node into its
    primary-concept supernode (position = member centroid); concept nodes
    disappear into their supernodes; the skeleton stays atomic."""
    if not collapsed:
        return VisibleGraph(
            nodes=tuple(
                VNode(
                    id=n.id, type=n.type, label=n.label, x=node_pos(n)[0], y=node_pos(n)[1],
                    concepts=tuple(n.concepts), first_seen=n.first_seen,
                )
                for n in graph.nodes
            ),
            edges=tuple(VEdge(e.src, e.dst, e.type, e.weight) for e in graph.edges),
        )
    remap: dict[str, str] = {}
    groups: dict[str, list[GraphNode]] = {}
    atoms: list[GraphNode] = []
    for node in graph.nodes:
        if node.type == "concept":
            remap[node.id] = f"supernode:{node.label}"
            groups.setdefault(node.label, [])
        elif node.concepts:
            primary = node.concepts[0]
            remap[node.id] = f"supernode:{primary}"
            groups.setdefault(primary, []).append(node)
        else:
            atoms.append(node)
    nodes = [
        VNode(
            id=n.id, type=n.type, label=n.label, x=node_pos(n)[0], y=node_pos(n)[1],
            first_seen=n.first_seen,
        )
        for n in atoms
    ]
    for concept, members in sorted(groups.items()):
        if members:
            xs = [node_pos(m)[0] for m in members]
            ys = [node_pos(m)[1] for m in members]
            x, y = sum(xs) / len(xs), sum(ys) / len(ys)
        else:
            x, y = fallback_pos(f"supernode:{concept}")
        nodes.append(VNode(
            id=f"supernode:{concept}", type="supernode",
            label=f"{concept} ({len(members)})", x=x, y=y,
            concepts=(concept,), count=max(len(members), 1),
            members=tuple(m.id for m in members),
        ))
    aggregated: dict[tuple[str, str, str], float] = {}
    for edge in graph.edges:
        src = remap.get(edge.src, edge.src)
        dst = remap.get(edge.dst, edge.dst)
        if src == dst:
            continue
        key = (src, dst, edge.type)
        aggregated[key] = aggregated.get(key, 0.0) + edge.weight
    edges = tuple(
        VEdge(src, dst, kind, weight)
        for (src, dst, kind), weight in sorted(aggregated.items())
    )
    return VisibleGraph(nodes=tuple(nodes), edges=edges)


def world_bounds(nodes: tuple[VNode, ...] | list[VNode]) -> WorldBounds:
    if not nodes:
        return WorldBounds(-1.0, -1.0, 1.0, 1.0)
    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]
    pad_x = max((max(xs) - min(xs)) * 0.2, 0.1)
    pad_y = max((max(ys) - min(ys)) * 0.2, 0.1)
    return WorldBounds(min(xs) - pad_x, min(ys) - pad_y, max(xs) + pad_x, max(ys) + pad_y)


def dot_size(cells_w: int, cells_h: int) -> tuple[int, int]:
    return cells_w * 2, cells_h * 4


def world_to_dot(cam: Camera, dots_w: int, dots_h: int, wx: float, wy: float) -> tuple[float, float]:
    return (
        (wx - cam.cx) * cam.scale + dots_w / 2,
        (wy - cam.cy) * cam.scale + dots_h / 2,
    )


def dot_to_world(cam: Camera, dots_w: int, dots_h: int, dx: float, dy: float) -> tuple[float, float]:
    return (
        (dx - dots_w / 2) / cam.scale + cam.cx,
        (dy - dots_h / 2) / cam.scale + cam.cy,
    )


def clamp_camera(cam: Camera, bounds: WorldBounds) -> Camera:
    return replace(
        cam,
        cx=min(max(cam.cx, bounds.min_x), bounds.max_x),
        cy=min(max(cam.cy, bounds.min_y), bounds.max_y),
    )


def zoom_about(
    cam: Camera, dots_w: int, dots_h: int,
    cursor_dx: float, cursor_dy: float, factor: float, bounds: WorldBounds,
) -> Camera:
    """Invariant: the world point under the cursor stays under the cursor."""
    wx, wy = dot_to_world(cam, dots_w, dots_h, cursor_dx, cursor_dy)
    scale = min(max(cam.scale * factor, SCALE_MIN), SCALE_MAX)
    return clamp_camera(
        Camera(
            cx=wx - (cursor_dx - dots_w / 2) / scale,
            cy=wy - (cursor_dy - dots_h / 2) / scale,
            scale=scale,
        ),
        bounds,
    )


def pan_camera(cam: Camera, ddx_dots: float, ddy_dots: float, bounds: WorldBounds) -> Camera:
    return clamp_camera(
        replace(cam, cx=cam.cx + ddx_dots / cam.scale, cy=cam.cy + ddy_dots / cam.scale),
        bounds,
    )


def fit_camera(bounds: WorldBounds, dots_w: int, dots_h: int) -> Camera:
    span_x = max(bounds.max_x - bounds.min_x, 1e-6)
    span_y = max(bounds.max_y - bounds.min_y, 1e-6)
    scale = min(dots_w / span_x, dots_h / span_y)
    scale = min(max(scale, SCALE_MIN), SCALE_MAX)
    return Camera(
        cx=(bounds.min_x + bounds.max_x) / 2,
        cy=(bounds.min_y + bounds.max_y) / 2,
        scale=scale,
    )


class CellBuffer:
    """A cells_w x cells_h frame: an edge-stroke layer under a whole-cell
    glyph/label layer (chars win over lines at flush), one style string per
    cell."""

    def __init__(self, cells_w: int, cells_h: int):
        self.w = cells_w
        self.h = cells_h
        self.lines: list[list[str | None]] = [[None] * cells_w for _ in range(cells_h)]
        self.chars: list[list[str | None]] = [[None] * cells_w for _ in range(cells_h)]
        self.styles: list[list[str | None]] = [[None] * cells_w for _ in range(cells_h)]

    def set_line(self, cx: int, cy: int, char: str, style: str) -> None:
        if 0 <= cx < self.w and 0 <= cy < self.h:
            self.lines[cy][cx] = char  # crossing edges: last drawn wins the cell
            if self.chars[cy][cx] is None:  # a glyph keeps its own style
                self.styles[cy][cx] = style

    def set_char(self, cx: int, cy: int, char: str, style: str) -> None:
        if 0 <= cx < self.w and 0 <= cy < self.h:
            self.chars[cy][cx] = char
            self.styles[cy][cx] = style

    def char_free(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.w and 0 <= cy < self.h and self.chars[cy][cx] is None

    def to_segments(self) -> list[list[Segment]]:
        rows: list[list[Segment]] = []
        for y in range(self.h):
            segments: list[Segment] = []
            run: list[str] = []
            run_style: str | None = None
            for x in range(self.w):
                char = self.chars[y][x] or self.lines[y][x] or " "
                style = self.styles[y][x] if char != " " else None
                if style != run_style and run:
                    segments.append(Segment("".join(run), Style.parse(run_style) if run_style else None))
                    run = []
                run_style = style
                run.append(char)
            if run:
                segments.append(Segment("".join(run), Style.parse(run_style) if run_style else None))
            rows.append(segments)
        return rows


def clip_segment(
    x0: float, y0: float, x1: float, y1: float, w: int, h: int
) -> tuple[float, float, float, float] | None:
    """Liang-Barsky against [0,w) x [0,h) — clip BEFORE Bresenham so a deep
    zoom never rasterizes miles of off-screen line."""
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x0), (dx, w - 1 - x0),
        (-dy, y0), (dy, h - 1 - y0),
    ):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    return (x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy)


def draw_line(buf: CellBuffer, x0: int, y0: int, x1: int, y1: int, style: str) -> None:
    """Rasterize an edge as a thin stroke: Bresenham over CELLS (endpoints
    arrive in dot coordinates), each step stamping ─ │ ╱ or ╲ by the step's
    direction so consecutive cells read as one continuous line."""
    cx, cy = x0 // 2, y0 // 4
    ex, ey = x1 // 2, y1 // 4
    dx = abs(ex - cx)
    dy = -abs(ey - cy)
    sx = 1 if cx < ex else -1
    sy = 1 if cy < ey else -1
    if cx == ex and cy == ey:
        buf.set_line(cx, cy, LINE_H if abs(x1 - x0) >= abs(y1 - y0) else LINE_V, style)
        return
    err = dx + dy
    while not (cx == ex and cy == ey):
        e2 = 2 * err
        stepped_x = stepped_y = False
        if e2 >= dy:
            err += dy
            cx += sx
            stepped_x = True
        if e2 <= dx:
            err += dx
            cy += sy
            stepped_y = True
        if stepped_x and stepped_y:
            char = LINE_DOWN if sx == sy else LINE_UP  # y grows downward
        elif stepped_x:
            char = LINE_H
        else:
            char = LINE_V
        buf.set_line(cx, cy, char, style)


def concept_color(concept: str) -> str:
    digest = int(hashlib.sha1(concept.encode()).hexdigest()[:8], 16)
    return CONCEPT_COLOR_POOL[digest % len(CONCEPT_COLOR_POOL)]


def node_color(node: VNode, color_by: str, latest: str = "") -> str:
    if color_by == "concept" and node.concepts:
        return concept_color(node.concepts[0])
    if color_by == "recency":
        # 3 buckets, no gradients — gradients band in 256-color terminals
        if not node.first_seen or not latest:
            return "bright_black"
        if node.first_seen == latest:
            return "bold white"
        return "white" if node.first_seen >= min(latest, node.first_seen) else "dim white"
    return NODE_COLORS.get(node.type, "white")


def render_frame(
    vg: VisibleGraph,
    cam: Camera,
    cells_w: int,
    cells_h: int,
    *,
    selected: str | None = None,
    hovered: str | None = None,
    color_by: str = "type",
) -> tuple[list[list[Segment]], list[PlacedNode]]:
    """Rasterize one frame. Z-order: edge strokes < node glyphs < labels <
    selection ring. Returns the rows plus the on-screen node positions
    (in dots) that hit-testing consumes."""
    buf = CellBuffer(cells_w, cells_h)
    dots_w, dots_h = dot_size(cells_w, cells_h)
    positions = {
        n.id: world_to_dot(cam, dots_w, dots_h, n.x, n.y) for n in vg.nodes
    }
    latest = max((n.first_seen for n in vg.nodes if n.first_seen), default="")

    for edge in vg.edges:
        a, b = positions.get(edge.src), positions.get(edge.dst)
        if a is None or b is None:
            continue
        clipped = clip_segment(a[0], a[1], b[0], b[1], dots_w, dots_h)
        if clipped is None:
            continue
        style = EDGE_COLORS.get(edge.type, "dim white")
        draw_line(buf, round(clipped[0]), round(clipped[1]),
                  round(clipped[2]), round(clipped[3]), style)

    placed: list[PlacedNode] = []
    on_screen: list[VNode] = []
    for node in vg.nodes:
        dx, dy = positions[node.id]
        if not (0 <= dx < dots_w and 0 <= dy < dots_h):
            continue
        placed.append(PlacedNode(node_id=node.id, dot_x=dx, dot_y=dy))
        on_screen.append(node)
        color = node_color(node, color_by, latest)
        glyph = NODE_GLYPHS.get(node.type, "●")
        cx, cy = int(dx // 2), int(dy // 4)
        if node.id == selected:
            buf.set_char(cx, cy, glyph, f"reverse bold {_base_color(color)}")
            if cx > 0:
                buf.set_char(cx - 1, cy, "(", "bold white")
            buf.set_char(cx + 1, cy, ")", "bold white")
        elif node.id == hovered:
            buf.set_char(cx, cy, glyph, f"bold {_base_color(color)}")
        else:
            buf.set_char(cx, cy, glyph, color)

    show_labels = cam.scale >= LABEL_SCALE
    by_priority = sorted(
        on_screen,
        key=lambda n: (n.id == selected, n.id == hovered, n.count, n.label),
        reverse=True,
    )
    for node in by_priority:
        if not (show_labels or node.id in (selected, hovered) or node.type == "supernode"):
            continue
        dx, dy = positions[node.id]
        cx, cy = int(dx // 2) + 2, int(dy // 4)
        label = node.label[:LABEL_MAX] + ("…" if len(node.label) > LABEL_MAX else "")
        span = range(cx, cx + len(label))
        if not all(buf.char_free(x, cy) for x in span if x < cells_w):
            continue  # greedy collision mask: lower priority label skipped
        for offset, char in enumerate(label):
            if cx + offset >= cells_w:
                break
            buf.set_char(cx + offset, cy, char, "bright_black" if node.id not in (selected, hovered) else "bold white")
    return buf.to_segments(), placed


def _base_color(style: str) -> str:
    return style.split()[-1]


def hit_test(
    placed: list[PlacedNode], dot_x: float, dot_y: float, radius: float = HIT_RADIUS_DOTS
) -> str | None:
    """Nearest on-screen node within radius (dot space is ~square, so this
    is a true Euclidean nearest)."""
    best: str | None = None
    best_dist = radius
    for node in placed:
        dist = math.hypot(node.dot_x - dot_x, node.dot_y - dot_y)
        if dist <= best_dist:
            best = node.node_id
            best_dist = dist
    return best


def fuzzy_match(nodes: list[GraphNode], query: str, limit: int = 20) -> list[GraphNode]:
    """Subsequence scorer: all query chars must appear in order; contiguity
    and prefix matches score higher."""
    query = query.strip().lower()
    if not query:
        return []
    scored: list[tuple[float, str, GraphNode]] = []
    for node in nodes:
        haystack = f"{node.label} {node.id}".lower()
        score = _subsequence_score(haystack, query)
        if score > 0:
            scored.append((score, node.id, node))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [node for _, _, node in scored[:limit]]


def _subsequence_score(haystack: str, query: str) -> float:
    index = 0
    score = 0.0
    streak = 0
    for char in query:
        found = haystack.find(char, index)
        if found < 0:
            return 0.0
        streak = streak + 1 if found == index else 1
        score += streak
        if found == 0:
            score += 2  # prefix bonus
        index = found + 1
    return score / (1 + len(haystack) / 40)


def snap_to_event(events: list[str], fraction: float) -> int:
    """Scrubber state IS an event index — never a continuous time."""
    if not events:
        return 0
    return min(max(round(fraction * (len(events) - 1)), 0), len(events) - 1)


def render_scrubber(events: list[str], index: int | None, width: int) -> Text:
    """Two rows: a tick-per-event track with the cursor, then the label."""
    track = ["─"] * max(width, 10)
    for i in range(len(events)):
        pos = 0 if len(events) == 1 else round(i / (len(events) - 1) * (len(track) - 1))
        track[pos] = "┬"
    if events:
        cursor = len(track) - 1 if index is None else (
            0 if len(events) == 1 else round(index / (len(events) - 1) * (len(track) - 1))
        )
        track[cursor] = "◉"
    if index is None:
        label = f"(live) {len(events)} search event(s)   [ / ] scrub, end = live"
    else:
        label = f"as of {events[index]}  (event {index + 1}/{len(events)})"
    text = Text("".join(track) + "\n")
    text.append(label, style="bold" if index is not None else "dim")
    return text


def node_detail_renderables(graph: KnowledgeGraph, node_id: str) -> list:
    """Right-panel content for one node — mirror of watch.py's candidate
    detail builder: plain data, grouped neighbors, provenance."""
    nodes = graph.node_map()
    node = nodes.get(node_id)
    if node is None:
        return [Text(f"{node_id} (not in graph)")]
    parts: list = []
    header = Text()
    header.append(f"{node.label}\n", style="bold")
    header.append(f"{node.type}", style="cyan")
    if node.concepts:
        header.append("  " + ", ".join(node.concepts), style="magenta")
    parts.append(header)
    meta = Text()
    if node.first_seen:
        meta.append(f"first seen {node.first_seen}\n", style="dim")
    if node.superseded_at:
        meta.append(f"superseded {node.superseded_at}\n", style="red")
    for key, value in node.data.items():
        if value not in (None, "", [], {}):
            meta.append(f"{key}: {value}\n")
    if meta.plain:
        parts.append(meta)
    by_kind: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.src == node_id and edge.dst in nodes:
            by_kind.setdefault(edge.type, []).append(nodes[edge.dst].label)
        elif edge.dst == node_id and edge.src in nodes:
            by_kind.setdefault(f"{edge.type} <-", []).append(nodes[edge.src].label)
    for kind, labels in sorted(by_kind.items()):
        listing = ", ".join(sorted(labels)[:8])
        if len(labels) > 8:
            listing += f" +{len(labels) - 8}"
        parts.append(Text(f"{kind}: {listing}", style="dim"))
    if node.type == "search":
        parts.append(Text("enter: open this search's candidates", style="italic dim"))
    return parts


def search_dir_from_run_ref(config: Config, run_ref: str) -> Path | None:
    """search node -> runs/<run-id>/searches/<search-id>, if it still exists."""
    run_id, _, search_id = run_ref.partition("/")
    if not run_id or not search_id:
        return None
    search_dir = config.paths.runs_dir / run_id / "searches" / search_id
    return search_dir if search_dir.exists() else None


# --- Textual widgets ---

from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual import events  # noqa: E402
from textual.containers import Vertical  # noqa: E402
from textual.message import Message  # noqa: E402
from textual.screen import Screen  # noqa: E402
from textual.strip import Strip  # noqa: E402
from textual.widget import Widget  # noqa: E402
from textual.widgets import (  # noqa: E402
    Footer, Header, Input, OptionList, RichLog, Select, SelectionList, Static,
)
from textual.widgets.option_list import Option  # noqa: E402

from hillclimb.watch import REFRESH_S, _mouse_event_x, _mouse_event_y  # noqa: E402

DRAG_THRESHOLD = 1  # cells before a press becomes a pan instead of a click
PAN_STEP_FRACTION = 8  # arrow keys pan viewport/8


class GraphCanvas(Widget):
    """The zoomable/pannable braille canvas. All state that must survive a
    data refresh (camera, hover) lives here, never derived from data."""

    can_focus = True

    class NodeSelected(Message):
        def __init__(self, node_id: str | None):
            super().__init__()
            self.node_id = node_id

    class NodeActivated(Message):
        def __init__(self, node_id: str):
            super().__init__()
            self.node_id = node_id

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera: Camera | None = None
        self.color_by = "type"
        self.selected: str | None = None
        self._graph: KnowledgeGraph | None = None
        self._visible: VisibleGraph | None = None
        self._collapsed = False
        self._strips: list[Strip] = []
        self._placed: list[PlacedNode] = []
        self._hover: str | None = None
        self._press: tuple[int, int] | None = None
        self._dragged = False

    @property
    def dragging(self) -> bool:
        return self._press is not None and self._dragged

    def set_graph(self, graph: KnowledgeGraph) -> None:
        self._graph = graph
        if self.selected is not None and self.selected not in graph.node_map():
            self.selected = None
            self.post_message(self.NodeSelected(None))
        self.rebuild()

    def set_camera(self, camera: Camera) -> None:
        self.camera = camera
        self.rebuild()

    def _bounds(self) -> WorldBounds:
        nodes = self._visible.nodes if self._visible else ()
        return world_bounds(list(nodes))

    def fit(self) -> None:
        if self._graph is None or not self._graph.nodes:
            return
        # fit against the uncollapsed extents so LOD flips can't shrink the view
        all_nodes = apply_lod(self._graph, False).nodes
        dots_w, dots_h = dot_size(max(self.size.width, 1), max(self.size.height, 1))
        self.camera = fit_camera(world_bounds(list(all_nodes)), dots_w, dots_h)
        self.rebuild()

    def rebuild(self) -> None:
        if self._graph is None or self.size.width <= 0 or self.size.height <= 0:
            return
        if self.camera is None:
            all_nodes = apply_lod(self._graph, False).nodes
            dots_w, dots_h = dot_size(self.size.width, self.size.height)
            self.camera = fit_camera(world_bounds(list(all_nodes)), dots_w, dots_h)
        self._collapsed = lod_collapsed(self.camera.scale, self._collapsed)
        self._visible = apply_lod(self._graph, self._collapsed)
        rows, self._placed = render_frame(
            self._visible, self.camera, self.size.width, self.size.height,
            selected=self.selected, hovered=self._hover, color_by=self.color_by,
        )
        self._strips = [Strip(row, cell_length=self.size.width) for row in rows]
        self.refresh()

    def render_line(self, y: int) -> Strip:
        if y < len(self._strips):
            return self._strips[y]
        return Strip.blank(self.size.width)

    def on_resize(self, event: events.Resize) -> None:
        self.rebuild()

    # -- interaction --

    def _event_dots(self, event: events.MouseEvent) -> tuple[float, float]:
        cx = _mouse_event_x(event) - self.region.x
        cy = _mouse_event_y(event) - self.region.y
        return cx * 2 + 1, cy * 4 + 2  # center of the cell, in dots

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._press = (_mouse_event_x(event), _mouse_event_y(event))
        self._dragged = False
        self.capture_mouse()
        self.focus()
        event.prevent_default()
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._press is not None:
            x, y = _mouse_event_x(event), _mouse_event_y(event)
            dx, dy = x - self._press[0], y - self._press[1]
            if self._dragged or max(abs(dx), abs(dy)) >= DRAG_THRESHOLD:
                self._dragged = True
                if self.camera is not None:
                    # drag right -> world moves with the pointer (camera left);
                    # cells -> dots is the one place the 2x4 factor leaks in
                    self.camera = pan_camera(self.camera, -dx * 2, -dy * 4, self._bounds())
                self._press = (x, y)
                self.rebuild()
            event.prevent_default()
            event.stop()
            return
        dot_x, dot_y = self._event_dots(event)
        hover = hit_test(self._placed, dot_x, dot_y)
        if hover != self._hover:
            self._hover = hover
            self.rebuild()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        was_click = self._press is not None and not self._dragged
        self._press = None
        self._dragged = False
        self.release_mouse()
        if was_click:
            dot_x, dot_y = self._event_dots(event)
            node_id = hit_test(self._placed, dot_x, dot_y)
            if node_id != self.selected:
                self.selected = node_id
                self.rebuild()
                self.post_message(self.NodeSelected(node_id))
            elif node_id is not None:
                self.post_message(self.NodeActivated(node_id))
        event.prevent_default()
        event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self._zoom_at_event(event, ZOOM_FACTOR)
        event.prevent_default()
        event.stop()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self._zoom_at_event(event, 1 / ZOOM_FACTOR)
        event.prevent_default()
        event.stop()

    def _zoom_at_event(self, event: events.MouseEvent, factor: float) -> None:
        if self.camera is None:
            return
        dots_w, dots_h = dot_size(self.size.width, self.size.height)
        dot_x, dot_y = self._event_dots(event)
        self.camera = zoom_about(self.camera, dots_w, dots_h, dot_x, dot_y, factor, self._bounds())
        self.rebuild()

    def zoom(self, factor: float) -> None:
        if self.camera is None:
            return
        dots_w, dots_h = dot_size(self.size.width, self.size.height)
        self.camera = zoom_about(
            self.camera, dots_w, dots_h, dots_w / 2, dots_h / 2, factor, self._bounds()
        )
        self.rebuild()

    def pan(self, dx_fraction: float, dy_fraction: float) -> None:
        if self.camera is None:
            return
        dots_w, dots_h = dot_size(self.size.width, self.size.height)
        self.camera = pan_camera(
            self.camera, dx_fraction * dots_w, dy_fraction * dots_h, self._bounds()
        )
        self.rebuild()

    def center_on(self, node_id: str, *, select: bool = True) -> None:
        if self._graph is None:
            return
        node = self._graph.node_map().get(node_id)
        if node is None:
            return
        x, y = node_pos(node)
        scale = max(self.camera.scale if self.camera else LABEL_SCALE, LABEL_SCALE)
        self.camera = Camera(cx=x, cy=y, scale=scale)
        if select:
            self.selected = node_id
            self.post_message(self.NodeSelected(node_id))
        self.rebuild()

    def zoom_to_members(self, member_ids: tuple[str, ...]) -> None:
        if self._graph is None or not member_ids:
            return
        nodes = [n for n in self._graph.nodes if n.id in set(member_ids)]
        if not nodes:
            return
        vnodes = [
            VNode(id=n.id, type=n.type, label=n.label, x=node_pos(n)[0], y=node_pos(n)[1])
            for n in nodes
        ]
        dots_w, dots_h = dot_size(self.size.width, self.size.height)
        camera = fit_camera(world_bounds(vnodes), dots_w, dots_h)
        # land past the LOD exit threshold so the supernode actually expands
        self.camera = replace(camera, scale=max(camera.scale, COLLAPSE_EXIT * 1.1))
        self.rebuild()


class TimeScrubber(Static):
    """Discrete scrubber over search-finish events; state IS an event index
    (None = live)."""

    class TimeChanged(Message):
        def __init__(self, index: int | None):
            super().__init__()
            self.index = index

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events_list: list[str] = []
        self.index: int | None = None
        self._scrub_active = False

    def set_events(self, events_list: list[str]) -> None:
        if events_list != self.events_list:
            # clamp a historical cursor if events changed under it
            if self.index is not None and self.index >= len(events_list):
                self.index = len(events_list) - 1 if events_list else None
            self.events_list = list(events_list)
            self._refresh_scrubber()

    def set_index(self, index: int | None) -> None:
        self.index = index
        self._refresh_scrubber()

    def _refresh_scrubber(self) -> None:
        width = self.size.width or 60
        self.update(render_scrubber(self.events_list, self.index, width))

    def on_resize(self, event: events.Resize) -> None:
        self._refresh_scrubber()

    def _index_from_event(self, event: events.MouseEvent) -> int:
        x = _mouse_event_x(event) - self.region.x
        width = max(self.size.width - 1, 1)
        return snap_to_event(self.events_list, x / width)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if not self.events_list:
            return
        self._scrub_active = True
        self.capture_mouse()
        self._apply(event)
        event.prevent_default()
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if not self._scrub_active:
            return
        self._apply(event)
        event.prevent_default()
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if not self._scrub_active:
            return
        self._scrub_active = False
        self.release_mouse()
        event.prevent_default()
        event.stop()

    def _apply(self, event: events.MouseEvent) -> None:
        index = self._index_from_event(event)
        # snapping to the last event means "live"
        new_index = None if index == len(self.events_list) - 1 else index
        if new_index != self.index:
            self.index = new_index
            self._refresh_scrubber()
            self.post_message(self.TimeChanged(new_index))

    def step(self, delta: int) -> None:
        if not self.events_list:
            return
        current = len(self.events_list) - 1 if self.index is None else self.index
        target = min(max(current + delta, 0), len(self.events_list) - 1)
        self.set_index(None if target == len(self.events_list) - 1 else target)
        self.post_message(self.TimeChanged(self.index))


class ConceptSidebar(Vertical):
    """Concept filter checkboxes plus the color-by selector."""

    class FiltersChanged(Message):
        def __init__(self, enabled: frozenset[str] | None, color_by: str):
            super().__init__()
            self.enabled = enabled
            self.color_by = color_by

    def compose(self) -> ComposeResult:
        yield Static(" concepts", id="sidebar-title")
        yield SelectionList[str](id="concept-filters")
        yield Static(" color by", id="sidebar-color-title")
        yield Select(
            [("node type", "type"), ("concept", "concept"), ("recency", "recency")],
            value="type", allow_blank=False, id="color-by",
        )

    def set_concepts(self, concepts: list[str]) -> None:
        selection = self.query_one("#concept-filters", SelectionList)
        current = set(selection.selected) or None
        known = {option.value for option in selection._values} if hasattr(selection, "_values") else set()
        if set(concepts) == known:
            return
        selection.clear_options()
        for concept in sorted(concepts):
            selected = current is None or concept in current
            selection.add_option((concept, concept, selected))

    def _emit(self) -> None:
        selection = self.query_one("#concept-filters", SelectionList)
        color_by = self.query_one("#color-by", Select).value
        enabled = frozenset(selection.selected)
        total = len(selection._values) if hasattr(selection, "_values") else 0
        # everything selected = no filter at all
        self.post_message(self.FiltersChanged(
            None if total and len(enabled) == total else enabled,
            str(color_by),
        ))

    def on_selection_list_selected_changed(self, event: SelectionList.SelectedChanged) -> None:
        self._emit()
        event.stop()

    def on_select_changed(self, event: Select.Changed) -> None:
        self._emit()
        event.stop()


class GraphScreen(Screen):
    """The knowledge graph: canvas + concept sidebar + node detail + time
    scrubber. Reached via `g` in watch or `hillclimb knowledge graph`."""

    BINDINGS = [
        Binding("escape", "dismiss_or_back", "back"),
        Binding("enter", "activate", "open", priority=True),
        Binding("+,=", "zoom_in", "zoom in"),
        Binding("-", "zoom_out", "zoom out"),
        Binding("f,0", "fit", "fit"),
        Binding("up", "pan_up", show=False),
        Binding("down", "pan_down", show=False),
        Binding("left", "pan_left", show=False),
        Binding("right", "pan_right", show=False),
        Binding("slash", "search", "search"),
        Binding("left_square_bracket", "scrub_back", "back in time"),
        Binding("right_square_bracket", "scrub_forward", "forward"),
        Binding("end", "scrub_live", "live"),
        Binding("c", "toggle_sidebar", "concepts"),
        Binding("q", "app.quit", "quit"),
    ]

    DEFAULT_CSS = """
    GraphScreen #concept-sidebar { dock: left; width: 26; display: none; background: $surface; }
    GraphScreen #concept-filters { height: auto; max-height: 70%; }
    GraphScreen #node-detail { dock: right; width: 42; display: none; padding: 0 1; }
    GraphScreen #time-scrubber { dock: bottom; height: 2; background: $surface; padding: 0 1; }
    GraphScreen #graph-search { dock: top; display: none; }
    GraphScreen #graph-search-results { dock: top; display: none; max-height: 10; }
    GraphScreen #graph-canvas { width: 1fr; height: 1fr; }
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._graph: KnowledgeGraph | None = None
        self._graph_mtime = 0.0
        self._filters: frozenset[str] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Input(placeholder="find node…", id="graph-search")
        yield OptionList(id="graph-search-results")
        yield ConceptSidebar(id="concept-sidebar")
        yield RichLog(id="node-detail", wrap=True, markup=False, auto_scroll=False)
        yield TimeScrubber(id="time-scrubber")
        yield GraphCanvas(id="graph-canvas")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_data()
        self.query_one("#graph-canvas", GraphCanvas).focus()
        self.set_interval(REFRESH_S, self.refresh_data)

    def _knowledge_dir(self) -> Path | None:
        from hillclimb.api import resolve_knowledge_dir

        return resolve_knowledge_dir(self.config)

    def refresh_data(self) -> None:
        from hillclimb.graph import graph_path, load_or_build_graph

        canvas = self.query_one("#graph-canvas", GraphCanvas)
        if canvas.dragging:
            return  # no disk I/O mid-drag; in-memory rebuilds keep flowing
        knowledge_dir = self._knowledge_dir()
        if knowledge_dir is None:
            self.notify("learning is disabled — no knowledge graph", severity="warning")
            return
        path = graph_path(knowledge_dir)
        mtime = path.stat().st_mtime if path.exists() else -1.0
        if self._graph is not None and mtime == self._graph_mtime:
            return
        self._graph = load_or_build_graph(knowledge_dir)
        self._graph_mtime = path.stat().st_mtime if path.exists() else -1.0
        self.query_one("#concept-sidebar", ConceptSidebar).set_concepts(
            [n.label for n in self._graph.nodes if n.type == "concept"]
        )
        self.query_one("#time-scrubber", TimeScrubber).set_events(self._graph.events)
        self._apply_view()

    def _apply_view(self) -> None:
        if self._graph is None:
            return
        scrubber = self.query_one("#time-scrubber", TimeScrubber)
        t = None
        if scrubber.index is not None and scrubber.events_list:
            t = scrubber.events_list[scrubber.index]
        view = graph_at(self._graph, t)
        view = filter_concepts(view, self._filters)
        self.query_one("#graph-canvas", GraphCanvas).set_graph(view)

    # -- messages --

    def on_graph_canvas_node_selected(self, message: GraphCanvas.NodeSelected) -> None:
        detail = self.query_one("#node-detail", RichLog)
        canvas = self.query_one("#graph-canvas", GraphCanvas)
        if message.node_id is None or canvas._graph is None:
            detail.styles.display = "none"
            detail.clear()
            return
        detail.styles.display = "block"
        detail.clear()
        source = canvas._graph
        for renderable in node_detail_renderables(source, message.node_id):
            detail.write(renderable, expand=True)

    def on_graph_canvas_node_activated(self, message: GraphCanvas.NodeActivated) -> None:
        self._activate(message.node_id)

    def on_time_scrubber_time_changed(self, message: TimeScrubber.TimeChanged) -> None:
        self._apply_view()

    def on_concept_sidebar_filters_changed(self, message: ConceptSidebar.FiltersChanged) -> None:
        self._filters = message.enabled
        canvas = self.query_one("#graph-canvas", GraphCanvas)
        canvas.color_by = message.color_by
        self._apply_view()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "graph-search" or self._graph is None:
            return
        matches = fuzzy_match(self._graph.nodes, event.value, limit=1)
        self._hide_search()
        if matches:
            self.query_one("#graph-canvas", GraphCanvas).center_on(matches[0].id)
        event.stop()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "graph-search" or self._graph is None:
            return
        results = self.query_one("#graph-search-results", OptionList)
        results.clear_options()
        matches = fuzzy_match(self._graph.nodes, event.value, limit=8)
        if matches:
            results.styles.display = "block"
            for node in matches:
                results.add_option(Option(f"{node.label}  ({node.type})", id=node.id))
        else:
            results.styles.display = "none"
        event.stop()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "graph-search-results":
            return
        self._hide_search()
        if event.option.id:
            self.query_one("#graph-canvas", GraphCanvas).center_on(event.option.id)
        event.stop()

    # -- actions --

    def _canvas(self) -> GraphCanvas:
        return self.query_one("#graph-canvas", GraphCanvas)

    def action_zoom_in(self) -> None:
        self._canvas().zoom(ZOOM_FACTOR)

    def action_zoom_out(self) -> None:
        self._canvas().zoom(1 / ZOOM_FACTOR)

    def action_fit(self) -> None:
        self._canvas().fit()

    def action_pan_up(self) -> None:
        self._canvas().pan(0, -1 / PAN_STEP_FRACTION)

    def action_pan_down(self) -> None:
        self._canvas().pan(0, 1 / PAN_STEP_FRACTION)

    def action_pan_left(self) -> None:
        self._canvas().pan(-1 / PAN_STEP_FRACTION, 0)

    def action_pan_right(self) -> None:
        self._canvas().pan(1 / PAN_STEP_FRACTION, 0)

    def action_scrub_back(self) -> None:
        self.query_one("#time-scrubber", TimeScrubber).step(-1)

    def action_scrub_forward(self) -> None:
        self.query_one("#time-scrubber", TimeScrubber).step(1)

    def action_scrub_live(self) -> None:
        scrubber = self.query_one("#time-scrubber", TimeScrubber)
        scrubber.set_index(None)
        self._apply_view()

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#concept-sidebar", ConceptSidebar)
        shown = sidebar.styles.display != "none"
        sidebar.styles.display = "none" if shown else "block"

    def action_search(self) -> None:
        search = self.query_one("#graph-search", Input)
        search.styles.display = "block"
        search.value = ""
        search.focus()

    def _hide_search(self) -> None:
        self.query_one("#graph-search", Input).styles.display = "none"
        self.query_one("#graph-search-results", OptionList).styles.display = "none"
        self._canvas().focus()

    def action_activate(self) -> None:
        canvas = self._canvas()
        if canvas.selected is not None:
            self._activate(canvas.selected)

    def _activate(self, node_id: str) -> None:
        canvas = self._canvas()
        if node_id.startswith("supernode:") and canvas._visible is not None:
            supernode = next((n for n in canvas._visible.nodes if n.id == node_id), None)
            if supernode is not None:
                canvas.zoom_to_members(supernode.members)
            return
        if node_id.startswith("search:"):
            search_dir = search_dir_from_run_ref(self.config, node_id.removeprefix("search:"))
            if search_dir is None:
                self.notify("search artifacts no longer on disk", severity="warning")
                return
            from hillclimb.watch import CandidateScreen

            self.app.push_screen(CandidateScreen(self.config, search_dir))

    def action_dismiss_or_back(self) -> None:
        search = self.query_one("#graph-search", Input)
        if search.styles.display != "none":
            self._hide_search()
            return
        canvas = self._canvas()
        if canvas.selected is not None:
            canvas.selected = None
            canvas.rebuild()
            self.on_graph_canvas_node_selected(GraphCanvas.NodeSelected(None))
            return
        self.app.pop_screen()


class GraphApp(App):
    """Standalone shell for `hillclimb knowledge graph` — same screen the
    watch TUI reaches via `g`."""

    TITLE = "hillclimb knowledge graph"

    def __init__(self, config: Config | None = None):
        super().__init__()
        self.config = config or Config.load()

    def on_mount(self) -> None:
        self.push_screen(GraphScreen(self.config))
