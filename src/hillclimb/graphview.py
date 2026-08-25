"""Interactive knowledge-graph screen for the watch TUI.

Strictly a *viewer* over `knowledge/graph.json` (built by graph.py): rotate,
pan, zoom, click nodes, scrub through time (one tick per finished search,
or per graph change — `g` toggles),
filter and color by concept. The graph renders as a true-3D scene through
plotui (Rust rasterizer → Kitty pixel graphics: placeholder placement in
kitty/Ghostty, direct placement in iTerm2 ≥ 3.5/WezTerm/Konsole, a support
notice elsewhere); node positions come from the 3D spring layout cached in
graph.json (`pos3`). The pure functions at the top carry LOD, color, marker
shape, and label geometry so they stay testable without driving Textual; the widgets
below are thin shells in the style of watch.py.

Zoom drives semantic LOD: below COLLAPSE_ENTER the concept-bearing nodes fold
into concept supernodes (hysteresis so the boundary doesn't flicker), and
labels appear past LABEL_ZOOM as a text overlay spliced over the image by
plotui's PlotWidget.
"""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from rich.color import Color as RichColor
from rich.table import Table
from rich.style import Style
from rich.text import Text

from hillclimb.config import Config
from hillclimb.graph import GraphNode, KnowledgeGraph, fuzzy_match, graph_at
from hillclimb.theme import HILLCLIMB_CSS, apply_theme, themed_plot

# --- pure data layer ---

ZOOM_FACTOR = 1.25
# plotui camera zoom units: 1.0 = the whole graph auto-framed to the widget.
LABEL_ZOOM = 1.6       # labels appear past this zoom
COLLAPSE_ENTER = 0.55  # below this zoom entities fold into concept supernodes
COLLAPSE_EXIT = COLLAPSE_ENTER * 1.4
LABEL_MAX = 18
NODE_SIZE = 3.5        # base node radius in framebuffer px (pre-supersampling)
SUPERNODE_SIZE_CAP = 9.0

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
    "generalizes": "dim green",
}
# A marker silhouette per node type (the plotui `node_shapes` names), so the
# type reads at a glance without matching a colour against a legend. Same
# table as the hillclimb.sh graph; anchors (family/problem) are rings,
# searches point, libraries are blocks, claims are the open twin of the
# technique they are about, operators recede to dots.
NODE_SHAPES = {
    "family": "ring", "problem": "ring", "search": "triangle", "library": "square",
    "concept": "disc", "technique": "diamond", "claim": "diamond-open", "operator": "dot",
    "supernode": "disc",
}
# The legend row, in the website's order. Each entry toggles its type off and
# on; the glyph is the closest text stand-in for the plotui marker.
LEGEND_TYPES = (
    "claim", "concept", "family", "library", "operator", "problem", "search", "technique",
)
LEGEND_GLYPHS = {
    "disc": "●", "ring": "◉", "square": "■", "triangle": "▲",
    "diamond": "◆", "diamond-open": "◇", "dot": "•",
}
# fixed 8-color pool for concept coloring — theme-safe, no RGB gradients
CONCEPT_COLOR_POOL = (
    "cyan", "magenta", "green", "yellow", "blue", "red", "bright_cyan", "bright_magenta",
)


@dataclass(frozen=True)
class VNode:
    """A drawable node: original graph node(s) resolved to a world position.
    count > 1 marks a concept supernode."""

    id: str
    type: str
    label: str
    x: float
    y: float
    z: float = 0.0
    concepts: tuple[str, ...] = ()
    first_seen: str = ""
    count: int = 1
    style: str = ""  # label style when not selected/hovered; "" = the dim default
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


def fallback_pos(node_id: str) -> tuple[float, float, float]:
    """Deterministic spherical placement for nodes the index left unplaced
    (engine-only installs without networkx)."""
    digest = int(hashlib.sha1(node_id.encode()).hexdigest()[:12], 16)
    theta = (digest % 3600) / 3600 * 2 * math.pi
    phi = ((digest // 3600) % 1800) / 1800 * math.pi
    radius = 0.4 + ((digest // (3600 * 1800)) % 600) / 1000
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def node_pos(node: GraphNode) -> tuple[float, float, float]:
    return node.pos3 if node.pos3 is not None else fallback_pos(node.id)


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


def filter_types(graph: KnowledgeGraph, hidden: frozenset[str]) -> KnowledgeGraph:
    """Legend filter: drop every node of a hidden type, and the edges that
    touched one. Runs before LOD, so hiding `concept` also empties the
    supernodes' labels but leaves the entities grouped under them."""
    if not hidden:
        return graph
    nodes = [n for n in graph.nodes if n.type not in hidden]
    ids = {n.id for n in nodes}
    edges = [e for e in graph.edges if e.src in ids and e.dst in ids]
    return KnowledgeGraph(
        schema_version=graph.schema_version, built_at=graph.built_at,
        events=graph.events, nodes=nodes, edges=edges,
    )


LEGEND_ROW, LEGEND_COL = 1, 1  # where the legend's first line sits on the canvas
LEGEND_WIDTH = 2 + 2 + max(len(t) for t in LEGEND_TYPES)  # "1 " + glyph + " " + name


def legend_spans(hidden: frozenset[str]) -> list[tuple[int, int, str, str]]:
    """The legend as text-overlay spans `(row, col, text, style)` — one line
    per type from LEGEND_ROW down, drawn over the plot the same way node
    labels are, so it costs the canvas no columns. Entries carry their hotkey
    (1-8, which hides/shows the type) and render in the truecolor the plot
    paints the type with; hidden types go dim and lose their glyph, so the
    legend itself shows what the canvas is not drawing."""
    spans: list[tuple[int, int, str, str]] = []
    for index, type_ in enumerate(LEGEND_TYPES):
        row = LEGEND_ROW + index
        spans.append((row, LEGEND_COL, f"{index + 1} ", "dim"))
        if type_ in hidden:
            spans.append((row, LEGEND_COL + 2, f"  {type_}", "dim strike"))
        else:
            r, g, b = style_to_rgb(NODE_COLORS.get(type_, "white"))
            glyph = LEGEND_GLYPHS[NODE_SHAPES.get(type_, "disc")]
            spans.append((row, LEGEND_COL + 2, f"{glyph} {type_}", f"rgb({r},{g},{b})"))
    return spans


def legend_entry_at(col: int, row: int) -> str | None:
    """The legend type under canvas cell `(col, row)`, or None off the legend."""
    index = row - LEGEND_ROW
    if 0 <= index < len(LEGEND_TYPES) and LEGEND_COL <= col < LEGEND_COL + LEGEND_WIDTH:
        return LEGEND_TYPES[index]
    return None


def lod_collapsed(zoom: float, was_collapsed: bool) -> bool:
    """Hysteresis so the boundary doesn't flicker while zooming."""
    if was_collapsed:
        return zoom < COLLAPSE_EXIT
    return zoom < COLLAPSE_ENTER


def apply_lod(graph: KnowledgeGraph, collapsed: bool) -> VisibleGraph:
    """Semantic zoom: collapsed folds every concept-bearing node into its
    primary-concept supernode (position = member centroid); concept nodes
    disappear into their supernodes; the skeleton stays atomic."""
    if not collapsed:
        return VisibleGraph(
            nodes=tuple(
                VNode(
                    id=n.id, type=n.type, label=n.label,
                    x=node_pos(n)[0], y=node_pos(n)[1], z=node_pos(n)[2],
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
            id=n.id, type=n.type, label=n.label,
            x=node_pos(n)[0], y=node_pos(n)[1], z=node_pos(n)[2],
            first_seen=n.first_seen,
        )
        for n in atoms
    ]
    for concept, members in sorted(groups.items()):
        if members:
            positions = [node_pos(m) for m in members]
            x, y, z = (sum(p[k] for p in positions) / len(positions) for k in range(3))
        else:
            x, y, z = fallback_pos(f"supernode:{concept}")
        nodes.append(VNode(
            id=f"supernode:{concept}", type="supernode",
            label=f"{concept} ({len(members)})", x=x, y=y, z=z,
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


# Saturated, high-contrast node colors — the standard-ANSI truecolors Rich
# returns (green #008000, yellow #808000/olive, cyan #008080/teal) read muddy
# over a dark background and blur together where nodes overlap. These vivid
# values give the graph the punchy per-category look of a molecular viewer.
_VIVID: dict[str, tuple[int, int, int]] = {
    "red": (239, 68, 68),
    "green": (34, 197, 94),
    "yellow": (234, 179, 8),
    "blue": (96, 165, 250),
    "magenta": (224, 33, 138),
    "cyan": (34, 211, 238),
    "white": (229, 231, 235),
    "black": (90, 99, 120),  # 'bright_black'/operator — visible on dark bg
}


@lru_cache(maxsize=256)
def style_to_rgb(style: str) -> tuple[int, int, int]:
    """A Rich style string ("bold yellow", "dim red", "bright_black") as the
    vivid RGB plotui wants. Base names resolve through the saturated `_VIVID`
    palette (not Rich's muddy standard-ANSI truecolors); ``bold``/``bright_``
    brightens, ``dim`` darkens."""
    parts = style.split()
    base = parts[-1].removeprefix("bright_")
    rgb = _VIVID.get(base)
    if rgb is None:  # a truecolor name/hex — trust Rich
        c = RichColor.parse(parts[-1]).get_truecolor()
        rgb = (c.red, c.green, c.blue)
    bright = "bold" in parts or parts[-1].startswith("bright_")
    factor = 1.25 if bright else 0.5 if "dim" in parts else 1.0
    return tuple(min(255, round(c * factor)) for c in rgb)


def node_shape(node: VNode) -> str:
    """The marker silhouette for a node's type; anything unmapped is a disc."""
    return NODE_SHAPES.get(node.type, "disc")


def node_size(node: VNode) -> float:
    """Supernodes grow with member count (log-scaled, capped); everything
    else renders at the base radius."""
    if node.type != "supernode":
        return NODE_SIZE
    return min(NODE_SIZE * (1.0 + 0.45 * math.log2(node.count + 1)), SUPERNODE_SIZE_CAP)


def build_plot(vg: VisibleGraph, *, color_by: str = "type", selected: str | None = None):
    """VisibleGraph -> a fresh plotui Plot plus the flat-index -> node-id list
    (a single Graph3d trace, so flat index == node order). Selection is set on
    the plot; camera state is the caller's to restore."""
    from hillclimb.theme import themed_plot

    ids = [n.id for n in vg.nodes]
    index_of = {node_id: i for i, node_id in enumerate(ids)}
    latest = max((n.first_seen for n in vg.nodes if n.first_seen), default="")
    plot = themed_plot()
    plot.set_show_box(False)  # the axis cube reads as clutter over a graph
    if vg.nodes:
        edges = [e for e in vg.edges if e.src in index_of and e.dst in index_of]
        plot.add_graph3d(
            [n.x for n in vg.nodes],
            [n.y for n in vg.nodes],
            [n.z for n in vg.nodes],
            edges=[(index_of[e.src], index_of[e.dst]) for e in edges],
            node_colors=[style_to_rgb(node_color(n, color_by, latest)) for n in vg.nodes],
            size=NODE_SIZE,
            node_sizes=[node_size(n) for n in vg.nodes],
            edge_colors=[style_to_rgb(EDGE_COLORS.get(e.type, "dim white")) for e in edges],
            node_shapes=[node_shape(n) for n in vg.nodes],
        )
        if selected in index_of:
            plot.set_selected(index_of[selected])
    return plot, ids


def place_labels(
    nodes: Sequence[VNode],
    projected: Sequence[tuple[float, float, float]],
    *,
    cols: int,
    rows: int,
    cell_px: tuple[int, int],
    zoom: float,
    selected: str | None = None,
    hovered: str | None = None,
) -> list[tuple[int, int, str, str]]:
    """Label spans for the overlay: `(row, col, text, style_str)` per label
    (see `place_labels_by_node` for the node each span belongs to).
    Rules carried over from the braille canvas: all labels past LABEL_ZOOM,
    always for selected/hovered/supernodes; priority order (selection, hover,
    supernode weight, nearer depth) with a greedy collision mask; text
    truncated to LABEL_MAX. Node cells themselves are masked so a label never
    covers another node's mark."""
    cell_w, cell_h = cell_px
    show_all = zoom >= LABEL_ZOOM
    on_screen: list[tuple[VNode, int, int, float]] = []
    claimed: set[tuple[int, int]] = set()
    for node, (sx, sy, depth) in zip(nodes, projected):
        col, row = int(sx // cell_w), int(sy // cell_h)
        if 0 <= col < cols and 0 <= row < rows:
            on_screen.append((node, col, row, depth))
            claimed.add((row, col))
    by_priority = sorted(
        on_screen,
        key=lambda item: (
            item[0].id == selected, item[0].id == hovered,
            item[0].count, -item[3], item[0].label,
        ),
        reverse=True,
    )
    return [span for span, _node_id in _place(by_priority, claimed, cols, show_all, selected, hovered)]


def place_labels_by_node(
    nodes: Sequence[VNode],
    projected: Sequence[tuple[float, float, float]],
    **kw,
) -> list[tuple[tuple[int, int, str, str], str]]:
    """`place_labels`, each span paired with its node id — so a label's cells
    can hit-test back to the node the way the mark itself does."""
    cell_w, cell_h = kw["cell_px"]
    cols, rows = kw["cols"], kw["rows"]
    show_all = kw["zoom"] >= LABEL_ZOOM
    selected, hovered = kw.get("selected"), kw.get("hovered")
    on_screen: list[tuple[VNode, int, int, float]] = []
    claimed: set[tuple[int, int]] = set()
    for node, (sx, sy, depth) in zip(nodes, projected):
        col, row = int(sx // cell_w), int(sy // cell_h)
        if 0 <= col < cols and 0 <= row < rows:
            on_screen.append((node, col, row, depth))
            claimed.add((row, col))
    by_priority = sorted(
        on_screen,
        key=lambda item: (
            item[0].id == selected, item[0].id == hovered,
            item[0].count, -item[3], item[0].label,
        ),
        reverse=True,
    )
    return _place(by_priority, claimed, cols, show_all, selected, hovered)


def _place(by_priority, claimed, cols, show_all, selected, hovered):
    placed: list[tuple[tuple[int, int, str, str], str]] = []
    for node, col, row, _depth in by_priority:
        if not (show_all or node.id in (selected, hovered) or node.type == "supernode"):
            continue
        label = node.label[:LABEL_MAX] + ("…" if len(node.label) > LABEL_MAX else "")
        start = col + 2
        cells = [(row, x) for x in range(start, min(start + len(label), cols))]
        if not cells or any(cell in claimed for cell in cells):
            continue  # greedy collision mask: lower-priority label skipped
        claimed.update(cells)
        label = label[: len(cells)]
        style = "bold white" if node.id in (selected, hovered) else node.style or "bright_black"
        placed.append(((row, start, label, style), node.id))
    return placed


def snap_to_event(events: list[str], fraction: float) -> int:
    """Scrubber state IS an event index — never a continuous time."""
    if not events:
        return 0
    return min(max(round(fraction * (len(events) - 1)), 0), len(events) - 1)


def event_stamp(event: str) -> str:
    """A search-finish timestamp for people: local time, to the minute. Not
    an ISO stamp? Shown as is."""
    from datetime import datetime

    try:
        return datetime.fromisoformat(event).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return event


SCRUBBER_STYLES = {  # component class -> fallback Rich style (outside Textual)
    "scrubber--elapsed": "bold cyan",
    "scrubber--remaining": "dim",
    "scrubber--cursor": "bold cyan",
    "scrubber--label": "bold",
    "scrubber--hint": "dim",
}


def render_scrubber(
    events: list[str], index: int | None, width: int,
    styles: dict[str, str | Style] | None = None,
    unit: str = "search",
) -> Text:
    """Two rows. A media-player track: the stretch up to the cursor is drawn
    solid in the accent, the rest thin and dim, one tick per event — a
    finished search, or every graph change when `unit` says so — and `●` at
    the cursor (the right end is live). Under it, what the cursor means —
    the event it rests on and its position in the sequence — with the key
    hints right-aligned and muted."""
    st = {**SCRUBBER_STYLES, **(styles or {})}
    width = max(width, 12)
    n = len(events)
    plural = f"{unit}es" if unit.endswith(("s", "ch")) else f"{unit}s"
    cursor = width - 1 if index is None else (0 if n <= 1 else round(index / (n - 1) * (width - 1)))
    ticks = {0 if n <= 1 else round(i / (n - 1) * (width - 1)) for i in range(n)}
    text = Text()
    for col in range(width):
        if col == cursor and n:
            text.append("●", st["scrubber--cursor"])
        elif col <= cursor:
            text.append("┿" if col in ticks else "━", st["scrubber--elapsed"])
        else:
            text.append("┼" if col in ticks else "─", st["scrubber--remaining"])
    text.append("\n")
    if not n:
        text.append("no finished searches yet", st["scrubber--hint"])
        return text
    if index is None:
        label = f"live · {n} {plural if n != 1 else unit}"
    else:
        label = f"as of {event_stamp(events[index])} · {unit} {index + 1} of {n}"
    hint = "drag · j/k step · g unit · end live"
    text.append(label, st["scrubber--label"])
    gap = width - len(label) - len(hint)
    if gap >= 2:
        text.append(" " * gap)
        text.append(hint, st["scrubber--hint"])
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

from plotui import Plot  # noqa: E402
from plotui.textual import PlotWidget  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual import events  # noqa: E402
from textual.containers import Vertical  # noqa: E402
from textual.message import Message  # noqa: E402
from textual.screen import Screen  # noqa: E402
from textual.widgets import (  # noqa: E402
    Footer, Input, OptionList, RichLog, Select, SelectionList, Static,
)
from textual.widgets.option_list import Option  # noqa: E402

from hillclimb.header import HillclimbHeader, TimezoneMixin  # noqa: E402
from hillclimb.keys import KEYS_BINDING, QUIT_BINDINGS, KeysMixin, KeysPanel  # noqa: E402
from hillclimb.watch import REFRESH_S, _mouse_event_x, _mouse_event_y  # noqa: E402


class GraphPlotWidget(PlotWidget):
    """The 3D plotui view of the knowledge graph. All state that must survive
    a data refresh (camera, selection, hover, LOD) lives here, never derived
    from data — the plotui Plot itself is rebuilt from the screen-supplied
    graph on every change (its traces are append-only), with the camera
    captured and restored around each rebuild.

    Interaction: drag rotates, shift-drag pans, scroll and +/- zoom (zoom
    drives the semantic-LOD collapse), click selects, re-click activates."""

    class NodeSelected(Message):
        def __init__(self, node_id: str | None):
            super().__init__()
            self.node_id = node_id

    class NodeActivated(Message):
        def __init__(self, node_id: str):
            super().__init__()
            self.node_id = node_id

    class TypeToggled(Message):
        """A legend entry was clicked."""

        def __init__(self, type_: str):
            super().__init__()
            self.type_ = type_

    def __init__(self, **kwargs):
        super().__init__(themed_plot(), **kwargs)
        self.color_by = "type"
        self.hidden_types: frozenset[str] = frozenset()
        self.selected: str | None = None
        self._graph: KnowledgeGraph | None = None
        self._visible: VisibleGraph | None = None
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._collapsed = False
        self._hover: str | None = None
        self._label_cells: dict[tuple[int, int], str] = {}  # (row, col) -> node id

    # -- data --

    def set_graph(self, graph: KnowledgeGraph) -> None:
        self._graph = graph
        if self.selected is not None and self.selected not in graph.node_map():
            self.selected = None
            self.post_message(self.NodeSelected(None))
        self.rebuild()

    def rebuild(self) -> None:
        if self._graph is None:
            return
        yaw, pitch, zoom, pan_x, pan_y = self._plot.camera_state()
        self._collapsed = lod_collapsed(zoom, self._collapsed)
        self._visible = apply_lod(self._graph, self._collapsed)
        plot, self._ids = build_plot(
            self._visible, color_by=self.color_by, selected=self.selected
        )
        self._index = {node_id: i for i, node_id in enumerate(self._ids)}
        if self._hover is not None:
            if self._hover in self._index:
                plot.set_hovered(self._index[self._hover])
            else:
                self._hover = None
        plot.set_camera_state(yaw, pitch, zoom, pan_x, pan_y)
        self._plot = plot
        self._refresh_overlay()
        self.invalidate()

    # -- geometry helpers --

    def _px_dims(self) -> tuple[int, int]:
        """Framebuffer pixel size (cells × the supersampling cell size)."""
        w, h = max(self.size.width, 1), max(self.size.height, 1)
        return w * self._cell_w, h * self._cell_h

    def _cell_px_size(self) -> tuple[int, int]:
        return self._cell_w, self._cell_h

    def _node_at(self, event: events.MouseEvent) -> str | None:
        # widget-relative cell coords via the watch.py shims — raw event.x/y
        # can be screen-relative while the mouse is captured
        x = _mouse_event_x(event) - self.region.x
        y = _mouse_event_y(event) - self.region.y
        px_w, px_h, px, py, radius = self._pixel_geometry(x, y)
        flat = self._plot.pick_px(px_w, px_h, px, py, radius)
        if flat is not None and flat < len(self._ids):
            return self._ids[flat]
        # not on a mark: the node's label next to it counts as the node too
        return self._label_cells.get((int(y), int(x)))

    def _refresh_overlay(self) -> None:
        if self._mode == "unsupported":
            return  # the widget shows its terminal-support notice instead
        legend = legend_spans(self.hidden_types)
        self._label_cells = {}
        if self._visible is None or not self._ids or self.size.width <= 0:
            self.set_overlay([(r, c, t, Style.parse(st)) for r, c, t, st in legend])
            return
        placed = place_labels_by_node(
            self._visible.nodes,
            self._plot.project_nodes(*self._px_dims()),
            cols=self.size.width,
            rows=self.size.height,
            cell_px=self._cell_px_size(),
            zoom=self._plot.camera_state()[2],
            selected=self.selected,
            hovered=self._hover,
        )
        for (row, col, text, _style), node_id in placed:
            for x in range(col, col + len(text)):
                self._label_cells[(row, x)] = node_id
        # legend first: set_overlay keeps the first span where two overlap,
        # so a node label never paints over a legend line
        spans = legend + [span for span, _node_id in placed]
        self.set_overlay(
            [(row, col, text, Style.parse(style)) for row, col, text, style in spans]
        )

    def _lod_check(self) -> None:
        """After any zoom change: rebuild when the collapse state flips,
        else just retrace the labels."""
        zoom = self._plot.camera_state()[2]
        if lod_collapsed(zoom, self._collapsed) != self._collapsed:
            self.rebuild()
        else:
            self._refresh_overlay()

    def on_resize(self, event: events.Resize) -> None:
        self._refresh_overlay()

    # -- interaction hooks (PlotWidget routes all input through these) --

    def apply_zoom(self, factor: float) -> None:
        super().apply_zoom(factor)
        self._lod_check()

    def apply_rotate(self, d_yaw: float, d_pitch: float) -> None:
        super().apply_rotate(d_yaw, d_pitch)
        self._refresh_overlay()

    def apply_pan(self, dx: float, dy: float) -> None:
        super().apply_pan(dx, dy)
        self._refresh_overlay()

    def apply_reset(self) -> None:
        super().apply_reset()
        self._lod_check()

    def set_hidden_types(self, hidden: frozenset[str]) -> None:
        self.hidden_types = hidden
        self._refresh_overlay()

    def on_click_at(self, event: events.MouseUp) -> None:
        type_ = legend_entry_at(
            _mouse_event_x(event) - self.region.x, _mouse_event_y(event) - self.region.y
        )
        if type_ is not None:
            self.post_message(self.TypeToggled(type_))
            return
        node_id = self._node_at(event)
        if node_id != self.selected:
            self.selected = node_id
            self._plot.set_selected(self._index.get(node_id) if node_id else None)
            self.invalidate()
            self._refresh_overlay()
            self.post_message(self.NodeSelected(node_id))
        elif node_id is not None:
            self.post_message(self.NodeActivated(node_id))

    # -- camera (the GraphScreen contract) --

    def zoom(self, factor: float) -> None:
        self.apply_zoom(factor)

    def pan(self, dx_fraction: float, dy_fraction: float) -> None:
        px_w, px_h = self._px_dims()
        self.apply_pan(dx_fraction * px_w, dy_fraction * px_h)

    def fit(self) -> None:
        # zoom 1.0 = the whole graph auto-framed; keep the current rotation
        yaw, pitch, _zoom, _px, _py = self._plot.camera_state()
        self._plot.set_camera_state(yaw, pitch, 1.0, 0.0, 0.0)
        self.invalidate()
        self._lod_check()

    def center_on(self, node_id: str, *, select: bool = True) -> None:
        if self._graph is None:
            return
        # centering implies label-visible zoom, which is past the LOD exit —
        # rebuild first so the node exists as its own mark, then pan its
        # projection to the view center (projector: cx = w/2 + pan_x).
        yaw, pitch, zoom, pan_x, pan_y = self._plot.camera_state()
        self._plot.set_camera_state(yaw, pitch, max(zoom, LABEL_ZOOM), pan_x, pan_y)
        if select:
            self.selected = node_id
        self.rebuild()
        flat = self._index.get(node_id)
        if flat is not None:
            px_w, px_h = self._px_dims()
            sx, sy, _depth = self._plot.project_nodes(px_w, px_h)[flat]
            self.apply_pan(px_w / 2 - sx, px_h / 2 - sy)
        if select:
            self.post_message(self.NodeSelected(node_id))

    def zoom_to_members(self, member_ids: tuple[str, ...]) -> None:
        if self._graph is None or not member_ids:
            return
        # land past the LOD exit threshold so the supernode actually expands,
        # then frame the members: zoom to their projected bounding box and
        # pan its center to the view center.
        yaw, pitch, zoom, pan_x, pan_y = self._plot.camera_state()
        self._plot.set_camera_state(
            yaw, pitch, max(zoom, COLLAPSE_EXIT * 1.1), pan_x, pan_y
        )
        self.rebuild()
        flats = [self._index[m] for m in member_ids if m in self._index]
        if not flats:
            return
        px_w, px_h = self._px_dims()
        projected = self._plot.project_nodes(px_w, px_h)
        xs = [projected[f][0] for f in flats]
        ys = [projected[f][1] for f in flats]
        span_x = max(max(xs) - min(xs), 1.0)
        span_y = max(max(ys) - min(ys), 1.0)
        factor = min(px_w / span_x, px_h / span_y) * 0.7
        if factor > 1.0:
            self._plot.zoom_by(factor)
            projected = self._plot.project_nodes(px_w, px_h)
            xs = [projected[f][0] for f in flats]
            ys = [projected[f][1] for f in flats]
        self.apply_pan(px_w / 2 - (max(xs) + min(xs)) / 2, px_h / 2 - (max(ys) + min(ys)) / 2)

    # -- hover (drag/click/scroll/keys are handled by PlotWidget's own
    #    handlers, which route through the apply_*/on_click_at hooks above;
    #    Textual dispatches this handler IN ADDITION to the base class's) --

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._dragging:
            return  # PlotWidget's handler rotates/pans via the hooks
        hover = self._node_at(event)
        if hover != self._hover:
            self._hover = hover
            element = self._index.get(hover) if hover is not None else None
            if self._plot.set_hovered(element):
                self.invalidate()
            self._refresh_overlay()


class TimeScrubber(Static):
    """Discrete scrubber over search-finish events; state IS an event index
    (None = live)."""

    # A drag here moves the cursor; without this Textual would also start a
    # text selection on mouse-down and paint it over the track as you drag.
    ALLOW_SELECT = False

    COMPONENT_CLASSES = set(SCRUBBER_STYLES)

    DEFAULT_CSS = """
    TimeScrubber > .scrubber--elapsed { color: $accent; }
    TimeScrubber > .scrubber--remaining { color: $foreground 30%; }
    TimeScrubber > .scrubber--cursor { color: $accent; text-style: bold; }
    TimeScrubber > .scrubber--label { color: $foreground; }
    TimeScrubber > .scrubber--hint { color: $text-muted; }
    """

    class TimeChanged(Message):
        def __init__(self, index: int | None):
            super().__init__()
            self.index = index

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events_list: list[str] = []
        self.index: int | None = None
        self.unit = "search"
        self._scrub_active = False

    def set_events(self, events_list: list[str], unit: str | None = None) -> None:
        if unit is not None and unit != self.unit:
            # granularity flip: keep the cursor on the same moment, not the
            # same ordinal — the last new event at or before where it stood
            if self.index is not None and self.events_list and events_list:
                at = self.events_list[self.index]
                self.index = max(bisect_right(events_list, at) - 1, 0)
            self.unit = unit
            self.events_list = list(events_list)
            self._refresh_scrubber()
            return
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
        width = self.content_size.width or 60
        styles = {name: self.get_component_rich_style(name) for name in SCRUBBER_STYLES}
        self.update(render_scrubber(self.events_list, self.index, width, styles, unit=self.unit))

    def on_resize(self, event: events.Resize) -> None:
        self._refresh_scrubber()

    def _index_from_event(self, event: events.MouseEvent) -> int:
        x = _mouse_event_x(event) - self.content_region.x
        width = max(self.content_size.width - 1, 1)
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


GraphKeys = KeysPanel  # the `?` panel; graph gestures come from GraphScreen.POINTER_HELP


class GraphScreen(KeysMixin, Screen):
    """The knowledge graph: canvas + concept sidebar + node detail + time
    scrubber. Reached via `g` in watch or `hillclimb knowledge graph`.

    `?` splits out the keys panel — every command in one list — so the footer
    only has to carry a few keys."""

    # Camera gestures, in the order they are worth discovering. They live
    # inside plotui's PlotWidget rather than in Textual bindings, so the keys
    # panel cannot read them off BINDINGS.
    POINTER_HELP = [
        ("drag", "rotate"),
        ("shift-drag", "pan"),
        ("scroll", "zoom"),
        ("arrows", "rotate"),
        ("shift+arrows", "pan"),
        ("r", "reset view"),
        ("click", "select"),
        ("click again", "open"),
        ("click legend", "hide a type"),
    ]

    # Only the handful worth a permanent slot stay in the footer; everything
    # else is show=False and lives in the `?` panel, which lists every active
    # binding regardless of `show`.
    BINDING_GROUP_TITLE = "graph"
    BINDINGS = [
        Binding("escape", "dismiss_or_back", "back"),
        Binding("enter", "activate", "open", show=False, priority=True),
        Binding("+,=", "zoom_in", "zoom in", show=False),
        Binding("-", "zoom_out", "zoom out", show=False),
        Binding("f,0", "fit", "fit", show=False, tooltip="frame the whole graph"),
        Binding("slash", "search", "search"),
        Binding("j", "scrub_back", "back in time", show=False, tooltip="one tick on the timeline"),
        Binding("k", "scrub_forward", "forward", show=False),
        Binding("g", "toggle_granularity", "timeline unit", show=False,
                tooltip="step per finished search, or per change to the graph"),
        Binding("end", "scrub_live", "live", show=False, tooltip="jump to now"),
        Binding("c", "toggle_sidebar", "concepts"),
        # One binding per legend slot; only the first is described, so the
        # `?` panel shows a single "1-8" row for the lot.
        Binding("1", "toggle_type(0)", "hide/show a type", show=False, key_display="1-8"),
        *(Binding(str(i + 1), f"toggle_type({i})", show=False) for i in range(1, len(LEGEND_TYPES))),
        KEYS_BINDING,
        *QUIT_BINDINGS,
    ]

    DEFAULT_CSS = """
    GraphScreen #concept-sidebar { dock: left; width: 26; display: none; background: $surface; }
    GraphScreen #concept-filters { height: auto; max-height: 70%; }
    GraphScreen #node-detail { dock: right; width: 42; display: none; padding: 0 1; }
    GraphScreen #graph-stage { width: 1fr; height: 1fr; }
    GraphScreen #time-scrubber { height: 2; background: $surface; padding: 0 1; }
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
        self._hidden_types: frozenset[str] = frozenset()
        self._fine_timeline = False  # False: per finished search; True: per graph change

    def compose(self) -> ComposeResult:
        yield HillclimbHeader()
        yield Input(placeholder="find node…", id="graph-search")
        yield OptionList(id="graph-search-results")
        yield ConceptSidebar(id="concept-sidebar")
        yield RichLog(id="node-detail", wrap=True, markup=False, auto_scroll=False)
        # Not docks: bottom docks overlap each other at the edge (the footer
        # would cover the scrubber's label row), so the canvas and scrubber
        # stack in a column of their own.
        with Vertical(id="graph-stage"):
            yield GraphPlotWidget(id="graph-canvas")
            yield TimeScrubber(id="time-scrubber")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_data()
        self.query_one("#graph-canvas", GraphPlotWidget).focus()
        self.set_interval(REFRESH_S, self.refresh_data)

    def _knowledge_dir(self) -> Path | None:
        from hillclimb.api import resolve_knowledge_dir

        return resolve_knowledge_dir(self.config)

    def refresh_data(self) -> None:
        from hillclimb.graph import graph_path, load_or_build_graph

        canvas = self.query_one("#graph-canvas", GraphPlotWidget)
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
        self.query_one("#time-scrubber", TimeScrubber).set_events(*self._timeline())
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
        view = filter_types(view, self._hidden_types)
        self.query_one("#graph-canvas", GraphPlotWidget).set_graph(view)

    # -- messages --

    def on_graph_plot_widget_node_selected(
        self, message: GraphPlotWidget.NodeSelected
    ) -> None:
        detail = self.query_one("#node-detail", RichLog)
        canvas = self.query_one("#graph-canvas", GraphPlotWidget)
        if message.node_id is None or canvas._graph is None:
            detail.styles.display = "none"
            detail.clear()
            return
        detail.styles.display = "block"
        detail.clear()
        source = canvas._graph
        for renderable in node_detail_renderables(source, message.node_id):
            detail.write(renderable, expand=True)

    def on_graph_plot_widget_node_activated(
        self, message: GraphPlotWidget.NodeActivated
    ) -> None:
        self._activate(message.node_id)

    def on_time_scrubber_time_changed(self, message: TimeScrubber.TimeChanged) -> None:
        self._apply_view()

    def on_graph_plot_widget_type_toggled(self, message: GraphPlotWidget.TypeToggled) -> None:
        self._toggle_type(message.type_)

    def action_toggle_type(self, index: int) -> None:
        if 0 <= index < len(LEGEND_TYPES):
            self._toggle_type(LEGEND_TYPES[index])

    def _toggle_type(self, type_: str) -> None:
        self._hidden_types = self._hidden_types ^ {type_}
        self._canvas().set_hidden_types(self._hidden_types)
        self._apply_view()

    def on_concept_sidebar_filters_changed(self, message: ConceptSidebar.FiltersChanged) -> None:
        self._filters = message.enabled
        canvas = self.query_one("#graph-canvas", GraphPlotWidget)
        canvas.color_by = message.color_by
        self._apply_view()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "graph-search" or self._graph is None:
            return
        matches = fuzzy_match(self._graph.nodes, event.value, limit=1)
        self._hide_search()
        if matches:
            self.query_one("#graph-canvas", GraphPlotWidget).center_on(matches[0].id)
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
            self.query_one("#graph-canvas", GraphPlotWidget).center_on(event.option.id)
        event.stop()

    # -- actions --

    def _canvas(self) -> GraphPlotWidget:
        return self.query_one("#graph-canvas", GraphPlotWidget)

    def action_zoom_in(self) -> None:
        self._canvas().zoom(ZOOM_FACTOR)

    def action_zoom_out(self) -> None:
        self._canvas().zoom(1 / ZOOM_FACTOR)

    def action_fit(self) -> None:
        self._canvas().fit()

    def action_scrub_back(self) -> None:
        self.query_one("#time-scrubber", TimeScrubber).step(-1)

    def action_scrub_forward(self) -> None:
        self.query_one("#time-scrubber", TimeScrubber).step(1)

    def action_scrub_live(self) -> None:
        scrubber = self.query_one("#time-scrubber", TimeScrubber)
        scrubber.set_index(None)
        self._apply_view()

    def _timeline(self) -> tuple[list[str], str]:
        """The scrubber's events and unit for the current granularity."""
        from hillclimb.graph import change_events

        if self._graph is None:
            return [], "search"
        if self._fine_timeline:
            return change_events(self._graph), "change"
        return list(self._graph.events), "search"

    def action_toggle_granularity(self) -> None:
        """Flip the timeline between one tick per finished search and one per
        change to the graph (per candidate, where claims carry their
        evidencing candidate's stamp). The cursor stays on the same moment."""
        if self._graph is None:
            return
        self._fine_timeline = not self._fine_timeline
        self.query_one("#time-scrubber", TimeScrubber).set_events(*self._timeline())
        self._apply_view()

    def action_toggle_help(self) -> None:
        self.action_toggle_keys()

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
            self.on_graph_plot_widget_node_selected(GraphPlotWidget.NodeSelected(None))
            return
        self.app.pop_screen()


class GraphApp(TimezoneMixin, App):
    """Standalone shell for `hillclimb knowledge graph` — same screen the
    watch TUI reaches via `g`."""

    BINDINGS = [
        Binding("t", "choose_timezone", "time zone", show=False),
        Binding("ctrl+c", "quit", "quit", show=False, priority=True),
    ]
    CSS = HILLCLIMB_CSS

    def __init__(self, config: Config | None = None):
        super().__init__()
        self.config = config or Config.load()

    def on_mount(self) -> None:
        apply_theme(self)
        self._init_timezone()
        self.push_screen(GraphScreen(self.config))
