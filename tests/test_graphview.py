"""Graph view: camera math, braille rasterization, hit-testing, LOD, and the
Textual screen (Pilot-driven pan/zoom/click/scrub)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hillclimb.config import Config
from hillclimb.graph import GraphEdge, GraphNode, KnowledgeGraph, rebuild_graph
from hillclimb.graphview import (
    COLLAPSE_ENTER,
    COLLAPSE_EXIT,
    Camera,
    CellBuffer,
    VNode,
    VisibleGraph,
    WorldBounds,
    apply_lod,
    clip_segment,
    dot_to_world,
    draw_line,
    filter_concepts,
    fit_camera,
    fuzzy_match,
    hit_test,
    lod_collapsed,
    node_detail_renderables,
    pan_camera,
    render_frame,
    render_scrubber,
    snap_to_event,
    world_bounds,
    world_to_dot,
    zoom_about,
    PlacedNode,
)
from tests.test_graph import claim, make_card

BOUNDS = WorldBounds(-10, -10, 10, 10)


class TestCamera:
    def test_world_dot_roundtrip(self):
        cam = Camera(cx=0.3, cy=-0.2, scale=50)
        dx, dy = world_to_dot(cam, 200, 100, 0.7, 0.1)
        wx, wy = dot_to_world(cam, 200, 100, dx, dy)
        assert abs(wx - 0.7) < 1e-9 and abs(wy - 0.1) < 1e-9

    def test_zoom_about_keeps_cursor_point_fixed(self):
        cam = Camera(cx=0.0, cy=0.0, scale=40)
        cursor = (37.0, 81.0)
        before = dot_to_world(cam, 200, 100, *cursor)
        zoomed = zoom_about(cam, 200, 100, *cursor, 1.25, BOUNDS)
        after = dot_to_world(zoomed, 200, 100, *cursor)
        assert abs(before[0] - after[0]) < 1e-9
        assert abs(before[1] - after[1]) < 1e-9
        assert zoomed.scale == pytest.approx(50)

    def test_pan_and_clamp(self):
        cam = Camera(cx=0, cy=0, scale=10)
        panned = pan_camera(cam, 20, -40, BOUNDS)
        assert panned.cx == pytest.approx(2.0)
        assert panned.cy == pytest.approx(-4.0)
        far = pan_camera(cam, 1e6, 1e6, BOUNDS)
        assert far.cx == BOUNDS.max_x and far.cy == BOUNDS.max_y

    def test_fit_contains_all_nodes(self):
        nodes = [VNode(id="a", type="search", label="a", x=-1, y=-1),
                 VNode(id="b", type="search", label="b", x=1, y=1)]
        bounds = world_bounds(nodes)
        cam = fit_camera(bounds, 200, 100)
        for node in nodes:
            dx, dy = world_to_dot(cam, 200, 100, node.x, node.y)
            assert 0 <= dx < 200 and 0 <= dy < 100


class TestRaster:
    def test_draw_line_horizontal_stroke(self):
        buf = CellBuffer(4, 2)
        draw_line(buf, 0, 0, 7, 0, "cyan")  # dots 0..7 -> cells 0..3
        rows = buf.to_segments()
        text = "".join(seg.text for seg in rows[0])
        assert text[1:4] == "───"
        assert len(text) == 4

    def test_draw_line_vertical_and_diagonal_strokes(self):
        buf = CellBuffer(4, 4)
        draw_line(buf, 0, 0, 0, 15, "cyan")  # dots rows 0..15 -> cells 0..3
        column = [buf.lines[y][0] for y in range(1, 4)]
        assert column == ["│", "│", "│"]
        buf2 = CellBuffer(4, 4)
        draw_line(buf2, 0, 0, 7, 15, "cyan")  # down-right diagonal
        assert "╲" in {buf2.lines[y][x] for y in range(4) for x in range(4)} - {None}
        buf3 = CellBuffer(4, 4)
        draw_line(buf3, 0, 15, 7, 0, "cyan")  # up-right diagonal
        assert "╱" in {buf3.lines[y][x] for y in range(4) for x in range(4)} - {None}

    def test_same_cell_edge_stub(self):
        buf = CellBuffer(2, 2)
        draw_line(buf, 0, 0, 1, 0, "cyan")  # both endpoints inside cell (0,0)
        assert buf.lines[0][0] == "─"

    def test_rows_are_exact_width(self):
        buf = CellBuffer(7, 3)
        draw_line(buf, 0, 0, 13, 11, "green")
        buf.set_char(2, 1, "●", "cyan")
        for row in buf.to_segments():
            assert sum(len(seg.text) for seg in row) == 7

    def test_glyph_wins_over_edges(self):
        buf = CellBuffer(2, 1)
        draw_line(buf, 0, 0, 3, 0, "dim blue")
        buf.set_char(1, 0, "●", "yellow")
        text = "".join(seg.text for seg in buf.to_segments()[0])
        assert text[1] == "●"

    def test_clip_segment(self):
        assert clip_segment(-5, -5, -1, -1, 100, 100) is None
        inside = clip_segment(1, 1, 5, 5, 100, 100)
        assert inside == (1, 1, 5, 5)
        clipped = clip_segment(-10, 2, 10, 2, 100, 100)
        assert clipped is not None and clipped[0] == 0

    def test_render_frame_places_nodes_and_labels_collide(self):
        vg = VisibleGraph(
            nodes=(
                VNode(id="a", type="technique", label="alpha", x=0.0, y=0.0),
                VNode(id="b", type="library", label="beta", x=0.02, y=0.0),
            ),
            edges=(),
        )
        cam = Camera(cx=0, cy=0, scale=100)  # above LABEL_SCALE -> labels on
        rows, placed = render_frame(vg, cam, 40, 10)
        assert {p.node_id for p in placed} == {"a", "b"}
        text = "\n".join("".join(seg.text for seg in row) for row in rows)
        # both glyphs land; the two labels overlap so only one survives
        assert "●" in text and "■" in text
        assert ("alpha" in text) != ("beta" in text)
        assert all(sum(len(seg.text) for seg in row) == 40 for row in rows)


class TestHitTest:
    def test_nearest_within_radius(self):
        placed = [PlacedNode("a", 10, 10), PlacedNode("b", 14, 10)]
        assert hit_test(placed, 11, 10) == "a"
        assert hit_test(placed, 13.5, 10) == "b"
        assert hit_test(placed, 50, 50) is None
        assert hit_test([], 0, 0) is None


class TestLod:
    def test_hysteresis(self):
        assert lod_collapsed(COLLAPSE_ENTER - 1, False) is True
        assert lod_collapsed(COLLAPSE_ENTER + 1, False) is False
        # inside the band the previous state sticks
        mid = (COLLAPSE_ENTER + COLLAPSE_EXIT) / 2
        assert lod_collapsed(mid, True) is True
        assert lod_collapsed(mid, False) is False

    def _graph(self):
        return KnowledgeGraph(nodes=[
            GraphNode(id="entity:a", type="technique", label="a",
                      concepts=["decision-trees"], pos=(0.0, 0.0)),
            GraphNode(id="entity:b", type="technique", label="b",
                      concepts=["decision-trees"], pos=(1.0, 1.0)),
            GraphNode(id="concept:decision-trees", type="concept",
                      label="decision-trees", pos=(0.5, 0.5)),
            GraphNode(id="search:r1/s1", type="search", label="r1/s1", pos=(2.0, 2.0)),
        ], edges=[
            GraphEdge(src="search:r1/s1", dst="entity:a", type="used"),
            GraphEdge(src="search:r1/s1", dst="entity:b", type="used"),
            GraphEdge(src="entity:a", dst="concept:decision-trees", type="has_concept"),
        ])

    def test_collapse_groups_by_primary_concept(self):
        vg = apply_lod(self._graph(), collapsed=True)
        by_id = {n.id: n for n in vg.nodes}
        supernode = by_id["supernode:decision-trees"]
        assert supernode.count == 2
        assert supernode.x == pytest.approx(0.5) and supernode.y == pytest.approx(0.5)
        assert set(supernode.members) == {"entity:a", "entity:b"}
        assert "search:r1/s1" in by_id  # skeleton stays atomic
        assert "entity:a" not in by_id and "concept:decision-trees" not in by_id
        used = [e for e in vg.edges if e.type == "used"]
        assert len(used) == 1 and used[0].weight == pytest.approx(2.0)

    def test_uncollapsed_passthrough(self):
        vg = apply_lod(self._graph(), collapsed=False)
        assert {n.id for n in vg.nodes} == {
            "entity:a", "entity:b", "concept:decision-trees", "search:r1/s1",
        }


class TestFilter:
    def test_filter_concepts(self):
        graph = KnowledgeGraph(nodes=[
            GraphNode(id="entity:a", type="technique", label="a", concepts=["tabular"]),
            GraphNode(id="entity:b", type="technique", label="b", concepts=["image"]),
            GraphNode(id="concept:tabular", type="concept", label="tabular"),
            GraphNode(id="concept:image", type="concept", label="image"),
            GraphNode(id="search:r", type="search", label="r"),
        ], edges=[
            GraphEdge(src="search:r", dst="entity:a", type="used"),
            GraphEdge(src="search:r", dst="entity:b", type="used"),
        ])
        filtered = filter_concepts(graph, frozenset({"tabular"}))
        ids = {n.id for n in filtered.nodes}
        assert ids == {"entity:a", "concept:tabular", "search:r"}
        assert len(filtered.edges) == 1
        assert filter_concepts(graph, None) is graph


class TestSearchAndScrub:
    def test_fuzzy_match(self):
        nodes = [
            GraphNode(id="entity:histgradientboosting", type="technique",
                      label="histgradientboosting"),
            GraphNode(id="entity:lightgbm", type="library", label="lightgbm"),
            GraphNode(id="search:r1/s1", type="search", label="r1/s1"),
        ]
        assert fuzzy_match(nodes, "hist")[0].label == "histgradientboosting"
        assert fuzzy_match(nodes, "lightgbm")[0].label == "lightgbm"
        assert fuzzy_match(nodes, "zzz") == []
        assert fuzzy_match(nodes, "") == []

    def test_snap_to_event(self):
        events = ["t1", "t2", "t3"]
        assert snap_to_event(events, 0.0) == 0
        assert snap_to_event(events, 0.49) == 1
        assert snap_to_event(events, 1.0) == 2
        assert snap_to_event(events, 5.0) == 2
        assert snap_to_event([], 0.5) == 0

    def test_render_scrubber(self):
        live = render_scrubber(["t1", "t2"], None, 40).plain
        assert "◉" in live and "(live)" in live
        historical = render_scrubber(["t1", "t2"], 0, 40).plain
        assert "as of t1" in historical


class TestDetail:
    def test_node_detail(self):
        graph = KnowledgeGraph(nodes=[
            GraphNode(id="entity:a", type="technique", label="alpha",
                      concepts=["tabular"], first_seen="2026-07-01T00:00:00Z"),
            GraphNode(id="search:r1/s1", type="search", label="r1/s1"),
        ], edges=[GraphEdge(src="search:r1/s1", dst="entity:a", type="used")])
        parts = node_detail_renderables(graph, "entity:a")
        text = "\n".join(p.plain for p in parts)
        assert "alpha" in text and "technique" in text and "tabular" in text
        assert "used <-" in text and "r1/s1" in text
        missing = node_detail_renderables(graph, "nope")
        assert "not in graph" in missing[0].plain


# --- Textual Pilot tests ---


@pytest.fixture
def graph_workspace(tmp_path):
    from hillclimb.claims import Entity, ensure_concepts, save_entities
    from hillclimb.knowledge import write_card

    kdir = tmp_path / "knowledge"
    ensure_concepts(kdir)
    save_entities(kdir, [
        Entity(slug="histgradientboosting", concepts=["decision-trees", "tabular"],
               first_seen="2026-07-01T00:00:00Z"),
    ])
    write_card(kdir, make_card(claims=[claim()]))
    write_card(kdir, make_card(run_ref="r2/s1", finished_at="2026-07-02T00:00:00Z"))
    rebuild_graph(kdir)
    config = Config()
    config.learning.dir = kdir
    config.paths.runs_dir = tmp_path / "runs"
    return config


def _scroll_stub(x: int, y: int) -> SimpleNamespace:
    return SimpleNamespace(
        x=x, y=y, screen_x=x, screen_y=y,
        prevent_default=lambda: None, stop=lambda: None,
    )


@pytest.mark.asyncio
async def test_graph_app_mounts_and_zooms(graph_workspace):
    from hillclimb.graphview import GraphApp, GraphCanvas

    app = GraphApp(graph_workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.screen.query_one("#graph-canvas", GraphCanvas)
        assert canvas.camera is not None
        assert canvas._placed  # nodes on screen
        scale = canvas.camera.scale
        await pilot.press("plus")
        assert canvas.camera.scale > scale
        await pilot.press("minus")
        assert canvas.camera.scale == pytest.approx(scale)
        # wheel zoom about a stubbed cursor keeps that world point fixed
        canvas.on_mouse_scroll_up(_scroll_stub(canvas.region.x + 10, canvas.region.y + 5))
        assert canvas.camera.scale > scale


@pytest.mark.asyncio
async def test_drag_pans_camera(graph_workspace):
    from hillclimb.graphview import GraphApp, GraphCanvas

    app = GraphApp(graph_workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.screen.query_one("#graph-canvas", GraphCanvas)
        cx_before = canvas.camera.cx
        await pilot.mouse_down("#graph-canvas", offset=(40, 10))
        await pilot.hover("#graph-canvas", offset=(50, 10))
        assert canvas.camera.cx < cx_before  # dragged right -> camera left
        assert canvas.dragging
        await pilot.mouse_up("#graph-canvas", offset=(50, 10))
        assert not canvas.dragging


@pytest.mark.asyncio
async def test_click_selects_node_and_opens_detail(graph_workspace):
    from textual.widgets import RichLog

    from hillclimb.graphview import GraphApp, GraphCanvas

    app = GraphApp(graph_workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.screen.query_one("#graph-canvas", GraphCanvas)
        target = canvas._placed[0]
        cell = (int(target.dot_x // 2), int(target.dot_y // 4))
        await pilot.mouse_down("#graph-canvas", offset=cell)
        await pilot.mouse_up("#graph-canvas", offset=cell)
        assert canvas.selected == target.node_id
        detail = app.screen.query_one("#node-detail", RichLog)
        assert detail.styles.display != "none"
        # escape clears the selection before popping the screen
        await pilot.press("escape")
        assert canvas.selected is None
        assert detail.styles.display == "none"


@pytest.mark.asyncio
async def test_scrubber_steps_and_refresh_keeps_state(graph_workspace):
    from hillclimb.graphview import GraphApp, GraphCanvas, TimeScrubber

    app = GraphApp(graph_workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.screen.query_one("#graph-canvas", GraphCanvas)
        scrubber = app.screen.query_one("#time-scrubber", TimeScrubber)
        assert scrubber.index is None and len(scrubber.events_list) == 2
        assert any(p.node_id == "search:r2/s1" for p in canvas._placed)
        await pilot.press("left_square_bracket")
        await pilot.pause()
        assert scrubber.index == 0
        assert all(p.node_id != "search:r2/s1" for p in canvas._placed)
        # live refresh must not move the time cursor or the camera
        camera = canvas.camera
        app.screen.refresh_data()
        await pilot.pause()
        assert scrubber.index == 0
        assert canvas.camera == camera
        await pilot.press("end")
        await pilot.pause()
        assert scrubber.index is None
        assert any(p.node_id == "search:r2/s1" for p in canvas._placed)


@pytest.mark.asyncio
async def test_watch_g_opens_graph_screen(graph_workspace):
    from hillclimb.graphview import GraphScreen
    from hillclimb.watch import WatchApp

    graph_workspace.paths.runs_dir.mkdir(parents=True, exist_ok=True)
    app = WatchApp(graph_workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("g")
        await pilot.pause()
        assert isinstance(app.screen, GraphScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, GraphScreen)
