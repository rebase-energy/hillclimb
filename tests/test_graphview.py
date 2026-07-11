"""Graph view: LOD/filter logic, the VisibleGraph→plotui adapter, label
placement, and the Textual screen (Pilot-driven rotate/zoom/click/scrub)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hillclimb.config import Config
from hillclimb.graph import GraphEdge, GraphNode, KnowledgeGraph, rebuild_graph
from hillclimb.graphview import (
    COLLAPSE_ENTER,
    COLLAPSE_EXIT,
    EDGE_COLORS,
    LABEL_MAX,
    LABEL_ZOOM,
    NODE_COLORS,
    NODE_SIZE,
    SUPERNODE_SIZE_CAP,
    VNode,
    VisibleGraph,
    apply_lod,
    build_plot,
    fallback_pos,
    filter_concepts,
    fuzzy_match,
    lod_collapsed,
    node_detail_renderables,
    node_size,
    place_labels,
    render_scrubber,
    snap_to_event,
    style_to_rgb,
)
from tests.test_graph import claim, make_card


class TestLod:
    def test_hysteresis(self):
        assert lod_collapsed(COLLAPSE_ENTER - 0.01, False) is True
        assert lod_collapsed(COLLAPSE_ENTER + 0.01, False) is False
        # inside the band the previous state sticks
        mid = (COLLAPSE_ENTER + COLLAPSE_EXIT) / 2
        assert lod_collapsed(mid, True) is True
        assert lod_collapsed(mid, False) is False

    def _graph(self):
        return KnowledgeGraph(nodes=[
            GraphNode(id="entity:a", type="technique", label="a",
                      concepts=["decision-trees"], pos=(0.0, 0.0), pos3=(0.0, 0.0, 0.0)),
            GraphNode(id="entity:b", type="technique", label="b",
                      concepts=["decision-trees"], pos=(1.0, 1.0), pos3=(1.0, 1.0, 1.0)),
            GraphNode(id="concept:decision-trees", type="concept",
                      label="decision-trees", pos=(0.5, 0.5), pos3=(0.5, 0.5, 0.5)),
            GraphNode(id="search:r1/s1", type="search", label="r1/s1",
                      pos=(2.0, 2.0), pos3=(2.0, 2.0, 2.0)),
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
        assert supernode.x == pytest.approx(0.5)
        assert supernode.y == pytest.approx(0.5)
        assert supernode.z == pytest.approx(0.5)
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
        by_id = {n.id: n for n in vg.nodes}
        assert by_id["entity:b"].z == pytest.approx(1.0)  # pos3 carried through

    def test_unplaced_nodes_get_deterministic_fallback(self):
        graph = KnowledgeGraph(nodes=[
            GraphNode(id="entity:a", type="technique", label="a"),
        ])
        first = apply_lod(graph, collapsed=False).nodes[0]
        second = apply_lod(graph, collapsed=False).nodes[0]
        assert (first.x, first.y, first.z) == (second.x, second.y, second.z)


class TestAdapter:
    def test_style_to_rgb(self):
        # Vivid palette: saturated base colors, bold brightens, dim darkens.
        assert style_to_rgb("yellow") == (234, 179, 8)
        assert style_to_rgb("bold yellow") > style_to_rgb("yellow")
        assert style_to_rgb("dim red") < style_to_rgb("red")
        assert style_to_rgb("bright_black") == style_to_rgb("bold black")
        for style in list(NODE_COLORS.values()) + list(EDGE_COLORS.values()):
            rgb = style_to_rgb(style)
            assert len(rgb) == 3 and all(0 <= c <= 255 for c in rgb)

    def test_fallback_pos_is_3d_and_deterministic(self):
        a = fallback_pos("entity:a")
        assert len(a) == 3
        assert fallback_pos("entity:a") == a
        assert fallback_pos("entity:b") != a
        assert sum(c * c for c in a) ** 0.5 <= 1.01  # inside the unit-ish ball

    def test_node_size_scales_supernodes(self):
        plain = VNode(id="a", type="technique", label="a", x=0, y=0)
        assert node_size(plain) == NODE_SIZE
        small = VNode(id="s", type="supernode", label="s", x=0, y=0, count=2)
        huge = VNode(id="h", type="supernode", label="h", x=0, y=0, count=500)
        assert NODE_SIZE < node_size(small) < node_size(huge) <= SUPERNODE_SIZE_CAP

    def test_build_plot_maps_ids_and_survives_rendering(self):
        from hillclimb.graphview import VEdge

        vg = VisibleGraph(
            nodes=(
                VNode(id="a", type="technique", label="alpha", x=0.0, y=0.0, z=0.0),
                VNode(id="b", type="library", label="beta", x=1.0, y=0.0, z=0.5),
                VNode(id="s", type="supernode", label="s (3)", x=0.5, y=1.0, z=0.0,
                      count=3),
            ),
            edges=(
                VEdge(src="a", dst="b", type="used"),
                VEdge(src="a", dst="ghost", type="used"),  # endpoint missing
            ),
        )
        plot, ids = build_plot(vg, color_by="type", selected="b")
        assert ids == ["a", "b", "s"]
        rgba = plot.render_rgba(120, 80)
        assert len(rgba) == 120 * 80 * 4
        # selection ring: the selected node renders with white pixels
        white = bytes((255, 255, 255, 255))
        assert any(rgba[i:i + 4] == white for i in range(0, len(rgba), 4))
        # a vanished selection or empty graph must not blow up
        plot2, ids2 = build_plot(vg, selected="gone")
        assert ids2 == ["a", "b", "s"]
        plot3, ids3 = build_plot(VisibleGraph(nodes=(), edges=()))
        assert ids3 == [] and plot3.render_rgba(20, 10) is not None


class TestLabels:
    def _nodes(self):
        return [
            VNode(id="a", type="technique", label="alpha", x=0, y=0),
            VNode(id="b", type="library", label="beta", x=0, y=0),
        ]

    def test_zoom_threshold_and_always_show(self):
        nodes = self._nodes()
        projected = [(5.0, 4.0, 0.0), (5.0, 12.0, 0.0)]  # cells (2,5) and (6,5)
        quiet = place_labels(nodes, projected, cols=40, rows=10, cell_px=(1, 2),
                             zoom=LABEL_ZOOM / 2)
        assert quiet == []
        selected = place_labels(nodes, projected, cols=40, rows=10, cell_px=(1, 2),
                                zoom=LABEL_ZOOM / 2, selected="a")
        assert [s[2] for s in selected] == ["alpha"]
        assert selected[0][3] == "bold white"
        loud = place_labels(nodes, projected, cols=40, rows=10, cell_px=(1, 2),
                            zoom=LABEL_ZOOM * 2)
        assert {s[2] for s in loud} == {"alpha", "beta"}
        supernode = [VNode(id="s", type="supernode", label="s (2)", x=0, y=0, count=2)]
        always = place_labels(supernode, [(5.0, 4.0, 0.0)], cols=40, rows=10,
                              cell_px=(1, 2), zoom=LABEL_ZOOM / 2)
        assert [s[2] for s in always] == ["s (2)"]

    def test_collision_keeps_higher_priority(self):
        nodes = self._nodes()
        # same row, one cell apart: labels overlap, only one survives
        projected = [(5.0, 4.0, 0.0), (6.0, 4.0, 0.0)]
        spans = place_labels(nodes, projected, cols=40, rows=10, cell_px=(1, 2),
                             zoom=LABEL_ZOOM * 2)
        assert len(spans) == 1
        # hovering the loser flips the priority
        hovered = place_labels(nodes, projected, cols=40, rows=10, cell_px=(1, 2),
                               zoom=LABEL_ZOOM * 2, hovered="b")
        assert hovered[0][2] == "beta"

    def test_truncation_and_offscreen(self):
        long_label = VNode(id="l", type="technique", label="x" * 40, x=0, y=0)
        spans = place_labels([long_label], [(5.0, 4.0, 0.0)], cols=80, rows=10,
                             cell_px=(1, 2), zoom=LABEL_ZOOM * 2)
        assert spans[0][2] == "x" * LABEL_MAX + "…"
        gone = place_labels([long_label], [(-5.0, 4.0, 0.0)], cols=80, rows=10,
                            cell_px=(1, 2), zoom=LABEL_ZOOM * 2)
        assert gone == []


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
def graph_workspace(tmp_path, monkeypatch):
    from hillclimb.claims import Entity, ensure_concepts, save_entities
    from hillclimb.knowledge import write_card

    # deterministic render path regardless of the terminal the tests run in
    # (placeholder emits escape strings — safe headlessly)
    monkeypatch.setenv("PLOTUI_RENDER", "placeholder")
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


def _cell_of_some_node(canvas) -> tuple[int, int, str]:
    """A widget cell whose center picks a node, plus that node's id — using
    the widget's own geometry so the click test can't drift from it."""
    px_w, px_h = canvas._px_dims()
    cell_w, cell_h = canvas._cell_px_size()
    for flat, (sx, sy, _depth) in enumerate(canvas._plot.project_nodes(px_w, px_h)):
        col, row = int(sx // cell_w), int(sy // cell_h)
        if not (0 <= col < canvas.size.width and 0 <= row < canvas.size.height):
            continue
        _pw, _ph, px, py, radius = canvas._pixel_geometry(col, row)
        picked = canvas._plot.pick_px(px_w, px_h, px, py, radius)
        if picked is not None:
            return col, row, canvas._ids[picked]
    raise AssertionError("no pickable node on screen")


@pytest.mark.asyncio
async def test_graph_app_mounts_and_zooms(graph_workspace):
    from hillclimb.graphview import GraphApp, GraphPlotWidget

    app = GraphApp(graph_workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.screen.query_one("#graph-canvas", GraphPlotWidget)
        assert canvas._ids  # nodes loaded
        zoom = canvas._plot.camera_state()[2]
        await pilot.press("plus")
        zoomed_in = canvas._plot.camera_state()[2]
        assert zoomed_in > zoom
        await pilot.press("minus")
        assert canvas._plot.camera_state()[2] < zoomed_in
        canvas.on_mouse_scroll_up(_scroll_stub(canvas.region.x + 10, canvas.region.y + 5))
        assert canvas._plot.camera_state()[2] > zoom * 0.99


@pytest.mark.asyncio
async def test_drag_rotates_and_pan_moves_camera(graph_workspace):
    from hillclimb.graphview import GraphApp, GraphPlotWidget

    app = GraphApp(graph_workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.screen.query_one("#graph-canvas", GraphPlotWidget)
        yaw_before = canvas._plot.camera_state()[0]
        await pilot.mouse_down("#graph-canvas", offset=(40, 10))
        await pilot.hover("#graph-canvas", offset=(50, 10))
        assert canvas._plot.camera_state()[0] > yaw_before  # dragged right -> yaw
        assert canvas.dragging
        await pilot.mouse_up("#graph-canvas", offset=(50, 10))
        assert not canvas.dragging
        # the screen-contract pan shifts the projection center
        pan_before = canvas._plot.camera_state()[3]
        canvas.pan(0.25, 0.0)
        assert canvas._plot.camera_state()[3] > pan_before


@pytest.mark.asyncio
async def test_click_selects_node_and_opens_detail(graph_workspace):
    from textual.widgets import RichLog

    from hillclimb.graphview import GraphApp, GraphPlotWidget

    app = GraphApp(graph_workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.screen.query_one("#graph-canvas", GraphPlotWidget)
        col, row, node_id = _cell_of_some_node(canvas)
        await pilot.mouse_down("#graph-canvas", offset=(col, row))
        await pilot.mouse_up("#graph-canvas", offset=(col, row))
        await pilot.pause()
        assert canvas.selected == node_id
        detail = app.screen.query_one("#node-detail", RichLog)
        assert detail.styles.display != "none"
        # escape clears the selection before popping the screen
        await pilot.press("escape")
        assert canvas.selected is None
        assert detail.styles.display == "none"


@pytest.mark.asyncio
async def test_scrubber_steps_and_refresh_keeps_state(graph_workspace):
    from hillclimb.graphview import GraphApp, GraphPlotWidget, TimeScrubber

    app = GraphApp(graph_workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.screen.query_one("#graph-canvas", GraphPlotWidget)
        scrubber = app.screen.query_one("#time-scrubber", TimeScrubber)
        assert scrubber.index is None and len(scrubber.events_list) == 2
        assert "search:r2/s1" in canvas._ids
        await pilot.press("left_square_bracket")
        await pilot.pause()
        assert scrubber.index == 0
        assert "search:r2/s1" not in canvas._ids
        # live refresh must not move the time cursor or the camera
        camera = canvas._plot.camera_state()
        app.screen.refresh_data()
        await pilot.pause()
        assert scrubber.index == 0
        assert canvas._plot.camera_state() == camera
        await pilot.press("end")
        await pilot.pause()
        assert scrubber.index is None
        assert "search:r2/s1" in canvas._ids


@pytest.mark.asyncio
async def test_zoom_out_collapses_and_members_expand(graph_workspace):
    from hillclimb.graphview import GraphApp, GraphPlotWidget

    app = GraphApp(graph_workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.screen.query_one("#graph-canvas", GraphPlotWidget)
        assert not any(i.startswith("supernode:") for i in canvas._ids)
        zoom = canvas._plot.camera_state()[2]
        canvas.zoom((COLLAPSE_ENTER * 0.9) / zoom)
        await pilot.pause()
        supernodes = [i for i in canvas._ids if i.startswith("supernode:")]
        assert supernodes, "zooming out must fold entities into supernodes"
        target = next(n for n in canvas._visible.nodes if n.id == supernodes[0])
        canvas.zoom_to_members(target.members)
        await pilot.pause()
        assert canvas._plot.camera_state()[2] >= COLLAPSE_EXIT
        assert all(m in canvas._ids for m in target.members)


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
