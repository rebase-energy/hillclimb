"""The archive tree (`hillclimb tree2`): per-node encoding, the best's
lineage, the single-trace plot with plotui borders/labels/star, the mark
sizing that keeps circles apart, the text legend (tree2.py), and the widget
hooks it rides on (tree2view.py)."""

from __future__ import annotations

import pytest

from hillclimb.tui.tree import build_tree, candidates_until
from hillclimb.tui.tree2 import (
    ASPECT,
    BEST_RING_RGB,
    BEST_SIZE_SCALE,
    BORDER,
    GAP_PX,
    LINEAGE_RGB,
    R_MAX_PX,
    R_MIN_PX,
    RAMP_ROWS,
    STAGE_RGB,
    STAGES,
    UNSCORED_RGB,
    VIRIDIS,
    best_lineage,
    build_tree2_plot,
    closest_pair_px,
    fill_rgb,
    fit_radius,
    label_nodes,
    legend_spans,
    nearest_px,
    node_label,
    node_number,
    node_radius,
    node_shape,
    radius_px,
    row_height,
    star_reach,
    star_scale,
    score_range,
    score_t,
    stage,
    stage_counts,
    viridis,
    world_xy,
)
from hillclimb.tui.treeview import ROW_H, build_tree_plot
from tests.test_tree import cand, forest, tree_workspace  # noqa: F401 (fixture)


def wide_forest(n: int = 40):
    return [cand("c000", "baseline", score=0.1)] + [
        cand(f"c{i:03d}", parent="c000", score=0.2, t=i) for i in range(1, n + 1)
    ]


def rgba_colours(plot, w: int, h: int) -> set[tuple[int, int, int]]:
    data = plot.render_rgba(w, h)
    return {tuple(data[i : i + 3]) for i in range(0, len(data), 4) if data[i + 3]}


def colour_pixels(plot, w: int, h: int, rgb: tuple[int, int, int]) -> int:
    """How many pixels of the frame are exactly `rgb` — the spine's white is
    shared by rings and the lineage line, so a hidden line is fewer of them."""
    data = plot.render_rgba(w, h)
    return sum(1 for i in range(0, len(data), 4) if data[i + 3] and tuple(data[i : i + 3]) == rgb)


class TestColour:
    def test_viridis_ends_on_plotui_stops_and_clamps(self):
        assert viridis(0.0) == VIRIDIS[0]
        assert viridis(1.0) == VIRIDIS[-1]
        assert viridis(-3.0) == VIRIDIS[0] and viridis(7.0) == VIRIDIS[-1]
        assert viridis(0.5) == VIRIDIS[2]  # a stop lands exactly

    def test_ramp_is_oriented_so_the_best_is_bright(self):
        assert score_t(0.9, 0.1, 0.9, higher_is_better=True) == 1.0
        assert score_t(0.1, 0.1, 0.9, higher_is_better=False) == 1.0
        assert score_t(0.5, 0.1, 0.9) == pytest.approx(0.5)
        assert score_t(3.0, 3.0, 3.0) == 1.0  # a flat search sits at the top

    def test_fill_follows_the_search_range(self):
        tree = build_tree(forest())
        span = score_range(tree)
        assert span == (0.1, 0.9)
        by_id = {n.id: n for n in tree.nodes}
        assert fill_rgb(by_id["c006"], span) == VIRIDIS[-1]  # 0.9, the top (pruned or not)
        assert fill_rgb(by_id["c000"], span) == VIRIDIS[0]
        assert fill_rgb(by_id["c005"], span) == UNSCORED_RGB  # buggy: off the ramp
        assert fill_rgb(by_id["c000"], None) == UNSCORED_RGB

    def test_unscored_tree_has_no_range(self):
        assert score_range(build_tree([cand("c001", status="buggy")])) is None


class TestEncoding:
    def test_number_is_the_candidate_number(self):
        assert node_number("c000") == "0"
        assert node_number("c017") == "17"
        assert node_number("c1234") == "1234"
        assert node_number("gepa-7") == "gepa-7"

    def test_stage_is_the_fate_ladder(self):
        tree = build_tree(forest())
        stages = {n.id: stage(n) for n in tree.nodes}
        assert stages == {
            "c000": "scored", "c001": "scored", "c002": "expanded", "c003": "scored",
            "c004": "expanded", "c005": "failed", "c006": "failed", "c007": "expanded",
        }
        assert stage_counts(tree) == {"expanded": 2, "scored": 3, "failed": 2}  # the best is its own entry
        assert set(STAGE_RGB) >= set(STAGES) | {"pending"}

    def test_rings_never_borrow_a_viridis_hue(self):
        """Fill is the only place hue means score: the spine (expanded, the
        best, the lineage) is one white, a scored node's ring is the ground — the
        same tone a failed node is filled with, so it reads hollow — and red
        is reserved for failure."""
        from hillclimb.tui.theme import PLOT_BG
        from hillclimb.tui.tree2 import HOLLOW_RGB, LINEAGE_RGB, WHITE_RGB

        assert HOLLOW_RGB == PLOT_BG == STAGE_RGB["scored"] == UNSCORED_RGB
        assert STAGE_RGB["expanded"] == BEST_RING_RGB == LINEAGE_RGB == WHITE_RGB
        red = STAGE_RGB["failed"]
        assert red[0] > 200 and red[1] < 80 and red[2] < 80
        ramp = {viridis(i / 50) for i in range(51)}
        assert not ramp & {WHITE_RGB, red, STAGE_RGB["pending"]}

    def test_best_is_a_bigger_star_everything_else_a_disc(self):
        tree = build_tree(forest())
        best, other = tree.node("c007"), tree.node("c003")
        assert node_shape(best) == "star" and node_shape(other) == "disc"
        assert node_radius(best, 5.0) == 5.0 * BEST_SIZE_SCALE and node_radius(other, 5.0) == 5.0
        assert node_label(best) == "7" and node_label(other) == "3"

    def test_overlay_labels_carry_priority(self):
        tree = build_tree(forest())
        labels = {v.id: v for v in label_nodes(tree)}
        assert labels["c007"].count == 3           # best outranks the lineage
        assert labels["c004"].count == 2           # on the lineage
        assert labels["c001"].count == 1           # the ensemble input is not ancestry
        assert labels["c003"].label == "3"

    def test_best_lineage_is_the_parent_chain_not_the_staircase(self):
        tree = build_tree(forest())
        assert tree.accepted == ("c000", "c001", "c002", "c004", "c007")
        assert best_lineage(tree) == ("c002", "c004", "c007")
        assert best_lineage(build_tree([])) == ()
        assert best_lineage(build_tree([cand("c001", status="buggy")])) == ()


class TestRowHeight:
    def test_small_tree_keeps_the_tree_views_row_height(self):
        assert row_height(build_tree(forest())) == ROW_H
        assert row_height(build_tree([])) == ROW_H
        assert row_height(None) == ROW_H
        wide = build_tree(wide_forest())
        assert wide.depth == 2
        assert row_height(wide) == pytest.approx(39.0 / ASPECT)
        assert ASPECT > 1

    def test_frame_sets_the_row_height_of_a_scrubbed_tree(self):
        wide = build_tree(wide_forest())
        early = build_tree(candidates_until(wide_forest(), "2026-08-22T10:03:30+00:00"), layout=wide)
        assert row_height(early) < row_height(wide)
        plot, ids = build_tree2_plot(early, frame=wide)
        full, full_ids = build_tree2_plot(wide)
        px = dict(zip(ids, plot.project_nodes(800, 400)))
        full_px = dict(zip(full_ids, full.project_nodes(800, 400)))
        assert all(px[i][:2] == full_px[i][:2] for i in ids)
        labels = {v.id: v for v in label_nodes(early, frame=wide)}
        assert labels["c001"].y == world_xy(wide.node("c001"), row_height(wide))[1]


class TestMarkSizing:
    def test_closest_pair_is_the_tightest_row_neighbours_or_rows(self):
        assert closest_pair_px([]) is None and closest_pair_px([(0.0, 0.0)]) is None
        # two rows 30 px apart, neighbours 10 px apart in a row
        points = [(0.0, 0.0), (10.0, 0.0), (25.0, 0.0), (5.0, 30.0), (40.0, 30.0)]
        assert closest_pair_px(points) == 10.0
        assert closest_pair_px([(0.0, 0.0), (100.0, 0.0), (0.0, 8.0)]) == 8.0
        # a row is one bucket even when projection jitters by a sub-pixel
        assert closest_pair_px([(0.0, 0.0), (12.0, 0.1), (24.0, -0.1)]) == 12.0

    def test_radius_keeps_rings_apart_and_is_clamped(self):
        assert radius_px(None) == R_MAX_PX
        assert radius_px(1.0) == R_MIN_PX
        assert radius_px(10_000.0) == R_MAX_PX
        closest = 40.0
        r = radius_px(closest)
        # two discs with their borders plus the gap fit the pair exactly
        assert 2 * r * (1 + BORDER) + 2 * GAP_PX == pytest.approx(closest)
        assert R_MIN_PX < r < R_MAX_PX

    def test_star_gets_its_own_budget(self):
        assert star_scale(10.0, None) == BEST_SIZE_SCALE
        # a far neighbour: the full star
        assert star_scale(10.0, 100.0) == BEST_SIZE_SCALE
        # a neighbour as close as the discs allow: the star, whose points
        # reach past its nominal radius, is held to the discs' size
        r = 10.0
        tight = 2 * r * (1 + BORDER) + 2 * GAP_PX
        assert star_scale(r, tight) == 1.0
        assert star_scale(r, tight / 2) == 1.0  # never below the discs
        # in between: exactly what fits, points included
        near = star_reach(r) + r * (1 + BORDER) + GAP_PX + 4.0
        mid = star_scale(r, near)
        assert 1.0 < mid < BEST_SIZE_SCALE
        assert star_reach(r * mid) + r * (1 + BORDER) + GAP_PX == pytest.approx(near)
        assert nearest_px([(0.0, 0.0), (3.0, 4.0), (30.0, 0.0)], 0) == 5.0
        assert nearest_px([(0.0, 0.0)], 0) is None

    def test_fit_radius_grows_with_zoom_and_shrinks_with_crowding(self):
        from plotui import Plot

        wide = build_tree(wide_forest())
        plot, _ = build_tree2_plot(wide)
        fit, _star = fit_radius(plot, 800, 400)
        plot.set_camera_state(0.0, 0.0, 4.0, 0.0, 0.0)
        zoomed, _star = fit_radius(plot, 800, 400)
        assert zoomed > fit
        small, _ = build_tree2_plot(build_tree(forest()))
        assert fit_radius(small, 800, 400)[0] > fit
        # the budget is in plotui units: a frame twice as wide projects the
        # pair twice as far apart but draws marks at twice the scale, so the
        # fitted radius is the same and the picture keeps its proportions
        assert Plot.mark_scale(1000) == 2 * Plot.mark_scale(500)
        assert fit_radius(plot, 1000, 500)[0] == pytest.approx(fit_radius(plot, 500, 250)[0], rel=0.05)
        # a zoom past the cap stops growing the marks
        plot.set_camera_state(0.0, 0.0, 40.0, 0.0, 0.0)
        assert fit_radius(plot, 800, 400) == (R_MAX_PX, BEST_SIZE_SCALE)

    def test_no_two_marks_overlap_at_the_fitted_radius(self):
        from plotui import Plot

        wide = build_tree(wide_forest())
        probe, ids = build_tree2_plot(wide)
        px_w, px_h = 800, 400
        best = ids.index(wide.best_id)
        # packed to the floor at the fit view: the floor wins over the gap
        assert fit_radius(probe, px_w, px_h, best_index=best)[0] == R_MIN_PX
        # zoomed so the row has room, the rule holds pair by pair — the
        # star, a sibling in the packed row, gets only what its neighbours leave
        probe.set_camera_state(0.0, 0.0, 3.0, 0.0, 0.0)
        r, star = fit_radius(probe, px_w, px_h, best_index=best)
        assert R_MIN_PX < r < R_MAX_PX
        assert 1.0 <= star < BEST_SIZE_SCALE
        plot, ids = build_tree2_plot(wide, radius=r, star=star)
        plot.set_camera_state(0.0, 0.0, 3.0, 0.0, 0.0)
        projected = plot.project_nodes(px_w, px_h)
        ms = Plot.mark_scale(px_w)
        reach = [
            (star_reach(r * star) if n.fate == "best" else node_radius(n, r) * (1 + BORDER)) * ms
            for n in wide.nodes
        ]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                dx = projected[i][0] - projected[j][0]
                dy = projected[i][1] - projected[j][1]
                assert (dx * dx + dy * dy) ** 0.5 >= reach[i] + reach[j]


class TestPlot:
    def test_one_trace_same_layout_as_tree(self):
        tree = build_tree(forest())
        plot, ids = build_tree2_plot(tree, selected="c004")
        assert ids == [n.id for n in tree.nodes]
        assert plot.node_count() == len(ids)
        assert plot.is_3d()
        projected = plot.project_nodes(1200, 800)
        base, _ = build_tree_plot(tree)
        assert [p[:2] for p in base.project_nodes(1200, 800)] == [p[:2] for p in projected]

    def test_rings_star_and_numbers_are_drawn(self):
        tree = build_tree(forest())
        plot, _ = build_tree2_plot(tree, radius=8.0)
        colours = rgba_colours(plot, 1200, 800)
        assert BEST_RING_RGB in colours                  # the spine's white: the star, the built-on rings, the line
        assert STAGE_RGB["failed"] in colours            # a red ring
        assert STAGE_RGB["expanded"] in colours
        assert VIRIDIS[-1] in colours                    # the top score's fill
        assert (242, 242, 246) in colours or (18, 18, 22) in colours  # a number's ink
        # marks too small to carry a number draw no ink
        tiny, _ = build_tree2_plot(tree, radius=1.0)
        small_colours = rgba_colours(tiny, 400, 300)
        assert (242, 242, 246) not in small_colours and (18, 18, 22) not in small_colours

    def test_lineage_line_needs_two_nodes(self):
        one = build_tree([cand("c000", "baseline", score=1.0)])
        plot, ids = build_tree2_plot(one)
        assert ids == ["c000"] and plot.node_count() == 1
        empty, ids = build_tree2_plot(build_tree([]))
        assert ids == [] and empty.node_count() == 0

    def test_frame_pins_a_scrubbed_tree(self):
        live = build_tree(forest())
        past = build_tree(candidates_until(forest(), "2026-08-22T10:04:30+00:00"), layout=live)
        live_plot, live_ids = build_tree2_plot(live)
        pinned_plot, pinned_ids = build_tree2_plot(past, frame=live)
        live_px = dict(zip(live_ids, live_plot.project_nodes(800, 400)))
        pinned_px = dict(zip(pinned_ids, pinned_plot.project_nodes(800, 400)))
        assert all(pinned_px[i][:2] == live_px[i][:2] for i in pinned_ids)

    def test_plot_arguments(self):
        # the border, label and shape channels are handed to plotui per node
        import hillclimb.tui.theme as theme

        calls: dict[str, list] = {}

        class FakePlot:
            def __getattr__(self, name):
                def record(*args, **kwargs):
                    calls.setdefault(name, []).append((args, kwargs))
                    return 0
                return record

        original = theme.themed_plot
        theme.themed_plot = lambda: FakePlot()
        try:
            build_tree2_plot(build_tree(forest()), radius=6.0)
        finally:
            theme.themed_plot = original
        (_a, graph), = calls["add_graph3d"]
        ids = [n.id for n in build_tree(forest()).nodes]
        best = ids.index("c007")
        assert graph["node_shapes"][best] == "star" and set(graph["node_shapes"]) == {"star", "disc"}
        assert graph["node_sizes"][best] == 6.0 * BEST_SIZE_SCALE
        (args, _), = calls["set_graph_borders"]
        assert args[1][best] == BEST_RING_RGB and args[1][ids.index("c005")] == STAGE_RGB["failed"]
        (args, _), = calls["set_graph_labels"]
        assert args[1] == [n.lstrip("c").lstrip("0") or "0" for n in ids]
        assert calls["add_line3d"]  # the lineage
        # the two lineage edges (c002→c004→c007) are drawn by the line, not as grey edges
        tree = build_tree(forest())
        assert len(graph["edges"]) == len(tree.edges) - 2
        lineage_pairs = {(ids.index("c002"), ids.index("c004")), (ids.index("c004"), ids.index("c007"))}
        assert not lineage_pairs & set(graph["edges"])


class TestLegend:
    def test_ladder_counts_star_and_ramp(self):
        tree = build_tree(forest())
        spans = legend_spans(tree, metric="score", higher_is_better=True, cols=60)
        text = {s[2].strip() for s in spans}
        assert {"score", "0.9", "0.1"} <= text  # the ramp; the ladder is plotui's legend (legend_entries)
        assert not any("expanded" in t for t in text)
        blocks = [s for s in spans if s[2] == "█"]
        assert len(blocks) == RAMP_ROWS
        assert all(s[1] == 58 for s in blocks)  # one column, at the right edge
        assert blocks[0][3] == "rgb(253,231,37)"  # top is the best end
        top = next(s for s in spans if s[2].strip() == "0.9")
        assert top[0] == blocks[0][0] and top[1] < 58

    def test_lower_is_better_puts_the_low_score_on_top(self):
        spans = legend_spans(build_tree(forest()), higher_is_better=False, cols=60)
        blocks = [s for s in spans if s[2] == "█"]
        bar_col = blocks[0][1]
        top = next(s for s in spans if s[0] == blocks[0][0] and 10 < s[1] < bar_col)
        assert top[2].strip() == "0.1"

    def test_entries_are_plotui_legend_rows_with_node_swatches(self):
        from hillclimb.tui.tree2 import LEGEND_ENTRIES, WHITE_RGB, hidden_fates, legend_entries

        assert LEGEND_ENTRIES == ("expanded", "scored", "failed", "best", "lineage")
        tree = build_tree(forest())
        rows = legend_entries(tree)
        assert [r[0] for r in rows] == ["1 expanded 2", "2 scored 3", "3 failed 2", "4 best", "5 lineage"]
        expanded, scored, failed, best, lineage = rows
        # the same fill for expanded and scored; the white ring is the only difference
        assert expanded[1:4] == ("disc", viridis(0.5), WHITE_RGB)
        assert scored[1:4] == ("disc", viridis(0.5), None)
        assert failed[1:4] == ("disc", UNSCORED_RGB, STAGE_RGB["failed"])
        assert best[1] == "star" and best[3] == BEST_RING_RGB
        assert lineage[1:4] == ("line", LINEAGE_RGB, None)
        assert all(r[4] for r in rows)
        # a hidden entry keeps its row, switched off
        hidden = legend_entries(tree, hidden={"failed", "lineage"})
        assert [r[4] for r in hidden] == [True, True, False, True, False]
        assert [r[0] for r in legend_entries(None)] == ["1 expanded", "2 scored", "3 failed", "4 best", "5 lineage"]
        # stages map onto the tree widget's fate filter; the lineage hides no node
        assert hidden_fates({"scored"}) == {"discontinued"}
        assert hidden_fates({"failed", "best"}) == {"failed", "pruned", "best"}
        assert hidden_fates({"lineage"}) == frozenset()

    def test_legend_is_drawn_in_the_plot_top_left_and_hit_by_row(self):
        from hillclimb.tui.tree2 import legend_entries

        tree = build_tree(forest())
        plot, _ids = build_tree2_plot(tree, legend=legend_entries(tree))
        assert [r[0] for r in plot.legend_entries()] == ["1 expanded 2", "2 scored 3", "3 failed 2", "4 best", "5 lineage"]
        w, h = 400, 300
        data = plot.render_rgba(w, h)
        top_left = {tuple(data[(y * w + x) * 4:(y * w + x) * 4 + 3]) for y in range(h // 2) for x in range(w // 2)}
        assert viridis(0.5) in top_left and STAGE_RGB["failed"] in top_left  # the discs' fill, the red ring
        assert plot.legend_entry_hit(w, h, 40.0, 12.0) == 0
        assert plot.legend_entry_hit(w, h, w - 5.0, h - 5.0) is None
        bare, _ids = build_tree2_plot(tree)
        assert bare.legend_entries() == [] and bare.legend_entry_hit(w, h, 40.0, 12.0) is None

    def test_show_lineage_off_drops_the_thick_line(self):
        tree = build_tree(forest())
        with_line = colour_pixels(build_tree2_plot(tree)[0], 320, 200, LINEAGE_RGB)
        without = colour_pixels(build_tree2_plot(tree, show_lineage=False)[0], 320, 200, LINEAGE_RGB)
        assert 0 < without < with_line  # the white rings stay, the thick line goes

    def test_hiding_the_best_keeps_the_lineage(self):
        from hillclimb.tui.treeview import filter_hidden
        from hillclimb.tui.tree2 import hidden_fates, lineage_nodes

        tree = build_tree(forest())
        without_best = filter_hidden(tree, hidden_fates({"best"}))
        assert [n.id for n in lineage_nodes(tree)] == ["c002", "c004", "c007"]
        assert lineage_nodes(without_best) == ()  # nothing to walk back from once the best is gone...
        # ...but drawn with the unfiltered tree's chain, the path still runs to the star's place:
        # the line trace adds one vertex per ancestor over the node marks
        plot, ids = build_tree2_plot(without_best, lineage=lineage_nodes(tree))
        assert "c007" not in ids and plot.vertex_count() == len(ids) + 3
        bare, _ids = build_tree2_plot(without_best)
        assert bare.vertex_count() == len(ids)

    def test_unscored_tree_has_no_ramp(self):
        spans = legend_spans(build_tree([cand("c001", status="buggy")]), cols=60)
        assert not [s for s in spans if s[2] == "█"]
        assert legend_spans(None, cols=40) == []  # no ramp without scores; the ladder is plotui's


class TestWidget:
    def test_screen_is_a_tree_screen_with_its_own_legend_keys(self):
        from hillclimb.tui.tree2view import Tree2Screen
        from hillclimb.tui.treeview import TreeScreen

        assert issubclass(Tree2Screen, TreeScreen)
        keys = [b.key for b in Tree2Screen.BINDINGS]
        assert {"1", "5", "j", "n", "l"} <= set(keys) and "6" not in keys  # five legend entries, no more
        assert keys.index("b") + 1 == keys.index("l")  # lineage sits right after best in the footer

    def test_hooks(self):
        from hillclimb.tui.tree2view import Tree2PlotWidget

        widget = Tree2PlotWidget()
        widget._ids = ["c000", "c001"]
        assert widget._flat_to_id(1) == "c001" and widget._flat_to_id(2) is None
        assert widget._legend_entry_at(1, 1) is None  # unmounted: no frame to hit-test against


def test_cli_lists_tree2():
    from typer.testing import CliRunner

    from hillclimb.cli import app

    result = CliRunner().invoke(app, ["tree2", "--help"])
    assert result.exit_code == 0
    assert "archive tree" in result.output


# --- Textual Pilot tests ---


@pytest.mark.asyncio
async def test_tree2_app_mounts_sizes_marks_selects_and_scrubs(tree_workspace):
    from hillclimb.tui.tree2view import Tree2App, Tree2Keys, Tree2PlotWidget

    _search_dir, config = tree_workspace
    app = Tree2App(config, "r1/circle-packing")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.screen.query_one("#tree-canvas", Tree2PlotWidget)
        assert len(canvas._ids) == 8
        assert canvas._plot.node_count() == 8
        assert canvas._plot.camera_state()[:2] == (0.0, 0.0)
        fitted = canvas.radius
        assert R_MIN_PX / 3 <= fitted <= R_MAX_PX
        assert 1.0 <= canvas.star <= BEST_SIZE_SCALE
        assert not canvas._label_cells  # nothing hovered or selected: no overlay labels
        # zooming in re-fits the marks: larger, never overlapping
        await pilot.press("plus")
        await pilot.pause()
        assert canvas.radius >= fitted
        await pilot.press("f")
        await pilot.pause()
        assert canvas._plot.camera_state()[2] == 1.0
        assert canvas.radius == pytest.approx(fitted)
        # b selects the best: its number is named beside the mark, detail opens
        await pilot.press("b")
        await pilot.pause()
        assert canvas.selected == "c007"
        assert set(canvas._label_cells.values()) == {"c007"}
        assert app.screen.query_one("#node-detail").styles.display == "block"
        # the legend filters: 1 hides the built-on nodes, again shows them; 5 hides the lineage line
        await pilot.press("1")
        await pilot.pause()
        assert canvas.hidden == {"expanded"} and len(canvas._ids) == 6
        rows = canvas._plot.legend_entries()
        assert rows[0][0] == "1 expanded 2" and rows[0][4] is False and rows[1][4] is True
        # a click on the first legend row resolves through plotui's hit test
        assert canvas._legend_entry_at(4, 1) == "expanded"
        # the lit legend row survives a rebuild (a hover elsewhere rebuilds the plot)
        canvas._plot.set_legend_hover_index(2)
        canvas.rebuild()
        assert canvas._plot.legend_hover() == 2
        canvas._plot.set_legend_hover_index(None)
        await pilot.press("1")
        await pilot.pause()
        assert not canvas.hidden and len(canvas._ids) == 8
        with_line = colour_pixels(canvas._plot, 320, 200, LINEAGE_RGB)
        await pilot.press("5")
        await pilot.pause()
        assert canvas.hidden == {"lineage"} and len(canvas._ids) == 8
        assert 0 < colour_pixels(canvas._plot, 320, 200, LINEAGE_RGB) < with_line  # rings stay, line goes
        await pilot.press("l")  # the lineage's own key brings it back
        await pilot.pause()
        assert not canvas.hidden and LINEAGE_RGB in rgba_colours(canvas._plot, 320, 200)
        await pilot.press("4")  # hiding the best removes the star only: the lineage stays
        await pilot.pause()
        assert canvas.hidden == {"best"} and len(canvas._ids) == 7 and "c007" not in canvas._ids
        assert LINEAGE_RGB in rgba_colours(canvas._plot, 320, 200)
        await pilot.press("4")
        await pilot.pause()
        await pilot.press("j")
        await pilot.pause()
        assert app.screen._tree.best_id != "c007"
        await pilot.press("end")
        await pilot.pause()
        assert app.screen._tree.best_id == "c007"
        # the footer reads esc, b, l, n …: lineage right after best
        keys = list(app.screen._bindings.key_to_bindings)
        assert keys.index("b") < keys.index("l") < keys.index("n")
        await pilot.press("question_mark")
        await pilot.pause()
        assert app.screen.query(Tree2Keys)
        await pilot.press("escape")  # closes the selection first
        await pilot.pause()
        assert canvas.selected is None
