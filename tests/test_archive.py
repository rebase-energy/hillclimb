"""The two-panel archive view (`hillclimb archive`): the progress chart's
candidate-number axis, best-so-far in number order, the best's lineage as a
line, the scrub cursor, the plot (archive.py), and the screen that steps
both panels together (archiveview.py)."""

from __future__ import annotations

import pytest

from hillclimb.archive import (
    CURSOR_RGB,
    LEGEND_COL,
    LEGEND_TOP_ROW,
    SELECTED_RGB,
    SERIES,
    build_progress_plot,
    counts_as_scored,
    candidate_numbers,
    cursor_number,
    legend_entry_at,
    legend_rows,
    legend_series,
    legend_spans,
    lineage_points,
    progress_bounds,
    progress_extent,
    progress_legend,
    progress_points,
    staircase,
)
from hillclimb.chart import CYAN, MISS_RGB
from hillclimb.tree import build_tree, candidates_until, tree_events
from hillclimb.tree2 import LINEAGE_RGB, best_lineage
from tests.test_tree import cand, forest, tree_workspace  # noqa: F401 (fixture)
from tests.test_tree2 import colour_pixels, rgba_colours


def test_candidate_numbers_are_the_circle_numbers_or_creation_order():
    tree = build_tree(forest())
    assert candidate_numbers(tree) == {f"c{i:03d}": i for i in range(8)}
    odd = build_tree([cand("seed", "baseline", score=0.1, t=0), cand("c001", parent="seed", score=0.2, t=1)])
    assert candidate_numbers(odd) == {"seed": 0, "c001": 1}


def test_progress_points_climb_by_number_and_flag_the_lineage():
    tree = build_tree(forest())
    points = progress_points(tree)
    # pruned c006 and buggy c005 are not dots; the rest sit at their number
    assert [p.id for p in points] == ["c000", "c001", "c002", "c003", "c004", "c007"]
    assert [p.number for p in points] == [0, 1, 2, 3, 4, 7]
    assert [p.id for p in points if p.best] == ["c000", "c001", "c002", "c004", "c007"]
    assert best_lineage(tree) == ("c002", "c004", "c007")
    assert [p.id for p in lineage_points(points)] == ["c002", "c004", "c007"]
    assert not any(counts_as_scored(n) for n in tree.nodes if n.id in ("c005", "c006"))
    # lower-is-better flips what counts as an improvement, not the order
    lower = progress_points(build_tree(forest(), higher_is_better=False), higher_is_better=False)
    assert [p.id for p in lower if p.best] == ["c000"]


def test_landing_order_does_not_move_the_staircase():
    # c003 (a later number) lands before c002 (t=1 vs t=5): the chart still walks 0,1,2,3
    tree = build_tree([
        cand("c000", "baseline", score=0.1, t=0), cand("c001", "draft", score=0.3, t=1),
        cand("c003", parent="c001", score=0.5, t=1), cand("c002", parent="c001", score=0.4, t=5),
    ])
    points = progress_points(tree)
    assert [(p.number, p.best) for p in points] == [(0, True), (1, True), (2, True), (3, True)]
    assert tree.best_id == "c003"  # the same final best as the landing-order staircase


def test_staircase_runs_flat_to_the_extent():
    tree = build_tree(forest())
    points = progress_points(tree)
    xs, ys = staircase(points, progress_extent(tree))
    assert progress_extent(tree) == 7
    assert xs[0] == 0.0 and xs[-1] == 7.0 and ys[-1] == 0.8
    assert staircase([], 5) == ([], [])


def test_cursor_is_the_candidate_that_landed_at_the_tick():
    candidates = forest()
    tree = build_tree(candidates)
    events = tree_events(candidates)
    assert cursor_number(tree, None) is None
    assert cursor_number(tree, events[0]) == 0
    assert cursor_number(tree, events[-1]) == 7
    assert cursor_number(tree, "2030-01-01T00:00:00+00:00") is None
    # the scrubbed tree, framed by the live one, keeps the cursor at the tick
    scrubbed = build_tree(candidates_until(candidates, events[3]), layout=tree)
    assert cursor_number(scrubbed, events[3]) == 3
    assert [p.id for p in progress_points(scrubbed)] == ["c000", "c001", "c002", "c003"]


def test_bounds_come_from_the_frame_and_pad_both_axes():
    tree = build_tree(forest())
    (x_lo, x_hi), y = progress_bounds(tree)
    assert x_lo < 0 < 7 < x_hi
    assert y is not None and y[0] < 0.1 and y[1] > 0.9  # pruned c006's 0.9 is in the score range
    empty = build_tree([cand("c000", "baseline", status="buggy")])
    assert progress_bounds(empty) == ((-1.0, 1.0), None)


def test_progress_plot_draws_lineage_cursor_and_selection():
    tree = build_tree(forest())
    plot = build_progress_plot(tree, cursor=4, selected="c001")
    colours = rgba_colours(plot, 320, 200)
    assert LINEAGE_RGB in colours and CYAN in colours and MISS_RGB in colours
    assert CURSOR_RGB in colours and SELECTED_RGB in colours
    assert not plot.is_3d()
    bare = build_progress_plot(tree)
    assert CURSOR_RGB not in rgba_colours(bare, 320, 200)
    assert SELECTED_RGB not in rgba_colours(bare, 320, 200)
    # hiding the lineage drops its colour; the frame pins the axes while scrubbing
    without = build_progress_plot(tree, hidden={"lineage"})
    assert LINEAGE_RGB not in rgba_colours(without, 320, 200)
    first = build_tree(candidates_until(forest(), tree_events(forest())[0]), layout=tree)
    framed = build_progress_plot(first, frame=tree)
    assert framed.x_range() == bare.x_range() and framed.y_range() == bare.y_range()
    # ...but the staircase stops at the scrubbed tree's own last candidate
    # (c001 is already created, still pending, at c000's landing tick)
    assert staircase(progress_points(first), progress_extent(first))[0][-1] == 1.0
    assert staircase(progress_points(tree), progress_extent(tree))[0][-1] == 7.0
    assert rgba_colours(build_progress_plot(build_tree([])), 64, 32)  # an empty tree still draws the frame


def test_legend_names_the_lineage_length_and_maps_back_to_the_series():
    points = progress_points(build_tree(forest()))
    labels = [label for label, _rgb, _glyph in progress_legend(points)]
    assert labels == ["attempt", "best so far", "new best", "lineage (3)"]
    assert legend_series("lineage (3)") == "lineage" and legend_series("attempt") == "attempt"
    assert [label for label, *_ in progress_legend([])][-1] == "lineage"


def test_legend_sits_in_the_empty_corner_and_hit_tests():
    points = progress_points(build_tree(forest()))
    top = legend_spans(points, True, rows=30)
    assert top[0][0] == LEGEND_TOP_ROW and top[0][1] == LEGEND_COL
    texts = [t for _r, _c, t, _s in top]
    assert "● attempt" in texts and "━ lineage (3)" in texts and "1 " in texts
    # lower is better: the climb falls to the bottom right, so the legend goes bottom left
    bottom = legend_spans(points, False, rows=30)
    assert legend_rows(False, 4, 30) == 30 - 4 - 4
    assert {r for r, *_ in bottom} == set(range(22, 26))
    assert legend_rows(False, 4, 5) == LEGEND_TOP_ROW  # a tiny pane never pushes it above the top
    # a hidden series is struck through
    struck = legend_spans(points, True, hidden={"lineage"}, rows=30)
    assert any(t == "  lineage (3)" and st == "dim strike" for _r, _c, t, st in struck)
    # hit test: the row picks the series, the column must be on the text
    assert legend_entry_at(LEGEND_COL + 3, LEGEND_TOP_ROW + 3, True, rows=30) == "lineage"
    assert legend_entry_at(LEGEND_COL + 3, 22, False, rows=30) == SERIES[0]
    assert legend_entry_at(0, LEGEND_TOP_ROW, True, rows=30) is None
    assert legend_entry_at(LEGEND_COL, LEGEND_TOP_ROW + 4, True, rows=30) is None


def test_cli_lists_archive():
    from typer.testing import CliRunner

    from hillclimb.cli import app

    result = CliRunner().invoke(app, ["archive", "--help"])
    assert result.exit_code == 0
    assert "progress chart" in result.output


# --- Textual Pilot tests ---


@pytest.mark.asyncio
async def test_archive_app_scrubs_both_panels_and_rings_the_selection(tree_workspace):
    from hillclimb.archiveview import ArchiveApp, ArchiveKeys, ProgressPlotWidget
    from hillclimb.tree2view import Tree2PlotWidget

    def overlay_text(widget) -> str:
        return " ".join(text for spans in widget._overlay.values() for _col, text, _style in spans)

    _search_dir, config = tree_workspace
    app = ArchiveApp(config, "r1/circle-packing")
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        tree = app.screen.query_one("#tree-canvas", Tree2PlotWidget)
        chart = app.screen.query_one("#progress-canvas", ProgressPlotWidget)
        assert len(tree._ids) == 8
        assert tree.image_ids != chart.image_ids  # two plots, two Kitty image-id pairs
        assert chart.cursor is None and chart.selected is None
        assert LINEAGE_RGB in rgba_colours(chart._plot, 320, 200)
        assert "lineage (3)" in overlay_text(chart)  # the legend is drawn over the chart's top left
        assert min(chart._overlay) == LEGEND_TOP_ROW
        # j steps both: the tree loses its best, the chart gains a cursor at the tick's candidate
        await pilot.press("j")
        await pilot.pause()
        assert app.screen._tree.best_id != "c007"
        assert chart.cursor == 6
        assert CURSOR_RGB in rgba_colours(chart._plot, 320, 200)
        assert "at 10:07" not in overlay_text(chart)  # the scrubber's label carries the time, not the legend
        assert "candidate 4 of 5" in str(app.screen.query_one("#time-scrubber").render())
        await pilot.press("k")
        await pilot.pause()
        assert chart.cursor is None and app.screen._tree.best_id == "c007"
        # b selects the best in the tree and rings it on the chart
        await pilot.press("b")
        await pilot.pause()
        assert tree.selected == "c007" and chart.selected == "c007"
        assert SELECTED_RGB in rgba_colours(chart._plot, 320, 200)
        spine = colour_pixels(tree._plot, 320, 200, LINEAGE_RGB)
        # two legends, one row of hotkeys: 1-5 filter the tree, 6-9 the chart's series
        await pilot.press("5")
        await pilot.pause()
        assert tree.hidden == {"lineage"} and not chart.hidden
        assert 0 < colour_pixels(tree._plot, 320, 200, LINEAGE_RGB) < spine  # the rings stay, the line goes
        assert LINEAGE_RGB in rgba_colours(chart._plot, 320, 200)
        await pilot.press("5")
        await pilot.pause()
        assert not tree.hidden
        assert "6 " in overlay_text(chart) and "1 " not in overlay_text(chart)
        # l hides the lineage in both panels at once, and brings them back in step
        await pilot.press("l")
        await pilot.pause()
        assert tree.hidden == {"lineage"} and chart.hidden == {"lineage"}
        assert colour_pixels(tree._plot, 320, 200, LINEAGE_RGB) < spine
        assert LINEAGE_RGB not in rgba_colours(chart._plot, 320, 200)
        await pilot.press("l")
        await pilot.pause()
        assert not tree.hidden and not chart.hidden
        await pilot.press("5")  # out of step: only the tree's is hidden...
        await pilot.press("l")  # ...so l shows both rather than hiding the chart's
        await pilot.pause()
        assert not tree.hidden and not chart.hidden
        # the chart's lineage: its hotkey, then a click on the entry
        await pilot.press("9")
        await pilot.pause()
        assert chart.hidden == {"lineage"} and not tree.hidden
        assert LINEAGE_RGB not in rgba_colours(chart._plot, 320, 200)
        assert "  lineage (3)" in overlay_text(chart)
        chart.post_message(ProgressPlotWidget.SeriesToggled("lineage"))
        await pilot.pause()
        assert not chart.hidden
        await pilot.press("question_mark")
        await pilot.pause()
        assert app.screen.query(ArchiveKeys)
        await pilot.press("escape")
        await pilot.pause()
        assert tree.selected is None and chart.selected is None
