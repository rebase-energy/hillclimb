"""Exploration tree: layout + fates (tree.py), the plotui adapter and legend
(treeview.py), the chart's detail layout, and the Textual screen."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.style import Style

from hillclimb.candidate import Candidate, Trial
from hillclimb.chart import build_detail_plot, detail_layout
from hillclimb.config import Config
from hillclimb.journal import Journal
from hillclimb.tree import (
    FATES,
    accepted_lineage,
    build_tree,
    candidates_until,
    minutes_since,
    tree_events,
)
from hillclimb.treeview import (
    BEST_LABEL_STYLE,
    FATE_SHAPE,
    OPERATOR_RGB,
    build_tree_plot,
    label_nodes,
    legend_spans,
    node_label,
    statusline,
    world_xy,
)
from tests.test_watch import make_run_with_search


def cand(
    cid: str, operator: str = "improve", parent: str | None = None, score: float | None = None,
    status: str = "ok", t: int = 0, **kwargs,
) -> Candidate:
    """A candidate created at minute `t` and finished a minute later."""
    return Candidate(
        candidate_id=cid, operator=operator, parent_id=parent, status=status,
        trials=[Trial(val_score=score)] if score is not None else [],
        created_at=f"2026-08-22T10:{t:02d}:00+00:00",
        finished_at=f"2026-08-22T10:{t + 1:02d}:00+00:00",
        **kwargs,
    )


def forest() -> list[Candidate]:
    """baseline + two drafts; c002 expanded twice, c004 once more, c005 failed,
    c006 pruned, one ensemble with an extra input."""
    return [
        cand("c000", "baseline", score=0.1, t=0),
        cand("c001", "draft", score=0.5, t=1),
        cand("c002", "draft", score=0.6, t=2),
        cand("c003", parent="c002", score=0.55, t=3),
        cand("c004", parent="c002", score=0.7, t=4),
        cand("c005", parent="c004", status="buggy", t=5),
        cand("c006", parent="c004", score=0.9, t=6, pruned=True),
        cand("c007", "ensemble", parent="c004", score=0.8, t=7,
             policy_meta={"inspiration_ids": ["c004", "c001"]}),
    ]


class TestBuildTree:
    def test_fates_and_lineage(self):
        tree = build_tree(forest(), higher_is_better=True)
        fate = {n.id: n.fate for n in tree.nodes}
        assert fate == {
            "c000": "discontinued", "c001": "discontinued", "c002": "expanded",
            "c003": "discontinued", "c004": "expanded", "c005": "failed",
            "c006": "pruned", "c007": "best",
        }
        # pruned c006 (0.9) never enters the lineage
        assert tree.accepted == ("c000", "c001", "c002", "c004", "c007")
        assert tree.best_id == "c007"
        assert tree.depth == 3
        assert tree.fate_counts() == {
            "expanded": 2, "best": 1, "discontinued": 3, "failed": 1, "pruned": 1,
        }
        assert set(tree.fate_counts()) == set(FATES)

    def test_lower_is_better_flips_the_lineage(self):
        assert accepted_lineage(forest(), higher_is_better=False) == ["c000"]

    def test_tidy_layout(self):
        tree = build_tree(forest())
        at = {n.id: (n.x, n.depth) for n in tree.nodes}
        # roots at depth 0 in creation order; leaves take successive columns
        assert at["c000"] == (0.0, 0)
        assert at["c001"] == (1.0, 0)
        assert at["c003"] == (2.0, 1)
        assert [at[c][0] for c in ("c005", "c006", "c007")] == [3.0, 4.0, 5.0]
        # a parent sits over the centre of its children
        assert at["c004"] == (4.0, 1)
        assert at["c002"] == (3.0, 0)
        # siblings never share a column with each other or a cousin
        columns = [n.x for n in tree.nodes if n.n_children == 0]
        assert len(columns) == len(set(columns))

    def test_edges(self):
        tree = build_tree(forest())
        parents = {(e.src, e.dst): e.on_path for e in tree.edges if e.kind == "parent"}
        assert parents[("c002", "c004")] is True
        assert parents[("c002", "c003")] is False
        assert parents[("c004", "c007")] is True
        extra = [(e.src, e.dst) for e in tree.edges if e.kind == "ensemble-input"]
        assert extra == [("c001", "c007")]  # the parent itself is not repeated

    def test_unknown_parent_becomes_a_root(self):
        tree = build_tree([cand("c009", parent="ghost", score=1.0)])
        assert tree.nodes[0].depth == 0 and tree.nodes[0].parent_id is None
        assert tree.edges == ()

    def test_empty(self):
        tree = build_tree([])
        assert tree.nodes == () and tree.depth == 0 and tree.best_id is None

    def test_pending_fate(self):
        pending = Candidate(candidate_id="c001", operator="draft", status="pending")
        assert build_tree([pending]).nodes[0].fate == "pending"


class TestTimeScrub:
    def test_events_are_landed_results_oldest_first(self):
        assert tree_events(forest())[:2] == ["2026-08-22T10:01:00+00:00", "2026-08-22T10:02:00+00:00"]

    def test_frame_pins_the_projection_of_a_scrubbed_tree(self):
        from hillclimb.treeview import build_tree_plot

        live = build_tree(forest())
        past = build_tree(candidates_until(forest(), "2026-08-22T10:04:30+00:00"), layout=live)
        assert len(past.nodes) < len(live.nodes)
        live_plot, live_ids = build_tree_plot(live)
        loose_plot, _ = build_tree_plot(past)
        pinned_plot, pinned_ids = build_tree_plot(past, frame=live)
        live_px = dict(zip(live_ids, live_plot.project_nodes(800, 400)))
        loose_px = dict(zip(pinned_ids, loose_plot.project_nodes(800, 400)))
        pinned_px = dict(zip(pinned_ids, pinned_plot.project_nodes(800, 400)))
        # without the frame the subset re-centres; with it every node keeps its pixel
        assert any(loose_px[i][:2] != live_px[i][:2] for i in pinned_ids)
        assert all(pinned_px[i][:2] == live_px[i][:2] for i in pinned_ids)

    def test_layout_pins_a_scrubbed_tree_to_the_live_positions(self):
        live = build_tree(forest())
        past = build_tree(candidates_until(forest(), "2026-08-22T10:04:30+00:00"))
        pinned = build_tree(candidates_until(forest(), "2026-08-22T10:04:30+00:00"), layout=live)
        assert len(pinned.nodes) == len(past.nodes) < len(live.nodes)
        # on its own the smaller tree re-packs its columns; pinned, every node
        # sits exactly where the live tree draws it
        assert any(past.node(n.id).x != live.node(n.id).x for n in past.nodes)
        assert all((n.x, n.depth) == (live.node(n.id).x, live.node(n.id).depth) for n in pinned.nodes)
        assert pinned.accepted == past.accepted  # fates are still the past's

    def test_candidates_until_hides_the_future_and_pends_the_in_flight(self):
        view = candidates_until(forest(), "2026-08-22T10:04:30+00:00")
        ids = [c.candidate_id for c in view]
        assert ids == ["c000", "c001", "c002", "c003", "c004"]
        # c004 was created at :04 and finished at :05 — in flight at :04:30
        assert view[-1].status == "pending" and view[-1].val_score is None
        assert [c.candidate_id for c in candidates_until(forest(), None)] == [c.candidate_id for c in forest()]

    def test_minutes_since(self):
        assert minutes_since("2026-08-22T10:05:00+00:00", "2026-08-22T10:00:00+00:00") == 5.0
        assert minutes_since("2026-08-22T09:00:00+00:00", "2026-08-22T10:00:00+00:00") == 0.0
        assert minutes_since(None, "2026-08-22T10:00:00+00:00") is None
        assert minutes_since("nope", "2026-08-22T10:00:00+00:00") is None


class TestTreePlot:
    def test_plot_is_flat_and_indexed(self):
        tree = build_tree(forest())
        plot, ids = build_tree_plot(tree, selected="c004")
        assert ids == [n.id for n in tree.nodes]
        assert plot.node_count() == len(ids)
        assert plot.is_3d()  # Graph3d trace, face-on camera
        px_w, px_h = 1200, 800
        projected = plot.project_nodes(px_w, px_h)
        # depth goes down the screen, columns go right
        y_of = {nid: projected[i][1] for i, nid in enumerate(ids)}
        x_of = {nid: projected[i][0] for i, nid in enumerate(ids)}
        assert y_of["c000"] < y_of["c003"] < y_of["c005"]
        assert x_of["c000"] < x_of["c001"] < x_of["c003"]
        assert all(p[2] == pytest.approx(0.0) for p in projected)

    def test_empty_plot(self):
        plot, ids = build_tree_plot(build_tree([]))
        assert ids == [] and plot.node_count() == 0

    def test_shapes_cover_every_fate(self):
        assert set(FATE_SHAPE) >= set(FATES) | {"pending"}

    def test_world_xy_and_labels(self):
        tree = build_tree(forest())
        best = tree.node("c007")
        assert world_xy(best) == (5.0, -2 * 1.6)
        assert node_label(best) == "c007 ★ 0.8"  # the best wears a star
        assert node_label(tree.node("c001")) == "c001 0.5"
        assert node_label(tree.node("c005")) == "c005"
        labels = label_nodes(tree)
        assert [v.id for v in labels] == [n.id for n in tree.nodes]
        by_id = {v.id: v for v in labels}
        # the best's label is gold and wins any crowd; lineage nodes outrank the rest
        assert by_id["c007"].style == BEST_LABEL_STYLE and by_id["c007"].count == 3
        assert by_id["c004"].style == "" and by_id["c004"].count == 2
        assert by_id["c003"].count == 1

    def test_legend_spans_and_hit_test(self):
        from hillclimb.treeview import (
            LEGEND_COL, LEGEND_ENTRIES, LEGEND_ROW, LEGEND_WIDTH, filter_hidden, legend_entry_at,
        )

        tree = build_tree(forest())
        hidden = frozenset({"failed", "draft"})
        lines: dict[int, list[tuple[str, str]]] = {}
        for row, col, text, style in legend_spans(tree, hidden):
            assert LEGEND_COL <= col < LEGEND_COL + LEGEND_WIDTH
            lines.setdefault(row, []).append((text, style))
        assert sorted(lines) == list(range(LEGEND_ROW, LEGEND_ROW + len(LEGEND_ENTRIES)))
        by_entry = {LEGEND_ENTRIES[r - LEGEND_ROW]: parts for r, parts in lines.items()}
        assert by_entry["improve"][0] == ("4 ", "dim")
        assert by_entry["improve"][1] == ("● improve", "rgb(47,191,113)")
        assert by_entry["discontinued"][1] == ("◯ discontinued 3", "white")
        assert by_entry["pruned"][0][0] == "0 "  # the tenth entry
        # hidden entries lose their glyph and go dim
        assert by_entry["failed"][1] == ("  failed 1", "dim strike")
        assert by_entry["draft"][1] == ("  draft", "dim strike")
        # the empty legend still names everything
        assert len(legend_spans(None)) == 2 * len(LEGEND_ENTRIES)
        # each line's cells hit its entry; around the legend is nothing
        for index, entry in enumerate(LEGEND_ENTRIES):
            assert legend_entry_at(LEGEND_COL, LEGEND_ROW + index) == entry
            assert legend_entry_at(LEGEND_COL + LEGEND_WIDTH - 1, LEGEND_ROW + index) == entry
        assert legend_entry_at(LEGEND_COL + LEGEND_WIDTH, LEGEND_ROW) is None
        assert legend_entry_at(LEGEND_COL, LEGEND_ROW + len(LEGEND_ENTRIES)) is None
        assert legend_entry_at(LEGEND_COL, LEGEND_ROW - 1) is None
        # the filter drops the nodes and their edges, keeps positions
        shown = filter_hidden(tree, hidden)
        assert {n.id for n in shown.nodes} == {"c000", "c003", "c004", "c006", "c007"}
        assert all(e.src in {n.id for n in shown.nodes} for e in shown.edges)
        assert shown.node("c004").x == tree.node("c004").x
        assert filter_hidden(tree, frozenset()) is tree

    def test_statusline(self):
        line = statusline("r1/p", "running", build_tree(forest()), "score", True)
        assert "8 candidates · depth 3" in line
        assert "best [bold]c007 ★[/] score=0.8 (higher is better)" in line
        assert "n/p" not in line
        line = statusline("r1/p", "running", build_tree(forest()), "score", True, position=(1, 3))
        assert "(2/3, n/p to switch)" in line


class TestChartDetail:
    def test_marks_edges_and_curve_agree(self):
        layout = detail_layout(
            forest(), label="r1/p", state="done", started_at="2026-08-22T10:00:00+00:00"
        )
        by_id = {m.id: m for m in layout.marks}
        # scored, unpruned only — c005 failed and c006 pruned are left out
        assert set(by_id) == {"c000", "c001", "c002", "c003", "c004", "c007"}
        assert layout.unscored == 2
        assert by_id["c007"].x == 8.0 and by_id["c007"].y == 0.8 and by_id["c007"].on_path
        assert by_id["c003"].on_path is False
        edges = {((e.x0, e.y0), (e.x1, e.y1)): e.on_path for e in layout.edges}
        assert edges[((3.0, 0.6), (5.0, 0.7))] is True   # c002 -> c004
        assert edges[((3.0, 0.6), (4.0, 0.55))] is False  # c002 -> c003
        assert len(layout.edges) == 3  # c005/c006 children are not drawn
        # the staircase is the chart's own curve
        assert layout.curve.ys == [0.1, 0.5, 0.6, 0.6, 0.7, 0.8]

    def test_empty_detail(self):
        layout = detail_layout([cand("c001", status="buggy")], label="x", state="done")
        assert layout.marks == [] and layout.edges == [] and layout.unscored == 1
        assert build_detail_plot(layout).is_3d() is False

    def test_detail_plot_builds(self):
        layout = detail_layout(forest(), label="r1/p", state="done")
        plot = build_detail_plot(layout)
        assert plot.is_3d() is False


# --- Textual Pilot tests ---


@pytest.fixture
def tree_workspace(tmp_path, monkeypatch) -> tuple[Path, Config]:
    monkeypatch.setenv("PLOTUI_RENDER", "placeholder")
    search_dir = make_run_with_search(tmp_path / "runs", "r1")
    journal = Journal(search_dir / "journal.jsonl")
    for c in forest()[3:]:
        journal.candidate_result(c.model_copy(update={"parent_id": "c001" if c.parent_id == "c002" else c.parent_id}))
    config = Config()
    config.paths.runs_dir = tmp_path / "runs"
    return search_dir, config


@pytest.mark.asyncio
async def test_tree_app_mounts_selects_scrubs_and_opens(tree_workspace):
    from hillclimb.treeview import TreeApp, TreePlotWidget, TreeKeys
    from hillclimb.watch import CandidateScreen

    _search_dir, config = tree_workspace
    app = TreeApp(config, "r1/circle-packing")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.screen.query_one("#tree-canvas", TreePlotWidget)
        assert len(canvas._ids) == 8
        yaw, pitch, zoom, _px, _py = canvas._plot.camera_state()
        assert (yaw, pitch) == (0.0, 0.0)
        await pilot.press("plus")
        assert canvas._plot.camera_state()[2] > zoom
        # rotation is locked: arrows pan, the camera stays face-on
        await pilot.press("left")
        assert canvas._plot.camera_state()[:2] == (0.0, 0.0)
        await pilot.press("f")
        assert canvas._plot.camera_state()[2] == 1.0
        # b selects the best and opens its detail
        await pilot.press("b")
        await pilot.pause()
        assert canvas.selected == "c007"
        assert app.screen.query_one("#node-detail").styles.display == "block"
        # scrub one tick back: the last landed result drops out
        await pilot.press("j")
        await pilot.pause()
        assert app.screen._tree.best_id != "c007"  # c007 is still in flight at that tick
        await pilot.press("end")
        await pilot.pause()
        assert app.screen._tree.best_id == "c007"
        await pilot.press("question_mark")
        await pilot.pause()
        assert app.screen.query(TreeKeys)
        # legend: `4` hides improve (most of the tree), again shows it; a click
        # on the legend line toggles too
        from hillclimb.treeview import LEGEND_COL, LEGEND_ROW

        await pilot.press("4")
        await pilot.pause()
        assert canvas.hidden == {"improve"} and len(canvas._ids) == 3
        await pilot.press("4")
        await pilot.pause()
        assert not canvas.hidden and len(canvas._ids) == 8
        await pilot.click("#tree-canvas", offset=(LEGEND_COL + 3, LEGEND_ROW + 3))
        await pilot.pause()
        assert canvas.hidden == {"improve"}
        await pilot.press("4")
        await pilot.pause()
        # enter opens the selected candidate in the candidate screen
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, CandidateScreen)


@pytest.mark.asyncio
async def test_tree_app_drag_pans_and_switches_searches(tree_workspace):
    from hillclimb.treeview import TreeApp, TreePlotWidget

    _search_dir, config = tree_workspace
    # a second, later search with a smaller (3-node) tree: the one n/p moves to
    second = make_run_with_search(config.paths.runs_dir, "r2")
    meta = second / "search.yaml"
    meta.write_text(meta.read_text().replace("started_at:", "started_at_old:") + "\nstarted_at: '2099-01-01T00:00:00'\n")
    app = TreeApp(config, "r1/circle-packing")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.screen.query_one("#tree-canvas", TreePlotWidget)
        assert len(canvas._ids) == 8
        assert "(1/2, n/p to switch)" in str(app.screen.query_one("#treeline").render())
        # a plain drag pans by exactly the cells the pointer moved, and the
        # camera stays face-on
        await pilot.press("b")
        await pilot.pause()
        before = canvas._plot.camera_state()
        await pilot.mouse_down("#tree-canvas", offset=(40, 20))
        await pilot.hover("#tree-canvas", offset=(50, 20))
        await pilot.mouse_up("#tree-canvas", offset=(50, 20))
        await pilot.pause()
        after = canvas._plot.camera_state()
        assert after[:2] == (0.0, 0.0)
        assert after[3] - before[3] == pytest.approx(10 * canvas._cell_w)
        assert after[4] == before[4]
        assert canvas.selected == "c007"  # a drag is not a click
        # n moves to the next search: new tree, selection cleared, pan kept
        await pilot.press("n")
        await pilot.pause()
        assert app.screen._record.ref == "r2/circle-packing"
        assert len(canvas._ids) == 3
        assert canvas.selected is None
        assert canvas._plot.camera_state()[3] == after[3]
        assert "(2/2, n/p to switch)" in str(app.screen.query_one("#treeline").render())
        # and p wraps back
        await pilot.press("p")
        await pilot.pause()
        assert app.screen._record.ref == "r1/circle-packing"
        assert len(canvas._ids) == 8
        # hovering a candidate's name (not just its mark) highlights it, and
        # clicking the name selects it; the best's label is drawn in gold
        await pilot.press("f")
        await pilot.pause()
        cells = {node_id: cell for cell, node_id in canvas._label_cells.items()}
        assert "c007" in cells
        row, col = cells["c007"]
        gold = [
            (text, style) for spans in canvas._overlay.values() for _c, text, style in spans
            if text.startswith("c007 ★")
        ]
        assert gold and gold[0][1] == Style.parse(BEST_LABEL_STYLE)
        await pilot.hover("#tree-canvas", offset=(col, row))
        await pilot.pause()
        assert canvas._hover == "c007"
        await pilot.click("#tree-canvas", offset=(col, row))
        await pilot.pause()
        assert canvas.selected == "c007"


@pytest.mark.asyncio
async def test_tree_app_reports_a_missing_search(tmp_path, monkeypatch):
    from hillclimb.treeview import TreeApp

    monkeypatch.setenv("PLOTUI_RENDER", "placeholder")
    config = Config()
    config.paths.runs_dir = tmp_path / "runs"
    app = TreeApp(config, "nope/none")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert "No search at" in str(app.screen.query_one("#treeline").render())


@pytest.mark.asyncio
async def test_chart_detail_toggle(tree_workspace):
    from hillclimb.chart import ChartApp
    from plotui.textual import PlotWidget

    _search_dir, config = tree_workspace
    app = ChartApp(config, "r1/circle-packing", detail=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        line = str(app.screen.query_one("#chartline").render())
        assert "detail:" in line and "scored" in line
        first = app.screen.query_one("#chart-canvas", PlotWidget)
        await pilot.press("d")
        await pilot.pause()
        assert "detail:" not in str(app.screen.query_one("#chartline").render())
        # the widget is reused — the plot is swapped, not remounted
        assert app.screen.query_one("#chart-canvas", PlotWidget) is first
