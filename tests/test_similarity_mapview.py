"""`similarity map`'s rendering layer (styles, highlight, replay frames,
statusline, the plot build) and the CLI's routing of the two views."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from hillclimb import cli
from hillclimb.similarity import clear_caches
from hillclimb.similarity_map import METRICS, build_map, build_run_map
from hillclimb.similarity_mapview import (
    BEST_RGB,
    REFERENCE_RGB,
    MapPlotWidget,
    build_map_plot,
    edge_colours,
    highlight_colours,
    hover_readout,
    lineage_of,
    next_metric,
    nice_extent,
    node_styles,
    prefix_view,
    statusline,
)
from hillclimb.similarityview import RunScopeMixin, SimilarityBase, SimilarityScreen
from tests.test_similarity_map import make_search
from tests.test_similarity_run import make_arm


@pytest.fixture(autouse=True)
def _fresh_caches():
    clear_caches()
    yield
    clear_caches()


@pytest.fixture
def view(tmp_path):
    return build_map(make_search(tmp_path), tmp_path, True)


class TestStyles:
    def test_origin_and_champion_stand_out(self, view):
        colours, sizes, shapes = node_styles(view)
        ids = [n.id for n in view.nodes]
        assert colours[ids.index("c000")] == REFERENCE_RGB and shapes[ids.index("c000")] == "diamond-open"
        assert colours[ids.index("c003")] == BEST_RGB and shapes[ids.index("c003")] == "diamond"
        assert sizes[ids.index("c003")] == sizes[ids.index("c000")] > sizes[ids.index("c002")]
        assert len(edge_colours(view, colours)) == len(view.edges)

    def test_run_scope_colours_by_arm(self, tmp_path):
        from hillclimb.chart import ARM_PALETTE

        run = build_run_map([
            make_arm(tmp_path, "p", "greedy"), make_arm(tmp_path, "p-2", "openevolve"),
        ], True)
        colours, _sizes, _shapes = node_styles(run)
        by_id = dict(zip((n.id for n in run.nodes), colours))
        assert by_id["p/c002"] == ARM_PALETTE[0] and by_id["p-2/c002"] == ARM_PALETTE[1]
        assert by_id["p/c001"] == REFERENCE_RGB  # the seed

    def test_lineage_highlight_keeps_ancestors_and_descendants(self, view):
        ids = [n.id for n in view.nodes]
        assert {ids[i] for i in lineage_of(view, ids.index("c001"))} == {"c000", "c001", "c003"}
        assert {ids[i] for i in lineage_of(view, ids.index("c000"))} == set(ids)
        colours, _s, _sh = node_styles(view)
        edges = edge_colours(view, colours)
        lit_nodes, lit_edges = highlight_colours(view, colours, edges, ids.index("c001"))
        assert lit_nodes[ids.index("c001")] == colours[ids.index("c001")]
        assert lit_nodes[ids.index("c002")] != colours[ids.index("c002")]  # dimmed
        dimmed = [e for (a, b), e, orig in zip(view.edges, lit_edges, edges) if e != orig]
        assert len(dimmed) == 1  # only c000→c002 leaves the lineage
        assert highlight_colours(view, colours, edges, None) == (colours, edges)

    def test_nice_extent_rounds_up_and_holds(self, view):
        biggest = max(abs(v) for n in view.nodes for v in (n.x, n.y, n.z))
        extent = nice_extent(view)
        assert extent >= biggest and extent / biggest < 5.0
        assert nice_extent(prefix_view(view, 0)) == 1.0


class TestReplayAndReadout:
    def test_prefix_view_restricts_edges_and_trail(self, view):
        two = prefix_view(view, 2)
        assert [n.id for n in two.nodes] == ["c000", "c001"]
        assert two.edges == ((0, 1),) and two.trail == (0, 1)
        assert two.behavioral.shape == (2, 2)
        assert prefix_view(view, 99).nodes == view.nodes

    def test_hover_reads_distances_to_the_anchor(self, view):
        ids = [n.id for n in view.nodes]
        text = hover_readout(view, ids.index("c002"), None)
        assert text.startswith("[bold]c002[/] draft 0.4") and "to c000" in text and "lineage 1" in text
        text = hover_readout(view, ids.index("c003"), ids.index("c001"))
        assert "to c001" in text and "behav 0" in text
        assert "to" not in hover_readout(view, ids.index("c000"), None)  # the anchor itself

    def test_statusline(self, view, tmp_path):
        text = statusline("r", "running", view, position=(0, 3), hover="HOVER", spinning=True)
        assert "map by [bold]behavioral[/]" in text and "4 placed" in text and "(1/3)" in text
        assert "HOVER" in text and "spinning" in text and "stress" in text
        assert "replay 2/4" in statusline("r", "done", view, replaying=2)
        assert statusline("r", "done", None) == "[bold]r[/] [cyan]done[/]"
        none = build_map([], tmp_path, True)
        assert "no candidates yet" in statusline("r", "done", none)

    def test_metric_cycle(self):
        seen = [METRICS[0]]
        for _ in METRICS:
            seen.append(next_metric(seen[-1]))
        assert seen[: len(METRICS)] == list(METRICS) and seen[len(METRICS)] == METRICS[0]


class TestPlot:
    def test_graph_indices_match_the_view(self, view):
        plot, handle = build_map_plot(view, selected=1)
        assert handle is not None
        assert plot.node_count() == len(view.nodes)  # the trail adds no pickable nodes

    def test_empty_view_has_no_graph(self, view):
        plot, handle = build_map_plot(prefix_view(view, 0))
        assert handle is None and plot.node_count() == 0

    def test_widget_survives_select_replay_and_clear(self, view):
        widget = MapPlotWidget()
        widget.set_view(view)
        widget.select(3)
        assert widget.selected == "c003"
        widget.set_view(prefix_view(view, 2))  # the champion is gone: selection drops
        assert widget.selected is None
        widget.select(None)
        assert widget.spinning
        widget.apply_rotate(0.1, 0.0)
        assert not widget.spinning
        widget.clear_view()
        assert widget.view is None


class TestScreens:
    def test_the_two_views_share_a_base_and_swap_keys(self):
        from hillclimb.similarity_mapview import MapScreen, RunMapScreen

        assert issubclass(MapScreen, SimilarityBase) and issubclass(SimilarityScreen, SimilarityBase)
        assert not issubclass(MapScreen, SimilarityScreen)
        assert issubclass(RunMapScreen, RunScopeMixin)
        map_keys = {b.key: b.action for b in MapScreen.BINDINGS}
        cube_keys = {b.key: b.action for b in SimilarityScreen.BINDINGS}
        assert map_keys["v"] == "open_reference" and map_keys["m"] == "cycle_metric"
        assert cube_keys["v"] == "open_map" and cube_keys["c"] == "toggle_reference"
        assert "c" not in map_keys

    def test_unknown_metric_is_rejected(self):
        from hillclimb.similarity_mapview import MapScreen

        with pytest.raises(ValueError):
            MapScreen(config=None, metric="vibes")  # type: ignore[arg-type]


class TestCli:
    def test_bare_similarity_routes_to_map(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        assert "map" in runner.invoke(cli.app, ["similarity", "--help"]).output
        assert "reference" in runner.invoke(cli.app, ["similarity", "--help"]).output
        # a first token that is no subcommand is a search ref for `map`
        result = runner.invoke(cli.app, ["similarity", "no-such-search"])
        assert "No such command" not in result.output
        result = runner.invoke(cli.app, ["similarity", "map", "--metric", "vibes"])
        assert result.exit_code != 0 and "behavioral" in result.output


# ---------------------------------------------------------------------------
# the screens, mounted


def _seeded_search(runs_dir, run_id: str, search_id: str, arm: str | None) -> None:
    """An experiment-arm search with a seed and two children of it — enough
    for both views to have something to draw."""
    from tests.test_similarity import cand, sub_csv, write_candidate
    from tests.test_similarity_run import _experiment_search

    _experiment_search(runs_dir, run_id, search_id, arm or "greedy", seed=True)
    search_dir = runs_dir / run_id / "searches" / search_id
    from hillclimb.journal import Journal

    journal = Journal(search_dir / "journal.jsonl")
    for i, (score, offset) in enumerate(((0.6, 1.0), (0.8, 2.0)), start=2):
        cid = f"c{i:03d}"
        write_candidate(search_dir, cid, solution=f"s = {i}\n", submission=sub_csv([1.0 + offset, 2.0 + offset]))
        journal.candidate_result(cand(cid, parent="c001", score=score, t=i))


@pytest.mark.asyncio
async def test_map_screen_mounts_swaps_and_replays(config):
    from hillclimb.similarity_mapview import MapScreen, RunMapScreen
    from hillclimb.similarityview import RunSimilarityScreen, SimilarityApp, SimilarityScreen

    for search_id, arm in (("p", "greedy"), ("p-2", "gepa")):
        _seeded_search(config.paths.runs_dir, "r1", search_id, arm)

    # run scope, opened on the map
    app = SimilarityApp(config, run=("r1", "p"), view="map")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, RunMapScreen)
        line = str(app.screen.query_one("#similarityline").content)
        assert "map by [bold]behavioral[/]" in line and "over 2 searches" in line
        await pilot.press("m")  # metric cycles
        await pilot.pause()
        assert "map by [bold]structural[/]" in str(app.screen.query_one("#similarityline").content)
        await pilot.press("s")  # spin toggles
        assert not app.screen._canvas().spinning
        await pilot.press("space")  # replay runs to the end
        for _ in range(20):
            await pilot.pause(0.1)
            if app.screen._replay is None:
                break
        assert app.screen._replay is None
        assert len(app.screen._canvas().view.nodes) == len(app.screen._view.nodes)
        await pilot.press("v")  # to the reference cube ...
        await pilot.pause()
        assert isinstance(app.screen, RunSimilarityScreen)
        assert "vs [bold]seed[/]" in str(app.screen.query_one("#similarityline").content)
        await pilot.press("v")  # ... and back
        await pilot.pause()
        assert isinstance(app.screen, RunMapScreen)

    # search scope, opened on the cube
    app = SimilarityApp(config, "r1/p", view="reference")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, SimilarityScreen)
        await pilot.press("v")
        await pilot.pause()
        assert isinstance(app.screen, MapScreen) and not isinstance(app.screen, RunMapScreen)
        assert "4 placed" in str(app.screen.query_one("#similarityline").content)
        await pilot.press("n")  # the next search in the store
        await pilot.pause()
        assert app.screen.search == "r1/p-2"
