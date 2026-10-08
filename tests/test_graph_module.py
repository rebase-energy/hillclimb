"""Graph modules: the registry and its three ref forms, the cache stamp, and
the seam every consumer goes through."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hillclimb.config import Config

from hillclimb.modules.memory.base import GraphModule, GraphNode, KnowledgeGraph
from hillclimb.modules.memory.graph import (
    build_graph,
    graph_path,
    load_graph,
    load_or_build_graph,
    query_graph,
    retrieve_claims,
)
from hillclimb.modules.memory.graphs import _GRAPHS, get_graph, register_graph, registered_graphs
from tests.test_graph import claim, knowledge_dir, make_card  # noqa: F401 — fixture
from tests.catalog_fixture import pinned

# a module that only changes the structure: one note node, no claims, no layout
NOTES_PY = '''
from hillclimb.sdk import GraphModule, GraphNode, KnowledgeGraph


class Notes(GraphModule):
    name = "notes"

    def build(self, knowledge_dir, previous=None):
        return KnowledgeGraph(nodes=[
            GraphNode(id="note:a", type="note", label="a", first_seen="2026-01-01T00:00:00+00:00"),
            GraphNode(id="note:b", type="note", label="b", first_seen="2026-01-02T00:00:00+00:00"),
        ])
'''


class Notes(GraphModule):
    name = "notes"
    key = "notes"

    def build(self, knowledge_dir, previous=None):
        return KnowledgeGraph(nodes=[
            GraphNode(id="note:a", type="note", label="a", first_seen="2026-01-01T00:00:00+00:00"),
        ])


def _scope(graph: KnowledgeGraph) -> dict:
    family = next(n.label for n in graph.nodes if n.type == "family")
    problem = next(n.label for n in graph.nodes if n.type == "problem")
    return {"family": family, "problem_id": problem, "concepts": []}


class TestResolve:
    def test_the_default_is_the_built_in(self, knowledge_dir):  # noqa: F811
        module = get_graph("knowledge-graph")
        assert module.name == module.key == "knowledge-graph"
        built, reference = module.build(knowledge_dir), build_graph(knowledge_dir)
        assert [n.id for n in built.nodes] == [n.id for n in reference.nodes]
        assert [(e.src, e.dst, e.type) for e in built.edges] == [(e.src, e.dst, e.type) for e in reference.edges]
        # retrieve and query are the built-in walks unless a module overrides them
        scope = _scope(built)
        assert [n.id for n in module.retrieve(built, **scope)] == [n.id for n in retrieve_claims(built, **scope)]
        assert module.query(built, "hist") == query_graph(built, "hist")

    def test_a_file_module_by_path(self, tmp_path):
        path = tmp_path / "mygraph.py"
        path.write_text(NOTES_PY)
        module = get_graph("mygraph.py", base_dir=tmp_path)
        assert isinstance(module, GraphModule) and module.name == "notes"
        assert module.key.startswith("mygraph.py#") and len(module.key) == len("mygraph.py#") + 12
        assert [n.id for n in module.build(tmp_path).nodes] == ["note:a", "note:b"]
        # the key follows the file's bytes: an edited module is a different builder
        path.write_text(NOTES_PY + "\n# edited\n")
        assert get_graph("mygraph.py", base_dir=tmp_path).key != module.key
        # a class without a name is named after the file
        (tmp_path / "unnamed.py").write_text(NOTES_PY.replace('    name = "notes"\n', ""))
        assert get_graph("unnamed.py", base_dir=tmp_path).name == "unnamed"

    def test_a_file_must_expose_one_module_or_the_attr(self, tmp_path):
        two = tmp_path / "two.py"
        two.write_text(NOTES_PY + "\nclass Other(Notes):\n    name = 'other'\n")
        with pytest.raises(ValueError, match="two.py must define exactly one GraphModule subclass"):
            get_graph(str(two))
        two.write_text(two.read_text() + "\nKNOWLEDGE_GRAPH = Other\n")
        assert get_graph(str(two)).name == "other"
        (tmp_path / "broken.py").write_text("import nothing_here\n")
        with pytest.raises(ValueError, match="broken.py failed to import: ModuleNotFoundError"):
            get_graph(str(tmp_path / "broken.py"))
        with pytest.raises(ValueError, match="graph module file not found"):
            get_graph(str(tmp_path / "missing.py"))

    def test_the_dotted_form_and_the_registry(self):
        module = get_graph("hillclimb.modules.memory.graph:KnowledgeGraphBuilder")
        assert module.name == "knowledge-graph"
        assert module.key == "hillclimb.modules.memory.graph:KnowledgeGraphBuilder"
        with pytest.raises(ValueError, match="is not a GraphModule subclass"):
            get_graph("hillclimb.modules.memory.base:GraphNode")
        with pytest.raises(ValueError, match="cannot import graph module 'nowhere.mod:X'"):
            get_graph("nowhere.mod:X")
        with pytest.raises(ValueError, match="unknown graph module 'nope' \\(available: knowledge-graph"):
            get_graph("nope")

    def test_register_and_its_name_rules(self):
        register_graph(Notes)
        try:
            assert get_graph("notes").name == "notes" and "notes" in registered_graphs()
        finally:
            _GRAPHS.pop("notes")
        with pytest.raises(ValueError, match="would read as a file or module:Class"):
            register_graph(Notes, "notes.py")
        with pytest.raises(ValueError, match="would read as a file or module:Class"):
            register_graph(Notes, "pkg:Notes")


class TestCache:
    def test_switching_modules_rebuilds_despite_a_fresh_index(self, knowledge_dir):  # noqa: F811
        default = load_or_build_graph(knowledge_dir)
        assert default.builder == "knowledge-graph" and len(default.nodes) > 1
        path = graph_path(knowledge_dir)
        os.utime(path, (time.time() + 60, time.time() + 60))  # newer than every input
        assert load_or_build_graph(knowledge_dir).built_at == default.built_at  # served from cache
        mine = load_or_build_graph(knowledge_dir, module=Notes())
        assert mine.builder == "notes" and [n.id for n in mine.nodes] == ["note:a"]
        assert json.loads(path.read_text())["builder"] == "notes"
        assert json.loads(path.read_text())["schema_version"] == 2
        back = load_or_build_graph(knowledge_dir)  # the default finds a foreign index: rebuilds
        assert back.builder == "knowledge-graph" and len(back.nodes) == len(default.nodes)

    def test_an_index_without_a_builder_reads_as_the_built_in(self, knowledge_dir):  # noqa: F811
        graph = load_or_build_graph(knowledge_dir)
        path = graph_path(knowledge_dir)
        data = json.loads(path.read_text())
        del data["builder"]
        path.write_text(json.dumps(data))
        os.utime(path, (time.time() + 60, time.time() + 60))
        assert load_graph(path).builder == "knowledge-graph"
        assert load_or_build_graph(knowledge_dir).built_at == graph.built_at  # still the cached index

    def test_a_module_that_sets_no_positions_gets_them(self, knowledge_dir):  # noqa: F811
        pytest.importorskip("networkx")
        graph = load_or_build_graph(knowledge_dir, module=Notes())
        assert all(n.pos is not None and n.pos3 is not None for n in graph.nodes)


class Claimy(GraphModule):
    """A graph of its own shape that still hands the harness claim nodes."""

    name = "claimy"
    key = "claimy"

    def build(self, knowledge_dir, previous=None):
        return KnowledgeGraph(nodes=[GraphNode(
            id="claim:custom", type="claim", label="custom helps", first_seen="2026-01-01T00:00:00+00:00",
            data={"subject": "custom", "relation": "helps", "object": "", "confidence": 0.9,
                  "evidence": ["c001"], "scope": {"family": "spaceship-titanic"}},
        )])

    def retrieve(self, graph, *, family, problem_id, concepts, limit=8):
        return [n for n in graph.nodes if n.type == "claim"]

    def query(self, graph, terms, *, family="", limit=5):
        return [{"id": n.id, "type": n.type, "label": n.label} for n in graph.nodes]


PROBLEM = SimpleNamespace(problem_id="spaceship-titanic", metric_name="accuracy", higher_is_better=True, runtime="csv")


class TestSeam:
    """Every consumer asks the climber's module — retrieval, the query
    command, consolidation, the graph screen."""

    def test_retrieval_is_whatever_the_module_returns(self, knowledge_dir, monkeypatch):  # noqa: F811
        from hillclimb.api import build_knowledge_context
        from hillclimb.modules.memory.files import FilesMemory

        config = pinned()
        config.learning.dir = knowledge_dir
        monkeypatch.setattr(FilesMemory, "graph_module", lambda self: Claimy())
        text, _, injected = build_knowledge_context(config, PROBLEM, "", lambda m: None)
        assert injected == ["custom"] and "custom helps" in text
        monkeypatch.setattr(FilesMemory, "graph_module", lambda self: Notes())
        text, _, injected = build_knowledge_context(config, PROBLEM, "", lambda m: None)
        assert injected == [] and "Distilled claims" not in (text or "")  # a graph without claims injects nothing

    def test_knowledge_query_answers_through_the_module(self, knowledge_dir, monkeypatch, capsys):  # noqa: F811
        import hillclimb.harness.glue as glue
        from hillclimb.cli import common
        from hillclimb.cli.knowledge import knowledge_query

        config = pinned()
        config.learning.dir = knowledge_dir
        monkeypatch.setattr(common, "load_config", lambda *a, **k: config)
        monkeypatch.setattr(glue, "build_graph_module", lambda *a, **k: Claimy())
        knowledge_query("custom", family="", limit=5, as_json=True)
        assert json.loads(capsys.readouterr().out)[0]["id"] == "claim:custom"
        monkeypatch.setattr(glue, "build_graph_module", lambda *a, **k: Notes())
        knowledge_query("zzz", family="", limit=5, as_json=False)
        assert "no matches" in capsys.readouterr().out

    def test_consolidate_is_a_no_op_on_a_graph_without_claims(self, knowledge_dir):  # noqa: F811
        from hillclimb.modules.memory.consolidate import consolidate

        notes: list[str] = []
        summary = consolidate(knowledge_dir, pinned(), notes.append, graph_module=Notes())
        assert summary["generalized"] == [] and summary["playbooks_written"] == []
        assert any("yielded no claim nodes" in n for n in notes)


@pytest.mark.asyncio
async def test_the_graph_screen_shows_a_climbers_own_graph(knowledge_dir, tmp_path, monkeypatch):  # noqa: F811
    import hillclimb.harness.glue as glue
    from hillclimb.tui.graphview import GraphApp, GraphPlotWidget

    monkeypatch.setenv("PLOTUI_RENDER", "placeholder")
    monkeypatch.setattr(glue, "build_graph_module", lambda *a, **k: Notes())
    config = pinned()
    config.learning.dir = knowledge_dir
    config.paths.runs_dir = tmp_path / "runs"
    app = GraphApp(config)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        canvas = app.screen.query_one("#graph-canvas", GraphPlotWidget)
        assert canvas._ids and app.screen._graph.builder == "notes"  # an unknown node type still draws
