"""Knowledge query tool: graph lookup output and the contract tools clause."""

from __future__ import annotations

import pytest

from hillclimb.modules.memory.graph import build_graph, fuzzy_match, query_graph, render_query_hits
from tests.test_graph import claim, knowledge_dir, make_card  # noqa: F401 — fixture


class TestQuery:
    def test_entity_hit_carries_claims_and_searches(self, knowledge_dir):  # noqa: F811
        from hillclimb.modules.memory.credit import CreditEvent, write_credit_event

        write_credit_event(knowledge_dir, CreditEvent(
            run_ref="r9/s1", problem_id="spaceship-titanic",
            family="spaceship-titanic", claim_ids=["cl1"], reward=1.0,
            observed_at="2026-07-09T00:00:00Z",
        ))
        graph = build_graph(knowledge_dir)
        hits = query_graph(graph, "histgradient")
        assert hits and hits[0]["id"] == "entity:histgradientboosting"
        claims = hits[0]["claims"]
        assert claims[0]["statement"].startswith("histgradientboosting helps")
        assert claims[0]["track"]["injections"] == 1
        text = render_query_hits(hits)
        assert "histgradientboosting (technique)" in text
        assert "measured" in text

    def test_family_filter_and_no_match(self, knowledge_dir):  # noqa: F811
        graph = build_graph(knowledge_dir)
        hits = query_graph(graph, "histgradient", family="some-other-family")
        assert "claims" not in hits[0]  # scoped out
        assert query_graph(graph, "zzzzzz") == []
        assert "no matches" in render_query_hits([])

    def test_claim_hit_summarized(self, knowledge_dir):  # noqa: F811
        graph = build_graph(knowledge_dir)
        hits = query_graph(graph, "helps")
        claim_hits = [h for h in hits if h["type"] == "claim"]
        assert claim_hits and "claim" in claim_hits[0]

    def test_json_round_trips(self, knowledge_dir):  # noqa: F811
        import json

        hits = query_graph(build_graph(knowledge_dir), "histgradient")
        assert json.loads(json.dumps(hits)) == hits

    def test_fuzzy_match_moved_home(self):
        from hillclimb.modules.memory.graph import GraphNode

        nodes = [GraphNode(id="entity:lightgbm", type="library", label="lightgbm")]
        assert fuzzy_match(nodes, "light")[0].label == "lightgbm"
        # graphview still re-exports it for the TUI search box
        from hillclimb.tui.graphview import fuzzy_match as gv_fuzzy

        assert gv_fuzzy is fuzzy_match


class TestToolsClause:
    @pytest.fixture
    def searcher(self, task, config, tmp_path):
        import sys
        from pathlib import Path

        from hillclimb.backends.fake import FakeBackend
        from hillclimb.harness.budget import BudgetManager
        from hillclimb.harness.journal import Journal
        from tests.harness_factory import SearchRig

        config.paths.runtime_python = Path(sys.executable)
        return SearchRig(
            problem=task,
            config=config,
            journal=Journal(tmp_path / "journal.jsonl"),
            backend=FakeBackend(),
            executor=None,
            budget=BudgetManager(600),
            search_dir=tmp_path,
        )

    def test_clause_present_with_interpreter_path(self, searcher):
        import sys

        prompt = searcher.build_prompt("draft", None, "minimal", None)
        assert "knowledge query" in prompt
        assert f"{sys.executable} -m hillclimb.cli" in prompt
        assert "uv run" not in prompt.split("knowledge query")[0].splitlines()[-1]

    def test_clause_gated_by_flag_and_learning(self, searcher):
        searcher.config.learning.tool = False
        assert "knowledge query" not in searcher.build_prompt("draft", None, "minimal", None)
        searcher.config.learning.tool = True
        searcher.config.learning.enabled = False
        assert "knowledge query" not in searcher.build_prompt("draft", None, "minimal", None)
