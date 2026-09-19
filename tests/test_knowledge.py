"""Cross-search learning: card distillation, retrieval, prompt injection."""

from __future__ import annotations

from tests.factories import trial as mk_trial

from pathlib import Path

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.candidate import BackendInfo, Candidate
from hillclimb.journal import Journal
from hillclimb.knowledge import (
    KnowledgeCard,
    complexity_offset,
    distill_card,
    extract_libraries,
    load_cards,
    load_live_cards,
    problem_family,
    render_live_experience,
    render_prior_experience,
    write_card,
    write_live_card,
)
from tests.conftest import ok_script


def make_journal(tmp_path, entries) -> Journal:
    journal = Journal(tmp_path / "journal.jsonl")
    for e in entries:
        journal.candidate_result(Candidate(**e))
    return journal


class FakeProblem:
    problem_id = "gefcom2014-solar"
    metric_name = "PinballLoss"
    higher_is_better = False


def scored(cid, op, val, summary="", complexity=None, candidate_dir="w", holdout=None):
    return dict(
        candidate_id=cid, operator=op, status="passing", complexity=complexity,
        candidate_dir=candidate_dir, summary=summary,
        trials=[mk_trial(val_score=val, holdout_score=holdout)],
    )


class TestDistill:
    def test_card_shape(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "solution.py").write_text("import lightgbm\nfrom scipy.stats import beta\n")
        journal = make_journal(tmp_path, [
            dict(candidate_id="c000", operator="baseline", status="passing", candidate_dir="w"),
            scored("c001", "draft", 0.02, "GBM with lag features", "minimal", str(ws)),
            scored("c002", "improve", 0.015, "added clearsky ratio", "minimal", str(ws), holdout=0.016),
            dict(candidate_id="c003", operator="draft", status="buggy", candidate_dir="w",
                 trials=[mk_trial(returncode=1, stdout_tail="ValueError: bad shape")]),
        ])
        card = distill_card(
            journal, problem=FakeProblem(), run_ref="run1/gefcom2014-solar",
            target="emflow://gefcom2014:solar", budget_s=1800, cost_usd=2.5,
        )
        assert card.family == "gefcom2014"
        assert card.n_candidates == 3  # baseline excluded
        assert card.n_ok == 2 and card.n_failing == 0 and card.n_buggy == 1
        assert card.selected_val == 0.015
        assert card.top_approaches[0].summary == "added clearsky ratio"
        assert "lightgbm" in card.top_approaches[0].libraries
        assert "scipy" in card.top_approaches[0].libraries
        assert card.operator_stats["draft"].attempts == 2
        assert card.operator_stats["draft"].ok == 1
        assert any("ValueError" in f for f in card.failure_modes)

    def test_write_load_roundtrip(self, tmp_path):
        ws = tmp_path / "ws"; ws.mkdir()
        journal = make_journal(tmp_path, [scored("c001", "draft", 0.5, "x", candidate_dir=str(ws))])
        card = distill_card(journal, problem=FakeProblem(), run_ref="r/s",
                            target="emflow://gefcom2014:solar")
        kdir = tmp_path / "knowledge"
        write_card(kdir, card)
        loaded = load_cards(kdir, problem_id="gefcom2014-solar", family="gefcom2014")
        assert len(loaded) == 1
        assert loaded[0].selected_val == 0.5

    def test_family_retrieval_and_same_problem_priority(self, tmp_path):
        kdir = tmp_path / "knowledge"
        for pid, when in (("gefcom2014-wind", "2026-01-02"), ("gefcom2014-solar", "2026-01-01")):
            card = KnowledgeCard(problem_id=pid, family="gefcom2014",
                                 run_ref=f"r-{pid}/s", finished_at=when)
            write_card(kdir, card)
        loaded = load_cards(kdir, problem_id="gefcom2014-solar", family="gefcom2014")
        assert [c.problem_id for c in loaded] == ["gefcom2014-solar", "gefcom2014-wind"]

    def test_corrupt_card_skipped(self, tmp_path):
        kdir = tmp_path / "knowledge" / "gefcom2014"
        kdir.mkdir(parents=True)
        (kdir / "bad.yaml").write_text("{{{{not yaml")
        assert load_cards(tmp_path / "knowledge", problem_id="x", family="gefcom2014") == []


class TestHelpers:
    def test_problem_family(self):
        assert problem_family("gefcom2014-solar", "emflow://gefcom2014:solar") == "gefcom2014"
        assert problem_family("gefcom2014-solar") == "gefcom2014"
        assert problem_family("circle-packing") == "circle-packing"

    def test_extract_libraries_filters_noise(self, tmp_path):
        f = tmp_path / "s.py"
        f.write_text("import os\nimport pandas as pd\nimport lightgbm\nfrom torch import nn\n")
        assert extract_libraries(f) == ["lightgbm", "torch"]

    def test_render_and_offset(self):
        card = KnowledgeCard(
            problem_id="gefcom2014-solar", family="gefcom2014", metric="PinballLoss",
            n_candidates=8, selected_val=0.013, selected_operator="ensemble",
            top_approaches=[dict(candidate_id="c009", operator="improve",
                                 val_score=0.013, complexity="advanced",
                                 summary="anchor-quantile GBM", libraries=["lightgbm"])],
            failure_modes=["execution timed out (x2)"],
        )
        text = render_prior_experience([card])
        assert "anchor-quantile GBM" in text
        assert "lightgbm" in text
        assert "0.013" in text
        assert complexity_offset([card]) == 1  # winner was not minimal
        card.top_approaches[0].complexity = "minimal"
        assert complexity_offset([card]) == 0
        assert render_prior_experience([]) == ""


class TestLiveSharing:
    def test_roundtrip_excludes_own_search(self, tmp_path):
        run_dir = tmp_path / "run"
        solar = KnowledgeCard(
            problem_id="gefcom2014-solar", family="gefcom2014", run_ref="r/solar", n_ok=2
        )
        wind = KnowledgeCard(
            problem_id="gefcom2014-wind", family="gefcom2014", run_ref="r/wind"
        )
        write_live_card(run_dir, solar, "gefcom2014-solar")
        write_live_card(run_dir, wind, "gefcom2014-wind")
        loaded = load_live_cards(
            run_dir, exclude_search_id="gefcom2014-wind", family="gefcom2014"
        )
        assert [c.problem_id for c in loaded] == ["gefcom2014-solar"]
        assert len(load_live_cards(run_dir)) == 2

    def test_republish_overwrites_same_search(self, tmp_path):
        run_dir = tmp_path / "run"
        write_live_card(run_dir, KnowledgeCard(problem_id="p", family="p", n_ok=1), "s")
        write_live_card(run_dir, KnowledgeCard(problem_id="p", family="p", n_ok=2), "s")
        loaded = load_live_cards(run_dir)
        assert len(loaded) == 1 and loaded[0].n_ok == 2

    def test_corrupt_live_card_skipped(self, tmp_path):
        knowledge = tmp_path / "run" / "knowledge"
        knowledge.mkdir(parents=True)
        (knowledge / "live--x.yaml").write_text("{{{{not yaml")
        assert load_live_cards(tmp_path / "run") == []

    def test_render_live_experience(self):
        card = KnowledgeCard(
            problem_id="gefcom2014-solar", family="gefcom2014", metric="PinballLoss",
            n_ok=3, selected_val=0.013,
            top_approaches=[dict(candidate_id="c004", operator="improve",
                                 val_score=0.013, summary="clearsky ratio feature",
                                 libraries=["lightgbm"])],
            failure_modes=["execution timed out (x2)"],
        )
        text = render_live_experience([card])
        assert "CONCURRENTLY" in text
        assert "clearsky ratio feature" in text
        assert "lightgbm" in text
        assert "gefcom2014-solar" in text
        assert render_live_experience([]) == ""

    def test_sibling_searches_in_one_run_share_discoveries(
        self, task, task_larger, config, tmp_path, monkeypatch
    ):
        from hillclimb.api import create_search, execute_search
        from hillclimb.budget import BudgetManager
        from hillclimb.dirs import create_run_dir
        import sys

        config.learning.dir = tmp_path / "knowledge"
        config.paths.runtime_python = Path(sys.executable)
        config.budget.stop_margin_s = 1
        config.holdout.enabled = False
        config.search.num_drafts = 1
        backend = FakeBackend()
        monkeypatch.setattr("hillclimb.api.get_backend", lambda *a, **k: backend)
        config.budget.max_evaluations = 2  # draft + improve, then stop
        run_dir = create_run_dir(config.paths.runs_dir, "suite-run")

        def run_search(problem, notes):
            for note, val in zip(notes, (0.7, 0.75)):
                backend.queue(script=ok_script(val), notes=note + "\n")
            backend.queue(operator="distill", files={"claims.yaml": "claims: []\n"})
            search_dir = create_search(config, problem, run_dir, "suite-run", 3600)
            outcome = execute_search(
                config, problem, search_dir, BudgetManager(3600, stop_margin_s=1),
                log=lambda *_: None,
            )
            assert outcome.state == "done"
            return search_dir

        search_a = run_search(task, ["clearsky ratio feature wins", "added lag features"])
        live = list((run_dir / "knowledge").glob("live--*.yaml"))
        assert [p.name for p in live] == ["live--synthetic.yaml"]
        # a search must never be fed its own live card back
        a_prompts = [p.read_text() for p in search_a.glob("candidates/*/prompt.md")]
        assert a_prompts and all("CONCURRENTLY" not in p for p in a_prompts)

        # sibling search in the SAME run: draft and improve prompts both
        # carry search A's discoveries via the live channel
        search_b = run_search(task_larger, ["baseline try", "tweak"])
        b_prompts = [p.read_text() for p in search_b.glob("candidates/*/prompt.md")]
        draft = next(p for p in b_prompts if "drafting a new candidate" in p)
        improve = next(p for p in b_prompts if "improving the current best" in p)
        assert "clearsky ratio feature wins" in draft and "CONCURRENTLY" in draft
        assert "clearsky ratio feature wins" in improve
        assert "# Discoveries from concurrent searches" in improve
        assert "{{live_experience}}" not in improve


class TestEndToEnd:
    def test_search_writes_card_and_next_search_reads_it(self, task, config, tmp_path, monkeypatch):
        from hillclimb.api import execute_search, create_search
        from hillclimb.budget import BudgetManager
        from hillclimb.dirs import create_run_dir
        import sys


        config.learning.dir = tmp_path / "knowledge"
        config.paths.runtime_python = Path(sys.executable)
        config.budget.stop_margin_s = 1
        config.holdout.enabled = False
        monkeypatch.setattr("hillclimb.api.get_backend", lambda *a, **k: backend)
        config.budget.max_evaluations = 1  # stop after one draft

        def run_once(name, val, note):
            backend.queue(script=ok_script(val), notes=note + "\n")
            backend.queue(operator="distill", files={"claims.yaml": (
                "claims:\n"
                "  - subject: gradient-boosting\n"
                "    relation: helps\n"
                "    confidence: 0.7\n"
                "entities:\n"
                "  - slug: gradient-boosting\n"
                "    concepts: [decision-trees]\n"
            )})
            run_dir = create_run_dir(config.paths.runs_dir, name)
            search_dir = create_search(config, task, run_dir, name, 3600)
            return execute_search(
                config, task, search_dir, BudgetManager(3600, stop_margin_s=1),
                log=lambda *_: None, target="demo-task",
            )

        backend = FakeBackend()
        # first search: 1 draft then out of responses -> cap candidates
        config.search.num_drafts = 1
        outcome1 = run_once("run-one", 0.7, "winning approach: gradient boosting")
        assert outcome1.state == "done"
        # cards live in family subdirs; the knowledge-dir root holds the
        # concept/entity registries the distill pass maintains
        cards = list((tmp_path / "knowledge").glob("*/*.yaml"))
        assert len(cards) == 1
        card_text = cards[0].read_text()
        assert "gradient-boosting" in card_text and "claims:" in card_text

        # second search: the draft prompt must carry the first search's learnings
        outcome2 = run_once("run-two", 0.8, "second approach\n")
        prompts = list(outcome2.search_dir.glob("candidates/*/prompt.md"))
        draft_prompt = next(p.read_text() for p in prompts)
        assert "winning approach: gradient boosting" in draft_prompt
        assert "PREVIOUS searches" in draft_prompt
        # ...including the graph-retrieved distilled claim from run one
        assert "Distilled claims" in draft_prompt
        assert "gradient-boosting helps" in draft_prompt

        # credit assignment: run-two improved 0.7 -> 0.8 over run-one's
        # record, so the injected claim earned a full-reward event...
        from hillclimb.credit import load_credit_events
        from hillclimb.graph import graph_path, load_graph

        events = load_credit_events(tmp_path / "knowledge")
        assert len(events) == 1
        assert events[0].reward == 1.0 and events[0].basis == "prior-best"
        assert len(events[0].claim_ids) == 1
        # ...and the rebuilt graph carries the track on the claim node
        graph = load_graph(graph_path(tmp_path / "knowledge"))
        claim_node = graph.node_map()[f"claim:{events[0].claim_ids[0]}"]
        assert claim_node.data["track"]["injections"] == 1
        assert claim_node.data["track"]["mean_reward"] == 1.0
