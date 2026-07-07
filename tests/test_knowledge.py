"""Cross-search learning: card distillation, retrieval, prompt injection."""

from __future__ import annotations

from pathlib import Path

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.candidate import BackendInfo, Candidate, Trial
from hillclimb.journal import Journal
from hillclimb.knowledge import (
    KnowledgeCard,
    complexity_offset,
    distill_card,
    extract_libraries,
    load_cards,
    problem_family,
    render_prior_experience,
    write_card,
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
    lower_is_better = True


def scored(cid, op, val, summary="", complexity=None, workspace="w", holdout=None):
    return dict(
        candidate_id=cid, operator=op, status="ok", complexity=complexity,
        workspace=workspace, summary=summary,
        trials=[Trial(val_score=val, holdout_score=holdout)],
    )


class TestDistill:
    def test_card_shape(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "solution.py").write_text("import lightgbm\nfrom scipy.stats import beta\n")
        journal = make_journal(tmp_path, [
            dict(candidate_id="c000", operator="baseline", status="ok", workspace="w"),
            scored("c001", "draft", 0.02, "GBM with lag features", "minimal", str(ws)),
            scored("c002", "improve", 0.015, "added clearsky ratio", "minimal", str(ws), holdout=0.016),
            dict(candidate_id="c003", operator="draft", status="buggy", workspace="w",
                 trials=[Trial(returncode=1, stdout_tail="ValueError: bad shape")]),
        ])
        card = distill_card(
            journal, problem=FakeProblem(), run_ref="run1/gefcom2014-solar",
            target="emflow://gefcom2014:solar", budget_s=1800, cost_usd=2.5,
        )
        assert card.family == "gefcom2014"
        assert card.n_candidates == 3  # baseline excluded
        assert card.n_ok == 2 and card.n_buggy == 1
        assert card.selected_val == 0.015
        assert card.top_approaches[0].summary == "added clearsky ratio"
        assert "lightgbm" in card.top_approaches[0].libraries
        assert "scipy" in card.top_approaches[0].libraries
        assert card.operator_stats["draft"].attempts == 2
        assert card.operator_stats["draft"].ok == 1
        assert any("ValueError" in f for f in card.failure_modes)

    def test_write_load_roundtrip(self, tmp_path):
        ws = tmp_path / "ws"; ws.mkdir()
        journal = make_journal(tmp_path, [scored("c001", "draft", 0.5, "x", workspace=str(ws))])
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


class TestEndToEnd:
    def test_search_writes_card_and_next_search_reads_it(self, task, config, tmp_path, monkeypatch):
        from hillclimb.api import execute_search, create_search
        from hillclimb.budget import BudgetManager
        from hillclimb.workspace import create_run_dir
        import sys

        from hillclimb.search import GreedySearcher

        config.learning.dir = tmp_path / "knowledge"
        config.paths.runtime_python = Path(sys.executable)
        config.budget.stop_margin_s = 1
        config.holdout.enabled = False
        monkeypatch.setattr("hillclimb.api.get_backend", lambda *a, **k: backend)
        monkeypatch.setattr(  # stop after baseline + one draft
            "hillclimb.api.GreedySearcher",
            lambda **kw: GreedySearcher(**{**kw, "max_candidates": 2}),
        )

        def run_once(name, val, note):
            backend.queue(script=ok_script(val), notes=note + "\n")
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
        cards = list((tmp_path / "knowledge").rglob("*.yaml"))
        assert len(cards) == 1

        # second search: the draft prompt must carry the first search's learnings
        outcome2 = run_once("run-two", 0.8, "second approach\n")
        prompts = list(outcome2.search_dir.glob("candidates/*/prompt.md"))
        draft_prompt = next(p.read_text() for p in prompts)
        assert "winning approach: gradient boosting" in draft_prompt
        assert "PREVIOUS searches" in draft_prompt
