"""Skill library: harvest quality gate, per-family cap, selection, injection."""

from __future__ import annotations

from tests.factories import trial as mk_trial

from pathlib import Path

import pytest

from hillclimb.candidate import Candidate
from hillclimb.journal import Journal
from hillclimb.knowledge import KnowledgeCard
from hillclimb.skills import (
    SKILL_CODE_FILENAME,
    SKILLS_PER_FAMILY,
    harvest_skill,
    load_skills,
    select_skill,
)


class Problem:
    problem_id = "comp-a"
    metric_name = "accuracy"
    higher_is_better = True


def make_journal(tmp_path, entries) -> Journal:
    journal = Journal(tmp_path / "journal.jsonl")
    for e in entries:
        journal.candidate_result(Candidate(**e))
    return journal


def winner_journal(tmp_path, val=0.8, operator="improve", code="import sklearn\n"):
    ws = tmp_path / f"ws-{val}"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "solution.py").write_text(code)
    return make_journal(tmp_path, [dict(
        candidate_id="c001", operator=operator, status="passing", candidate_dir=str(ws),
        summary="gradient boosting", trials=[mk_trial(val_score=val)],
    )])


def card(problem_id="comp-a", run_ref="r1/s1", metric="accuracy"):
    return KnowledgeCard(problem_id=problem_id, family=problem_id,
                         run_ref=run_ref, metric=metric)


class TestHarvest:
    def test_winner_harvested_with_metadata(self, tmp_path):
        kdir = tmp_path / "knowledge"
        journal = winner_journal(tmp_path)
        skill_dir = harvest_skill(
            journal, problem=Problem(), card=card(), knowledge_dir=kdir,
            selection="rank-blend", log=lambda m: None,
        )
        assert skill_dir is not None
        assert (skill_dir / SKILL_CODE_FILENAME).read_text() == "import sklearn\n"
        skills = load_skills(kdir)
        assert len(skills) == 1
        skill = skills[0][0]
        assert skill.family == "comp-a" and skill.score == 0.8
        assert skill.libraries == ["sklearn"]
        assert "tabular" in skill.concepts and "classification" in skill.concepts

    def test_quality_gate(self, tmp_path):
        kdir = tmp_path / "knowledge"
        # baseline-selected -> no skill
        journal = winner_journal(tmp_path, operator="baseline")
        assert harvest_skill(journal, problem=Problem(), card=card(),
                             knowledge_dir=kdir, selection="rank-blend",
                             log=lambda m: None) is None
        # nothing scored -> no skill
        empty = make_journal(tmp_path / "e", [])
        assert harvest_skill(empty, problem=Problem(), card=card(),
                             knowledge_dir=kdir, selection="rank-blend",
                             log=lambda m: None) is None
        assert load_skills(kdir) == []

    def test_family_cap_replaces_worst(self, tmp_path):
        kdir = tmp_path / "knowledge"
        for i, val in enumerate((0.6, 0.7)):
            harvest_skill(
                winner_journal(tmp_path / str(i), val=val), problem=Problem(),
                card=card(run_ref=f"r{i}/s1"), knowledge_dir=kdir,
                selection="rank-blend", log=lambda m: None,
            )
        assert len(load_skills(kdir)) == SKILLS_PER_FAMILY
        # a better winner replaces the worst (0.6)
        harvest_skill(
            winner_journal(tmp_path / "hi", val=0.9), problem=Problem(),
            card=card(run_ref="r9/s1"), knowledge_dir=kdir,
            selection="rank-blend", log=lambda m: None,
        )
        scores = sorted(s.score for s, _ in load_skills(kdir))
        assert scores == [0.7, 0.9]
        # a worse one is rejected
        assert harvest_skill(
            winner_journal(tmp_path / "lo", val=0.5), problem=Problem(),
            card=card(run_ref="r5/s1"), knowledge_dir=kdir,
            selection="rank-blend", log=lambda m: None,
        ) is None


class TestSelect:
    def _seed(self, tmp_path) -> Path:
        kdir = tmp_path / "knowledge"
        harvest_skill(winner_journal(tmp_path / "a", val=0.7), problem=Problem(),
                      card=card(run_ref="r1/s1"), knowledge_dir=kdir,
                      selection="rank-blend", log=lambda m: None)
        harvest_skill(winner_journal(tmp_path / "b", val=0.9), problem=Problem(),
                      card=card(run_ref="r2/s1"), knowledge_dir=kdir,
                      selection="rank-blend", log=lambda m: None)
        return kdir

    def test_same_family_best_score(self, tmp_path):
        kdir = self._seed(tmp_path)
        match = select_skill(kdir, family="comp-a", concepts=["tabular"],
                             higher_is_better=True)
        assert match is not None and match[0].score == 0.9

    def test_concept_fallback_and_no_match(self, tmp_path):
        kdir = self._seed(tmp_path)
        # different family, shared concept -> newest sibling
        match = select_skill(kdir, family="other", concepts=["tabular"],
                             higher_is_better=True)
        assert match is not None and match[0].family == "comp-a"
        # no concept overlap -> nothing
        assert select_skill(kdir, family="other", concepts=["image"],
                            higher_is_better=True) is None


class TestInjection:
    def test_first_draft_gets_reference_later_drafts_do_not(self, task, config, tmp_path, monkeypatch):
        import sys

        from hillclimb.api import create_search, execute_search
        from hillclimb.backends.fake import FakeBackend
        from hillclimb.budget import BudgetManager
        from hillclimb.search import GreedySearcher
        from hillclimb.dirs import create_run_dir
        from tests.conftest import ok_script

        config.learning.dir = tmp_path / "knowledge"
        config.paths.runtime_python = Path(sys.executable)
        config.budget.stop_margin_s = 1
        config.holdout.enabled = False
        config.search.num_drafts = 2
        backend = FakeBackend()
        monkeypatch.setattr("hillclimb.api.get_backend", lambda *a, **k: backend)
        monkeypatch.setattr(
            "hillclimb.search.GreedySearcher",
            lambda **kw: GreedySearcher(**{**kw, "max_candidates": 3}),
        )

        def run_once(name, vals):
            for val in vals:
                backend.queue(script=ok_script(val), notes=f"approach {val}\n")
            backend.queue(operator="distill", files={"claims.yaml": "claims: []\n"})
            run_dir = create_run_dir(config.paths.runs_dir, name)
            search_dir = create_search(config, task, run_dir, name, 3600)
            outcome = execute_search(
                config, task, search_dir, BudgetManager(3600, stop_margin_s=1),
                log=lambda *_: None,
            )
            assert outcome.state == "done"
            return search_dir

        # search 1: no skills yet -> no reference anywhere; harvests a winner
        search_1 = run_once("run-one", [0.7, 0.75])
        assert not list(search_1.glob("candidates/*/reference_solution.py"))
        from hillclimb.skills import load_skills

        assert len(load_skills(tmp_path / "knowledge")) == 1

        # search 2: first draft gets the reference file + starter cue
        search_2 = run_once("run-two", [0.8, 0.85])
        drafts = sorted(
            p.parent for p in search_2.glob("candidates/*/prompt.md")
            if "drafting a new candidate" in p.read_text()
        )
        assert len(drafts) == 2
        first, second = drafts
        assert (first / "reference_solution.py").exists()
        assert "Starter reference" in (first / "prompt.md").read_text()
        assert not (second / "reference_solution.py").exists()
        assert "Starter reference" not in (second / "prompt.md").read_text()
