"""Parallel-execution features: multi-seed trials, top-k holdout gating,
and (phase 2) the worker-pool scheduler."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.executor import LocalExecutor
from hillclimb.journal import Journal
from hillclimb.search import GreedySearcher
from hillclimb.workspace import create_search_dir
from tests.conftest import ok_script

SEEDED_SCRIPT = """\
import os
import shutil
shutil.copy("data/sample_submission.csv", "submission.csv")
seed = os.environ.get("HILLCLIMB_TRIAL_SEED", "0")
print(f"val_score: 0.{int(seed) + 5}")
"""

FLAKY_SCRIPT = """\
import os
import shutil
shutil.copy("data/sample_submission.csv", "submission.csv")
seed = int(os.environ.get("HILLCLIMB_TRIAL_SEED", "0"))
if seed == 1:
    raise RuntimeError("flaky on seed 1")
print("val_score: 0.5")
"""


def make_searcher(task, config, backend, **kwargs):
    search_dir = create_search_dir(config.paths.runs_dir, "test-search")
    journal = Journal(search_dir / "journal.jsonl")
    searcher = GreedySearcher(
        problem=task,
        config=config,
        journal=journal,
        backend=backend,
        executor=LocalExecutor(Path(sys.executable)),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir,
        log=lambda *_: None,
        **kwargs,
    )
    return searcher, journal, search_dir


class TestMultiSeedTrials:
    def test_trials_recorded_and_val_is_mean(self, task, config):
        config.search.n_trials = 3
        backend = FakeBackend()
        backend.queue(script=SEEDED_SCRIPT, notes="seeded draft\n")
        searcher, journal, _ = make_searcher(task, config, backend)

        candidate = searcher.run_operator("draft", None)

        assert candidate.status == "ok"
        assert len(candidate.trials) == 3
        assert [t.seed for t in candidate.trials] == [0, 1, 2]
        # seeds 0,1,2 -> scores 0.5, 0.6, 0.7 -> mean 0.6
        assert candidate.val_score == pytest.approx(0.6)

    def test_trial_zero_artifacts_at_workspace_root(self, task, config):
        config.search.n_trials = 2
        backend = FakeBackend()
        backend.queue(script=SEEDED_SCRIPT, notes="seeded draft\n")
        searcher, _, _ = make_searcher(task, config, backend)

        candidate = searcher.run_operator("draft", None)

        workspace = Path(candidate.workspace)
        assert (workspace / "submission.csv").exists()
        assert (workspace / "trials" / "t0" / "solution.py").exists()
        assert (workspace / "trials" / "t1" / "solution.py").exists()

    def test_seed_flaky_candidate_is_buggy(self, task, config):
        config.search.n_trials = 2
        backend = FakeBackend()
        backend.queue(script=FLAKY_SCRIPT, notes="flaky draft\n")
        searcher, _, _ = make_searcher(task, config, backend)

        candidate = searcher.run_operator("draft", None)

        assert candidate.status == "buggy"
        assert len(candidate.trials) == 2

    def test_single_trial_path_unchanged(self, task, config):
        assert config.search.n_trials == 1
        backend = FakeBackend()
        backend.queue(script=ok_script(0.7), notes="draft\n")
        searcher, _, _ = make_searcher(task, config, backend)

        candidate = searcher.run_operator("draft", None)

        assert candidate.status == "ok"
        assert len(candidate.trials) == 1
        assert candidate.trials[0].seed is None
        assert candidate.val_score == 0.7
        assert not (Path(candidate.workspace) / "trials").exists()


class TestHoldoutTopK:
    def make_holdout_searcher(self, task_larger, config, backend, top_k):
        from hillclimb.holdout import build_data_view

        config.holdout.top_k = top_k
        search_dir = create_search_dir(config.paths.runs_dir, "test-search")
        info = build_data_view(
            task_larger.data_dir, search_dir, task_larger.sample_submission, 0.3, 42
        )
        journal = Journal(search_dir / "journal.jsonl")
        searcher = GreedySearcher(
            problem=task_larger,
            config=config,
            journal=journal,
            backend=backend,
            executor=LocalExecutor(Path(sys.executable)),
            budget=BudgetManager(3600, stop_margin_s=1),
            search_dir=search_dir,
            log=lambda *_: None,
            holdout=info,
        )
        return searcher, journal

    def holdout_script(self, val):
        return f'''
import pandas as pd
sample = pd.read_csv("data/sample_submission.csv")
sample.to_csv("submission.csv", index=False)
hold = pd.read_csv("data/holdout.csv")
truth = (hold["feature"] > 0)
pd.DataFrame({{"id": hold["id"], "target": truth.astype(int)}}).to_csv(
    "holdout_predictions.csv", index=False)
print("val_score: {val}")
'''

    def test_gate_skips_holdout_below_top_k(self, task_larger, config):
        backend = FakeBackend()
        # two strong drafts fill top_k=2, then a weak one
        for val in (0.9, 0.8, 0.1):
            backend.queue(script=self.holdout_script(val), notes="d\n")
        searcher, journal = self.make_holdout_searcher(task_larger, config, backend, top_k=2)
        for _ in range(3):
            searcher.run_operator("draft", None)

        strong1, strong2, weak = journal.get("c000"), journal.get("c001"), journal.get("c002")
        assert strong1.holdout_score is not None
        assert strong2.holdout_score is not None
        assert weak.status == "ok"  # still climbs on val
        assert weak.holdout_score is None  # gated: no holdout query spent

    def test_gate_disabled_scores_everyone(self, task_larger, config):
        backend = FakeBackend()
        for val in (0.9, 0.8, 0.1):
            backend.queue(script=self.holdout_script(val), notes="d\n")
        searcher, journal = self.make_holdout_searcher(task_larger, config, backend, top_k=0)
        for _ in range(3):
            searcher.run_operator("draft", None)
        assert all(journal.get(f"c00{i}").holdout_score is not None for i in (0, 1, 2))

    def test_gated_candidate_not_selectable(self, task_larger, config):
        backend = FakeBackend()
        for val in (0.9, 0.8, 0.1):
            backend.queue(script=self.holdout_script(val), notes="d\n")
        searcher, journal = self.make_holdout_searcher(task_larger, config, backend, top_k=2)
        for _ in range(3):
            searcher.run_operator("draft", None)
        selected = journal.selected_candidate(False, "rank-blend")
        assert selected.candidate_id != "c002"
