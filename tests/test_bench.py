"""Learning A/B benchmark: pairing math, report rendering, flag plumbing."""

from __future__ import annotations

from pathlib import Path

import yaml

from hillclimb.bench import (
    BenchRow,
    bench_run_name,
    collect_bench_results,
    pair_and_summarize,
    render_bench_report,
)
from hillclimb.run import RunMeta, SearchMeta, write_run_meta, write_search_meta
from hillclimb.status import SearchStatus, ScoreRef, write_status
from hillclimb.workspace import create_run_dir, create_search_dir


def row(problem="p", learning=True, holdout=None, val=None, lower=False, started="2026-07-01"):
    return BenchRow(
        run_id="r", run_name="bench-p", problem_id=problem, learning=learning,
        state="done", holdout=holdout, val=val, lower_is_better=lower,
        started_at=started,
    )


class TestPairing:
    def test_pairs_chronologically_and_scores(self):
        rows = [
            row(learning=False, holdout=0.7, started="t1"),
            row(learning=True, holdout=0.8, started="t2"),
            row(learning=False, holdout=0.9, started="t3"),
            row(learning=True, holdout=0.85, started="t4"),
        ]
        summary = pair_and_summarize(rows)[0]
        assert len(summary.pairs) == 2
        assert summary.on_wins == 1 and summary.off_wins == 1 and summary.ties == 0
        assert summary.unpaired == []

    def test_direction_aware(self):
        rows = [row(learning=False, holdout=0.03, lower=True),
                row(learning=True, holdout=0.02, lower=True)]
        summary = pair_and_summarize(rows)[0]
        assert summary.on_wins == 1  # lower is better -> 0.02 wins

    def test_unpaired_and_missing_scores(self):
        rows = [
            row(learning=False, holdout=None, val=None),
            row(learning=True, holdout=0.8),
            row(learning=True, holdout=0.9, started="t9"),  # no off partner
        ]
        summary = pair_and_summarize(rows)[0]
        assert summary.ties == 1  # unscored arm -> tie, not a crash
        assert len(summary.unpaired) == 1

    def test_val_fallback_when_no_holdout(self):
        assert row(holdout=None, val=0.5).score == 0.5
        assert row(holdout=0.6, val=0.5).score == 0.6


class TestReport:
    def test_render(self):
        rows = [row(learning=False, holdout=0.7), row(learning=True, holdout=0.8)]
        text = render_bench_report(pair_and_summarize(rows))
        assert "| pair | off (no memory) | on (memory) | winner |" in text
        assert "| 1 | 0.7 | 0.8 | on |" in text
        assert "learning arm wins 1/1" in text
        assert "higher is better" in text

    def test_empty(self):
        assert "no finished benchmark searches" in render_bench_report([])


class TestCollect:
    def _make_search(self, runs_dir: Path, run_name: str, problem_id: str,
                     learning: bool, holdout: float | None, state="done"):
        run_id = f"20260710-{run_name}"
        run_dir = create_run_dir(runs_dir, run_id)
        write_run_meta(run_dir, RunMeta(
            run_id=run_id, name=run_name, kind="problem",
            target=problem_id, problem_ids=[problem_id],
        ))
        search_dir = create_search_dir(run_dir, problem_id)
        write_search_meta(search_dir, SearchMeta(
            search_id=problem_id, run_id=run_id, problem=problem_id,
            problem_id=problem_id, backend="dummy", model="-", metric="accuracy",
            learning_enabled=learning,
        ))
        write_status(search_dir, SearchStatus(
            search_id=problem_id, run_id=run_id, state=state, pid=0,
            selected=ScoreRef(candidate_id="c001", val_score=holdout, holdout_score=holdout),
        ))
        return search_dir

    def test_collect_filters_bench_runs(self, tmp_path):
        runs = tmp_path / "runs"
        self._make_search(runs, "bench-p-p1-off", "p", learning=False, holdout=0.7)
        self._make_search(runs, "bench-p-p1-on", "p", learning=True, holdout=0.8)
        self._make_search(runs, "ordinary-run", "p", learning=True, holdout=0.99)
        rows = collect_bench_results(runs)
        assert len(rows) == 2
        assert {r.learning for r in rows} == {False, True}
        everything = collect_bench_results(runs, include_all=True)
        assert len(everything) == 3

    def test_running_searches_excluded(self, tmp_path):
        runs = tmp_path / "runs"
        self._make_search(runs, "bench-p-p1-off", "p", learning=False,
                          holdout=None, state="running")
        # pid=0 is dead -> effective_state reports crashed, still not "done"
        rows = collect_bench_results(runs)
        assert all(r.state != "running" for r in rows)

    def test_meta_records_learning_flag(self, tmp_path):
        from hillclimb.api import create_search
        from hillclimb.config import Config
        from hillclimb.problem import ProblemSpec
        from hillclimb.run import load_search_meta

        config = Config()
        config.learning.enabled = False
        problem = ProblemSpec(
            problem_id="p", problem_dir=tmp_path, data_dir=tmp_path,
            description="", metric_name="accuracy", lower_is_better=False,
            verifier_cmd=["./verifier.sh"], time_budget_s=60,
        )
        run_dir = create_run_dir(tmp_path / "runs", "r1")
        search_dir = create_search(config, problem, run_dir, "r1", 60)
        meta = load_search_meta(search_dir)
        assert meta.learning_enabled is False


def test_bench_run_name():
    assert bench_run_name("circle-packing", 2, False) == "bench-circle-packing-p2-off"
    assert bench_run_name("circle-packing", 2, True) == "bench-circle-packing-p2-on"
