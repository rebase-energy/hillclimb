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


# --- phase 2: worker pool ---

import threading  # noqa: E402
import time  # noqa: E402

from hillclimb.backends.fake import GateBackend  # noqa: E402
from hillclimb.control import ControlCommand, write_command  # noqa: E402
from hillclimb.search import ParkedSearch, StopRequested  # noqa: E402


def pool_searcher(task, config, backend, n, max_candidates=10, **kwargs):
    config.search.parallel_agents = n
    return make_searcher(task, config, backend, max_candidates=max_candidates, **kwargs)


class TestWorkerPool:
    def test_three_drafts_in_flight_concurrently(self, task, config):
        backend = GateBackend()
        for score in (0.6, 0.7, 0.8):
            backend.queue(script=ok_script(score), notes="d\n")
        searcher, journal, _ = pool_searcher(task, config, backend, n=3, max_candidates=4)

        runner = threading.Thread(target=searcher.run)
        runner.start()
        for _ in range(3):
            assert backend.started.acquire(timeout=10)
        pending = [c for c in journal.candidates.values() if c.status == "pending"]
        assert len(pending) == 3
        assert len({c.candidate_id for c in pending}) == 3
        assert len({c.workspace for c in pending}) == 3
        backend.release_all()
        runner.join(timeout=30)
        assert not runner.is_alive()
        assert len(journal.scored_candidates()) == 3

    def test_pool_at_one_matches_serial_sequence(self, task, config):
        """Golden equivalence: the pool path at n=1 produces the same journal
        event sequence as the serial loop."""

        def scenario(backend):
            backend.queue(script=ok_script(0.6), notes="a\n")
            backend.queue(script=ok_script(0.7), notes="b\n")
            backend.queue(script=ok_script(0.5), notes="c\n")
            backend.queue(script=ok_script(0.8), notes="improve\n")

        def sequence(journal_path):
            import json

            events = []
            for line in journal_path.read_text().splitlines():
                r = json.loads(line)
                if r.get("event") in ("candidate_created", "candidate_result"):
                    events.append((r["event"], r["candidate_id"], r["operator"], r["status"]))
            return events

        serial_backend = FakeBackend()
        scenario(serial_backend)
        serial, s_journal, s_dir = make_searcher(task, config, serial_backend, max_candidates=5)
        serial.run()

        config.paths.runs_dir = config.paths.runs_dir / "pool"
        pool_backend = FakeBackend()
        scenario(pool_backend)
        pooled, p_journal, p_dir = make_searcher(task, config, pool_backend, max_candidates=5)
        # _run_pool is invoked below run(), so write the baseline like run() does
        from hillclimb.baseline import write_baseline

        p_journal.candidate_result(write_baseline(task, p_dir))
        pooled._run_pool(1)

        assert sequence(p_dir / "journal.jsonl") == sequence(s_dir / "journal.jsonl")

    def test_rate_limit_drains_in_flight_then_parks(self, task, config):
        # GateBackend pops responses at RELEASE time (FIFO), so the release
        # order maps to the response order below. The scheduler refills the
        # freed slot after the first commit — the third gate/response is that
        # refill, still in flight when the park lands (the drain case).
        backend = GateBackend()
        backend.queue(script=ok_script(0.6), notes="good draft\n")       # 1st release
        backend.queue(script=None, result={"ok": False, "error_kind": "rate_limited",
                                           "error_message": "limit"})    # 2nd release
        backend.queue(script=ok_script(0.7), notes="drained draft\n")    # 3rd release
        searcher, journal, _ = pool_searcher(task, config, backend, n=2, max_candidates=5)

        outcome: dict = {}

        def run():
            try:
                searcher.run()
            except Exception as exc:  # noqa: BLE001
                outcome["exc"] = exc

        runner = threading.Thread(target=run)
        runner.start()
        assert backend.started.acquire(timeout=10)
        assert backend.started.acquire(timeout=10)
        backend.release(0)  # first worker commits ok; scheduler refills a slot
        assert backend.started.acquire(timeout=10)  # the refill arrives
        backend.release(1)  # rate-limited -> park committed, drain begins
        time.sleep(1.5)
        backend.release(2)  # in-flight refill finishes and is committed
        runner.join(timeout=30)
        assert not runner.is_alive()

        assert isinstance(outcome.get("exc"), ParkedSearch)
        statuses = sorted(c.status for c in journal.candidates.values())
        # baseline ok + two scored drafts + the parked one
        assert statuses == ["ok", "ok", "ok", "parked"]

    def test_graceful_stop_drains_in_flight(self, task, config):
        backend = GateBackend()
        backend.queue(script=ok_script(0.6), notes="a\n")
        backend.queue(script=ok_script(0.7), notes="b\n")
        searcher, journal, search_dir = pool_searcher(task, config, backend, n=2, max_candidates=6)

        outcome: dict = {}

        def run():
            try:
                searcher.run()
            except Exception as exc:  # noqa: BLE001
                outcome["exc"] = exc

        runner = threading.Thread(target=run)
        runner.start()
        assert backend.started.acquire(timeout=10)
        assert backend.started.acquire(timeout=10)
        write_command(search_dir, ControlCommand(action="stop", source="cli"))
        time.sleep(2.0)  # control poll notices, sets drain
        backend.release_all()
        runner.join(timeout=30)
        assert not runner.is_alive()

        assert isinstance(outcome.get("exc"), StopRequested)
        # both in-flight operators finished and were committed as real work
        assert len(journal.scored_candidates()) == 2

    def test_journal_integrity_under_parallelism(self, task, config):
        import json

        backend = FakeBackend()
        for i in range(8):
            backend.queue(script=ok_script(0.5 + i / 100), notes=f"d{i}\n")
        searcher, journal, search_dir = pool_searcher(task, config, backend, n=3, max_candidates=8)
        searcher.run()

        created, terminal = set(), set()
        for line in (search_dir / "journal.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("event") == "candidate_created":
                created.add(r["candidate_id"])
            elif r.get("event") == "candidate_result" and r["status"] != "pending":
                terminal.add(r["candidate_id"])
        assert created <= terminal  # every created candidate reached a terminal record
        reloaded = Journal(search_dir / "journal.jsonl")
        assert len(reloaded.candidates) == 8
        assert len({c.workspace for c in reloaded.candidates.values()}) == 8


class TestDecideNextPolicy:
    def test_prospective_branches_counts_pending(self, task, config):
        backend = FakeBackend()
        searcher, journal, _ = pool_searcher(task, config, backend, n=2)
        from hillclimb.candidate import Candidate

        journal.candidate_created(Candidate(candidate_id="c000", operator="draft", workspace="w"))
        assert searcher._prospective_branches() == 1  # pending draft counts

    def test_debuggable_tip_skips_active_child_and_depth(self, task, config):
        backend = FakeBackend()
        searcher, journal, _ = pool_searcher(task, config, backend, n=2)
        from hillclimb.candidate import Candidate

        journal.candidate_result(Candidate(candidate_id="c000", operator="draft", status="buggy", workspace="w"))
        assert searcher._debuggable_tip().candidate_id == "c000"
        # pending debug child blocks the tip
        journal.candidate_created(
            Candidate(candidate_id="c001", operator="debug", parent_id="c000", status="pending", workspace="w")
        )
        assert searcher._debuggable_tip() is None


class TestMachineSlots:
    def test_cap_and_release(self, tmp_path):
        from hillclimb.slots import MachineSlots

        slots = MachineSlots(tmp_path, limit=1)
        first = slots.try_acquire()
        assert first is not None
        assert slots.try_acquire() is None  # cap reached
        first.release()
        second = slots.try_acquire()
        assert second is not None
        second.release()

    def test_zero_limit_is_noop(self, tmp_path):
        from hillclimb.slots import MachineSlots

        slots = MachineSlots(tmp_path / "slots", limit=0)
        assert slots.try_acquire() is not None
        assert not (tmp_path / "slots").exists()  # no-op cap creates nothing

    def test_cross_process_exclusion(self, tmp_path):
        import subprocess
        import sys as _sys

        from hillclimb.slots import MachineSlots

        holder = subprocess.Popen(
            [_sys.executable, "-c", (
                "import sys, time; sys.path.insert(0, 'src');"
                "from hillclimb.slots import MachineSlots;"
                f"h = MachineSlots(__import__('pathlib').Path({str(tmp_path)!r}), 1).try_acquire();"
                "print('held', flush=True); time.sleep(30)"
            )],
            stdout=subprocess.PIPE, text=True,
        )
        try:
            assert holder.stdout.readline().strip() == "held"
            assert MachineSlots(tmp_path, 1).try_acquire() is None
        finally:
            holder.kill()
            holder.wait()


class TestResumeAccounting:
    def test_reads_persisted_wall_clock(self, tmp_path):
        from hillclimb.api import resume_spent_seconds
        from hillclimb.status import BudgetStatus, SearchStatus, write_status

        write_status(
            tmp_path,
            SearchStatus(
                search_id="s", state="parked",
                budget=BudgetStatus(total_s=1800, spent_s=1234.0, remaining_s=566.0),
            ),
        )
        journal = Journal(tmp_path / "journal.jsonl")
        assert resume_spent_seconds(tmp_path, journal) == 1234.0

    def test_falls_back_to_work_sum(self, tmp_path):
        from hillclimb.api import resume_spent_seconds
        from hillclimb.candidate import BackendInfo, Candidate, Trial

        journal = Journal(tmp_path / "journal.jsonl")
        journal.candidate_result(
            Candidate(
                candidate_id="c001", operator="draft",
                backend=BackendInfo(agent_duration_s=100.0),
                trials=[Trial(duration_s=50.0)],
            )
        )
        assert resume_spent_seconds(tmp_path, journal) == 150.0
