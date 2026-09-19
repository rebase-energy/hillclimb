"""Parallel-execution features: multi-seed trials, top-k holdout gating,
and (phase 2) the worker-pool scheduler."""

from __future__ import annotations

from tests.factories import trial as mk_trial

import sys
from pathlib import Path

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from tests.conftest import local_executor
from hillclimb.journal import Journal
from hillclimb.search import GreedySearcher
from hillclimb.dirs import create_search_dir
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
    kwargs.setdefault("budget", BudgetManager(3600, stop_margin_s=1))
    searcher = GreedySearcher(
        problem=task,
        config=config,
        journal=journal,
        backend=backend,
        executor=local_executor(),
        search_dir=search_dir,
        log=lambda *_: None,
        **kwargs,
    )
    return searcher, journal, search_dir


class TestMultiSeedTrials:
    def test_trials_recorded_and_val_is_mean(self, task, config):
        config.search.n_replicates = 3
        backend = FakeBackend()
        backend.queue(script=SEEDED_SCRIPT, notes="seeded draft\n")
        searcher, journal, _ = make_searcher(task, config, backend)

        candidate = searcher.run_operator("draft", None)

        assert candidate.status == "passing"
        assert len(candidate.trials) == 1
        assert [r.seed for r in candidate.trials[0].replicates] == [0, 1, 2]
        # seeds 0,1,2 -> scores 0.5, 0.6, 0.7 -> median 0.6
        assert candidate.val_score == pytest.approx(0.6)

    def test_trial_zero_artifacts_at_workspace_root(self, task, config):
        config.search.n_replicates = 2
        backend = FakeBackend()
        backend.queue(script=SEEDED_SCRIPT, notes="seeded draft\n")
        searcher, _, _ = make_searcher(task, config, backend)

        candidate = searcher.run_operator("draft", None)

        candidate_dir = Path(candidate.candidate_dir)
        assert (candidate_dir / "submission.csv").exists()
        t0 = candidate_dir / "trials" / "t0"
        assert (t0 / "solution.py").exists()
        assert (t0 / "replicates" / "r0" / "solution.py").exists()
        assert (t0 / "replicates" / "r1" / "solution.py").exists()
        assert not (t0 / "params.json").exists()  # nothing declared

    def test_seed_flaky_candidate_is_buggy(self, task, config):
        config.search.n_replicates = 2
        backend = FakeBackend()
        backend.queue(script=FLAKY_SCRIPT, notes="flaky draft\n")
        searcher, _, _ = make_searcher(task, config, backend)

        candidate = searcher.run_operator("draft", None)

        assert candidate.status == "buggy"
        assert len(candidate.trials[0].replicates) == 2

    def test_single_replicate_still_gets_a_trial_dir(self, task, config):
        assert config.search.n_replicates == 1
        backend = FakeBackend()
        backend.queue(script=ok_script(0.7), notes="draft\n")
        searcher, _, _ = make_searcher(task, config, backend)

        candidate = searcher.run_operator("draft", None)

        assert candidate.status == "passing"
        assert len(candidate.trials) == 1
        assert candidate.trials[0].replicates[0].seed is None
        assert candidate.val_score == 0.7
        candidate_dir = Path(candidate.candidate_dir)
        assert (candidate_dir / "trials" / "t0" / "replicates" / "r0" / "eval_result.json").exists()
        # r0 is hoisted: artifacts, result and exec logs at the candidate root
        for name in ("submission.csv", "eval_result.json", "exec_stdout.log"):
            assert (candidate_dir / name).exists()


class TestHoldoutTopK:
    def make_holdout_searcher(self, task, config, backend, top_k):
        from tests.test_search import FileHoldoutScorer

        config.holdout.top_k = top_k
        from hillclimb.evaluation import CandidateEvaluator

        search_dir = create_search_dir(config.paths.runs_dir, "test-search")
        journal = Journal(search_dir / "journal.jsonl")
        problem = task.model_copy(update={"holdout_cmd": task.verifier_cmd + ["--holdout"]})
        evaluator = CandidateEvaluator(
            executor=local_executor(), problem=problem, config=config,
            holdout_scorer=FileHoldoutScorer(), journal=journal,
        )
        searcher = GreedySearcher(
            problem=problem,
            config=config,
            journal=journal,
            backend=backend,
            executor=evaluator.executor,
            budget=BudgetManager(3600, stop_margin_s=1),
            search_dir=search_dir,
            log=lambda *_: None,
            evaluator=evaluator,
        )
        return searcher, journal

    def holdout_script(self, val):
        return f'''
import pandas as pd
sample = pd.read_csv("data/sample_submission.csv")
sample.to_csv("submission.csv", index=False)
open("holdout_predictions.csv", "w").write("{val}")
print("val_score: {val}")
'''

    def test_gate_skips_holdout_below_top_k(self, task, config):
        backend = FakeBackend()
        # two strong drafts fill top_k=2, then a weak one
        for val in (0.9, 0.8, 0.1):
            backend.queue(script=self.holdout_script(val), notes="d\n")
        searcher, journal = self.make_holdout_searcher(task, config, backend, top_k=2)
        for _ in range(3):
            searcher.run_operator("draft", None)

        strong1, strong2, weak = journal.get("c000"), journal.get("c001"), journal.get("c002")
        assert strong1.holdout_score is not None
        assert strong2.holdout_score is not None
        assert weak.status == "passing"  # still climbs on val
        assert weak.holdout_score is None  # gated: no holdout query spent

    def test_gate_disabled_scores_everyone(self, task, config):
        backend = FakeBackend()
        for val in (0.9, 0.8, 0.1):
            backend.queue(script=self.holdout_script(val), notes="d\n")
        searcher, journal = self.make_holdout_searcher(task, config, backend, top_k=0)
        for _ in range(3):
            searcher.run_operator("draft", None)
        assert all(journal.get(f"c00{i}").holdout_score is not None for i in (0, 1, 2))

    def test_gated_candidate_not_selectable(self, task, config):
        backend = FakeBackend()
        for val in (0.9, 0.8, 0.1):
            backend.queue(script=self.holdout_script(val), notes="d\n")
        searcher, journal = self.make_holdout_searcher(task, config, backend, top_k=2)
        for _ in range(3):
            searcher.run_operator("draft", None)
        selected = journal.selected_candidate(True, "rank-blend")
        assert selected.candidate_id != "c002"


# --- phase 2: worker pool ---

import threading  # noqa: E402
import time  # noqa: E402

from hillclimb.backends.fake import GateBackend  # noqa: E402
from hillclimb.control import ControlCommand, write_command  # noqa: E402
from hillclimb.search import ParkedSearch, StopRequested  # noqa: E402


def pool_searcher(task, config, backend, n, max_candidates=10, **kwargs):
    config.search.parallel_operators = n
    return make_searcher(task, config, backend, max_candidates=max_candidates, **kwargs)


CRASH = 'raise RuntimeError("boom")\n'


class GoldenScenario:
    """One scripted history for the golden-equivalence check. `budget_spent`
    pre-spends the budget so the ensemble window opens immediately."""

    def __init__(self, queue_fn, max_candidates, budget_spent=0.0):
        self._queue_fn = queue_fn
        self.max_candidates = max_candidates
        self.budget_spent = budget_spent

    def queue(self, backend):
        self._queue_fn(backend)

    def searcher_kwargs(self):
        if not self.budget_spent:
            return {}
        return {"budget": BudgetManager(3600, stop_margin_s=1, spent_s=self.budget_spent)}


def _drafts_then_improve(backend):
    backend.queue(script=ok_script(0.6), notes="a\n")
    backend.queue(script=ok_script(0.7), notes="b\n")
    backend.queue(script=ok_script(0.5), notes="c\n")
    backend.queue(script=ok_script(0.8), notes="improve\n")


def _debug_chain(backend):
    backend.queue(script=CRASH, notes="buggy draft\n")
    backend.queue(script=CRASH, notes="failed fix\n")
    backend.queue(script=ok_script(0.6), notes="fixed\n")
    backend.queue(script=ok_script(0.7), notes="draft two\n")
    backend.queue(script=ok_script(0.5), notes="draft three\n")


def _ensemble_window(backend):
    backend.queue(script=ok_script(0.6), notes="a\n")
    backend.queue(script=ok_script(0.7), notes="b\n")
    backend.queue(script=ok_script(0.9), notes="ensemble\n")
    backend.queue(script=ok_script(0.5), notes="post-ensemble draft\n")


def _improve_tie(backend):
    backend.queue(script=ok_script(0.7), notes="a\n")
    backend.queue(script="print('val_score: 0.7')\nimport shutil\nshutil.copy('data/sample_submission.csv', 'submission.csv')\n", notes="b same score\n")
    backend.queue(script=ok_script(0.5), notes="c\n")
    backend.queue(script=ok_script(0.8), notes="improve\n")


GOLDEN_SCENARIOS = {
    "drafts-then-improve": GoldenScenario(_drafts_then_improve, max_candidates=5),
    "debug-chain": GoldenScenario(_debug_chain, max_candidates=6),
    # 3600*0.2 reserve + 1s margin: remaining 600s opens the window at once
    "ensemble-window": GoldenScenario(_ensemble_window, max_candidates=5, budget_spent=3000),
    "improve-tie": GoldenScenario(_improve_tie, max_candidates=5),
}

# Journal event sequences recorded from the historical serial loop (before it
# was deleted in favor of the unified pool loop). The unified loop must
# reproduce them exactly; a diff here means the greedy schedule changed.
# Repeated candidate_result lines are the selection resync flipping is_selected.
GOLDEN_SEQUENCES = {
    "drafts-then-improve": [
        ("candidate_result", "c000", "baseline", "passing", None),
        ("candidate_created", "c001", "draft", "pending", None),
        ("candidate_result", "c001", "draft", "passing", None),
        ("candidate_result", "c001", "draft", "passing", None),
        ("candidate_created", "c002", "draft", "pending", None),
        ("candidate_result", "c002", "draft", "passing", None),
        ("candidate_result", "c002", "draft", "passing", None),
        ("candidate_created", "c003", "draft", "pending", None),
        ("candidate_result", "c003", "draft", "passing", None),
        ("candidate_created", "c004", "improve", "pending", "c002"),
        ("candidate_result", "c004", "improve", "passing", "c002"),
        ("candidate_result", "c004", "improve", "passing", "c002"),
    ],
    "debug-chain": [
        ("candidate_result", "c000", "baseline", "passing", None),
        ("candidate_created", "c001", "draft", "pending", None),
        ("candidate_result", "c001", "draft", "buggy", None),
        ("candidate_created", "c002", "debug", "pending", "c001"),
        ("candidate_result", "c002", "debug", "buggy", "c001"),
        ("candidate_created", "c003", "debug", "pending", "c002"),
        ("candidate_result", "c003", "debug", "passing", "c002"),
        ("candidate_result", "c003", "debug", "passing", "c002"),
        ("candidate_created", "c004", "draft", "pending", None),
        ("candidate_result", "c004", "draft", "passing", None),
        ("candidate_result", "c004", "draft", "passing", None),
        ("candidate_created", "c005", "draft", "pending", None),
        ("candidate_result", "c005", "draft", "passing", None),
    ],
    "ensemble-window": [
        ("candidate_result", "c000", "baseline", "passing", None),
        ("candidate_created", "c001", "draft", "pending", None),
        ("candidate_result", "c001", "draft", "passing", None),
        ("candidate_result", "c001", "draft", "passing", None),
        ("candidate_created", "c002", "draft", "pending", None),
        ("candidate_result", "c002", "draft", "passing", None),
        ("candidate_result", "c002", "draft", "passing", None),
        ("candidate_created", "c003", "ensemble", "pending", "c002"),
        ("candidate_result", "c003", "ensemble", "passing", "c002"),
        ("candidate_result", "c003", "ensemble", "passing", "c002"),
        ("candidate_created", "c004", "draft", "pending", None),
        ("candidate_result", "c004", "draft", "passing", None),
    ],
    "improve-tie": [
        ("candidate_result", "c000", "baseline", "passing", None),
        ("candidate_created", "c001", "draft", "pending", None),
        ("candidate_result", "c001", "draft", "passing", None),
        ("candidate_result", "c001", "draft", "passing", None),
        ("candidate_created", "c002", "draft", "pending", None),
        ("candidate_result", "c002", "draft", "passing", None),
        ("candidate_created", "c003", "draft", "pending", None),
        ("candidate_result", "c003", "draft", "passing", None),
        ("candidate_created", "c004", "improve", "pending", "c001"),
        ("candidate_result", "c004", "improve", "passing", "c001"),
        ("candidate_result", "c004", "improve", "passing", "c001"),
    ],
}


def journal_sequence(journal_path):
    import json

    events = []
    for line in journal_path.read_text().splitlines():
        r = json.loads(line)
        if r.get("event") in ("candidate_created", "candidate_result"):
            events.append(
                (r["event"], r["candidate_id"], r["operator"], r["status"], r["parent_id"])
            )
    return events


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
        assert len({c.candidate_dir for c in pending}) == 3
        backend.release_all()
        runner.join(timeout=30)
        assert not runner.is_alive()
        assert len(journal.scored_candidates()) == 3

    @pytest.mark.parametrize("scenario_name", sorted(GOLDEN_SCENARIOS))
    def test_unified_loop_matches_recorded_serial_sequence(self, task, config, scenario_name):
        """Golden succession: the unified loop (pool at n=1) reproduces the
        journal event sequences recorded from the historical serial loop,
        across draft/improve, debug-chain, ensemble-window, and improve-tie
        histories."""
        scenario = GOLDEN_SCENARIOS[scenario_name]
        backend = FakeBackend()
        scenario.queue(backend)
        searcher, journal, search_dir = make_searcher(
            task, config, backend,
            max_candidates=scenario.max_candidates, **scenario.searcher_kwargs(),
        )
        searcher.run()

        assert (
            journal_sequence(search_dir / "journal.jsonl")
            == GOLDEN_SEQUENCES[scenario_name]
        )

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
        assert statuses == ["parked", "passing", "passing", "passing"]

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

    def test_graceful_deadline_lets_in_flight_finish(self, task, config):
        backend = GateBackend()
        backend.queue(script=ok_script(0.6), notes="a\n")
        backend.queue(script=ok_script(0.7), notes="b\n")
        searcher, journal, _ = pool_searcher(task, config, backend, n=2, max_candidates=6)
        runner = threading.Thread(target=searcher.run)
        runner.start()
        assert backend.started.acquire(timeout=10)
        assert backend.started.acquire(timeout=10)
        # the budget runs out while both operators are in flight
        searcher.budget._started = time.monotonic() - searcher.budget.total_s - 1
        time.sleep(2.0)
        assert runner.is_alive()  # graceful (default): still waiting on the operators
        backend.release_all()
        runner.join(timeout=30)
        assert not runner.is_alive()
        # both finished and were committed as real work, past the deadline
        assert len(journal.scored_candidates()) == 2
        assert searcher.budget.elapsed() > searcher.budget.total_s

    def test_hard_deadline_aborts_in_flight(self, task, config):
        config.budget.deadline = "hard"
        backend = GateBackend()
        backend.queue(script=ok_script(0.6), notes="a\n")
        backend.queue(script=ok_script(0.7), notes="b\n")
        searcher, journal, _ = pool_searcher(task, config, backend, n=2, max_candidates=6)
        backend.abort = searcher.abort
        outcome: dict = {}

        def run():
            try:
                outcome["selected"] = searcher.run()
            except Exception as exc:  # noqa: BLE001
                outcome["exc"] = exc

        runner = threading.Thread(target=run)
        runner.start()
        assert backend.started.acquire(timeout=10)
        assert backend.started.acquire(timeout=10)
        # nobody releases the gates: the deadline alone must cut the operators off
        searcher.budget._started = time.monotonic() - searcher.budget.total_s - 1
        runner.join(timeout=30)
        assert not runner.is_alive()
        assert "exc" not in outcome
        abandoned = [c for c in journal.candidates.values() if c.status == "abandoned"]
        assert len(abandoned) == 2
        assert all(c.summary == "cut off at the budget deadline" for c in abandoned)
        assert not [c for c in journal.scored_candidates() if c.operator != "baseline"]

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
        assert len({c.candidate_dir for c in reloaded.candidates.values()}) == 8


class TestDecideNextPolicy:
    def test_prospective_branches_counts_pending(self, task, config):
        backend = FakeBackend()
        searcher, journal, _ = pool_searcher(task, config, backend, n=2)
        from hillclimb.candidate import Candidate

        journal.candidate_created(Candidate(candidate_id="c000", operator="draft", candidate_dir="w"))
        assert searcher._prospective_branches() == 1  # pending draft counts

    def test_debuggable_tip_skips_active_child_and_depth(self, task, config):
        backend = FakeBackend()
        searcher, journal, _ = pool_searcher(task, config, backend, n=2)
        from hillclimb.candidate import Candidate

        journal.candidate_result(Candidate(candidate_id="c000", operator="draft", status="buggy", candidate_dir="w"))
        assert searcher._debuggable_tip().candidate_id == "c000"
        # pending debug child blocks the tip
        journal.candidate_created(
            Candidate(candidate_id="c001", operator="debug", parent_id="c000", status="pending", candidate_dir="w")
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
        from hillclimb.status import read_status
        assert resume_spent_seconds(read_status(tmp_path), journal) == 1234.0

    def test_falls_back_to_work_sum(self, tmp_path):
        from hillclimb.api import resume_spent_seconds
        from hillclimb.candidate import BackendInfo, Candidate

        journal = Journal(tmp_path / "journal.jsonl")
        journal.candidate_result(
            Candidate(
                candidate_id="c001", operator="draft",
                backend=BackendInfo(agent_duration_s=100.0),
                trials=[mk_trial(duration_s=50.0)],
            )
        )
        assert resume_spent_seconds(None, journal) == 150.0


class TestCostCeiling:
    def test_parks_when_ceiling_reached(self, task, config):
        config.budget.max_cost_usd = 0.05
        backend = FakeBackend()
        backend.queue(script=ok_script(0.6), notes="d\n", result={"cost_usd": 0.06})
        searcher, journal, _ = make_searcher(task, config, backend)

        searcher.run_operator("draft", None)  # spends past the ceiling
        with pytest.raises(ParkedSearch, match="cost ceiling"):
            searcher._check_cost_ceiling()
        assert searcher.total_cost_usd() == pytest.approx(0.06)

    def test_no_ceiling_by_default(self, task, config):
        backend = FakeBackend()
        backend.queue(script=ok_script(0.6), notes="d\n", result={"cost_usd": 999.0})
        searcher, _, _ = make_searcher(task, config, backend)
        searcher.run_operator("draft", None)
        searcher._check_cost_ceiling()  # no raise

    def test_cost_in_status(self, task, config, tmp_path):
        from hillclimb.budget import BudgetManager as BM
        from hillclimb.status import SearchStatus, StatusWriter, read_status

        backend = FakeBackend()
        backend.queue(script=ok_script(0.6), notes="d\n", result={"cost_usd": 1.25})
        searcher, _, search_dir = make_searcher(task, config, backend)
        searcher.status = StatusWriter(
            search_dir, SearchStatus(search_id="s"), budget=BM(100, stop_margin_s=0)
        )
        searcher.run_operator("draft", None)
        assert read_status(search_dir).cost_usd == pytest.approx(1.25)


class TestIncumbentSeeding:
    def test_seed_scored_as_floor_candidate(self, task, config, tmp_path):
        seed = tmp_path / "incumbent.py"
        seed.write_text(ok_script(0.8))
        backend = FakeBackend()
        backend.queue(script=ok_script(0.6), notes="worse draft\n")
        searcher, journal, _ = make_searcher(task, config, backend, seed_solution=seed)
        searcher.max_candidates = 3  # baseline + seed + one draft

        searcher.run()

        seeds = [c for c in journal.candidates.values() if c.operator == "seed"]
        assert len(seeds) == 1
        assert seeds[0].status == "passing"
        assert seeds[0].val_score == 0.8
        # the weaker draft cannot displace the incumbent floor
        best = journal.best_candidate(task.higher_is_better)
        assert best.candidate_id == seeds[0].candidate_id

    def test_improve_targets_the_seed(self, task, config, tmp_path):
        seed = tmp_path / "incumbent.py"
        seed.write_text(ok_script(0.9))
        backend = FakeBackend()
        for _ in range(3):  # drafts all weaker than the incumbent
            backend.queue(script=ok_script(0.5), notes="d\n")
        backend.queue(script=ok_script(0.95), notes="improved incumbent\n")
        config.search.num_drafts = 3
        searcher, journal, _ = make_searcher(task, config, backend, seed_solution=seed)
        searcher.max_candidates = 6  # baseline + seed + 3 drafts + 1 improve

        searcher.run()

        improves = [c for c in journal.candidates.values() if c.operator == "improve"]
        assert improves, "expected an improve after drafting completed"
        seed_id = next(c.candidate_id for c in journal.candidates.values() if c.operator == "seed")
        assert improves[0].parent_id == seed_id

    def test_resume_does_not_reseed(self, task, config, tmp_path):
        seed = tmp_path / "incumbent.py"
        seed.write_text(ok_script(0.8))
        backend = FakeBackend()
        searcher, journal, search_dir = make_searcher(task, config, backend, seed_solution=seed)
        searcher.max_candidates = 2  # baseline + seed, then stop
        searcher.run()

        from hillclimb.budget import BudgetManager as BM

        resumed = GreedySearcher(
            problem=task, config=config, journal=Journal(search_dir / "journal.jsonl"),
            backend=backend, executor=local_executor(),
            budget=BM(3600, stop_margin_s=1), search_dir=search_dir,
            log=lambda *_: None, seed_solution=seed, max_candidates=2,
        )
        resumed.run()
        seeds = [c for c in resumed.journal.candidates.values() if c.operator == "seed"]
        assert len(seeds) == 1


def test_worker_crash_does_not_hang_the_scheduler(task, config):
    """A worker that dies without reporting used to leave the candidate in
    flight and the scheduler blocked on the done-queue forever."""
    config.search.parallel_operators = 2
    backend = FakeBackend()
    backend.queue(script=ok_script(0.5), notes="d\n")
    search_dir = create_search_dir(config.paths.runs_dir, "crash-search")
    journal = Journal(search_dir / "journal.jsonl")

    class ExplodingExecutor:
        def execute(self, script, candidate_dir, timeout_s, seed=None):
            raise RuntimeError("executor blew up")

    searcher = GreedySearcher(
        problem=task, config=config, journal=journal, backend=backend,
        executor=ExplodingExecutor(),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir, max_candidates=3, log=lambda *_: None,
    )
    searcher.run()  # terminates instead of deadlocking
    assert all(c.status != "pending" for c in journal.candidates.values())
    assert any("orchestrator error" in (c.summary or "") for c in journal.candidates.values())
