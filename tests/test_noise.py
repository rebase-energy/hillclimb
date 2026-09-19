"""Noise handling: repeated trials, the median aggregate, and the accept band
that stops the search climbing measurement noise."""

from __future__ import annotations

import pytest

from tests.factories import trial as mk_trial

from math import isclose
from pathlib import Path

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.candidate import Candidate
from hillclimb.journal import Journal
from hillclimb.search import GreedySearcher
from hillclimb.dirs import create_search_dir
from tests.conftest import executor_for, ok_script


def candidate(cid: str, *scores: float, parent: str | None = None, trials=None) -> Candidate:
    if trials is not None:
        return Candidate(candidate_id=cid, operator="draft", parent_id=parent, status="passing",
                         candidate_dir="/tmp", trials=trials)
    return Candidate(
        candidate_id=cid,
        operator="draft",
        parent_id=parent,
        status="passing",
        candidate_dir="/tmp",
        trials=[mk_trial(*scores, submission_ok=True)] if scores else [],
    )


# --- aggregation ---


def test_val_score_is_the_median_of_replicates():
    """One slow run or unlucky seed must not drag the trial's score with
    it, which is exactly what a mean would do."""
    assert candidate("c1", 0.5).val_score == 0.5
    assert candidate("c1", 0.5, 0.6, 0.55).val_score == 0.55
    assert candidate("c1", 0.5, 0.5, 9.0).val_score == 0.5  # mean would say 3.33
    assert candidate("c1").val_score is None


def test_replicate_spread_needs_two_replicates():
    assert candidate("c1", 0.5).replicate_spread is None
    assert isclose(candidate("c1", 1.0, 1.2).replicate_spread, 0.1)  # MAD of two points
    assert candidate("c1", 1.0, 1.0, 1.0).replicate_spread == 0.0


def test_candidate_is_scored_by_its_best_trial():
    """Spread ACROSS parameter sets is signal, not noise: the candidate's
    score, metrics and holdout follow the best trial, and the noise floor
    only ever sees within-trial replicate spread."""
    c = candidate("c1", trials=[
        mk_trial(1.0, 1.2, params={"lr": 0.1}, holdout_score=0.9),
        mk_trial(2.0, 2.6, params={"lr": 0.2}, holdout_score=0.8, index=1),
    ])
    assert c.stamp_best_trial(higher_is_better=True).params == {"lr": 0.2}
    assert c.val_score == 2.3 and c.holdout_score == 0.8
    assert c.stamp_best_trial(higher_is_better=False).params == {"lr": 0.1}
    assert c.val_score == 1.1 and c.holdout_score == 0.9
    assert c.replicate_spreads == [pytest.approx(0.1), pytest.approx(0.3)]


# --- noise floor ---


def test_noise_floor_from_repeated_trials(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    assert journal.noise_floor() is None  # nothing measured twice yet

    journal.candidate_result(candidate("c1", 1.0))
    assert journal.noise_floor() is None  # a single trial says nothing

    journal.candidate_result(candidate("c2", 1.0, 1.2))  # spread 0.1
    journal.candidate_result(candidate("c3", 2.0, 2.6))  # spread 0.3
    assert journal.noise_floor() == 0.2  # median of the per-trial spreads


# --- the accept band ---


def make_searcher(task, config, backend=None, **kwargs):
    search_dir = create_search_dir(config.paths.runs_dir, "noise-run")
    journal = Journal(search_dir / "journal.jsonl")
    searcher = GreedySearcher(
        problem=task, config=config, journal=journal,
        backend=backend or FakeBackend(), executor=executor_for(task),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir, log=lambda *_: None, **kwargs,
    )
    return searcher, journal, search_dir


def test_band_defaults_to_off(task, config):
    searcher, _, _ = make_searcher(task, config)
    assert searcher.accept_band() == 0.0
    assert searcher._improves(0.5001, 0.5)  # strict comparison, as before
    assert not searcher._improves(0.5, 0.5)


def test_band_from_min_improvement(task, config):
    config.search.min_improvement = 0.01
    searcher, _, _ = make_searcher(task, config)
    assert searcher.accept_band() == 0.01
    assert not searcher._improves(0.505, 0.5)
    assert searcher._improves(0.52, 0.5)
    # ranking and gating still use the raw comparison
    assert searcher._improves(0.505, 0.5, band=0.0)


def test_band_from_measured_noise(task, config):
    config.search.noise_k = 2
    searcher, journal, _ = make_searcher(task, config)
    assert searcher.accept_band() == 0.0  # nothing measured yet

    journal.candidate_result(candidate("c1", 1.0, 1.2))  # spread 0.1
    assert isclose(searcher.accept_band(), 0.2)
    assert not searcher._improves(1.15, 1.0)
    assert searcher._improves(1.3, 1.0)


def test_band_respects_direction(task, config):
    config.search.min_improvement = 0.01
    task = task.model_copy(update={"higher_is_better": False})
    searcher, _, _ = make_searcher(task, config)
    assert searcher._improves(0.9, 1.0)
    assert not searcher._improves(0.995, 1.0)


def test_within_noise_candidate_is_not_promoted(task, config):
    """The end-to-end point of the band: a draft that is nominally ahead but
    inside the noise band must not become the thing the search climbs."""
    config.search.min_improvement = 0.05
    config.search.num_drafts = 3
    logs: list[str] = []
    backend = FakeBackend()
    backend.queue(script=ok_script(0.60), notes="first\n")
    backend.queue(script=ok_script(0.61), notes="noise-sized gain\n")
    backend.queue(script=ok_script(0.80), notes="real gain\n")
    searcher, journal, _ = make_searcher(task, config, backend)
    searcher.log = logs.append
    for _ in range(3):
        searcher.run_operator("draft", None)

    assert journal.get("c000").is_best  # first scored candidate
    assert not journal.get("c001").is_best  # +0.01 is inside the band
    assert journal.get("c002").is_best  # +0.20 clears it
    assert any("within noise, not promoted" in line for line in logs)


def test_bandit_reward_ignores_gains_inside_the_band():
    from hillclimb.bandit import REWARD_OK_NO_GAIN, candidate_reward

    parent = candidate("c1", 0.60)
    child = candidate("c2", 0.61, parent="c1")
    child.backend.model = "sonnet"
    assert candidate_reward(child, parent, higher_is_better=True) == 1.0
    assert candidate_reward(
        child, parent, higher_is_better=True, band=0.05
    ) == REWARD_OK_NO_GAIN


# --- trial execution mode ---

TIMED_SOLUTION = """\
import json, os, time
start = time.monotonic()
time.sleep(0.4)
open("submission.csv", "w").write("id\\n")
with open("../../../../overlap.log", "a") as fh:
    fh.write(f"{start},{time.monotonic()}\\n")
print("val_score: 0.5")
"""


def overlaps(path: Path) -> int:
    spans = [
        tuple(float(part) for part in line.split(","))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    return sum(
        1
        for i, (start_a, end_a) in enumerate(spans)
        for start_b, end_b in spans[i + 1:]
        if start_a < end_b and start_b < end_a
    )


def test_serial_trials_do_not_share_the_machine(task, config):
    """Anything that measures the machine (time, throughput, memory) measures
    its own sibling trials when they run concurrently."""
    config.search.n_replicates = 3
    config.search.replicate_mode = "serial"
    backend = FakeBackend()
    backend.queue(script=TIMED_SOLUTION, notes="timed\n")
    searcher, journal, _ = make_searcher(task, config, backend)
    node = searcher.run_operator("draft", None)

    replicates = node.trials[0].replicates
    assert len(node.trials) == 1 and len(replicates) == 3
    assert replicates[0].seed == 0 and replicates[2].seed == 2
    assert overlaps(Path(node.candidate_dir) / "overlap.log") == 0


def test_parallel_trials_run_concurrently(task, config):
    config.search.n_replicates = 3
    config.search.replicate_mode = "parallel"  # the default
    backend = FakeBackend()
    backend.queue(script=TIMED_SOLUTION, notes="timed\n")
    searcher, journal, _ = make_searcher(task, config, backend)
    node = searcher.run_operator("draft", None)

    assert len(node.trials[0].replicates) == 3
    assert overlaps(Path(node.candidate_dir) / "overlap.log") > 0
