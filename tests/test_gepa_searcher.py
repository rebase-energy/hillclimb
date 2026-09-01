"""GEPASearcher end to end with the fake driver: canonical candidates,
lineage, fitness, caching, control, resume — no gepa dependency."""

from __future__ import annotations

import threading

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.dirs import create_search_dir
from hillclimb.integrations.gepa.proposer import source_hash
from hillclimb.integrations.gepa.searcher import GEPASearcher
from hillclimb.journal import Journal
from hillclimb.search_runner import build_search_runner
from tests.conftest import executor_for, ok_script
from tests.gepa_fakes import FakeGEPADriver


def make_searcher(task, config, tmp_path, *, backend=None, driver=None, seed_score=0.5,
                  budget_s=3600, journal=None, search_dir=None):
    config.search.policy = "gepa"
    search_dir = search_dir or create_search_dir(tmp_path / "runs" / "r", "s")
    seed = tmp_path / "seed_solution.py"
    if not seed.exists():
        seed.write_text(ok_script(seed_score))
    return GEPASearcher(
        problem=task,
        config=config,
        journal=journal if journal is not None else Journal(search_dir / "journal.jsonl"),
        backend=backend or FakeBackend(),
        executor=executor_for(task),
        budget=BudgetManager(budget_s),
        search_dir=search_dir,
        log=lambda *_: None,
        seed_solution=seed,
        driver=driver or FakeGEPADriver(steps=0),
    ), search_dir


def test_seed_and_improvements_become_canonical_candidates(task, config, tmp_path):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), result={"cost_usd": 0.1, "model_id": "m-1"})
    backend.queue(script=ok_script(0.7), result={"cost_usd": 0.2, "model_id": "m-1"})
    driver = FakeGEPADriver(steps=2)
    searcher, search_dir = make_searcher(task, config, tmp_path, backend=backend, driver=driver)

    selected = searcher.run()

    journal = searcher.journal
    by_op = {c.operator: c for c in journal.candidates.values()}
    assert "baseline" in by_op and "seed" in by_op
    improves = sorted(
        (c for c in journal.candidates.values() if c.operator == "improve"),
        key=lambda c: c.candidate_id,
    )
    assert [c.val_score for c in improves] == [0.6, 0.7]
    # lineage: seed -> first improve -> second improve, via source hashes
    seed = by_op["seed"]
    assert improves[0].parent_id == seed.candidate_id
    assert improves[1].parent_id == improves[0].candidate_id
    for c in improves:
        assert c.policy_meta["optimizer"] == "gepa"
        assert c.policy_meta["source_hash"]
        assert c.policy_meta["gepa_fitness"] == c.val_score  # higher-is-better identity
    assert selected is not None and selected.val_score == 0.7
    assert improves[1].is_best
    # proposal scratch dirs and checkpoint state live under gepa/
    assert (search_dir / "gepa" / "proposals" / "p0001" / "prompt.md").exists()
    assert (search_dir / "gepa" / "state" / "gepa_state.bin").exists()
    assert (search_dir / "gepa" / "identity.json").exists()
    # agent costs are accounted
    assert searcher.total_cost_usd() == pytest.approx(0.3)


def test_lower_is_better_fitness_is_negated(task, config, tmp_path):
    task.higher_is_better = False
    backend = FakeBackend()
    backend.queue(script=ok_script(0.4))
    searcher, _ = make_searcher(
        task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=1), seed_score=0.5
    )
    selected = searcher.run()
    assert selected.val_score == 0.4  # raw journal direction, never negated
    improve = next(c for c in searcher.journal.candidates.values() if c.operator == "improve")
    assert improve.policy_meta["gepa_fitness"] == -0.4  # transformed at the boundary
    assert improve.is_best


def test_duplicate_source_reuses_the_candidate(task, config, tmp_path):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6))
    searcher, _ = make_searcher(
        task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=1)
    )
    searcher.run()
    n_before = len(searcher.journal.candidates)
    seed_source = (tmp_path / "seed_solution.py").read_text()
    result_a = searcher.bridge.eval_source(seed_source)
    result_b = searcher.bridge.eval_source(seed_source)
    assert result_a is result_b  # cache hit: no new id, no verifier call
    assert len(searcher.journal.candidates) == n_before


def test_buggy_proposal_gets_failure_fitness_and_feedback(task, config, tmp_path):
    backend = FakeBackend()
    backend.queue(script="raise RuntimeError('broken proposal')\n")
    backend.queue(script=ok_script(0.9))
    driver = FakeGEPADriver(steps=2)
    searcher, _ = make_searcher(task, config, tmp_path, backend=backend, driver=driver)
    searcher.run()
    journal = searcher.journal
    buggy = [c for c in journal.candidates.values() if c.status == "buggy"]
    assert len(buggy) == 1
    assert buggy[0].policy_meta["gepa_fitness"] <= -1.0e100
    # the next round's reflective dataset carried the failure's stderr
    last_feedback = driver.reflective[-1]["solution.py"][0]
    assert last_feedback["valid"] is False
    assert "broken proposal" in str(last_feedback["trials"])
    recovered = [c for c in journal.candidates.values() if c.val_score == 0.9]
    assert recovered and recovered[0].status == "ok"


def test_failure_fitness_dominates_all_valid_scores(task, config, tmp_path):
    task.higher_is_better = False  # valid fitnesses are negative
    config.search.policy = "gepa"
    config.search.policy_params = {"failure_fitness": -1.0}  # pathological config
    backend = FakeBackend()
    backend.queue(script="raise RuntimeError('boom')\n")
    searcher, _ = make_searcher(
        task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=1), seed_score=5.0
    )
    searcher.run()
    buggy = next(c for c in searcher.journal.candidates.values() if c.status == "buggy")
    seed = next(c for c in searcher.journal.candidates.values() if c.operator == "seed")
    assert buggy.policy_meta["gepa_fitness"] < seed.policy_meta["gepa_fitness"]


def test_stop_command_parks_the_loop_without_holdout(task, config, tmp_path):
    from hillclimb.control import ControlCommand
    from hillclimb.search_runner import StopRequested

    backend = FakeBackend()
    backend.queue(script=ok_script(0.6))
    commands = [[]]

    searcher, _ = make_searcher(
        task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=5)
    )
    searcher.drain_commands = lambda: commands.pop() if commands else []
    commands.insert(0, [ControlCommand(action="stop", source="cli")])  # second drain stops

    with pytest.raises(StopRequested):
        searcher.run()


def test_three_proposal_failures_park(task, config, tmp_path):
    from hillclimb.search_runner import ParkedSearch

    backend = FakeBackend()
    for _ in range(3):  # agent "succeeds" but returns the parent unchanged
        backend.queue(script=None)
    searcher, _ = make_searcher(
        task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=5)
    )
    with pytest.raises(ParkedSearch, match="3 consecutive proposal failures"):
        searcher.run()


def test_resume_reuses_ids_and_warm_cache(task, config, tmp_path):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6))
    searcher, search_dir = make_searcher(
        task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=1)
    )
    searcher.run()
    ids_before = sorted(searcher.journal.candidates)

    # resume: same dirs, fresh objects; driver proposes the SAME source again
    backend2 = FakeBackend()
    backend2.queue(script=ok_script(0.6))
    searcher2, _ = make_searcher(
        task, config, tmp_path, backend=backend2, driver=FakeGEPADriver(steps=1),
        search_dir=search_dir, journal=Journal(search_dir / "journal.jsonl"),
    )
    searcher2.run()
    ids_after = sorted(searcher2.journal.candidates)
    assert ids_after == ids_before  # no duplicate baseline/seed/candidate ids
    assert len([c for c in searcher2.journal.candidates.values() if c.operator == "seed"]) == 1


def test_resume_identity_mismatch_is_a_hard_error(task, config, tmp_path):
    backend = FakeBackend()
    searcher, search_dir = make_searcher(task, config, tmp_path, backend=backend)
    searcher.run()

    task.higher_is_better = False  # the same journal, a different optimization
    searcher2, _ = make_searcher(
        task, config, tmp_path, search_dir=search_dir,
        journal=Journal(search_dir / "journal.jsonl"),
    )
    with pytest.raises(RuntimeError, match="resume mismatch.*higher_is_better"):
        searcher2.run()


def test_dispatch_via_build_search_runner(task, config, tmp_path, monkeypatch):
    config.search.policy = "gepa"

    def explode(*a, **k):
        raise AssertionError("gepa dispatch must not touch the policy registry")

    monkeypatch.setattr("hillclimb.policies.get_policy", explode)
    search_dir = create_search_dir(tmp_path / "runs" / "r", "s")
    runner = build_search_runner(
        config=config,
        problem=task,
        journal=Journal(search_dir / "journal.jsonl"),
        backend=FakeBackend(),
        executor=executor_for(task),
        budget=BudgetManager(3600),
        search_dir=search_dir,
        log=lambda *_: None,
    )
    assert isinstance(runner, GEPASearcher)


def test_seedless_problem_fails_before_spend(task, config, tmp_path):
    config.search.policy = "gepa"
    search_dir = create_search_dir(tmp_path / "runs" / "r", "s")
    searcher = GEPASearcher(
        problem=task,
        config=config,
        journal=Journal(search_dir / "journal.jsonl"),
        backend=FakeBackend(),
        executor=executor_for(task),
        budget=BudgetManager(3600),
        search_dir=search_dir,
        log=lambda *_: None,
        seed_solution=None,
        driver=FakeGEPADriver(steps=0),
    )
    with pytest.raises(RuntimeError, match="--seed-from"):
        searcher.run()


def test_source_hash_normalizes_line_endings():
    assert source_hash("a\nb\n") == source_hash("a\r\nb")
    assert source_hash("a\nb") != source_hash("a\nc")
