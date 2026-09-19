"""GepaLoop end to end with the fake driver: gepa drives the iteration, the
harness does everything that costs or counts — canonical candidates, lineage,
fitness, caching, control, resume. No gepa dependency."""

from __future__ import annotations

import json

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.candidate import source_hash
from hillclimb.control import ControlCommand
from hillclimb.integrations.gepa.loop import GepaLoop
from hillclimb.integrations.gepa.operator import OPERATOR_NAME
from hillclimb.integrations.gepa.proposer import COMPONENT, ProposerError
from hillclimb.journal import Journal
from hillclimb.loop import PolicyLoop
from hillclimb.search_strategy import ParkedSearch, StopRequested, build_loop, holdout_timing
from tests.conftest import ok_script
from tests.gepa_fakes import FakeGEPADriver, make_gepa


def reflects(journal):
    return sorted(
        (c for c in journal.candidates.values() if c.operator == OPERATOR_NAME),
        key=lambda c: c.candidate_id,
    )


def test_seed_and_improvements_become_canonical_candidates(task, config, tmp_path):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), result={"cost_usd": 0.1, "model_id": "m-1"})
    backend.queue(script=ok_script(0.7), result={"cost_usd": 0.2, "model_id": "m-1"})
    search = make_gepa(task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=2))

    selected = search.run()

    journal = search.journal
    by_op = {c.operator: c for c in journal.candidates.values()}
    assert "baseline" in by_op and "seed" in by_op
    improves = reflects(journal)
    assert [c.val_score for c in improves] == [0.6, 0.7]
    # lineage is journaled at `created`: seed -> first mutation -> second
    assert improves[0].parent_id == by_op["seed"].candidate_id
    assert improves[1].parent_id == improves[0].candidate_id
    for c in improves:
        assert c.role == "refine" and c.policy_meta["optimizer"] == "gepa"
        assert c.solution_sha256 == source_hash(search.harness.source(c.candidate_id))
        assert c.backend.model_id == "m-1"
    assert selected is not None and selected.val_score == 0.7 and improves[1].is_best
    # every request went out under the operator's own name (routing keys on it)
    assert [r.operator for r in backend.requests] == [OPERATOR_NAME, OPERATOR_NAME]
    # the agent worked in an ordinary candidate dir, feedback beside the parent's solution
    first = improves[0].candidate_dir
    assert json.loads(open(f"{first}/feedback.json").read())[COMPONENT][0]["val_score"] == 0.5
    assert (search.search_dir / "loop" / "state" / "gepa_state.bin").exists()
    assert (search.search_dir / "loop" / "identity.json").exists()
    assert search.harness.total_cost_usd() == pytest.approx(0.3)


def test_the_seed_is_scored_once_by_the_harness(task, config, tmp_path):
    search = make_gepa(task, config, tmp_path, driver=FakeGEPADriver(steps=0))
    search.run()
    seeds = [c for c in search.journal.candidates.values() if c.operator == "seed"]
    assert len(seeds) == 1
    assert not [c for c in search.journal.candidates.values() if c.operator == "inject"]


def test_lower_is_better_fitness_is_negated(task, config, tmp_path):
    task.higher_is_better = False
    backend = FakeBackend()
    backend.queue(script=ok_script(0.4))
    search = make_gepa(task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=1))
    selected = search.run()
    assert selected.val_score == 0.4  # raw journal direction, never negated
    improve = reflects(search.journal)[0]
    assert improve.is_best
    scoring = search.loop.scoring
    assert scoring.fitness(scoring.results[improve.solution_sha256]) == -0.4  # transformed at the boundary


def test_duplicate_source_reuses_the_candidate(task, config, tmp_path):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6))
    search = make_gepa(task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=1))
    search.run()
    n_before = len(search.journal.candidates)
    seed_source = (tmp_path / "seed_solution.py").read_text()
    result_a = search.loop.scoring.eval_source(seed_source)
    result_b = search.loop.scoring.eval_source(ok_script(0.6))  # gepa re-evaluating its own proposal
    assert result_a.score == 0.5 and result_b.score == 0.6
    assert len(search.journal.candidates) == n_before  # cache hits: no new id, no verifier call


def test_a_text_gepa_did_not_propose_is_injected(task, config, tmp_path):
    search = make_gepa(task, config, tmp_path, driver=FakeGEPADriver(steps=0))
    search.harness.execute(search.loop)
    # the harness is still open (budget left): a merge-style text is scored agent-free
    result = search.loop.scoring.eval_source(ok_script(0.9))
    injected = [c for c in search.journal.candidates.values() if c.operator == "inject"]
    assert result.score == 0.9 and len(injected) == 1
    assert injected[0].policy_meta == {"optimizer": "gepa"} and not search.harness.backend.requests


def test_buggy_proposal_gets_failure_fitness_and_feedback(task, config, tmp_path):
    backend = FakeBackend()
    backend.queue(script="raise RuntimeError('broken proposal')\n")
    backend.queue(script=ok_script(0.9))
    driver = FakeGEPADriver(steps=2)
    search = make_gepa(task, config, tmp_path, backend=backend, driver=driver)
    search.run()
    journal, scoring = search.journal, search.loop.scoring
    buggy = [c for c in journal.candidates.values() if c.status == "buggy"]
    assert len(buggy) == 1
    assert scoring.fitness(scoring.results[buggy[0].solution_sha256]) <= -1.0e100
    # the next round's reflective dataset carried the failure's stderr
    last_feedback = driver.reflective[-1]["solution.py"][0]
    assert last_feedback["valid"] is False
    assert "broken proposal" in str(last_feedback["trials"])
    recovered = [c for c in journal.candidates.values() if c.val_score == 0.9]
    assert recovered and recovered[0].status == "passing"


def test_failure_fitness_dominates_all_valid_scores(task, config, tmp_path):
    task.higher_is_better = False  # valid fitnesses are negative
    config.climber.params = {"failure_fitness": -1.0}  # pathological config
    backend = FakeBackend()
    backend.queue(script="raise RuntimeError('boom')\n")
    search = make_gepa(
        task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=1), seed_score=5.0
    )
    search.run()
    scoring = search.loop.scoring
    buggy = next(c for c in search.journal.candidates.values() if c.status == "buggy")
    seed = next(c for c in search.journal.candidates.values() if c.operator == "seed")
    assert scoring.fitness(scoring.results[buggy.solution_sha256]) < scoring.fitness(
        scoring.results[seed.solution_sha256]
    )


def test_stop_command_ends_the_loop_and_spends_nothing_more(task, config, tmp_path):
    backend = FakeBackend()
    for score in (0.6, 0.7, 0.8):
        backend.queue(script=ok_script(score))
    commands = [[ControlCommand(action="stop", source="cli")], [], []]  # the third drain stops
    search = make_gepa(
        task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=5),
        drain_commands=lambda: commands.pop() if commands else [],
    )
    with pytest.raises(StopRequested):
        search.run()
    assert len(backend.requests) < 3  # gepa saw `should_stop` and the harness was closed anyway


def test_three_proposal_failures_park(task, config, tmp_path):
    backend = FakeBackend()
    for _ in range(3):  # the agent "succeeds" but hands the parent back unchanged
        backend.queue(script=None)
    search = make_gepa(task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=5))
    with pytest.raises(ParkedSearch, match="3 consecutive proposal failures"):
        search.run()
    # failed rounds are ordinary abandoned candidates: their cost is journaled, nothing was scored
    unchanged = [c for c in reflects(search.journal) if c.status == "abandoned"]
    assert len(unchanged) == 3 and all(not c.trials for c in unchanged)


def test_failed_agent_calls_keep_their_cost(task, config, tmp_path):
    backend = FakeBackend()
    backend.queue(script=None, result={"ok": False, "error_kind": "error", "error_message": "net down", "cost_usd": 0.4})
    backend.queue(script=ok_script(0.7), result={"cost_usd": 0.1})
    search = make_gepa(task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=2))
    search.run()
    assert search.harness.total_cost_usd() == pytest.approx(0.5)


def test_resume_reuses_ids_and_warm_cache(task, config, tmp_path):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6))
    search = make_gepa(task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=0))
    search.run()
    first = make_gepa(
        task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=1),
        search_dir=search.search_dir, journal=Journal(search.search_dir / "journal.jsonl"),
    )
    first.run()
    ids_before = sorted(first.journal.candidates)

    # resume: same dirs, fresh objects; nothing left to propose, everything cached
    resumed = make_gepa(
        task, config, tmp_path, backend=FakeBackend(), driver=FakeGEPADriver(steps=0),
        search_dir=search.search_dir, journal=Journal(search.search_dir / "journal.jsonl"),
    )
    resumed.run()
    assert sorted(resumed.journal.candidates) == ids_before  # no duplicate baseline/seed ids
    assert len([c for c in resumed.journal.candidates.values() if c.operator == "seed"]) == 1
    assert source_hash(ok_script(0.6)) in resumed.loop.scoring.results  # warm, no verifier call


def test_resume_refuses_a_modified_candidate_dir(task, config, tmp_path):
    search = make_gepa(task, config, tmp_path)
    search.run()
    seed = next(c for c in search.journal.candidates.values() if c.operator == "seed")
    open(f"{seed.candidate_dir}/solution.py", "a").write("\n# edited after scoring\n")
    resumed = make_gepa(
        task, config, tmp_path, search_dir=search.search_dir,
        journal=Journal(search.search_dir / "journal.jsonl"),
    )
    with pytest.raises(RuntimeError, match="resume mismatch.*candidate dir was modified"):
        resumed.run()


def test_resume_identity_mismatch_is_a_hard_error(task, config, tmp_path):
    search = make_gepa(task, config, tmp_path)
    search.run()
    task.higher_is_better = False  # the same journal, a different optimization
    resumed = make_gepa(
        task, config, tmp_path, search_dir=search.search_dir,
        journal=Journal(search.search_dir / "journal.jsonl"),
    )
    with pytest.raises(RuntimeError, match="resume mismatch.*higher_is_better"):
        resumed.run()


def test_gepa_is_a_loop_every_other_climber_a_policy(config):
    config.climber.ref = "gepa"

    assert isinstance(build_loop(config, log=lambda *_: None), GepaLoop)
    assert holdout_timing(config) == "after"  # the manifest asks; the user's holdout.timing cannot loosen it
    config.climber.ref = "greedy"
    assert isinstance(build_loop(config, complexity_start=2), PolicyLoop)
    assert holdout_timing(config) == "inline"
    config.holdout.timing = "after"
    assert holdout_timing(config) == "after"
    config.climber.ref = "nope"
    with pytest.raises(ValueError, match="Unknown climber: nope .bundled: gepa, greedy, openevolve"):
        build_loop(config)


def test_seedless_problem_fails_before_spend(task, config, tmp_path):
    search = make_gepa(task, config, tmp_path, seed_score=None)
    with pytest.raises(RuntimeError, match="--seed-from"):
        search.run()
    assert not search.harness.backend.requests


def test_a_problem_baseline_seeds_gepa_without_a_second_evaluation(task, config, tmp_path):
    """No --seed-from: the problem's own baseline solution is the seed — and
    the harness already scored it as the baseline candidate, so gepa roots
    its lineage there instead of paying for the same text twice."""
    task.baseline_text = ok_script(0.3)
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6))
    search = make_gepa(task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=1), seed_score=None)
    search.run()
    journal = search.journal
    assert not [c for c in journal.candidates.values() if c.operator in ("inject", "seed")]
    baseline = next(c for c in journal.candidates.values() if c.operator == "baseline")
    assert baseline.val_score == 0.3
    assert reflects(journal)[0].parent_id == baseline.candidate_id


def test_wrong_components_are_not_a_proposal(task, config, tmp_path):
    search = make_gepa(task, config, tmp_path)
    search.run()
    with pytest.raises(ProposerError, match="mutates only solution.py"):
        search.loop.propose({"solution.py": "x", "other.py": "y"}, {}, ["solution.py", "other.py"])


def test_the_evaluation_and_token_budgets_bind_gepa_too(task, config, tmp_path):
    config.budget.max_evaluations = 1
    backend = FakeBackend()
    for score in (0.6, 0.7, 0.8):
        backend.queue(script=ok_script(score))
    search = make_gepa(task, config, tmp_path, backend=backend, driver=FakeGEPADriver(steps=5))
    search.run()
    assert len(backend.requests) == 1 and search.harness.closed_reason == "evaluation budget spent"


def test_source_hash_normalizes_line_endings():
    assert source_hash("a\nb\n") == source_hash("a\r\nb")
    assert source_hash("a\nb") != source_hash("a\nc")
