"""OpenEvolve as a Policy: its MAP-Elites database picks parents and
inspirations, hillclimb's harness does the rest. Needs the `openevolve` extra."""

from __future__ import annotations

from tests.factories import trial as mk_trial

import random
from pathlib import Path

import pytest

pytest.importorskip("openevolve")

from hillclimb.agents.fake import FakeAgent
from hillclimb.harness.budget import BudgetManager
from hillclimb.harness.candidate import Candidate
from hillclimb.harness.dirs import create_search_dir
from hillclimb.harness.journal import Journal
from hillclimb.modules.policies import get_policy
from hillclimb.modules.policies.openevolve import OpenEvolvePolicy
from tests.harness_factory import SearchRig
from tests.conftest import local_executor, ok_script
from tests.test_policy import make_view

PARAMS = {"num_islands": 2, "num_inspirations": 2, "feature_dimensions": ["complexity", "score"]}


def scored(journal: Journal, tmp_path: Path, cid: str, op: str, score: float,
           parent_id: str | None = None, code: str = "x = 1\n", metrics: dict | None = None,
           climber_meta: dict | None = None) -> Candidate:
    d = tmp_path / "cands" / cid
    d.mkdir(parents=True, exist_ok=True)
    (d / "solution.py").write_text(code)
    cand = Candidate(
        candidate_id=cid, operator=op, status="passing", parent_id=parent_id,
        candidate_dir=str(d), climber_meta=climber_meta or {},
        trials=[mk_trial(val_score=score, submission_ok=True, metrics=metrics or {})],
    )
    journal.candidate_result(cand)
    return cand


def replayed(journal: Journal, config, params=PARAMS) -> tuple[OpenEvolvePolicy, object]:
    policy = get_policy("openevolve", params)
    view = make_view(journal, config)
    for cand in journal.candidates.values():
        policy.observe(view, cand)
    return policy, view


def test_registry_and_params_reach_openevolve(config):
    policy = get_policy("openevolve", PARAMS)
    assert policy.name == "openevolve"
    assert policy.db_config.num_islands == 2
    assert policy.feature_dimensions == ["complexity", "score"]
    assert policy.params == PARAMS  # persisted verbatim for resume


def test_drafts_until_population_seeded_then_evolves(config, tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    policy, view = replayed(journal, config, {**PARAMS, "num_drafts": 2})
    first = policy.propose(view)
    assert first.operator == "draft" and "island" in first.climber_meta

    scored(journal, tmp_path, "c000", "baseline", 0.1, code="pass\n")
    scored(journal, tmp_path, "c001", "draft", 0.5, code="a = 1\n" * 10)
    scored(journal, tmp_path, "c002", "draft", 0.7, code="b = 2\n" * 30)
    policy, view = replayed(journal, config, {**PARAMS, "num_drafts": 1})
    assert set(policy.db.programs) == {"c000", "c001", "c002"}

    action = policy.propose(view)
    assert action.operator == "improve"
    assert action.target_id in {"c000", "c001", "c002"}
    assert action.target_id not in action.inspiration_ids
    assert set(action.inspiration_ids) <= {"c000", "c001", "c002"}
    assert action.climber_meta["island"] in (0, 1)
    assert len(action.climber_meta["cell"]) == 2
    assert "MAP-Elites" in action.extra_prompt_context
    for i, _ in enumerate(action.inspiration_ids, 1):
        assert f"candidate_{i}.py" in action.extra_prompt_context


def test_proposals_are_replay_deterministic_and_rng_isolated(config, tmp_path):
    """Two fresh processes replaying the same journal must agree, and the
    policy must not disturb the harness's global RNG stream."""
    journal = Journal(tmp_path / "j.jsonl")
    scored(journal, tmp_path, "c000", "baseline", 0.1)
    for i in range(1, 9):
        scored(journal, tmp_path, f"c00{i}", "draft" if i < 3 else "improve",
               0.2 + 0.05 * i, parent_id=None if i < 3 else "c001", code=f"v = {i}\n" * i)
    random.seed(7)
    before = random.random()
    random.seed(7)
    p1, v1 = replayed(Journal(tmp_path / "j.jsonl"), config)
    a1 = p1.propose(v1)
    after = random.random()
    assert after == before  # global RNG state untouched by the policy
    p2, v2 = replayed(Journal(tmp_path / "j.jsonl"), config)
    a2 = p2.propose(v2)
    assert (a1.target_id, a1.inspiration_ids, a1.climber_meta) == (
        a2.target_id, a2.inspiration_ids, a2.climber_meta
    )


def test_custom_feature_dimension_reads_trial_metrics(config, tmp_path):
    params = {**PARAMS, "feature_dimensions": ["runtime_s", "score"]}
    journal = Journal(tmp_path / "j.jsonl")
    scored(journal, tmp_path, "c001", "draft", 0.5, metrics={"runtime_s": 1.0})
    policy, view = replayed(journal, config, params)
    assert policy.db.programs["c001"].metrics == {"combined_score": 0.5, "runtime_s": 1.0}

    journal2 = Journal(tmp_path / "j2.jsonl")
    scored(journal2, tmp_path, "c001", "draft", 0.5)  # verifier wrote no runtime_s
    with pytest.raises(ValueError, match="runtime_s"):
        replayed(journal2, config, params)


def test_lower_is_better_flips_fitness(config, tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    scored(journal, tmp_path, "c001", "draft", 2.0)
    scored(journal, tmp_path, "c002", "draft", 1.0)
    policy = get_policy("openevolve", PARAMS)
    view = make_view(journal, config, higher_is_better=False)
    for cand in journal.candidates.values():
        policy.observe(view, cand)
    assert policy.db.get_best_program().id == "c002"


def test_buggy_and_code_less_floor_are_not_programs(config, tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(Candidate(candidate_id="c000", operator="baseline", status="passing",
                                       trials=[mk_trial(val_score=0.3, submission_ok=True)]))
    journal.candidate_result(Candidate(candidate_id="c001", operator="draft", status="buggy",
                                       candidate_dir=str(tmp_path)))
    policy, view = replayed(journal, config, {**PARAMS, "num_drafts": 1})
    assert policy.db.programs == {}
    assert policy.propose(view).operator == "debug"  # hillclimb's debug rule survives


def test_openevolve_policy_drives_search_end_to_end(task, config):
    config.climber.ref = "openevolve"
    config.climber.params = {**PARAMS, "num_drafts": 2}
    agent = FakeAgent()
    agent.queue(script=ok_script(0.6), notes="one\n")
    agent.queue(script=ok_script(0.7), notes="two\n")
    agent.queue(script=ok_script(0.8), notes="three\n")
    search_dir = create_search_dir(config.paths.runs_dir, "test-run")
    policy = get_policy("openevolve", config.climber.params)
    searcher = SearchRig(
        problem=task, config=config, journal=Journal(search_dir / "journal.jsonl"),
        agent=agent, executor=local_executor(), budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir, max_candidates=4, log=lambda *_: None, policy=policy,
    )
    best = searcher.run()
    assert best.val_score == 0.8
    assert [r.operator for r in agent.requests] == ["draft", "draft", "improve"]
    evolved = searcher.journal.get("c003")
    assert evolved.operator == "improve" and evolved.parent_id in {"c001", "c002"}
    assert "island" in evolved.climber_meta and "cell" in evolved.climber_meta
    d = Path(evolved.candidate_dir)
    assert "MAP-Elites" in (d / "prompt.md").read_text()
    for i, _ in enumerate(evolved.climber_meta["inspirations"], 1):
        assert (d / f"candidate_{i}.py").exists()
    # MAP-Elites keeps the cell winner: c003 displaced c002 (same code length,
    # better score) rather than piling up alongside it
    assert policy.db.get_best_program().id == "c003"
    assert "c002" not in policy.db.programs


# --- live vs resume -----------------------------------------------------------
#
# A resumed search rebuilds the database by replaying the journal through
# `PolicyLoop.catch_up`; it must end up with the database the live run had,
# or the two would propose differently from the same journal.


def _catch_up(journal: Journal, config, params=PARAMS):
    """A fresh policy brought up to date the way a resumed search does it."""
    from types import SimpleNamespace

    from hillclimb.harness.loop import PolicyLoop

    policy = get_policy("openevolve", params)
    PolicyLoop(policy).catch_up(SimpleNamespace(view=lambda: make_view(journal, config)))
    return policy


def _observe_live(policy, journal: Journal, config, cid: str) -> None:
    """What `PolicyLoop.observe` does when a result lands: a fresh view."""
    view = make_view(journal, config)
    policy.observe(view, view.journal.candidates[cid])


def _pending(journal: Journal, cid: str, op: str, parent_id: str | None = None) -> None:
    journal.candidate_created(
        Candidate(candidate_id=cid, operator=op, status="pending", parent_id=parent_id)
    )


def _db_state(policy) -> dict:
    db = policy.db
    return {
        "iteration_found": {pid: p.iteration_found for pid, p in sorted(db.programs.items())},
        "fitness": {pid: p.metrics["combined_score"] for pid, p in sorted(db.programs.items())},
        "islands": [sorted(island) for island in db.islands],
        "cells": [dict(sorted(cells.items())) for cells in db.island_feature_maps],
    }


def _next(policy, journal: Journal, config):
    action = policy.propose(make_view(journal, config))
    return action.operator, action.target_id, action.inspiration_ids, action.climber_meta


LANDED = [
    ("c000", "baseline", 0.10, None, "pass\n"),
    ("c001", "draft", 0.50, None, "a = 1\n" * 10),
    ("c002", "draft", 0.70, None, "b = 2\n" * 30),
    ("c003", "improve", 0.72, "c002", "b = 3\n" * 45),
    ("c004", "improve", 0.64, "c001", "a = 4\n" * 70),
]


def test_resumed_database_is_the_live_one(config, tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    live = get_policy("openevolve", PARAMS)
    for cid, op, score, parent, code in LANDED:
        scored(journal, tmp_path, cid, op, score, parent_id=parent, code=code)
        _observe_live(live, journal, config, cid)
    resumed = _catch_up(Journal(tmp_path / "j.jsonl"), config)
    assert _db_state(resumed) == _db_state(live)
    assert _next(resumed, journal, config) == _next(live, journal, config)


def test_resumed_database_ignores_the_order_results_landed_in(config, tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    live = get_policy("openevolve", PARAMS)
    _pending(journal, "c001", "draft")
    _pending(journal, "c002", "draft")
    for cid, score, code in (("c002", 0.7, "b = 2\n" * 30), ("c001", 0.5, "a = 1\n" * 10)):
        scored(journal, tmp_path, cid, "draft", score, code=code)  # c002 lands first
        _observe_live(live, journal, config, cid)
    resumed = _catch_up(Journal(tmp_path / "j.jsonl"), config)
    assert _db_state(resumed) == _db_state(live)


def test_resumed_database_matches_after_a_tune_trial(config, tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    live = get_policy("openevolve", PARAMS)
    for cid, score in (("c001", 0.5), ("c002", 0.6)):
        scored(journal, tmp_path, cid, "draft", score, code=f"v = '{cid}'\n" * 10)
        _observe_live(live, journal, config, cid)
    scored(journal, tmp_path, "c001", "draft", 0.9, code="v = 'c001'\n" * 10)  # its best trial moved
    _observe_live(live, journal, config, "c001")
    resumed = _catch_up(Journal(tmp_path / "j.jsonl"), config)
    assert _db_state(resumed) == _db_state(live)
