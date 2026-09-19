"""UCB1 model routing: bandit math, reward mapping, Router pool precedence,
and journal-replay reconstruction (the resume contract)."""

from __future__ import annotations

from tests.factories import trial as mk_trial

import sys
from pathlib import Path

from hillclimb.backends.fake import FakeBackend
from hillclimb.bandit import UCB1, OperatorBandits, candidate_reward
from hillclimb.budget import BudgetManager
from hillclimb.candidate import BackendInfo, Candidate
from hillclimb.config import Config, RouteConfig
from tests.conftest import local_executor
from hillclimb.journal import Journal
from hillclimb.routing import Router
from tests.harness_factory import SearchRig
from hillclimb.dirs import create_search_dir
from tests.conftest import ok_script


def cand(cid, operator="improve", parent=None, status="passing", val=None, model="m1", best=False):
    return Candidate(
        candidate_id=cid,
        parent_id=parent,
        operator=operator,
        status=status,
        backend=BackendInfo(model=model),
        trials=[mk_trial(val_score=val)] if val is not None else [],
        is_best=best,
    )


# --- UCB1 mechanics ---


def test_ucb1_plays_unpulled_arms_first_then_exploits():
    bandit = UCB1(("a", "b"), exploration=0.0)
    assert bandit.select() == "a"
    bandit.update("a", 0.0)
    assert bandit.select() == "b"
    bandit.update("b", 1.0)
    assert bandit.select() == "b"  # exploration=0: pure exploitation


def test_ucb1_exploration_revisits_underplayed_arm():
    bandit = UCB1(("a", "b"), exploration=2.0)
    bandit.update("a", 1.0)
    for _ in range(20):
        bandit.update("b", 0.6)
    # a's bonus term dominates despite b's decent mean
    assert bandit.select() == "a"


def test_ucb1_ignores_unknown_arm():
    bandit = UCB1(("a",))
    bandit.update("gone-from-pool", 1.0)  # older journal, changed config
    assert bandit.pulls == {"a": 0}


# --- reward mapping (journal-derivable, scale-free) ---


def test_reward_improvement_over_parent():
    parent = cand("c001", val=0.6)
    assert candidate_reward(cand("c002", parent="c001", val=0.7), parent, True) == 1.0
    assert candidate_reward(cand("c002", parent="c001", val=0.5), parent, True) == 0.25
    # direction respected
    assert candidate_reward(cand("c002", parent="c001", val=0.5), parent, False) == 1.0


def test_reward_edges():
    assert candidate_reward(cand("c1", status="buggy"), None, False) == 0.0
    assert candidate_reward(cand("c1", status="abandoned"), None, False) is None
    assert candidate_reward(cand("c1", val=0.5, model=None), None, False) is None
    assert candidate_reward(cand("c1", val=0.5, best=True), None, True) == 1.0
    assert candidate_reward(cand("c1", val=0.5, best=False), None, True) == 0.25
    # debug that fixed a buggy (unscored) parent counts as best-or-ok
    buggy_parent = cand("c001", status="buggy")
    assert candidate_reward(cand("c2", parent="c001", val=0.4, best=True), buggy_parent, True) == 1.0


# --- Router pool precedence ---


def test_single_entry_pool_degrades_to_scalar():
    config = Config(routing={"draft": RouteConfig(models=["only"])})
    router = Router(config)
    assert router.bandits is None  # no 2+ pool anywhere -> no bandit
    assert router.resolve("draft").model == "only"


def test_operator_scalar_beats_default_pool():
    config = Config(
        model="global",
        routing={
            "improve": RouteConfig(model="opus"),
            "default": RouteConfig(models=["m1", "m2"]),
        },
    )
    router = Router(config)
    assert router.bandits is not None
    assert router.resolve("improve").model == "opus"
    assert router.resolve("draft").model in ("m1", "m2")
    # a scalar-governed operator never reaches the bandit
    assert router._pool_for("improve") is None
    assert router._pool_for("draft") == ("m1", "m2")


def test_router_observe_credits_pool_arm_only():
    config = Config(routing={"draft": RouteConfig(models=["m1", "m2"])})
    router = Router(config)
    router.observe("draft", "m1", 1.0)
    router.observe("draft", "not-in-pool", 1.0)
    router.observe("improve", "m1", 1.0)  # improve is not pool-governed
    snapshot = router.bandits.snapshot()
    assert snapshot == {"draft": {"m1": {"pulls": 1, "mean_reward": 1.0}, "m2": {"pulls": 0, "mean_reward": None}}}


# --- end-to-end through the searcher + replay reconstruction ---


def make_searcher(task, config, backend, search_dir=None, router=None):
    search_dir = search_dir or create_search_dir(config.paths.runs_dir, "test-run")
    journal = Journal(search_dir / "journal.jsonl")
    searcher = SearchRig(
        problem=task,
        config=config,
        journal=journal,
        backend=backend,
        executor=local_executor(),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir,
        log=lambda *_: None,
        router=router or Router(config),
    )
    return searcher, journal, search_dir


def test_pool_routes_models_and_journals_the_arm(task, config):
    config.routing = {"draft": RouteConfig(models=["m1", "m2"])}
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="a\n")
    backend.queue(script=ok_script(0.5), notes="b\n")
    searcher, journal, _ = make_searcher(task, config, backend)
    first = searcher.run_operator("draft", None)
    second = searcher.run_operator("draft", None)
    # unpulled-first: both arms get exercised before any exploitation
    assert [r.model for r in backend.requests] == ["m1", "m2"]
    assert first.backend.model == "m1" and second.backend.model == "m2"
    stats = searcher.router.bandits.snapshot()["draft"]
    assert stats["m1"] == {"pulls": 1, "mean_reward": 1.0}  # took the lead
    assert stats["m2"] == {"pulls": 1, "mean_reward": 0.25}  # ok, no gain


def test_bandit_state_rebuilds_from_journal_replay(task, config):
    """The resume contract: a fresh searcher over the same journal ends up
    with identical bandit statistics, with no extra persistence."""
    config.routing = {"draft": RouteConfig(models=["m1", "m2"])}
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="a\n")
    backend.queue(script=ok_script(0.5), notes="b\n")
    searcher, _, search_dir = make_searcher(task, config, backend)
    searcher.run_operator("draft", None)
    searcher.run_operator("draft", None)
    live = searcher.router.bandits.snapshot()

    resumed, _, _ = make_searcher(task, config, FakeBackend(), search_dir=search_dir)
    assert resumed.router.bandits.snapshot() == live


def test_no_pool_means_no_bandit_and_unchanged_routing(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="a\n")
    searcher, _, _ = make_searcher(task, config, backend)
    assert searcher.router.bandits is None
    candidate = searcher.run_operator("draft", None)
    assert backend.requests[0].model == config.model
    assert candidate.backend.model == config.model  # arm recorded regardless
