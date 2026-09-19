"""Strategy dispatch: build_search_strategy constructs the strategy
that search.policy names — a PolicySearch (Harness + PolicyLoop) through the
policy registry, full engines through _ENGINES without touching get_policy."""

from __future__ import annotations

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.dirs import create_search_dir
from hillclimb.journal import Journal
from hillclimb.policies.greedy import GreedyPolicy
from hillclimb.harness import Harness
from hillclimb.loop import PolicyLoop
from hillclimb.search import PolicySearch
from hillclimb.search_strategy import _ENGINES, SearchStrategy, build_search_strategy
from tests.conftest import executor_for


def build(task, config, tmp_path, **overrides):
    search_dir = create_search_dir(tmp_path / "runs" / "r", "s")
    deps = dict(
        config=config,
        problem=task,
        journal=Journal(search_dir / "journal.jsonl"),
        backend=FakeBackend(),
        executor=executor_for(task),
        budget=BudgetManager(3600),
        search_dir=search_dir,
        log=lambda *_: None,
    )
    deps.update(overrides)
    return build_search_strategy(**deps)


def test_greedy_dispatch_builds_a_harness_driven_by_a_policy_loop(task, config, tmp_path):
    config.search.policy = "greedy"
    strategy = build(task, config, tmp_path, complexity_start=2)
    assert isinstance(strategy, PolicySearch)
    assert type(strategy.harness) is Harness and isinstance(strategy.loop, PolicyLoop)
    assert isinstance(strategy.loop.policy, GreedyPolicy)
    # the learned complexity offset shapes the policy; the harness knows no policy
    assert strategy.loop.policy.complexity_start == 2
    assert not hasattr(strategy.harness, "policy")
    assert isinstance(strategy, SearchStrategy)  # protocol is runtime-checkable


def test_openevolve_dispatch_goes_through_policy_registry(task, config, tmp_path):
    pytest.importorskip("openevolve")
    from hillclimb.policies.openevolve import OpenEvolvePolicy

    config.search.policy = "openevolve"
    strategy = build(task, config, tmp_path)
    assert isinstance(strategy, PolicySearch)
    assert isinstance(strategy.loop.policy, OpenEvolvePolicy)


def test_unknown_policy_keeps_the_registry_error(task, config, tmp_path):
    config.search.policy = "nope"
    with pytest.raises(ValueError, match="Unknown policy: nope"):
        build(task, config, tmp_path)


def test_registered_engine_bypasses_get_policy(task, config, tmp_path, monkeypatch):
    seen = {}

    def fake_engine(**deps):
        seen.update(deps)

        class Stub:
            def run(self):
                return None

            def total_cost_usd(self) -> float:
                return 0.0

        return Stub()

    def explode(*_a, **_k):
        raise AssertionError("engine dispatch must not touch the policy registry")

    monkeypatch.setitem(_ENGINES, "stub-engine", fake_engine)
    monkeypatch.setattr("hillclimb.policies.get_policy", explode)

    config.search.policy = "stub-engine"
    strategy = build(task, config, tmp_path)
    assert isinstance(strategy, SearchStrategy)
    assert seen["problem"] is task
    assert seen["config"] is config
    assert "journal" in seen and "budget" in seen and "search_dir" in seen


def test_exceptions_are_the_same_objects_via_both_homes():
    from hillclimb import search, search_strategy

    assert search.ParkedSearch is search_strategy.ParkedSearch
    assert search.StopRequested is search_strategy.StopRequested
