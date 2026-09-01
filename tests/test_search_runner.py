"""Engine-level runner dispatch: build_search_runner constructs the runner
that search.policy names — GreedySearcher through the policy registry, full
engines through _ENGINES without touching get_policy."""

from __future__ import annotations

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.dirs import create_search_dir
from hillclimb.journal import Journal
from hillclimb.policies.greedy import GreedyPolicy
from hillclimb.search import GreedySearcher
from hillclimb.search_runner import _ENGINES, SearchRunner, build_search_runner
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
    return build_search_runner(**deps)


def test_greedy_dispatch_builds_greedy_searcher(task, config, tmp_path):
    config.search.policy = "greedy"
    runner = build(task, config, tmp_path, complexity_start=2)
    assert isinstance(runner, GreedySearcher)
    assert isinstance(runner.policy, GreedyPolicy)
    assert runner.complexity_start == 2
    assert isinstance(runner, SearchRunner)  # protocol is runtime-checkable


def test_openevolve_dispatch_goes_through_policy_registry(task, config, tmp_path):
    pytest.importorskip("openevolve")
    from hillclimb.policies.openevolve import OpenEvolvePolicy

    config.search.policy = "openevolve"
    runner = build(task, config, tmp_path)
    assert isinstance(runner, GreedySearcher)
    assert isinstance(runner.policy, OpenEvolvePolicy)


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
    runner = build(task, config, tmp_path)
    assert isinstance(runner, SearchRunner)
    assert seen["problem"] is task
    assert seen["config"] is config
    assert "journal" in seen and "budget" in seen and "search_dir" in seen


def test_exceptions_are_the_same_objects_via_both_homes():
    from hillclimb import search, search_runner

    assert search.ParkedSearch is search_runner.ParkedSearch
    assert search.StopRequested is search_runner.StopRequested
