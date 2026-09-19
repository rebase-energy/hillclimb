"""Per-operator routing: Router precedence, BackendPool identity, and the
routed path through _prepare/_execute_job."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.config import Config, RouteConfig
from tests.conftest import local_executor
from hillclimb.journal import Journal
from hillclimb.policy import Route
from hillclimb.routing import BackendPool, ResolvedRoute, Router
from hillclimb.search import GreedySearcher
from hillclimb.dirs import create_search_dir
from tests.conftest import ok_script


def test_router_falls_back_to_global_scalars():
    config = Config(backend="dummy", model="sonnet")
    assert Router(config).resolve("draft") == ResolvedRoute(
        backend="dummy", model="sonnet", backend_auth="subscription"
    )


def test_router_field_level_precedence():
    config = Config(
        backend="claude-code",
        model="sonnet",
        routing={
            "draft": RouteConfig(model="opus-4.8"),  # backend inherited
            "default": RouteConfig(backend="dummy"),
        },
    )
    router = Router(config)
    draft = router.resolve("draft")
    assert draft.model == "opus-4.8"
    assert draft.backend == "dummy"  # operator route says nothing -> default route
    improve = router.resolve("improve")
    assert improve.backend == "dummy"
    assert improve.model == "sonnet"  # default route says nothing -> global

    # a per-action override beats everything
    action_route = router.resolve("draft", Route(backend="claude-code", model="haiku"))
    assert action_route.backend == "claude-code"
    assert action_route.model == "haiku"


def test_router_sampling_precedence_and_explicit_empty_override():
    config = Config(
        backend="pi",
        routing={
            "default": RouteConfig(sampling={"temperature": 0.7}),
            "draft": RouteConfig(sampling={"temperature": 1.0, "top_p": 0.95}),
        },
    )
    router = Router(config)

    assert router.resolve("draft").sampling == {"temperature": 1.0, "top_p": 0.95}
    assert router.resolve("improve").sampling == {"temperature": 0.7}
    assert router.resolve("draft", Route(sampling={})).sampling == {}


def test_backend_pool_caches_and_wires_abort():
    abort = threading.Event()
    pool = BackendPool(abort=abort)
    first = pool.get("dummy", "subscription")
    assert pool.get("dummy", "subscription") is first  # stateful: one per search
    seeded = FakeBackend()
    pool.seed("fake", "subscription", seeded)
    assert pool.get("fake", "subscription") is seeded


def test_search_routes_operators_to_distinct_backends(task, config):
    """Draft routes to one backend/model, improve to another; the journal
    records which backend authored each candidate."""
    draft_backend, improve_backend = FakeBackend(), FakeBackend()
    draft_backend.name, improve_backend.name = "fake-draft", "fake-improve"
    for score in (0.5, 0.6, 0.7):
        draft_backend.queue(script=ok_script(score), notes="d\n")
    improve_backend.queue(script=ok_script(0.9), notes="i\n")

    config.backend = "fake-draft"
    config.model = "sonnet"
    config.routing = {"improve": RouteConfig(backend="fake-improve", model="opus-4.8")}
    pool = BackendPool()
    pool.seed("fake-draft", config.backend_auth, draft_backend)
    pool.seed("fake-improve", config.backend_auth, improve_backend)

    search_dir = create_search_dir(config.paths.runs_dir, "test-run")
    journal = Journal(search_dir / "journal.jsonl")
    searcher = GreedySearcher(
        problem=task,
        config=config,
        journal=journal,
        backend=draft_backend,
        executor=local_executor(),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir,
        max_candidates=5,  # baseline + 3 drafts + 1 improve
        log=lambda *_: None,
        router=Router(config),
        backends=pool,
    )
    best = searcher.run()

    assert best.val_score == 0.9
    assert [r.operator for r in draft_backend.requests] == ["draft", "draft", "draft"]
    assert [r.operator for r in improve_backend.requests] == ["improve"]
    assert all(r.model == "sonnet" for r in draft_backend.requests)
    assert improve_backend.requests[0].model == "opus-4.8"
    # routing is replay-visible: BackendInfo carries the authoring backend
    by_operator = {c.operator: c for c in journal.candidates.values()}
    assert by_operator["draft"].backend.name == "fake-draft"
    assert by_operator["improve"].backend.name == "fake-improve"


def test_no_routing_matches_default_backend_and_model(task, config):
    """Router present but empty routing block: identical requests to the
    router-less path."""
    backend = FakeBackend()
    backend.queue(script=ok_script(0.5), notes="d\n")
    config.backend = "fake"
    pool = BackendPool()
    pool.seed("fake", config.backend_auth, backend)
    search_dir = create_search_dir(config.paths.runs_dir, "test-run")
    searcher = GreedySearcher(
        problem=task,
        config=config,
        journal=Journal(search_dir / "journal.jsonl"),
        backend=backend,
        executor=local_executor(),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir,
        log=lambda *_: None,
        router=Router(config),
        backends=pool,
    )
    candidate = searcher.run_operator("draft", None)
    assert candidate.backend.name == "fake"
    assert backend.requests[0].model == config.model


def test_search_threads_sampling_and_resumes_pi_debug_session(task, config):
    backend = FakeBackend()
    backend.name = "pi"
    backend.queue(
        script='raise RuntimeError("boom")\n',  # a debug target must have failed
        result={"session_id": "pi-parent"},
    )
    backend.queue(script=ok_script(0.6), result={"session_id": "pi-child"})
    sampling = {"temperature": 0.9}
    config.backend = "pi"
    config.routing = {
        "default": RouteConfig(backend="pi", sampling=sampling),
    }
    pool = BackendPool()
    pool.seed("pi", config.backend_auth, backend)
    search_dir = create_search_dir(config.paths.runs_dir, "test-run")
    searcher = GreedySearcher(
        problem=task,
        config=config,
        journal=Journal(search_dir / "journal.jsonl"),
        backend=backend,
        executor=local_executor(),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir,
        log=lambda *_: None,
        router=Router(config),
        backends=pool,
    )

    parent = searcher.run_operator("draft", None)
    child = searcher.run_operator("debug", parent)

    assert backend.requests[0].sampling == sampling
    assert backend.requests[1].sampling == sampling
    assert backend.requests[1].resume_session_id == "pi-parent"
    assert child.backend.sampling == sampling
