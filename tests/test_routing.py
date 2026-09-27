"""Per-operator routing: Router precedence, AgentPool identity, and the
routed path through _prepare/_execute_job."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from hillclimb.agents.fake import FakeAgent
from hillclimb.harness.budget import BudgetManager
from hillclimb.config import Config, RouteConfig
from tests.conftest import local_executor
from hillclimb.harness.journal import Journal
from hillclimb.modules.policies.base import Route
from hillclimb.harness.routing import AgentPool, ResolvedRoute, Router
from tests.harness_factory import SearchRig
from hillclimb.harness.dirs import create_search_dir
from tests.conftest import ok_script


def test_router_falls_back_to_global_scalars():
    config = Config(agent="dummy", model="sonnet")
    assert Router(config).resolve("draft") == ResolvedRoute(
        agent="dummy", model="sonnet", agent_auth="subscription"
    )


def test_router_field_level_precedence():
    config = Config(
        agent="claude-code",
        model="sonnet",
        routing={
            "draft": RouteConfig(model="opus-4.8"),  # agent inherited
            "default": RouteConfig(agent="dummy"),
        },
    )
    router = Router(config)
    draft = router.resolve("draft")
    assert draft.model == "opus-4.8"
    assert draft.agent == "dummy"  # operator route says nothing -> default route
    improve = router.resolve("improve")
    assert improve.agent == "dummy"
    assert improve.model == "sonnet"  # default route says nothing -> global

    # a per-action override beats everything
    action_route = router.resolve("draft", Route(agent="claude-code", model="haiku"))
    assert action_route.agent == "claude-code"
    assert action_route.model == "haiku"


def test_router_sampling_precedence_and_explicit_empty_override():
    config = Config(
        agent="pi",
        routing={
            "default": RouteConfig(sampling={"temperature": 0.7}),
            "draft": RouteConfig(sampling={"temperature": 1.0, "top_p": 0.95}),
        },
    )
    router = Router(config)

    assert router.resolve("draft").sampling == {"temperature": 1.0, "top_p": 0.95}
    assert router.resolve("improve").sampling == {"temperature": 0.7}
    assert router.resolve("draft", Route(sampling={})).sampling == {}


def test_agent_pool_caches_and_wires_abort():
    abort = threading.Event()
    pool = AgentPool(abort=abort)
    first = pool.get("dummy", "subscription")
    assert pool.get("dummy", "subscription") is first  # stateful: one per search
    seeded = FakeAgent()
    pool.seed("fake", "subscription", seeded)
    assert pool.get("fake", "subscription") is seeded


def test_search_routes_operators_to_distinct_agents(task, config):
    """Draft routes to one agent/model, improve to another; the journal
    records which agent authored each candidate."""
    draft_agent, improve_agent = FakeAgent(), FakeAgent()
    draft_agent.name, improve_agent.name = "fake-draft", "fake-improve"
    for score in (0.5, 0.6, 0.7):
        draft_agent.queue(script=ok_script(score), notes="d\n")
    improve_agent.queue(script=ok_script(0.9), notes="i\n")

    config.agent = "fake-draft"
    config.model = "sonnet"
    config.routing = {"improve": RouteConfig(agent="fake-improve", model="opus-4.8")}
    pool = AgentPool()
    pool.seed("fake-draft", config.agent_auth, draft_agent)
    pool.seed("fake-improve", config.agent_auth, improve_agent)

    search_dir = create_search_dir(config.paths.runs_dir, "test-run")
    journal = Journal(search_dir / "journal.jsonl")
    searcher = SearchRig(
        problem=task,
        config=config,
        journal=journal,
        agent=draft_agent,
        executor=local_executor(),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir,
        max_candidates=5,  # baseline + 3 drafts + 1 improve
        log=lambda *_: None,
        router=Router(config),
        agents=pool,
    )
    best = searcher.run()

    assert best.val_score == 0.9
    assert [r.operator for r in draft_agent.requests] == ["draft", "draft", "draft"]
    assert [r.operator for r in improve_agent.requests] == ["improve"]
    assert all(r.model == "sonnet" for r in draft_agent.requests)
    assert improve_agent.requests[0].model == "opus-4.8"
    # routing is replay-visible: AgentInfo carries the authoring agent
    by_operator = {c.operator: c for c in journal.candidates.values()}
    assert by_operator["draft"].agent.name == "fake-draft"
    assert by_operator["improve"].agent.name == "fake-improve"


def test_no_routing_matches_default_agent_and_model(task, config):
    """Router present but empty routing block: identical requests to the
    router-less path."""
    agent = FakeAgent()
    agent.queue(script=ok_script(0.5), notes="d\n")
    config.agent = "fake"
    pool = AgentPool()
    pool.seed("fake", config.agent_auth, agent)
    search_dir = create_search_dir(config.paths.runs_dir, "test-run")
    searcher = SearchRig(
        problem=task,
        config=config,
        journal=Journal(search_dir / "journal.jsonl"),
        agent=agent,
        executor=local_executor(),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir,
        log=lambda *_: None,
        router=Router(config),
        agents=pool,
    )
    candidate = searcher.run_operator("draft", None)
    assert candidate.agent.name == "fake"
    assert agent.requests[0].model == config.model


def test_search_threads_sampling_and_resumes_pi_debug_session(task, config):
    agent = FakeAgent()
    agent.name = "pi"
    agent.queue(
        script='raise RuntimeError("boom")\n',  # a debug target must have failed
        result={"session_id": "pi-parent"},
    )
    agent.queue(script=ok_script(0.6), result={"session_id": "pi-child"})
    sampling = {"temperature": 0.9}
    config.agent = "pi"
    config.routing = {
        "default": RouteConfig(agent="pi", sampling=sampling),
    }
    pool = AgentPool()
    pool.seed("pi", config.agent_auth, agent)
    search_dir = create_search_dir(config.paths.runs_dir, "test-run")
    searcher = SearchRig(
        problem=task,
        config=config,
        journal=Journal(search_dir / "journal.jsonl"),
        agent=agent,
        executor=local_executor(),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir,
        log=lambda *_: None,
        router=Router(config),
        agents=pool,
    )

    parent = searcher.run_operator("draft", None)
    child = searcher.run_operator("debug", parent)

    assert agent.requests[0].sampling == sampling
    assert agent.requests[1].sampling == sampling
    assert agent.requests[1].resume_session_id == "pi-parent"
    assert child.agent.sampling == sampling
