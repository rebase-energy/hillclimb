"""The agentic proposer: scratch-dir materialization, routing, prompt
content, and rejection of unusable agent output."""

from __future__ import annotations

import json

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.config import RouteConfig
from hillclimb.integrations.gepa.proposer import (
    COMPONENT,
    GEPAProposer,
    ProposalBridge,
    ProposerError,
    source_hash,
)
from hillclimb.routing import BackendPool, Router
from tests.conftest import ok_script

PARENT = ok_script(0.5)
DATASET = {COMPONENT: [{"valid": True, "val_score": 0.5, "advice": "improve it"}]}


def make_proposer(task, config, tmp_path, backend, **overrides):
    backends = BackendPool()
    backends.seed(config.backend, config.backend_auth, backend)
    # route "gepa" to the fake regardless of the global backend name
    config.routing["gepa"] = RouteConfig(backend=config.backend, model="test-model")
    search_dir = tmp_path / "runs" / "r" / "searches" / "s"
    search_dir.mkdir(parents=True, exist_ok=True)
    defaults = dict(
        problem=task,
        config=config,
        router=Router(config),
        backends=backends,
        budget=BudgetManager(3600),
        search_dir=search_dir,
        bridge=ProposalBridge(),
        checkpoint=lambda: None,
        log=lambda *_: None,
    )
    defaults.update(overrides)
    return GEPAProposer(**defaults), search_dir


def test_proposal_materializes_and_returns_edited_source(task, config, tmp_path):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.8), result={"cost_usd": 0.05})
    proposer, search_dir = make_proposer(task, config, tmp_path, backend)

    out = proposer({COMPONENT: PARENT}, DATASET, [COMPONENT])

    assert out[COMPONENT] == ok_script(0.8)
    scratch = search_dir / "gepa" / "proposals" / "p0001"
    request = backend.requests[0]
    assert request.candidate_dir == scratch
    assert request.operator == "improve"
    assert request.model == "test-model"  # routing.gepa honored
    # the scratch dir carries everything the agent needs
    assert (scratch / "data").is_symlink() and (scratch / "problem").is_symlink()
    assert json.loads((scratch / "feedback.json").read_text()) == DATASET
    prompt = (scratch / "prompt.md").read_text()
    assert "higher is better" in prompt
    assert "holdout" in prompt  # the explicit do-not-touch-holdout line
    assert "improve it" in prompt  # reflective feedback rendered
    # lineage handoff recorded for the evaluator
    record = proposer.bridge.pop(source_hash(ok_script(0.8)))
    assert record is not None
    assert record.parent_hash == source_hash(PARENT)
    assert record.backend.cost_usd == 0.05


@pytest.mark.parametrize(
    "script,match",
    [
        (None, "no solution.py"),  # agent wrote nothing... parent copy exists though
        ("", "empty"),
        (PARENT, "unchanged"),
    ],
)
def test_unusable_output_is_rejected(task, config, tmp_path, script, match):
    backend = FakeBackend()
    if script is None:
        # simulate the agent deleting its file: respond with files that
        # remove solution.py via an empty write, then unlink in a hook —
        # simplest honest case is the unchanged parent, so patch the dir
        backend.queue(script=PARENT)
    else:
        backend.queue(script=script)
    failures = []
    proposer, search_dir = make_proposer(
        task, config, tmp_path, backend, on_failure=failures.append
    )
    if script is None:
        # remove the solution after the backend "runs"
        original = backend.invoke

        def invoke_and_delete(request):
            result = original(request)
            (request.candidate_dir / COMPONENT).unlink()
            return result

        backend.invoke = invoke_and_delete
        match = "no solution.py"
    with pytest.raises(ProposerError, match=match):
        proposer({COMPONENT: PARENT}, DATASET, [COMPONENT])
    assert len(failures) == 1  # burn accounting hook fired


def test_wrong_components_rejected(task, config, tmp_path):
    proposer, _ = make_proposer(task, config, tmp_path, FakeBackend())
    with pytest.raises(ProposerError, match="mutates only"):
        proposer({COMPONENT: PARENT}, DATASET, ["other.py"])


def test_failed_agent_call_carries_result(task, config, tmp_path):
    backend = FakeBackend()
    backend.queue(result={"ok": False, "error_kind": "error", "error_message": "rate limited-ish",
                          "cost_usd": 0.02})
    failures = []
    proposer, _ = make_proposer(task, config, tmp_path, backend, on_failure=failures.append)
    with pytest.raises(ProposerError, match="agent call failed"):
        proposer({COMPONENT: PARENT}, DATASET, [COMPONENT])
    assert failures[0].result.cost_usd == 0.02


def test_scratch_dirs_skip_existing_on_resume(task, config, tmp_path):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.9))
    proposer, search_dir = make_proposer(task, config, tmp_path, backend)
    (search_dir / "gepa" / "proposals" / "p0001").mkdir(parents=True)  # prior run's dir
    proposer({COMPONENT: PARENT}, DATASET, [COMPONENT])
    assert backend.requests[0].candidate_dir.name == "p0002"
