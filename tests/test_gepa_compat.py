"""Compatibility pins against the installed gepa release: every signature
and behavior the integration relies on (verified against 0.1.4). Skipped
without the optional extra; no network, no model calls."""

from __future__ import annotations

import inspect

import pytest

gepa = pytest.importorskip("gepa")

from hillclimb.agents.fake import FakeAgent  # noqa: E402
from tests.catalog_fixture import GEPA, gepa_module  # noqa: E402

CoreOptimizeDriver = gepa_module("driver").CoreOptimizeDriver
OPERATOR_NAME = gepa_module("operator").OPERATOR_NAME
from tests.conftest import ok_script  # noqa: E402
from tests.gepa_fakes import make_gepa  # noqa: E402
from tests.factories import name_climber


def test_optimize_signature_carries_every_relied_upon_kwarg():
    params = inspect.signature(gepa.optimize).parameters
    for kwarg in (
        "seed_candidate",
        "trainset",
        "valset",
        "adapter",
        "custom_candidate_proposer",
        "reflection_lm",
        "module_selector",
        "candidate_selection_strategy",
        "frontier_type",
        "use_merge",
        "max_metric_calls",
        "reflection_minibatch_size",
        "seed",
        "run_dir",
        "stop_callbacks",
        "skip_perfect_score",  # MANDATORY False for unbounded raw scores
        "cache_evaluation",
        "track_best_outputs",
        "display_progress_bar",
    ):
        assert kwarg in params, f"gepa.optimize lost kwarg {kwarg!r}"


def test_adapter_contract_shape():
    from gepa import EvaluationBatch, GEPAAdapter

    evaluate = inspect.signature(GEPAAdapter.evaluate).parameters
    assert list(evaluate) == ["self", "batch", "candidate", "capture_traces"]
    fields = EvaluationBatch.__annotations__
    assert "outputs" in fields and "scores" in fields and "trajectories" in fields


@pytest.mark.slow
def test_full_stack_with_real_gepa_loop(task, config, tmp_path):
    """GepaLoop -> CoreOptimizeDriver -> real gepa.optimize, with the
    fake agent as the mutation agent and the real executor as the
    verifier. Deterministic, no network."""
    name_climber(config, str(GEPA))
    config.climber.params = {"max_metric_calls": 6, "seed": 0}
    agent = FakeAgent()
    agent.queue(script=ok_script(0.6))
    agent.queue(script=ok_script(0.7))
    agent.queue(script=ok_script(0.8))
    search = make_gepa(task, config, tmp_path, agent=agent, driver=CoreOptimizeDriver())
    selected = search.run()

    assert selected is not None
    assert selected.val_score is not None and selected.val_score > 0.5
    improves = [c for c in search.journal.candidates.values() if c.operator == OPERATOR_NAME]
    assert improves, "real gepa never called the proposer"
    for candidate in improves:
        assert candidate.climber_meta["optimizer"] == "gepa"
        assert candidate.parent_id is not None
    # every text real gepa evaluated resolved to a journaled candidate: nothing was scored twice
    hashes = [c.solution_sha256 for c in search.journal.candidates.values() if c.solution_sha256]
    assert len(hashes) == len(set(hashes))
    # gepa checkpointed into the loop's own state dir
    assert (search.search_dir / "loop" / "state" / "gepa_state.bin").exists()
