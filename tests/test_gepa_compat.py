"""Compatibility pins against the installed gepa release: every signature
and behavior the integration relies on (verified against 0.1.4). Skipped
without the optional extra; no network, no model calls."""

from __future__ import annotations

import inspect

import pytest

gepa = pytest.importorskip("gepa")

from hillclimb.backends.fake import FakeBackend  # noqa: E402
from hillclimb.budget import BudgetManager  # noqa: E402
from hillclimb.dirs import create_search_dir  # noqa: E402
from hillclimb.integrations.gepa.driver import CoreOptimizeDriver  # noqa: E402
from hillclimb.integrations.gepa.searcher import GEPASearcher  # noqa: E402
from hillclimb.journal import Journal  # noqa: E402
from tests.conftest import executor_for, ok_script  # noqa: E402


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
    """GEPASearcher -> CoreOptimizeDriver -> real gepa.optimize, with the
    fake backend as the mutation agent and the real executor as the
    verifier. Deterministic, no network."""
    config.search.policy = "gepa"
    config.search.policy_params = {"max_metric_calls": 6, "seed": 0}
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6))
    backend.queue(script=ok_script(0.7))
    backend.queue(script=ok_script(0.8))
    search_dir = create_search_dir(tmp_path / "runs" / "r", "s")
    seed = tmp_path / "seed_solution.py"
    seed.write_text(ok_script(0.5))
    searcher = GEPASearcher(
        problem=task,
        config=config,
        journal=Journal(search_dir / "journal.jsonl"),
        backend=backend,
        executor=executor_for(task),
        budget=BudgetManager(3600),
        search_dir=search_dir,
        log=lambda *_: None,
        seed_solution=seed,
        driver=CoreOptimizeDriver(),
    )
    selected = searcher.run()

    assert selected is not None
    assert selected.val_score is not None and selected.val_score > 0.5
    improves = [c for c in searcher.journal.candidates.values() if c.operator == "improve"]
    assert improves, "real gepa never called the proposer"
    for candidate in improves:
        assert candidate.policy_meta["optimizer"] == "gepa"
        assert candidate.parent_id is not None
    # gepa checkpointed into the canonical state dir
    assert (search_dir / "gepa" / "state" / "gepa_state.bin").exists()
