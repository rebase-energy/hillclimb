"""Public programmatic API: run_search creates the Run/Search layout and
returns a SearchOutcome (dummy backend, no agent spend)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hillclimb.api import run_search
from hillclimb.run import load_run_meta, load_search_meta


@pytest.mark.slow
def test_run_search_end_to_end(config):
    config.paths.problems_dir = Path("problems")  # repo problems (circle-packing)
    logs: list[str] = []

    outcome = run_search(
        "circle-packing",
        budget_s=10,
        name="api-test",
        config=config,
        backend="dummy",
        holdout=False,
        log=logs.append,
    )

    assert outcome.state == "done"
    assert outcome.search_dir.name == "circle-packing"
    assert load_run_meta(outcome.run_dir) is not None
    assert load_search_meta(outcome.search_dir).budget_s == 10
    assert (outcome.search_dir / "journal.jsonl").exists()
    assert (outcome.search_dir / "best" / "submission.csv").exists()
    assert any("Search " in line for line in logs)
    assert outcome.selected is None or outcome.selected.val_score is not None
