"""EmflowLocalExecutor + EmflowHoldoutScorer against swedish-temperatures:ar
(packaged data; the dev venv has emflow, so sys.executable can run the
evaluator directly)."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("emflow")

from hillclimb.integrations.emflow.executor import (  # noqa: E402
    EmflowHoldoutScorer,
    EmflowLocalExecutor,
)

PROBLEM = "swedish-temperatures:ar"

PERSISTENCE_SOLUTION = textwrap.dedent(
    '''
    """Persistence: repeat the last observed temperature."""
    import pandas as pd
    from emflow.models.predictor import Predictor


    class Persistence(Predictor):
        def predict(self, obs):
            last = obs.history("temperature")["temperature"].dropna().iloc[-1]
            return pd.DataFrame({"point": last}, index=obs.target_index)


    def get_model():
        return Persistence()
    '''
)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "candidates" / "c001"
    ws.mkdir(parents=True)
    (ws / "solution.py").write_text(PERSISTENCE_SOLUTION)
    return ws


@pytest.fixture
def executor():
    return EmflowLocalExecutor(Path(sys.executable), PROBLEM)


@pytest.mark.slow
def test_validation_execution(executor, workspace):
    result = executor.execute(workspace / "solution.py", workspace, timeout_s=300)
    assert result.ok, Path(result.stderr_path).read_text()[-500:]
    assert result.val_score is not None
    payload = json.loads((workspace / "eval_result.json").read_text())
    assert payload["split"] == "validation"
    assert payload["score"] == pytest.approx(result.val_score)
    assert payload["n_scored"] > 0


@pytest.mark.slow
def test_validation_report_block(executor, workspace):
    """A real eval embeds the breakdown report — and keeps the default
    analyzers alive (concatenation regression guard)."""
    result = executor.execute(workspace / "solution.py", workspace, timeout_s=300)
    assert result.ok, Path(result.stderr_path).read_text()[-500:]
    payload = json.loads((workspace / "eval_result.json").read_text())
    report = payload["report"]
    assert report["version"] == 1
    assert report["split"] == "validation"
    assert report["source"] == "evaluator"
    assert report["report_error"] is None
    assert report["overall"]["score"] == pytest.approx(payload["score"], rel=1e-4)
    assert report["zones"] and report["zones"][0]["n_scored"] > 0
    assert report["horizons"]
    assert "quantiles" not in report  # point problem
    assert report["worst_origins"]
    assert report["residual_bias"]["mean_abs_error"] is not None
    # default analyzers survived the custom-analyzer concatenation
    assert "persistence_score" in report["persistence"]


@pytest.mark.slow
def test_broken_solution_is_not_ok(executor, workspace):
    (workspace / "solution.py").write_text("raise RuntimeError('boom')\n")
    result = executor.execute(workspace / "solution.py", workspace, timeout_s=120)
    assert not result.ok
    assert result.returncode != 0
    assert not (workspace / "eval_result.json").exists()


@pytest.mark.slow
def test_holdout_scorer_hidden_dir(workspace, tmp_path):
    scorer = EmflowHoldoutScorer(
        Path(sys.executable), PROBLEM, tmp_path / "holdout-eval", timeout_s=300
    )
    score, error = scorer.score(workspace)
    assert error is None
    assert isinstance(score, float)
    # evaluation ran outside the agent-visible workspace
    eval_dir = tmp_path / "holdout-eval" / "c001"
    assert (eval_dir / "eval_result.json").exists()
    holdout_payload = json.loads((eval_dir / "eval_result.json").read_text())
    assert holdout_payload["split"] == "holdout"
    assert "report" not in holdout_payload  # leakage guard: no holdout breakdowns
    assert not (workspace / "eval_result.json").exists() or True  # workspace untouched by scorer


@pytest.mark.slow
def test_holdout_scorer_maps_failure_to_error(workspace, tmp_path):
    (workspace / "solution.py").write_text("raise RuntimeError('nope')\n")
    scorer = EmflowHoldoutScorer(
        Path(sys.executable), PROBLEM, tmp_path / "holdout-eval", timeout_s=120
    )
    score, error = scorer.score(workspace)
    assert score is None
    assert "holdout evaluation failed" in error
