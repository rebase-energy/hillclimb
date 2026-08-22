"""End-to-end emflow search with the fake backend on swedish-temperatures:ar
(packaged data — no HF, no token, no agent spend)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("emflow")

from hillclimb.backends.fake import FakeBackend  # noqa: E402
from hillclimb.budget import BudgetManager  # noqa: E402
from hillclimb.executor import CommandExecutor, CommandHoldoutScorer  # noqa: E402
from hillclimb.journal import Journal  # noqa: E402
from hillclimb.problem import load_problem  # noqa: E402
from hillclimb.search import GreedySearcher  # noqa: E402

PROBLEM = "swedish-temperatures:ar"


def predictor_module(name: str, window: str) -> str:
    return textwrap.dedent(
        f'''
        """{name}: hour-of-day climatology over the trailing {window}."""
        import pandas as pd
        from emflow.models.predictor import Predictor


        class {name}(Predictor):
            def predict(self, obs):
                hist = obs.history("temperature", window="{window}")["temperature"].dropna()
                by_hour = hist.groupby(hist.index.hour).mean()
                vals = [by_hour.get(t.hour, hist.iloc[-1]) for t in obs.target_index]
                return pd.DataFrame({{"point": vals}}, index=obs.target_index)


        def get_model():
            return {name}()
        '''
    )


@pytest.mark.slow
def test_full_emflow_search(config, tmp_path):
    config.paths.runs_dir = tmp_path / "runs"
    config.budget.exec_timeout_s = 300
    config.search.num_drafts = 2
    config.ensemble.enabled = False
    spec = load_problem(f"emflow://{PROBLEM}", config)

    search_dir = config.paths.runs_dir / "e2e" / "searches" / spec.problem_id
    (search_dir / "candidates").mkdir(parents=True)
    (search_dir / "best").mkdir()

    backend = FakeBackend()
    backend.queue(script=predictor_module("Climatology7d", "7D"), notes="7-day climatology\n")
    backend.queue(script=predictor_module("Climatology30d", "30D"), notes="30-day climatology\n")
    backend.queue(script=predictor_module("Climatology60d", "60D"), notes="improve: 60-day window\n")

    python = Path(sys.executable)
    journal = Journal(search_dir / "journal.jsonl")
    searcher = GreedySearcher(
        problem=spec,
        config=config,
        journal=journal,
        backend=backend,
        executor=CommandExecutor(python, spec.verifier_cmd, spec.verifier_env),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir,
        max_candidates=4,
        log=lambda *_: None,
        holdout_scorer=CommandHoldoutScorer(
            python, spec.holdout_cmd, problem_dir=spec.problem_dir,
            data_dir=spec.data_dir, work_root=search_dir / "holdout-eval", timeout_s=300,
        ),
    )
    selected = searcher.run()

    # baseline: swedish-temperatures ships no baseline module -> placeholder
    baseline = journal.get("c000")
    assert baseline.operator == "baseline"
    assert baseline.trials == []

    # drafts + improve all scored on validation AND holdout via the evaluator
    scored = journal.scored_candidates()
    assert len(scored) == 3
    for candidate in scored:
        trial = candidate.last_trial
        assert trial.val_score is not None
        assert trial.holdout_score is not None
        assert trial.holdout_error is None

    # rank-blend selection ran over evaluator-produced scores
    assert selected is not None
    assert selected.is_selected
    assert (search_dir / "best" / "solution.py").exists()

    # holdout evaluation stayed outside agent-visible candidate dirs
    assert (search_dir / "holdout-eval").is_dir()
    for candidate in scored:
        assert not (Path(candidate.candidate_dir) / "holdout_predictions.csv").exists()

    # the improve prompt carried the emflow contract, not the CSV one
    improve_prompts = [r.prompt for r in backend.requests if r.operator == "improve"]
    assert improve_prompts and "get_model()" in improve_prompts[0]
