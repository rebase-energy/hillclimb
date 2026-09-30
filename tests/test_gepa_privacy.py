"""The holdout privacy boundary: gepa's prompts, feedback, and state never
contain holdout values, and holdout runs only after the optimizer is done.

Holdout is the host's: the searcher never holds a scorer. The host hands it
an evaluator whose holdout timing is `after` (the gepa registry entry) and
scores the top-k hidden splits itself once run() has returned
(`api._finish_holdout`); this file drives that host path directly."""

from __future__ import annotations

from pathlib import Path

from hillclimb.api import _finish_holdout
from hillclimb.agents.fake import FakeAgent
from hillclimb.harness.dirs import create_search_dir
from hillclimb.harness.evaluation import CandidateEvaluator
from hillclimb.harness.journal import Journal
from hillclimb.harness.glue import holdout_timing
from tests.conftest import executor_for, ok_script
from tests.gepa_fakes import FakeGEPADriver, make_gepa
from tests.factories import name_climber

SENTINEL_SCORE = 77.777
SENTINEL_TEXT = "HOLDOUT-SENTINEL-9f3a"


class SentinelHoldout:
    """Returns an unmistakable value and records when it was called relative
    to the driver's rounds."""

    def __init__(self):
        self.calls: list[str] = []
        self.driver_done = False

    def score(self, candidate_dir: Path, trial=None):
        assert self.driver_done, "holdout ran while the optimizer was still live"
        self.calls.append(candidate_dir.name)
        return SENTINEL_SCORE, None, 0.1


class DoneMarkingDriver(FakeGEPADriver):
    def __init__(self, holdout: SentinelHoldout, steps: int = 2):
        super().__init__(steps=steps)
        self.holdout = holdout

    def run(self, **kwargs):
        try:
            return super().run(**kwargs)
        finally:
            self.holdout.driver_done = True


def run_search(task, config, tmp_path):
    name_climber(config, "gepa")
    config.holdout.top_k = 2
    holdout = SentinelHoldout()
    driver = DoneMarkingDriver(holdout, steps=2)
    agent = FakeAgent()
    agent.queue(script=ok_script(0.6))
    agent.queue(script=ok_script(0.7))
    search_dir = create_search_dir(tmp_path / "runs" / "r", "s")
    journal = Journal(search_dir / "journal.jsonl")
    # exactly what execute_search builds for gepa: the climber asks for `after`
    assert holdout_timing(config) == "after"
    evaluator = CandidateEvaluator(
        executor=executor_for(task), problem=task, config=config,
        holdout_scorer=holdout, holdout_timing="after", journal=journal,
    )
    searcher = make_gepa(
        task, config, tmp_path, agent=agent, driver=driver,
        search_dir=search_dir, journal=journal, evaluator=evaluator,
    )
    selected = searcher.run()
    assert not holdout.calls  # the loop returned without a single holdout call
    selected = _finish_holdout(
        config, task, search_dir, journal, evaluator, selected, lambda *_: None
    )
    return searcher, search_dir, holdout, driver, selected


def test_holdout_runs_after_optimization_and_scores_top_k(task, config, tmp_path):
    searcher, _, holdout, _, selected = run_search(task, config, tmp_path)
    assert holdout.driver_done  # SentinelHoldout asserts ordering on every call
    assert len(holdout.calls) == 2  # top_k
    with_holdout = [
        c for c in searcher.journal.candidates.values() if c.holdout_score == SENTINEL_SCORE
    ]
    scores = sorted(c.val_score for c in with_holdout)
    assert scores == [0.6, 0.7]  # the top-2 by validation
    assert selected is not None and selected.holdout_score == SENTINEL_SCORE


def test_nothing_gepa_visible_carries_the_sentinel(task, config, tmp_path):
    _, search_dir, _, driver, _ = run_search(task, config, tmp_path)
    # every file gepa or its agents can see: the loop's state + identity, and
    # the candidate dirs its agents worked in (prompt, feedback, solution)
    offenders = []
    visible = [*(search_dir / "loop").rglob("*"), *(search_dir / "candidates").glob("*/*")]
    assert any(p.name == "feedback.json" for p in visible) and any(p.name == "prompt.md" for p in visible)
    for path in visible:
        if not path.is_file() or path.is_symlink():
            continue
        text = path.read_text(errors="replace")
        if SENTINEL_TEXT in text or str(SENTINEL_SCORE) in text or "holdout_score" in text:
            offenders.append(str(path))
    assert not offenders, offenders
    # and the reflective datasets handed to the proposer
    for dataset in driver.reflective:
        text = str(dataset)
        assert str(SENTINEL_SCORE) not in text
        assert "holdout" not in text.lower()


def test_gepa_state_is_frozen_after_holdout(task, config, tmp_path):
    _, search_dir, _, _, _ = run_search(task, config, tmp_path)
    state = search_dir / "loop" / "state" / "gepa_state.bin"
    assert state.read_bytes() == b"fake-checkpoint"  # untouched by finalization
