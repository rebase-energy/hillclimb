"""The holdout privacy boundary: gepa's prompts, feedback, and state never
contain holdout values, and holdout runs only after the optimizer is done."""

from __future__ import annotations

from pathlib import Path

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.dirs import create_search_dir
from hillclimb.integrations.gepa.searcher import GEPASearcher
from hillclimb.journal import Journal
from tests.conftest import executor_for, ok_script
from tests.gepa_fakes import FakeGEPADriver

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
    config.search.policy = "gepa"
    config.holdout.top_k = 2
    holdout = SentinelHoldout()
    driver = DoneMarkingDriver(holdout, steps=2)
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6))
    backend.queue(script=ok_script(0.7))
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
        holdout_scorer=holdout,
        seed_solution=seed,
        driver=driver,
    )
    selected = searcher.run()
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
    # every file gepa or its agents can see: proposal dirs, state, identity
    gepa_root = search_dir / "gepa"
    offenders = []
    for path in gepa_root.rglob("*"):
        if not path.is_file():
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
    state = search_dir / "gepa" / "state" / "gepa_state.bin"
    assert state.read_bytes() == b"fake-checkpoint"  # untouched by finalization
