"""Test doubles for the GEPA climber: a scripted driver so GepaLoop runs end
to end without the optional gepa dependency, and a helper that builds the
harness + loop pair the engine builds for `search.policy: gepa`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.dirs import create_search_dir
from hillclimb.harness import Harness
from hillclimb.integrations.gepa import build_gepa_loop
from hillclimb.integrations.gepa.loop import GepaLoop
from hillclimb.integrations.gepa.proposer import COMPONENT, ProposerError
from hillclimb.journal import Journal
from tests.conftest import executor_for, ok_script


class FakeGEPADriver:
    """A minimal serial reflective loop with the same contract as
    CoreOptimizeDriver: seed evaluated (a cache hit — the harness scored it),
    then `steps` rounds of proposer -> eval_source, honoring should_stop
    between rounds and swallowing ProposerError like real gepa swallows
    proposer exceptions."""

    def __init__(self, steps: int = 2):
        self.steps = steps
        self.reflective: list[dict] = []  # what the proposer was shown
        self.rounds = 0

    def run(self, *, seed_source, bridge, proposer, params, run_dir, instance_keys, should_stop):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "gepa_state.bin").write_bytes(b"fake-checkpoint")
        current = {COMPONENT: seed_source}
        last = bridge.eval_source(current[COMPONENT])
        for _ in range(self.steps):
            if should_stop():
                break
            self.rounds += 1
            dataset = {COMPONENT: [bridge.asi(last)]}
            self.reflective.append(dataset)
            try:
                current = proposer(current, dataset, [COMPONENT])
            except ProposerError:
                continue  # real gepa logs and moves on
            last = bridge.eval_source(current[COMPONENT])
        return {"best_candidate": current, "num_candidates": None}


@dataclass
class GepaSearch:
    harness: Harness
    loop: GepaLoop
    journal: Journal
    search_dir: Path

    def run(self):
        return self.harness.execute(self.loop)


def make_gepa(task, config, tmp_path, *, backend=None, driver=None, seed_score: float | None = 0.5,
              budget_s: int = 3600, journal=None, search_dir=None, **harness_kwargs) -> GepaSearch:
    config.search.policy = "gepa"
    search_dir = search_dir or create_search_dir(tmp_path / "runs" / "r", "s")
    seed = None
    if seed_score is not None:
        seed = tmp_path / "seed_solution.py"
        if not seed.exists():
            seed.write_text(ok_script(seed_score))
    journal = journal if journal is not None else Journal(search_dir / "journal.jsonl")
    harness = Harness(
        problem=task,
        config=config,
        journal=journal,
        backend=backend or FakeBackend(),
        executor=executor_for(task),
        budget=BudgetManager(budget_s),
        search_dir=search_dir,
        log=lambda *_: None,
        seed_solution=seed,
        **harness_kwargs,
    )
    loop = build_gepa_loop(config, driver=driver or FakeGEPADriver(steps=0), log=lambda *_: None)
    return GepaSearch(harness, loop, journal, search_dir)
