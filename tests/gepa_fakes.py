"""Test doubles for the GEPA engine: a scripted driver so GEPASearcher runs
end to end without the optional gepa dependency."""

from __future__ import annotations

from hillclimb.integrations.gepa.proposer import COMPONENT, ProposerError


class FakeGEPADriver:
    """A minimal serial reflective loop with the same contract as
    CoreOptimizeDriver: seed evaluated (cache hit from the searcher's own
    seed pass), then `steps` rounds of proposer -> eval_source, honoring
    should_stop between rounds and swallowing ProposerError like real gepa
    swallows proposer exceptions."""

    def __init__(self, steps: int = 2):
        self.steps = steps
        self.reflective: list[dict] = []  # what the proposer was shown
        self.rounds = 0

    def run(self, *, seed_source, bridge, proposer, params, run_dir, instance_keys, should_stop):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "gepa_state.bin").write_bytes(b"fake-checkpoint")
        current = {COMPONENT: seed_source}
        last = bridge.eval_source(current[COMPONENT])  # cached: searcher scored the seed
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
