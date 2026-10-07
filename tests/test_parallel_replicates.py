"""`concurrency.parallel_replicates`: how many of a trial's seeded runs
execute at once — a count like the other parallelism levels, with the old
`replicate_mode: parallel | serial` reading as 0 | 1."""

from __future__ import annotations

from pathlib import Path

from hillclimb.agents.fake import FakeAgent
from hillclimb.config import Config, ConcurrencyConfig, parse_set_overrides
from tests.test_noise import TIMED_SOLUTION, make_searcher, overlaps


def test_replicates_at_once():
    assert ConcurrencyConfig().replicates_at_once(5) == 5  # 0 = all of them
    assert ConcurrencyConfig(parallel_replicates=1).replicates_at_once(5) == 1
    assert ConcurrencyConfig(parallel_replicates=2).replicates_at_once(5) == 2
    assert ConcurrencyConfig(parallel_replicates=8).replicates_at_once(5) == 5  # never more than there are


def test_the_old_mode_reads_as_the_count():
    assert Config.model_validate({"evaluation": {"replicate_mode": "serial"}}).concurrency.parallel_replicates == 1
    assert Config.model_validate({"evaluation": {"replicate_mode": "parallel"}}).concurrency.parallel_replicates == 0
    assert Config.model_validate({"search": {"replicate_mode": "serial"}}).concurrency.parallel_replicates == 1
    explicit = Config.model_validate({"evaluation": {"replicate_mode": "serial"}, "concurrency": {"parallel_replicates": 2}})
    assert explicit.concurrency.parallel_replicates == 2  # the new spelling wins
    assert "replicate_mode" not in type(Config().evaluation).model_fields
    for pair in ("evaluation.replicate_mode=serial", "search.trial_mode=serial", "concurrency.parallel_replicates=1"):
        config = Config()
        config.apply_overrides(parse_set_overrides([pair]))
        assert config.concurrency.parallel_replicates == 1, pair
    config = Config()
    config.apply_overrides(parse_set_overrides(["evaluation.replicate_mode=parallel"]))
    assert config.concurrency.parallel_replicates == 0


def test_replicates_run_in_batches_of_the_count(task, config):
    """Four replicates, two at once: r0 runs first on its own, then the rest
    overlap at most two at a time — more than serial, less than all."""
    config.evaluation.n_replicates = 4
    config.concurrency.parallel_replicates = 2
    agent = FakeAgent()
    agent.queue(script=TIMED_SOLUTION, notes="timed\n")
    searcher, _, _ = make_searcher(task, config, agent)
    node = searcher.run_operator("draft", None)
    spans = [
        tuple(float(part) for part in line.split(","))
        for line in (Path(node.candidate_dir) / "overlap.log").read_text().splitlines() if line.strip()
    ]
    assert len(spans) == 4 and len(node.trials[0].replicates) == 4
    at_once = max(sum(1 for s, e in spans if s <= t < e) for t, _ in spans)
    assert at_once == 2
    assert overlaps(Path(node.candidate_dir) / "overlap.log") >= 1
