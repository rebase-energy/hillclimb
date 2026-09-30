"""Selectors: WHICH candidate a policy expands. `best` is what makes greedy
greedy; a block's `select:` swaps the exploration without touching the
schedule, and a selector is a module like any other — a file works."""

from __future__ import annotations

from pathlib import Path

import pytest

from hillclimb.agents.fake import FakeAgent
from hillclimb.climber import ClimberLoadError, resolve_climber
from hillclimb.harness.journal import Journal
from hillclimb.modules.policies.base import InflightRef
from hillclimb.modules.policies.greedy import Greedy
from hillclimb.modules.selectors import Selection, Selector, get_selector
from hillclimb.modules.selectors.best import Best
from tests.conftest import ok_script
from tests.harness_factory import make_harness
from tests.test_policy import add_candidate, make_view


def scored_journal(tmp_path: Path) -> Journal:
    journal = Journal(tmp_path / "j.jsonl")
    add_candidate(journal, "c000", "baseline", val_score=0.9)  # a declared floor: nothing to expand
    add_candidate(journal, "c001", "draft", val_score=0.5, solution="a\n", tmp_path=tmp_path)
    add_candidate(journal, "c002", "draft", val_score=0.7, solution="b\n", tmp_path=tmp_path)
    add_candidate(journal, "c003", "draft", status="buggy")
    return journal


def test_best_picks_the_best_candidate_nobody_is_expanding(config, tmp_path):
    journal = scored_journal(tmp_path)
    state = make_view(journal, config)
    best = get_selector("best")
    assert isinstance(best, Best) and best.name == "best"
    assert best.select(state) == Selection("c002")
    assert best.select(state, busy={"c002"}) == Selection("c001")  # spread over the top ones
    assert best.select(state, busy={"c001", "c002"}) == Selection("c002")  # all busy: the best gets another
    assert best.select(make_view(journal, config, higher_is_better=False)) == Selection("c001")
    assert best.select(make_view(Journal(tmp_path / "empty.jsonl"), config)) is None
    assert best.creation_meta(state) == {}


def test_greedy_expands_what_its_selector_picks(config, tmp_path):
    """The schedule is the policy's; the pick — parent, inspirations, prompt
    context, a note for the journal — is the selector's."""

    class SecondBest(Selector):
        name = "second-best"

        def select(self, state, *, busy=frozenset()):
            ranked = sorted(state.journal.scored_candidates(), key=lambda c: -c.val_score)
            return Selection(ranked[1].candidate_id, inspiration_ids=(ranked[0].candidate_id,),
                             prompt_context="Beat the leader.", meta={"why": "runner-up"})

        def creation_meta(self, state):
            return {"niche": len(state.journal.drafts())}

    journal = Journal(tmp_path / "j.jsonl")
    policy = Greedy({"num_drafts": 2, "tune_budget": 0}, selector=SecondBest())
    draft = policy.propose(make_view(journal, config))
    assert (draft.operator, draft.climber_meta) == ("draft", {"niche": 0})  # the selector tags new drafts
    add_candidate(journal, "c001", "draft", val_score=0.5, solution="a\n", tmp_path=tmp_path)
    add_candidate(journal, "c002", "draft", val_score=0.7, solution="b\n", tmp_path=tmp_path)
    action = policy.propose(make_view(journal, config))
    assert (action.operator, action.target_id, action.inspiration_ids) == ("improve", "c001", ("c002",))
    assert action.extra_prompt_context == "Beat the leader." and action.climber_meta == {"why": "runner-up"}
    # with the default selector the same journal expands the best, busy targets aside
    default = Greedy({"num_drafts": 2, "tune_budget": 0})
    assert default.propose(make_view(journal, config)).target_id == "c002"
    busy = (InflightRef(candidate_id="c003", operator="improve", parent_id="c002"),)
    assert default.propose(make_view(journal, config, inflight=busy)).target_id == "c001"


SELECTOR_PY = '''\
from hillclimb.sdk import Selection, Selector


class Oldest(Selector):
    """Expand the oldest scored candidate (a deliberately different exploration)."""
    DEFAULTS = {"skip": 0}

    def select(self, state, *, busy=frozenset()):
        scored = state.journal.scored_candidates()
        if len(scored) <= self.param("skip"):
            return None
        return Selection(scored[self.param("skip")].candidate_id, prompt_context="Oldest first.")
'''


def test_a_block_names_its_selector_like_any_module(task, config, tmp_path):
    (tmp_path / "oldest.py").write_text(SELECTOR_PY)
    block = {"policy": "greedy", "select": "oldest.py", "params": {"num_drafts": 2, "ensemble": False, "tune_budget": 0}}
    climber = resolve_climber(block, tmp_path)
    policy = climber.build_loop().policy
    assert type(policy.selector).__name__ == "Oldest" and policy.selector.name == "oldest"
    # the selector's file is part of what the climber IS, and its settings are `select_params`
    (tmp_path / "oldest.py").write_text(SELECTOR_PY + "# edited\n")
    assert resolve_climber(block, tmp_path).sha256 != climber.sha256
    assert resolve_climber({**block, "select_params": {"skip": 1}}, tmp_path).build_loop().policy.selector.param("skip") == 1

    agent = FakeAgent()
    for score in (0.5, 0.7, 0.6):
        agent.queue(script=ok_script(score), notes="x\n")
    config.budget.max_evaluations = 3
    harness, journal, _ = make_harness(task, config, agent, operators=climber.operator_set())
    harness.execute(resolve_climber(block, tmp_path).build_loop())
    third = journal.get("c003")
    assert (third.operator, third.parent_id) == ("improve", "c001")  # the OLDEST scored, not the best (c002)
    assert "Oldest first." in Path(third.candidate_dir, "prompt.md").read_text()


def test_a_selector_only_goes_where_a_policy_can_take_it(tmp_path):
    (tmp_path / "plain.py").write_text(
        "class Plain:\n    def __init__(self, params=None):\n        self.params = params or {}\n"
        "    def propose(self, view):\n        return None\n    def observe(self, view, candidate):\n        pass\n"
    )
    assert resolve_climber({"policy": "plain.py"}, tmp_path).build_loop().policy.params == {}  # no selector asked: fine
    with pytest.raises(ClimberLoadError, match="`select: best` — this policy takes no selector"):
        resolve_climber({"policy": "plain.py", "select": "best"}, tmp_path).build_loop()
    with pytest.raises(ClimberLoadError, match="a `loop:` does its own selection"):
        resolve_climber({"loop": "gepa", "select_params": {"k": 1}})
    with pytest.raises(ClimberLoadError, match="unknown selector 'nope' .available: best, map-elites"):
        resolve_climber({"select": "nope"}).build_loop()


def test_what_memory_learned_is_fixed_when_a_search_first_runs(task, config, tmp_path, monkeypatch):
    """A learned draft-complexity offset is a prior under the block's params.
    It is recorded when the search first runs, so a resume starts from the
    same one — not from whatever the knowledge says by then."""
    from hillclimb import api
    from hillclimb.harness.budget import BudgetManager
    from hillclimb.harness.run import RunMeta, load_search_meta

    seen = []
    real = api.build_loop

    def spy(config, *, priors=None, **kwargs):
        loop = real(config, priors=priors, **kwargs)
        seen.append(loop.policy.param("complexity_start"))
        return loop

    monkeypatch.setattr(api, "build_loop", spy)
    from hillclimb.modules.memory.files import FilesMemory

    offsets = iter([1, 0])
    monkeypatch.setattr(FilesMemory, "prior_experience", lambda self: ("", next(offsets), []))
    agent = FakeAgent()
    for _ in range(2):
        agent.queue(script=ok_script(0.6), notes="d\n")
    monkeypatch.setattr("hillclimb.api.get_agent", lambda *a, **k: agent)
    config.holdout.enabled = False
    config.budget.max_evaluations = 1
    run_dir = api.create_run(config, RunMeta(run_id="r1", name="r1", kind="problem", target="t", problem_ids=[task.problem_id]))
    search_dir = api.create_search(config, task, run_dir, "r1", 600)
    assert load_search_meta(search_dir).memory_priors is None  # not yet run

    api.execute_search(config, task, search_dir, BudgetManager(600, stop_margin_s=1), log=lambda *_: None)
    assert load_search_meta(search_dir).memory_priors == {"complexity_start": 1}
    api.execute_search(config, task, search_dir, BudgetManager(600, stop_margin_s=1), log=lambda *_: None)  # a resume
    assert seen == [1, 1]  # the knowledge said 0 the second time; the record won
