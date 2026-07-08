"""SearchPolicy seam: GreedyPolicy decision surface over synthetic journals,
and a scripted custom policy driving the harness end-to-end."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.candidate import Candidate, Trial
from hillclimb.executor import LocalExecutor
from hillclimb.journal import Journal
from hillclimb.policies import get_policy
from hillclimb.policies.greedy import GreedyPolicy
from hillclimb.policy import Action, BudgetView, InflightRef, SearchView
from hillclimb.search import GreedySearcher
from hillclimb.workspace import create_search_dir
from tests.conftest import ok_script


def make_view(
    journal: Journal,
    config,
    inflight: tuple[InflightRef, ...] = (),
    remaining_s: float = 3600.0,
    total_s: int = 3600,
    stop_margin_s: int = 1,
    lower_is_better: bool = False,
) -> SearchView:
    return SearchView(
        journal=journal,
        inflight=inflight,
        budget=BudgetView(
            remaining_s=remaining_s, total_s=total_s, stop_margin_s=stop_margin_s
        ),
        config=config,
        lower_is_better=lower_is_better,
    )


def add_candidate(
    journal: Journal,
    candidate_id: str,
    operator: str,
    status: str = "ok",
    val_score: float | None = None,
    parent_id: str | None = None,
    workspace: str = "",
    solution: str | None = None,
    tmp_path: Path | None = None,
) -> Candidate:
    if solution is not None:
        ws = tmp_path / candidate_id
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "solution.py").write_text(solution)
        workspace = str(ws)
    trials = (
        [Trial(val_score=val_score, submission_ok=True)] if val_score is not None else []
    )
    candidate = Candidate(
        candidate_id=candidate_id,
        operator=operator,
        status=status,
        parent_id=parent_id,
        workspace=workspace,
        trials=trials,
    )
    journal.candidate_result(candidate)
    return candidate


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    return Journal(tmp_path / "journal.jsonl")


def test_propose_drafts_on_empty_journal(journal, config):
    policy = GreedyPolicy()
    action = policy.propose(make_view(journal, config))
    assert action == Action(operator="draft", complexity="minimal")


def test_complexity_escalates_per_draft_with_offset(journal, config, tmp_path):
    assert GreedyPolicy(complexity_start=1).draft_complexity(
        make_view(journal, config)
    ) == "moderate"
    add_candidate(journal, "c001", "draft", status="buggy")
    add_candidate(journal, "c002", "draft", val_score=0.5)
    assert GreedyPolicy().draft_complexity(make_view(journal, config)) == "advanced"


def test_propose_debugs_newest_buggy_tip_until_depth_cap(journal, config):
    policy = GreedyPolicy()
    add_candidate(journal, "c001", "draft", status="buggy")
    action = policy.propose(make_view(journal, config))
    assert (action.operator, action.target_id) == ("debug", "c001")

    # a full-depth chain of failed fixes exhausts the cap -> back to drafting
    add_candidate(journal, "c002", "debug", status="buggy", parent_id="c001")
    add_candidate(journal, "c003", "debug", status="buggy", parent_id="c002")
    add_candidate(journal, "c004", "debug", status="buggy", parent_id="c003")
    action = policy.propose(make_view(journal, config))
    assert action.operator == "draft"


def test_propose_skips_tip_with_active_child(journal, config):
    policy = GreedyPolicy()
    add_candidate(journal, "c001", "draft", status="buggy")
    add_candidate(journal, "c002", "debug", status="pending", parent_id="c001")
    action = policy.propose(make_view(journal, config))
    assert action.operator == "draft"  # chain already being worked


def test_propose_spreads_improves_across_busy_targets(journal, config, tmp_path):
    policy = GreedyPolicy()
    for i, score in enumerate((0.9, 0.8, 0.7), start=1):
        add_candidate(journal, f"c00{i}", "draft", val_score=score)
    view = make_view(journal, config)
    assert policy.prospective_branches(view) == 3  # draft quota met
    assert policy.propose(view).target_id == "c001"  # best first

    busy = (InflightRef(candidate_id="c004", operator="improve", parent_id="c001"),)
    action = policy.propose(make_view(journal, config, inflight=busy))
    assert (action.operator, action.target_id) == ("improve", "c002")


def test_propose_ensemble_in_final_window_with_drain(journal, config, tmp_path):
    policy = GreedyPolicy()
    add_candidate(journal, "c001", "draft", val_score=0.9, solution="a\n", tmp_path=tmp_path)
    add_candidate(journal, "c002", "draft", val_score=0.8, solution="b\n", tmp_path=tmp_path)
    add_candidate(journal, "c003", "draft", val_score=0.7, solution="c\n", tmp_path=tmp_path)
    in_window = dict(remaining_s=100.0, total_s=3600, stop_margin_s=300)

    action = policy.propose(make_view(journal, config, **in_window))
    assert action.operator == "ensemble"
    assert action.target_id == "c001"
    assert action.inspiration_ids == ("c001", "c002", "c003")

    # in-flight work: hold the slot so ensemble inputs snapshot at launch
    busy = (InflightRef(candidate_id="c004", operator="improve", parent_id="c001"),)
    assert policy.propose(make_view(journal, config, inflight=busy, **in_window)) is None

    # identical scripts dedupe below the 2-candidate minimum -> no ensemble
    for cid in ("c001", "c002", "c003"):
        Path(journal.get(cid).workspace, "solution.py").write_text("same\n")
    action = policy.propose(make_view(journal, config, **in_window))
    assert action.operator != "ensemble"


def test_registry_resolves_greedy_and_rejects_unknown():
    policy = get_policy("greedy", {"note": "x"}, complexity_start=2)
    assert isinstance(policy, GreedyPolicy)
    assert policy.name == "greedy"
    assert policy.params == {"note": "x"}
    assert policy.complexity_start == 2
    with pytest.raises(ValueError, match="Unknown policy"):
        get_policy("map-elites")


class ScriptedPolicy:
    """Minimal non-greedy policy: drafts forever, stamping policy_meta and
    extra prompt context; records every observe() call."""

    name = "scripted"
    params: dict = {}

    def __init__(self):
        self.observed: list[str] = []
        self.proposals = 0

    def propose(self, view: SearchView) -> Action | None:
        self.proposals += 1
        return Action(
            operator="draft",
            policy_meta={"proposal": self.proposals},
            extra_prompt_context="Try simulated annealing.",
        )

    def observe(self, view: SearchView, candidate: Candidate) -> None:
        self.observed.append(candidate.candidate_id)


def test_custom_policy_drives_search(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="one\n")
    backend.queue(script=ok_script(0.7), notes="two\n")
    search_dir = create_search_dir(config.paths.runs_dir, "test-run")
    policy = ScriptedPolicy()
    searcher = GreedySearcher(
        problem=task,
        config=config,
        journal=Journal(search_dir / "journal.jsonl"),
        backend=backend,
        executor=LocalExecutor(Path(sys.executable)),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir,
        max_candidates=3,  # baseline + two drafts
        log=lambda *_: None,
        policy=policy,
    )
    best = searcher.run()

    assert best.val_score == 0.7
    assert [r.operator for r in backend.requests] == ["draft", "draft"]
    draft = searcher.journal.get("c001")
    assert draft.policy_meta == {"proposal": 1}
    prompt = Path(draft.workspace, "prompt.md").read_text()
    assert "# Additional context from the search strategy" in prompt
    assert "Try simulated annealing." in prompt
    # the policy saw every terminal result (baseline + both drafts)
    assert set(searcher.journal.candidates) <= set(policy.observed) | {"c001", "c002"}
    assert "c000" in policy.observed


def test_policy_replay_on_resume(task, config, tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    add_candidate(journal, "c000", "baseline", val_score=0.1)
    add_candidate(journal, "c001", "draft", val_score=0.5)
    policy = ScriptedPolicy()
    GreedySearcher(
        problem=task,
        config=config,
        journal=Journal(tmp_path / "j.jsonl"),  # fresh replay of the same file
        backend=FakeBackend(),
        executor=LocalExecutor(Path(sys.executable)),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=tmp_path,
        log=lambda *_: None,
        policy=policy,
    )
    assert policy.observed == ["c000", "c001"]  # journal order
