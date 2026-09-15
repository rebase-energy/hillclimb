"""SearchPolicy seam: GreedyPolicy decision surface over synthetic journals,
and a scripted custom policy driving the harness end-to-end."""

from __future__ import annotations

from tests.factories import trial as mk_trial

import sys
from pathlib import Path

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.candidate import Candidate
from tests.conftest import local_executor
from hillclimb.journal import Journal
from hillclimb.policies import get_policy
from hillclimb.policies.greedy import GreedyPolicy
from hillclimb.policy import Action, BudgetView, InflightRef, PolicyInput
from hillclimb.search import GreedySearcher
from hillclimb.dirs import create_search_dir
from tests.conftest import ok_script


def make_view(
    journal: Journal,
    config,
    inflight: tuple[InflightRef, ...] = (),
    remaining_s: float = 3600.0,
    total_s: int = 3600,
    stop_margin_s: int = 1,
    higher_is_better: bool = True,
) -> PolicyInput:
    return PolicyInput(
        journal=journal,
        inflight=inflight,
        budget=BudgetView(
            remaining_s=remaining_s, total_s=total_s, stop_margin_s=stop_margin_s
        ),
        config=config,
        higher_is_better=higher_is_better,
    )


def add_candidate(
    journal: Journal,
    candidate_id: str,
    operator: str,
    status: str = "ok",
    val_score: float | None = None,
    parent_id: str | None = None,
    candidate_dir: str = "",
    solution: str | None = None,
    tmp_path: Path | None = None,
) -> Candidate:
    if solution is not None:
        ws = tmp_path / candidate_id
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "solution.py").write_text(solution)
        candidate_dir = str(ws)
    trials = (
        [mk_trial(val_score=val_score, submission_ok=True)] if val_score is not None else []
    )
    candidate = Candidate(
        candidate_id=candidate_id,
        operator=operator,
        status=status,
        parent_id=parent_id,
        candidate_dir=candidate_dir,
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
        Path(journal.get(cid).candidate_dir, "solution.py").write_text("same\n")
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

    def propose(self, view: PolicyInput) -> Action | None:
        self.proposals += 1
        return Action(
            operator="draft",
            policy_meta={"proposal": self.proposals},
            extra_prompt_context="Try simulated annealing.",
        )

    def observe(self, view: PolicyInput, candidate: Candidate) -> None:
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
        executor=local_executor(),
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
    prompt = Path(draft.candidate_dir, "prompt.md").read_text()
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
        executor=local_executor(),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=tmp_path,
        log=lambda *_: None,
        policy=policy,
    )
    assert policy.observed == ["c000", "c001"]  # journal order


# --- one dict: strategy knobs in policy_params, config as the fallback ---


def test_strategy_knobs_fall_back_to_config_blocks(config):
    policy = GreedyPolicy()
    config.search.num_drafts = 5
    config.search.max_debug_depth = 1
    config.ensemble.top_k = 4
    resolved = policy.resolved_params(config)
    assert resolved["num_drafts"] == 5 and resolved["max_debug_depth"] == 1
    assert resolved["ensemble_top_k"] == 4 and resolved["ensemble"] is True
    assert resolved["tune_budget"] == 8
    # policy_params win over the config block
    policy = GreedyPolicy(params={"num_drafts": 1, "ensemble": False, "tune_budget": 0})
    resolved = policy.resolved_params(config)
    assert resolved["num_drafts"] == 1 and resolved["ensemble"] is False and resolved["tune_budget"] == 0
    assert set(resolved) == {
        "num_drafts", "max_debug_depth", "ensemble", "ensemble_reserve_fraction",
        "ensemble_top_k", "ensemble_max_attempts",
        "tune_budget", "tune_gate", "tune_parallel", "tune_burst",
    }


def test_num_drafts_and_debug_depth_from_policy_params(journal, config):
    add_candidate(journal, "c001", "draft", val_score=0.5)
    assert GreedyPolicy().propose(make_view(journal, config)).operator == "draft"  # config: 3 drafts
    action = GreedyPolicy(params={"num_drafts": 1}).propose(make_view(journal, config))
    assert (action.operator, action.target_id) == ("improve", "c001")

    add_candidate(journal, "c002", "draft", status="buggy")
    add_candidate(journal, "c003", "debug", status="buggy", parent_id="c002")
    assert GreedyPolicy().propose(make_view(journal, config)).operator == "debug"  # depth 1 < 3
    action = GreedyPolicy(params={"num_drafts": 1, "max_debug_depth": 1}).propose(make_view(journal, config))
    assert action.operator == "improve"  # chain exhausted at depth 1


def test_ensemble_knobs_from_policy_params(journal, config, tmp_path):
    for i, (score, text) in enumerate(zip((0.9, 0.8, 0.7, 0.6), "abcd"), start=1):
        add_candidate(journal, f"c00{i}", "draft", val_score=score, solution=text + "\n", tmp_path=tmp_path)
    in_window = dict(remaining_s=100.0, total_s=3600, stop_margin_s=300)
    assert GreedyPolicy().propose(make_view(journal, config, **in_window)).operator == "ensemble"
    assert GreedyPolicy(params={"ensemble": False}).propose(
        make_view(journal, config, **in_window)
    ).operator != "ensemble"
    # a wider reserve opens the window earlier; a bigger top_k blends more
    mid = dict(remaining_s=1500.0, total_s=3600, stop_margin_s=300)
    assert GreedyPolicy().propose(make_view(journal, config, **mid)).operator != "ensemble"
    action = GreedyPolicy(params={"ensemble_reserve_fraction": 0.4, "ensemble_top_k": 4}).propose(
        make_view(journal, config, **mid)
    )
    assert action.operator == "ensemble" and len(action.inspiration_ids) == 4
    # max_attempts=0 disables it outright
    assert GreedyPolicy(params={"ensemble_max_attempts": 0}).propose(
        make_view(journal, config, **in_window)
    ).operator != "ensemble"


# --- file policies: an edited exploration process loaded from a path ---

FILE_POLICY = '''
from hillclimb.policies.greedy import GreedyPolicy
from hillclimb.policy import Action


class DraftsOnly(GreedyPolicy):
    """Never improves: drafts forever (a deliberately different process)."""

    name = "drafts-only"

    def propose(self, view):
        tip = self.debuggable_tip(view)
        if tip is not None:
            return Action(operator="debug", target_id=tip.candidate_id)
        return self._draft_action(view)
'''


def test_file_policy_loads_by_path_and_is_hashed(tmp_path, journal, config):
    import hashlib

    from hillclimb.policies import get_policy, policy_label, policy_path, policy_sha256

    path = tmp_path / "hillclimb" / "policies" / "drafts_only.py"
    path.parent.mkdir(parents=True)
    path.write_text(FILE_POLICY)
    policy = get_policy(str(path), {"num_drafts": 1}, complexity_start=1)
    assert policy.name == "drafts-only" and policy.params == {"num_drafts": 1}
    assert policy.complexity_start == 1
    add_candidate(journal, "c001", "draft", val_score=0.5)
    assert policy.propose(make_view(journal, config)).operator == "draft"  # greedy would improve

    # relative paths anchor at the folder holding the hillclimb dir
    assert policy_path("hillclimb/policies/drafts_only.py", tmp_path) == path
    assert policy_path("greedy") is None and policy_sha256("greedy") is None
    assert policy_sha256("hillclimb/policies/drafts_only.py", tmp_path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert policy_label(str(path)) == "drafts_only" and policy_label("greedy") == "greedy"
    with pytest.raises(FileNotFoundError):
        policy_sha256("hillclimb/policies/missing.py", tmp_path)


def test_file_policy_exposes_POLICY_class_or_factory(tmp_path):
    from hillclimb.policies import get_policy

    factory_file = tmp_path / "factory.py"
    factory_file.write_text(
        "from hillclimb.policies.greedy import GreedyPolicy\n"
        "class A(GreedyPolicy):\n    name = 'a'\n"
        "class B(GreedyPolicy):\n    name = 'b'\n"
        "def POLICY(params, *, complexity_start=0):\n"
        "    return B(params=params, complexity_start=complexity_start + 10)\n"
    )
    policy = get_policy(str(factory_file), {"x": 1})
    assert policy.name == "b" and policy.complexity_start == 10 and policy.params == {"x": 1}

    # a bare protocol class with no constructor arguments still loads and
    # gets a name and params stamped on it
    bare = tmp_path / "bare.py"
    bare.write_text(
        "class Bare:\n"
        "    def propose(self, view):\n        return None\n"
        "    def observe(self, view, candidate):\n        pass\n"
    )
    policy = get_policy(str(bare), {"k": 2})
    assert policy.name == "bare" and policy.params == {"k": 2}


def test_file_policy_errors_name_the_file(tmp_path):
    from hillclimb.policies import get_policy

    with pytest.raises(ValueError, match="policy file not found"):
        get_policy(str(tmp_path / "nope.py"))
    broken = tmp_path / "broken.py"
    broken.write_text("import definitely_not_a_module\n")
    with pytest.raises(ValueError, match="failed to import: ModuleNotFoundError"):
        get_policy(str(broken))
    two = tmp_path / "two.py"
    two.write_text(
        "from hillclimb.policies.greedy import GreedyPolicy\n"
        "class A(GreedyPolicy): pass\nclass B(GreedyPolicy): pass\n"
    )
    with pytest.raises(ValueError, match="exactly one policy class"):
        get_policy(str(two))
    none = tmp_path / "none.py"
    none.write_text("x = 1\n")
    with pytest.raises(ValueError, match="exactly one policy class"):
        get_policy(str(none))
    not_policy = tmp_path / "notpolicy.py"
    not_policy.write_text("class Thing:\n    pass\nPOLICY = Thing\n")
    with pytest.raises(ValueError, match="has no propose"):
        get_policy(str(not_policy))
    with pytest.raises(ValueError, match="Unknown policy: map-elites"):
        get_policy("map-elites")


def test_file_policy_drives_a_search_and_is_recorded(task, config, tmp_path):
    """`search.policy: <file>` flows through build_search_strategy, and the
    search record pins the file's hash."""
    from hillclimb.api import create_run, create_search
    from hillclimb.problem import load_problem
    from hillclimb.run import RunMeta, load_search_meta
    from hillclimb.search_strategy import build_search_strategy
    from tests.test_cli import write_problem

    path = tmp_path / "drafts_only.py"
    path.write_text(FILE_POLICY)
    config.search.policy = str(path)
    config.search.policy_params = {"num_drafts": 1}
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="one\n")
    backend.queue(script=ok_script(0.7), notes="two\n")
    search_dir = create_search_dir(config.paths.runs_dir, "test-run")
    searcher = build_search_strategy(
        config=config, problem=task, journal=Journal(search_dir / "journal.jsonl"),
        backend=backend, executor=local_executor(), budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir, log=lambda *_: None,
    )
    searcher.max_candidates = 3
    assert searcher.policy.name == "drafts-only"
    searcher.run()
    assert [r.operator for r in backend.requests] == ["draft", "draft"]  # never improve

    root = tmp_path / "problems"
    write_problem(root, "p")
    config.paths.problems_dir = root
    run_dir = create_run(config, RunMeta(run_id="r1", name="r1", kind="problem", target="p", problem_ids=["p"]))
    meta = load_search_meta(create_search(config, load_problem("p", config), run_dir, "r1", 60))
    assert meta.policy == str(path)
    import hashlib
    assert meta.policy_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    config.search.policy = "greedy"
    meta = load_search_meta(create_search(config, load_problem("p", config), run_dir, "r1", 60))
    assert meta.policy_sha256 is None


def test_mixed_fleet_names_file_policy_arms_by_stem():
    from hillclimb.api import mixed_fleet

    engines = mixed_fleet(["greedy", "hillclimb/policies/drafts_only.py", "hillclimb/policies/drafts_only.py"])
    assert [(e.arm, e.policy) for e in engines] == [
        ("greedy", "greedy"),
        ("drafts_only", "hillclimb/policies/drafts_only.py"),
        ("drafts_only-2", "hillclimb/policies/drafts_only.py"),
    ]
