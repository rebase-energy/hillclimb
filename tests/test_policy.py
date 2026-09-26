"""SearchPolicy seam: GreedyPolicy decision surface over synthetic journals,
and a scripted custom policy driving the harness end-to-end."""

from __future__ import annotations

from tests.factories import trial as mk_trial

import sys
from pathlib import Path

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.harness.budget import BudgetManager
from hillclimb.harness.candidate import Candidate
from tests.conftest import local_executor
from hillclimb.harness.journal import Journal
from hillclimb.harness.evaluation import accept_band
from hillclimb.modules.policies import get_policy
from hillclimb.modules.policies.greedy import GreedyPolicy
from hillclimb.modules.policies.base import Action, BudgetView, InflightRef, PolicyInput
from tests.harness_factory import SearchRig
from hillclimb.harness.dirs import create_search_dir
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
        higher_is_better=higher_is_better,
        accept_band=accept_band(config, journal),
    )


def add_candidate(
    journal: Journal,
    candidate_id: str,
    operator: str,
    status: str = "passing",
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
    assert action == Action(operator="draft", args={"complexity": "minimal"})


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
    searcher = SearchRig(
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
    SearchRig(
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


def test_every_knob_is_one_dict_with_defaults(config):
    """A policy never sees the harness's config: knobs come from the
    climber's params (the manifest's, with the user's `climber.params` laid
    over them), else DEFAULTS."""
    from hillclimb.harness.glue import build_loop

    assert GreedyPolicy().resolved_params()["num_drafts"] == 3
    resolved = build_loop(config).policy.resolved_params()  # the bundled manifest's params
    assert resolved["num_drafts"] == 3 and resolved["ensemble_top_k"] == 3 and resolved["tune_budget"] == 8
    # the user's overlay wins over the manifest
    config.climber.params = {"num_drafts": 1, "ensemble": False, "tune_budget": 0}
    resolved = build_loop(config).policy.resolved_params()
    assert resolved["num_drafts"] == 1 and resolved["ensemble"] is False and resolved["tune_budget"] == 0
    assert resolved["max_debug_depth"] == 3  # untouched keys keep the manifest's value
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
from hillclimb.modules.policies.greedy import GreedyPolicy
from hillclimb.modules.policies.base import Action


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

    from hillclimb.modules.policies import get_policy, policy_label, policy_path, policy_sha256

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
    from hillclimb.modules.policies import get_policy

    factory_file = tmp_path / "factory.py"
    factory_file.write_text(
        "from hillclimb.modules.policies.greedy import GreedyPolicy\n"
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
    from hillclimb.modules.policies import get_policy

    with pytest.raises(ValueError, match="policy file not found"):
        get_policy(str(tmp_path / "nope.py"))
    broken = tmp_path / "broken.py"
    broken.write_text("import definitely_not_a_module\n")
    with pytest.raises(ValueError, match="failed to import: ModuleNotFoundError"):
        get_policy(str(broken))
    two = tmp_path / "two.py"
    two.write_text(
        "from hillclimb.modules.policies.greedy import GreedyPolicy\n"
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
    """`search.policy: <file>` flows through build_loop, and the search
    record pins the file's hash."""
    from hillclimb.api import create_run, create_search
    from hillclimb.problem import load_problem
    from hillclimb.harness.run import RunMeta, load_search_meta
    from hillclimb.harness.glue import build_loop
    from tests.harness_factory import make_harness
    from tests.test_cli import write_problem

    path = tmp_path / "drafts_only.py"
    path.write_text(FILE_POLICY)
    config.climber.ref = str(path)
    config.climber.params = {"num_drafts": 1}
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="one\n")
    backend.queue(script=ok_script(0.7), notes="two\n")
    config.budget.max_evaluations = 2
    loop = build_loop(config)
    assert loop.policy.name == "drafts-only"
    harness, _journal, _search_dir = make_harness(task, config, backend, name="test-run")
    harness.execute(loop)
    assert [r.operator for r in backend.requests] == ["draft", "draft"]  # never improve

    root = tmp_path / "problems"
    write_problem(root, "p")
    config.paths.problems_dir = root
    run_dir = create_run(config, RunMeta(run_id="r1", name="r1", kind="problem", target="p", problem_ids=["p"]))
    meta = load_search_meta(create_search(config, load_problem("p", config), run_dir, "r1", 60))
    assert meta.climber == str(path)
    import hashlib
    assert meta.climber_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    config.climber.ref = "greedy"
    meta = load_search_meta(create_search(config, load_problem("p", config), run_dir, "r1", 60))
    from hillclimb.climber import load_climber
    assert meta.climber_sha256 == load_climber("greedy").sha256  # a bundled climber has an identity too


def test_mixed_fleet_names_file_policy_arms_by_stem():
    from hillclimb.api import mixed_fleet

    engines = mixed_fleet(["greedy", "hillclimb/policies/drafts_only.py", "hillclimb/policies/drafts_only.py"])
    assert [(e.arm, e.climber) for e in engines] == [
        ("greedy", "greedy"),
        ("drafts_only", "hillclimb/policies/drafts_only.py"),
        ("drafts_only-2", "hillclimb/policies/drafts_only.py"),
    ]


# --- the policy seam is holdout-blind ---


def _add_with_holdout(journal, tmp_path, candidate_id, val, holdout, selected=False):
    ws = tmp_path / candidate_id
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "solution.py").write_text(f"{candidate_id}\n")
    journal.candidate_result(
        Candidate(
            candidate_id=candidate_id, operator="draft", status="passing",
            candidate_dir=str(ws), is_selected=selected,
            trials=[mk_trial(val_score=val, submission_ok=True, holdout_score=holdout, holdout_cpu_s=1.5)],
        )
    )


def test_policy_view_never_carries_holdout(journal, config, tmp_path):
    _add_with_holdout(journal, tmp_path, "c001", val=0.9, holdout=0.1)
    _add_with_holdout(journal, tmp_path, "c002", val=0.8, holdout=0.95, selected=True)
    view = make_view(journal, config)

    for candidate in view.journal.candidates.values():
        assert candidate.holdout_score is None and not candidate.is_selected
        assert all(t.holdout_score is None and t.holdout_cpu_s is None for t in candidate.trials)
    assert "holdout" not in "".join(
        k for c in view.journal.candidates.values() for k, v in c.trials[0].model_dump().items() if v is not None
    )
    # every selection mode degrades to val order on the view
    for mode in ("rank-blend", "holdout", "val"):
        assert [c.candidate_id for c in view.journal.ranked_candidates(True, mode)] == ["c001", "c002"]
    assert view.journal.path is None
    # the engine's journal is untouched, and the view cannot write
    assert journal.get("c002").holdout_score == 0.95 and journal.get("c002").is_selected
    assert journal.selected_candidate(True, "holdout").candidate_id == "c002"
    with pytest.raises(TypeError):
        view.journal.candidate_result(view.journal.get("c001"))


def test_greedy_ensemble_inputs_ignore_holdout(journal, config, tmp_path):
    # holdout ranks c003 > c002 > c001, val the reverse; selection=holdout
    # must still not steer which candidates a policy ensembles
    for cid, val, hold in (("c001", 0.9, 0.1), ("c002", 0.8, 0.5), ("c003", 0.7, 0.9)):
        _add_with_holdout(journal, tmp_path, cid, val, hold)
    config.holdout.selection = "holdout"
    action = GreedyPolicy(params={"ensemble_top_k": 2}).propose(
        make_view(journal, config, remaining_s=100.0, total_s=3600, stop_margin_s=300)
    )
    assert action.operator == "ensemble"
    assert action.inspiration_ids == ("c001", "c002")


def test_searcher_hands_observe_a_holdout_blind_candidate(task, config, tmp_path):
    seen: list[Candidate] = []

    class Spy(GreedyPolicy):
        def observe(self, view, candidate):
            seen.append(candidate)
            seen.extend(view.journal.candidates.values())

    search_dir = create_search_dir(config.paths.runs_dir, "test-run")
    journal = Journal(search_dir / "journal.jsonl")
    _add_with_holdout(journal, tmp_path, "c001", val=0.5, holdout=0.42, selected=True)
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="improved\n")
    searcher = SearchRig(
        problem=task, config=config, journal=journal, backend=backend,
        executor=local_executor(), budget=BudgetManager(60, stop_margin_s=1),
        search_dir=search_dir, log=lambda *_: None, policy=Spy(),
    )
    assert seen, "the policy replays the journal on construction"
    replayed = len(seen)
    searcher.run_operator("improve", journal.get("c001"))

    assert len(seen) > replayed  # ... and sees every result as it lands
    assert all(c.holdout_score is None and not c.is_selected for c in seen)
    assert journal.get("c001").holdout_score == 0.42
