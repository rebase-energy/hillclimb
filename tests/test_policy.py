"""Policy seam: Greedy decision surface over synthetic journals,
and a scripted custom policy driving the harness end-to-end."""

from __future__ import annotations

from tests.factories import make_policy, trial as mk_trial, name_climber

import sys
from pathlib import Path

import pytest

from hillclimb.agents.fake import FakeAgent
from hillclimb.harness.budget import BudgetManager
from hillclimb.harness.candidate import Candidate
from tests.conftest import local_executor
from hillclimb.harness.journal import Journal
from hillclimb.harness.evaluation import accept_band
from hillclimb.climber import ClimberLoadError, climber_label
from hillclimb.harness.loop import PolicyLoop
from tests.catalog_fixture import greedy_classes

Greedy, Best = greedy_classes()
from hillclimb.modules.policies.base import Action, BudgetView, InflightRef, SearchState
from tests.harness_factory import SearchRig
from hillclimb.harness.dirs import create_search_dir
from tests.conftest import ok_script


from tests.catalog_fixture import GREEDY
def make_view(
    journal: Journal,
    config,
    inflight: tuple[InflightRef, ...] = (),
    remaining_s: float = 3600.0,
    total_s: int = 3600,
    stop_margin_s: int = 1,
    higher_is_better: bool = True,
) -> SearchState:
    return SearchState(
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


def decide(view: SearchState, *, select=None, policy=None) -> Action | None:
    """One decision as the loop makes it: the selector (π_sel) first, then
    the policy (π_op) on what it chose."""
    policy = policy or Greedy(selector=select or Best())
    return PolicyLoop(policy).propose(view)


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    return Journal(tmp_path / "journal.jsonl")


def test_propose_drafts_on_empty_journal(journal, config):
    action = decide(make_view(journal, config))
    assert action == Action(operator="draft", args={"complexity": "minimal"})


def test_complexity_escalates_per_draft_with_offset(journal, config, tmp_path):
    assert Greedy({"complexity_start": 1}).draft_complexity(
        make_view(journal, config)
    ) == "moderate"
    add_candidate(journal, "c001", "draft", status="buggy")
    add_candidate(journal, "c002", "draft", val_score=0.5)
    assert Greedy().draft_complexity(make_view(journal, config)) == "advanced"


def test_propose_debugs_newest_buggy_tip_until_depth_cap(journal, config):
    add_candidate(journal, "c001", "draft", status="buggy")
    action = decide(make_view(journal, config))
    assert (action.operator, action.target_id) == ("debug", "c001")

    # a full-depth chain of failed fixes exhausts the cap -> back to drafting
    add_candidate(journal, "c002", "debug", status="buggy", parent_id="c001")
    add_candidate(journal, "c003", "debug", status="buggy", parent_id="c002")
    add_candidate(journal, "c004", "debug", status="buggy", parent_id="c003")
    action = decide(make_view(journal, config))
    assert action.operator == "draft"


def test_propose_skips_tip_with_active_child(journal, config):
    add_candidate(journal, "c001", "draft", status="buggy")
    add_candidate(journal, "c002", "debug", status="pending", parent_id="c001")
    action = decide(make_view(journal, config))
    assert action.operator == "draft"  # chain already being worked


def test_propose_spreads_improves_across_busy_targets(journal, config, tmp_path):
    for i, score in enumerate((0.9, 0.8, 0.7), start=1):
        add_candidate(journal, f"c00{i}", "draft", val_score=score)
    view = make_view(journal, config)
    assert Best().prospective_branches(view) == 3  # draft quota met
    assert decide(view).target_id == "c001"  # best first

    busy = (InflightRef(candidate_id="c004", operator="improve", parent_id="c001"),)
    action = decide(make_view(journal, config, inflight=busy))
    assert (action.operator, action.target_id) == ("improve", "c002")


def test_propose_ensemble_in_final_window_with_drain(journal, config, tmp_path):
    add_candidate(journal, "c001", "draft", val_score=0.9, solution="a\n", tmp_path=tmp_path)
    add_candidate(journal, "c002", "draft", val_score=0.8, solution="b\n", tmp_path=tmp_path)
    add_candidate(journal, "c003", "draft", val_score=0.7, solution="c\n", tmp_path=tmp_path)
    in_window = dict(remaining_s=100.0, total_s=3600, stop_margin_s=300)

    action = decide(make_view(journal, config, **in_window))
    assert action.operator == "ensemble"
    assert action.target_id == "c001"
    assert action.inspiration_ids == ("c001", "c002", "c003")

    # in-flight work: hold the slot so ensemble inputs snapshot at launch
    busy = (InflightRef(candidate_id="c004", operator="improve", parent_id="c001"),)
    assert decide(make_view(journal, config, inflight=busy, **in_window)) is None

    # identical scripts dedupe below the 2-candidate minimum -> no ensemble
    for cid in ("c001", "c002", "c003"):
        Path(journal.get(cid).candidate_dir, "solution.py").write_text("same\n")
    action = decide(make_view(journal, config, **in_window))
    assert action.operator != "ensemble"


def test_a_bundled_name_resolves_greedy_and_an_unknown_one_is_refused():
    policy = make_policy(str(GREEDY), {"num_drafts": 5, "tune_budget": 4}, priors={"complexity_start": 2, "not_a_knob": 1})
    assert isinstance(policy, Greedy)
    assert policy.name == "greedy"
    # the caller's params over the block's; what memory learned under both,
    # and only the knobs the policy declares — the schedule's went to the selector
    assert policy.params == {"tune_budget": 4, "complexity_start": 2}
    assert policy.selector.param("num_drafts") == 5
    assert make_policy(str(GREEDY), {"complexity_start": 0}, priors={"complexity_start": 2}).param("complexity_start") == 0
    # a param the policy does not have is refused before anything runs
    with pytest.raises(ClimberLoadError, match="has no param 'note' .it has: .*tune_budget.*selector_params"):
        make_policy(str(GREEDY), {"note": "x"})
    with pytest.raises(ClimberLoadError, match="Unknown climber: map-elites"):
        make_policy("map-elites")


class ScriptedPolicy:
    """Minimal non-greedy policy: drafts forever, stamping climber_meta and
    extra prompt context; records every observe() call."""

    name = "scripted"
    params: dict = {}

    def __init__(self):
        self.observed: list[str] = []
        self.proposals = 0

    def propose(self, view: SearchState, selection=None) -> Action | None:
        self.proposals += 1
        return Action(
            operator="draft",
            climber_meta={"proposal": self.proposals},
            extra_prompt_context="Try simulated annealing.",
        )

    def observe(self, view: SearchState, candidate: Candidate) -> None:
        self.observed.append(candidate.candidate_id)


def test_custom_policy_drives_search(task, config):
    agent = FakeAgent()
    agent.queue(script=ok_script(0.6), notes="one\n")
    agent.queue(script=ok_script(0.7), notes="two\n")
    search_dir = create_search_dir(config.paths.runs_dir, "test-run")
    policy = ScriptedPolicy()
    searcher = SearchRig(
        problem=task,
        config=config,
        journal=Journal(search_dir / "journal.jsonl"),
        agent=agent,
        executor=local_executor(),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir,
        max_candidates=3,  # baseline + two drafts
        log=lambda *_: None,
        policy=policy,
    )
    best = searcher.run()

    assert best.val_score == 0.7
    assert [r.operator for r in agent.requests] == ["draft", "draft"]
    draft = searcher.journal.get("c001")
    assert draft.climber_meta == {"proposal": 1}
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
        agent=FakeAgent(),
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
    climber's `params`, else the class's DEFAULTS."""
    from hillclimb.harness.glue import build_loop

    assert Greedy().resolved_params()["tune_budget"] == 8 and Best().param("num_drafts") == 3
    policy = build_loop(config).policy  # the folder's block: greedy over best, no params
    assert policy.resolved_params()["tune_budget"] == 8
    assert policy.selector.param("num_drafts") == 3 and policy.selector.param("ensemble_top_k") == 3
    # the block's params win over the defaults; the schedule's knobs under
    # `params` reach the selector (every block before 0.7 wrote them there)
    config.climber.params = {"num_drafts": 1, "ensemble": False, "tune_budget": 0}
    policy = build_loop(config).policy
    assert policy.resolved_params()["tune_budget"] == 0 and "num_drafts" not in policy.params
    assert policy.selector.param("num_drafts") == 1 and policy.selector.param("ensemble") is False
    assert policy.selector.param("max_debug_depth") == 3  # untouched knobs keep their default
    assert set(policy.resolved_params()) == {
        "complexity_start",  # every Policy's
        "tune_budget", "tune_gate", "tune_parallel", "tune_burst",
    } == set(Greedy.defaults())  # DEFAULTS merge over the class hierarchy
    assert set(Best.defaults()) == {
        "num_drafts", "debug", "max_debug_depth",
        "ensemble", "ensemble_reserve_fraction", "ensemble_top_k", "ensemble_max_attempts",
    }


def test_num_drafts_and_debug_depth_from_policy_params(journal, config):
    add_candidate(journal, "c001", "draft", val_score=0.5)
    assert decide(make_view(journal, config)).operator == "draft"  # config: 3 drafts
    action = decide(make_view(journal, config), select=Best(num_drafts=1))
    assert (action.operator, action.target_id) == ("improve", "c001")

    add_candidate(journal, "c002", "draft", status="buggy")
    add_candidate(journal, "c003", "debug", status="buggy", parent_id="c002")
    assert decide(make_view(journal, config)).operator == "debug"  # depth 1 < 3
    action = decide(make_view(journal, config), select=Best(num_drafts=1, max_debug_depth=1))
    assert action.operator == "improve"  # chain exhausted at depth 1


def test_ensemble_knobs_from_policy_params(journal, config, tmp_path):
    for i, (score, text) in enumerate(zip((0.9, 0.8, 0.7, 0.6), "abcd"), start=1):
        add_candidate(journal, f"c00{i}", "draft", val_score=score, solution=text + "\n", tmp_path=tmp_path)
    in_window = dict(remaining_s=100.0, total_s=3600, stop_margin_s=300)
    assert decide(make_view(journal, config, **in_window)).operator == "ensemble"
    assert decide(select=Best(ensemble=False), view=
        make_view(journal, config, **in_window)
    ).operator != "ensemble"
    # a wider reserve opens the window earlier; a bigger top_k blends more
    mid = dict(remaining_s=1500.0, total_s=3600, stop_margin_s=300)
    assert decide(make_view(journal, config, **mid)).operator != "ensemble"
    action = decide(select=Best(ensemble_reserve_fraction=0.4, ensemble_top_k=4), view=
        make_view(journal, config, **mid)
    )
    assert action.operator == "ensemble" and len(action.inspiration_ids) == 4
    # max_attempts=0 disables it outright
    assert decide(select=Best(ensemble_max_attempts=0), view=
        make_view(journal, config, **in_window)
    ).operator != "ensemble"


# --- file policies: an edited exploration process loaded from a path ---

FILE_POLICY = '''
import hillclimb as hc
Greedy = hc.catalog.module("greedy").Greedy
from hillclimb.modules.policies.base import Action


class DraftsOnly(Greedy):
    """Never improves: drafts forever (a deliberately different process)."""

    name = "drafts-only"

    def propose(self, view, selection):
        node = view.journal.candidates[selection.target_id] if selection is not None else None
        if node is not None and node.status in ("failing", "buggy"):
            return Action(operator="debug", target_id=node.candidate_id)
        return self.draft_action(view)
'''


def test_file_policy_loads_by_path_and_is_hashed(tmp_path, journal, config):
    from hillclimb.climber import load_climber

    path = tmp_path / "hillclimb" / "policies" / "drafts_only.py"
    path.parent.mkdir(parents=True)
    path.write_text(FILE_POLICY)
    policy = make_policy(str(path), {"num_drafts": 1}, priors={"complexity_start": 1})
    assert policy.name == "drafts-only" and policy.params == {"complexity_start": 1}
    assert policy.param("complexity_start") == 1 and policy.selector.param("num_drafts") == 1
    add_candidate(journal, "c001", "draft", val_score=0.5)
    assert decide(make_view(journal, config), policy=policy).operator == "draft"  # greedy would improve

    # relative paths anchor at the folder holding the hillclimb dir; a
    # one-file climber's identity follows its bytes, not its place
    relative = load_climber("hillclimb/policies/drafts_only.py", tmp_path)
    assert Path(relative.source) == path
    assert relative.sha256 == load_climber(str(path)).sha256
    elsewhere = tmp_path / "copy" / "drafts_only.py"
    elsewhere.parent.mkdir()
    elsewhere.write_text(FILE_POLICY)
    assert load_climber(str(elsewhere)).sha256 == relative.sha256
    elsewhere.write_text(FILE_POLICY + "# edited\n")
    assert load_climber(str(elsewhere)).sha256 != relative.sha256
    assert climber_label(str(path)) == "drafts_only" and climber_label("greedy") == "greedy"
    with pytest.raises(ClimberLoadError, match="climber file not found"):
        load_climber("hillclimb/policies/missing.py", tmp_path)


def test_file_policy_exposes_POLICY_class_or_factory(tmp_path):
    factory_file = tmp_path / "factory.py"
    factory_file.write_text(
        "import hillclimb as hc\nGreedy = hc.catalog.module('greedy').Greedy\n"
        "class A(Greedy):\n    name = 'a'\n"
        "class B(Greedy):\n    name = 'b'\n"
        "def POLICY(params):\n"
        "    return B(params={**params, 'num_drafts': 9})\n"
    )
    policy = make_policy(str(factory_file), {"x": 1})  # a factory takes whatever params it likes
    assert policy.name == "b" and policy.params == {"x": 1, "num_drafts": 9}

    # a bare protocol class with no constructor arguments still loads and
    # gets a name and params stamped on it
    bare = tmp_path / "bare.py"
    bare.write_text(
        "class Bare:\n"
        "    def propose(self, view):\n        return None\n"
        "    def observe(self, view, candidate):\n        pass\n"
    )
    policy = make_policy(str(bare), {"k": 2})
    assert policy.name == "bare" and policy.params == {"k": 2}


def test_file_policy_errors_name_the_file(tmp_path):
    with pytest.raises(ValueError, match="climber file not found"):
        make_policy(str(tmp_path / "nope.py"))
    broken = tmp_path / "broken.py"
    broken.write_text("import definitely_not_a_module\n")
    with pytest.raises(ValueError, match="failed to import: ModuleNotFoundError"):
        make_policy(str(broken))
    two = tmp_path / "two.py"
    two.write_text(
        "import hillclimb as hc\nGreedy = hc.catalog.module('greedy').Greedy\n"
        "class A(Greedy): pass\nclass B(Greedy): pass\n"
    )
    with pytest.raises(ValueError, match="exactly one operator policy class"):
        make_policy(str(two))
    none = tmp_path / "none.py"
    none.write_text("x = 1\n")
    with pytest.raises(ValueError, match="exactly one operator policy class"):
        make_policy(str(none))
    not_policy = tmp_path / "notpolicy.py"
    not_policy.write_text("class Thing:\n    pass\nPOLICY = Thing\n")
    with pytest.raises(ValueError, match="has no propose"):
        make_policy(str(not_policy))


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
    name_climber(config, str(path))
    config.climber.params = {"num_drafts": 1}
    agent = FakeAgent()
    agent.queue(script=ok_script(0.6), notes="one\n")
    agent.queue(script=ok_script(0.7), notes="two\n")
    config.budget.max_evaluations = 2
    loop = build_loop(config)
    assert loop.policy.name == "drafts-only"
    harness, _journal, _search_dir = make_harness(task, config, agent, name="test-run")
    harness.execute(loop)
    assert [r.operator for r in agent.requests] == ["draft", "draft"]  # never improve

    root = tmp_path / "problems"
    write_problem(root, "p")
    config.paths.problems_dir = root
    run_dir = create_run(config, RunMeta(run_id="r1", name="r1", kind="problem", target="p", problem_ids=["p"]))
    meta = load_search_meta(create_search(config, load_problem("p", config), run_dir, "r1", 60))
    assert meta.climber == "drafts_only" and meta.climber_spec["operator_policy"] == str(path)
    from hillclimb.harness.glue import search_climber
    assert meta.climber_sha256 == search_climber(config).sha256  # the block (its params too) + the file's bytes
    name_climber(config, str(GREEDY))
    meta = load_search_meta(create_search(config, load_problem("p", config), run_dir, "r1", 60))
    assert meta.climber == "greedy" and meta.climber_sha256 == search_climber(config).sha256  # a preset has an identity too


def test_mixed_fleet_names_file_policy_experiments_by_stem():
    from hillclimb.api import mixed_fleet

    engines = mixed_fleet([str(GREEDY), "hillclimb/policies/drafts_only.py", "hillclimb/policies/drafts_only.py"])
    assert [(e.experiment, e.climber) for e in engines] == [
        ("greedy", str(GREEDY)),
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
    action = decide(select=Best(ensemble_top_k=2), view=
        make_view(journal, config, remaining_s=100.0, total_s=3600, stop_margin_s=300)
    )
    assert action.operator == "ensemble"
    assert action.inspiration_ids == ("c001", "c002")


def test_searcher_hands_observe_a_holdout_blind_candidate(task, config, tmp_path):
    seen: list[Candidate] = []

    class Spy(Greedy):
        def observe(self, view, candidate):
            seen.append(candidate)
            seen.extend(view.journal.candidates.values())

    search_dir = create_search_dir(config.paths.runs_dir, "test-run")
    journal = Journal(search_dir / "journal.jsonl")
    _add_with_holdout(journal, tmp_path, "c001", val=0.5, holdout=0.42, selected=True)
    agent = FakeAgent()
    agent.queue(script=ok_script(0.6), notes="improved\n")
    searcher = SearchRig(
        problem=task, config=config, journal=journal, agent=agent,
        executor=local_executor(), budget=BudgetManager(60, stop_margin_s=1),
        search_dir=search_dir, log=lambda *_: None, policy=Spy(),
    )
    assert seen, "the policy replays the journal on construction"
    replayed = len(seen)
    searcher.run_operator("improve", journal.get("c001"))

    assert len(seen) > replayed  # ... and sees every result as it lands
    assert all(c.holdout_score is None and not c.is_selected for c in seen)
    assert journal.get("c001").holdout_score == 0.42
