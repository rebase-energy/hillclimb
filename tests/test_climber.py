"""Climbers: a block of config — a preset's name, one .py file, or the full
`climber:` block — one resolver, one identity, operators and prompts scoped
to the search. What 0.4/0.5 wrote (a directory with a manifest, a snapshot of
one) still loads."""

from __future__ import annotations

from pathlib import Path

import pytest

from hillclimb.agents.fake import FakeAgent
import yaml

from hillclimb.climber import (
    ClimberLoadError,
    ClimberSpec,
    bundled_climbers,
    load_climber,
    load_snapshot,
    resolve_climber,
    snapshot_climber,
    tree_sha256,
)
from hillclimb.harness.loop import PolicyLoop, Loop
from hillclimb.modules.policies.greedy import GreedyPolicy
from hillclimb.modules.policies.base import Action
from tests.conftest import ok_script
from tests.harness_factory import make_harness

POLICY_PY = '''\
from hillclimb.sdk import Action

class DraftsThenCross:
    """Two drafts, then cross the two best."""
    def __init__(self, params=None):
        self.params = params or {}
    def propose(self, view):
        scored = [c for c in view.journal.scored_candidates() if c.role == "create"]
        if len(scored) + len(view.inflight) < int(self.params.get("drafts", 2)):
            return Action(operator="draft")
        if any(c.operator == "cross" for c in view.journal.candidates.values()) or view.inflight:
            return None
        best = sorted(scored, key=lambda c: -c.val_score)[:2]
        return Action(operator="cross", target_id=best[0].candidate_id,
                      inspiration_ids=tuple(c.candidate_id for c in best))
    def observe(self, view, candidate):
        pass
'''

OPERATORS_PY = '''\
from hillclimb.sdk import Operator, Attempt, inspiration_filename

class Cross(Operator):
    name, role, needs_target = "cross", "combine", True
    def prepare(self, ctx):
        files = ", ".join(inspiration_filename(i) for i, _ in enumerate(ctx.inspirations, 1))
        return Attempt(prompt=ctx.render("cross", files=files, style=self.params.get("style", "plain")))
'''


BLOCK = {
    "policy": "policy.py",
    "params": {"drafts": 2},
    "operators": ["draft", {"operators.py:Cross": {"style": "bold"}}],
    "memory": "none",
    "prompts": "prompts",
}
LEGACY_SNAPSHOTS = Path(__file__).parent / "fixtures" / "legacy_snapshots"


def write_climber(root: Path, *, manifest: str | None = None) -> Path:
    """The files of a climber — and, as 0.4/0.5 had it, a `climber.yaml`
    manifest beside them (`BLOCK` is the same climber as a block)."""
    root.mkdir(parents=True)
    (root / "policy.py").write_text(POLICY_PY)
    (root / "operators.py").write_text(OPERATORS_PY)
    (root / "prompts").mkdir()
    (root / "prompts" / "cross.md").write_text("Cross {{files}} in a {{style}} way.\n")
    (root / "prompts" / "draft.md").write_text("MY OWN DRAFT PROMPT for {{metric_name}}.\n\n{{contract}}\n")
    (root / "climber.yaml").write_text(manifest or (
        "policy: policy.py\n"
        "params: {drafts: 2}\n"
        "operators:\n  - draft\n  - operators.py:Cross: {style: bold}\n"
        "memory: none\n"
        "prompts: prompts\n"
    ))
    return root


def test_the_bundled_climbers_load_and_name_their_modules():
    assert bundled_climbers() == ["gepa", "greedy", "openevolve"]
    greedy = load_climber("greedy")
    loop = greedy.build_loop(params={"num_drafts": 1})
    assert isinstance(loop, PolicyLoop) and isinstance(loop.policy, GreedyPolicy)
    assert loop.policy.param("num_drafts") == 1 and loop.policy.param("ensemble_top_k") == 3  # overlay, then the class's default
    assert greedy.operator_set().names() == ("draft", "debug", "improve", "ensemble")
    assert greedy.operator_set().get("draft").params == {}  # the operator's own defaults
    # a preset is a complete block because the classes declare what they need
    gepa = load_climber("gepa")
    assert gepa.spec.block() == {"loop": "gepa", "tuner": "random", "memory": "files", "graph": "knowledge-graph"}
    assert gepa.is_loop and gepa.holdout_timing == "after"
    assert gepa.operator_set().names() == ("gepa-reflect",)


def test_a_block_is_a_climber(tmp_path):
    """The `climber:` block: a bare string is a preset or one file, a mapping
    names each module; with neither `policy:` nor `loop:` it is greedy."""
    assert ClimberSpec.model_validate("openevolve").block()["policy"] == "openevolve"
    assert ClimberSpec.model_validate({"params": {"num_drafts": 5}}).policy == "greedy"
    spec = ClimberSpec.model_validate({"policy": "mine.py:Mine", "operators": ["draft", {"ops.py:Cross": None}]})
    assert spec.label == "mine" and spec.operator_items() == [("draft", {}), ("ops.py:Cross", {})]
    assert ClimberSpec.model_validate({"name": "x", "policy": "pkg.mod:Cls"}).label == "x"
    anchored = spec.anchored(tmp_path)
    assert anchored.policy == f"{tmp_path / 'mine.py'}:Mine"
    assert anchored.operators == ["draft", {f"{tmp_path / 'ops.py'}:Cross": None}]
    assert [p.name for p in anchored.file_paths()] == ["mine.py", "ops.py"]
    with pytest.raises(ValueError, match="Unknown climber: climbers/mine .*hillclimb climber show climbers/mine"):
        ClimberSpec.model_validate("climbers/mine")


def test_a_block_runs_with_its_own_operator_and_prompts(task, config, tmp_path):
    root = write_climber(tmp_path / "crosser")
    climber = resolve_climber({**BLOCK, "name": "crosser"}, root)  # file refs resolve from the block's folder
    assert climber.name == "crosser" and not climber.is_loop and climber.lint_prompts() == []
    agent = FakeAgent()
    for score in (0.5, 0.7, 0.9):
        agent.queue(script=ok_script(score), notes="x\n")
    config.budget.max_evaluations = 3  # the policy holds after its cross; a hold never ends a search
    harness, journal, _ = make_harness(
        task, config, agent, operators=climber.operator_set(), prompts_dir=climber.prompts_dir,
    )

    selected = harness.execute(climber.build_loop())

    assert [r.operator for r in agent.requests] == ["draft", "draft", "cross"]
    assert selected.operator == "cross" and selected.role == "combine" and selected.val_score == 0.9
    cross_prompt = Path(selected.candidate_dir, "prompt.md").read_text()
    assert cross_prompt.startswith("Cross candidate_1.py, candidate_2.py in a bold way.")  # manifest params reached it
    assert "# Output contract" in cross_prompt  # no {{contract}} token: the harness appended it
    draft_prompt = Path(journal.get("c001").candidate_dir, "prompt.md").read_text()
    assert draft_prompt.startswith("MY OWN DRAFT PROMPT for accuracy.")  # shadows the built-in by name
    # scoped to this search: the built-in registry never heard of `cross`
    from hillclimb.modules.operators import operator_names
    assert "cross" not in operator_names()


def test_an_operator_the_climber_did_not_list_is_refused(task, config, tmp_path):
    climber = resolve_climber(BLOCK, write_climber(tmp_path / "crosser"))
    agent = FakeAgent()
    agent.queue(script=ok_script(0.5), notes="x\n")
    harness, _journal, _ = make_harness(task, config, agent, operators=climber.operator_set())
    draft = harness.run(Action(operator="draft")).candidate
    refused = harness.run(Action(operator="improve", target_id=draft.candidate_id))
    assert refused.kind == "rejected" and "Unknown operator 'improve'" in refused.ticket.rejected


def test_a_one_file_climber_is_the_ten_line_story(task, config, tmp_path):
    path = tmp_path / "drafts_only.py"
    path.write_text(OPERATORS_PY + "\n" + POLICY_PY.replace("DraftsThenCross", "DraftsOnly"))
    climber = load_climber(str(path))
    assert (climber.name, climber.is_loop) == ("drafts_only", False)
    assert climber.spec.policy == str(path)  # a bare file is `policy: <file>`
    # the default four, plus the Operator the file itself defines
    assert climber.operator_set().names() == ("draft", "debug", "improve", "ensemble", "cross")
    assert climber.build_loop(params={"drafts": 5}).policy.params == {"drafts": 5}
    # relative refs resolve from a base dir (the folder holding the hillclimb dir)
    assert load_climber("drafts_only.py", base_dir=tmp_path).sha256 == climber.sha256


def test_a_one_file_loop_is_recognised(tmp_path):
    path = tmp_path / "two_shots.py"
    path.write_text(
        "from hillclimb.sdk import Action, Loop\n\n"
        "class TwoShots(Loop):\n"
        "    name = 'two-shots'\n"
        "    def __init__(self, params=None, parallelism=1):\n"
        "        self.shots = int((params or {}).get('shots', 2)); self.parallelism = parallelism\n"
        "    def run(self, harness):\n"
        "        for _ in range(self.shots):\n"
        "            harness.run(Action(operator='draft'))\n"
    )
    climber = load_climber(str(path))
    loop = climber.build_loop(params={"shots": 3}, parallelism=4)
    assert climber.is_loop and isinstance(loop, Loop) and (loop.shots, loop.parallelism) == (3, 4)


def test_identity_is_the_block_and_every_file_it_reaches(tmp_path):
    root = write_climber(tmp_path / "c")
    (root / "policy.py").write_text("from .helpers import LIMIT\n" + POLICY_PY)
    (root / "helpers.py").write_text("LIMIT = 2\n")
    before = resolve_climber(BLOCK, root).sha256
    assert resolve_climber({**BLOCK, "name": "renamed"}, root).sha256 == before  # a label is not identity
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "policy.cpython-312.pyc").write_bytes(b"junk")
    (root / "prompts" / ".DS_Store").write_bytes(b"junk")
    (root / "unrelated.py").write_text("x = 1\n")  # a file the block does not reach
    assert resolve_climber(BLOCK, root).sha256 == before
    assert resolve_climber({**BLOCK, "params": {"drafts": 3}}, root).sha256 != before  # a param
    (root / "prompts" / "cross.md").write_text("Cross {{files}} differently.\n")
    prompt_edit = resolve_climber(BLOCK, root).sha256
    assert prompt_edit != before  # a prompt edit is a different climber
    (root / "helpers.py").write_text("LIMIT = 3\n")
    assert resolve_climber(BLOCK, root).sha256 != prompt_edit  # so is a file it only reaches by import


def test_a_snapshot_is_the_climber_it_was_taken_of(tmp_path):
    """`<search_dir>/climber/`: the block, its local files and its prompts —
    same identity, loadable with the originals gone."""
    import shutil

    root = write_climber(tmp_path / "c")
    (root / "policy.py").write_text("from .helpers import LIMIT\n" + POLICY_PY)
    (root / "helpers.py").write_text("LIMIT = 2\n")
    live = resolve_climber({**BLOCK, "name": "crosser"}, root)
    snapshot = snapshot_climber(live, tmp_path / "search")
    assert sorted(p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file()) == [
        "climber.yaml", "files/helpers.py", "files/operators.py", "files/policy.py",
        "prompts/cross.md", "prompts/draft.md",
    ]
    block = yaml.safe_load((snapshot / "climber.yaml").read_text())
    assert block["snapshot"] == 2 and block["policy"] == "files/policy.py" and block["prompts"] == "prompts"
    assert block["operators"] == ["draft", {"files/operators.py:Cross": {"style": "bold"}}]
    shutil.rmtree(root)  # editing — or losing — the live files never changes a started search
    loaded = load_snapshot(tmp_path / "search")
    assert (loaded.name, loaded.sha256) == ("crosser", live.sha256)
    assert loaded.operator_set().names() == ("draft", "cross")
    assert loaded.build_loop().policy.params == {"drafts": 2}
    assert snapshot_climber(loaded, tmp_path / "search") == snapshot  # idempotent


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ("policy: policy.py\nloop: policy.py\n", "`policy:` .* or `loop:` .*, not both"),
        ("policy: policy.py\nrouting: {draft: {model: opus}}\n", "`routing` is reserved"),
        ("policy: policy.py\nmemory: sqlite\n", "memory"),
        ("policy: policy.py\nnum_drafts: 3\n", "num_drafts"),  # a typo'd top-level key, not silently ignored
        ("policy: policy.py\ndescription: mine\n", "`description`: .*YAML comment"),
        ("policy: policy.py\nsimilarity: [api-calls]\n", "`similarity`: .*viewer's setting"),
        ("policy: policy.py\nholdout_timing: after\n", "`holdout_timing`: a loop declares it on its class"),
        ("policy: nope.py\n", "climber file not found: .*nope.py"),
        ("policy: nonsense\n", r"unknown policy 'nonsense' \(available: greedy, openevolve"),
        ("loop: nonsense\n", r"unknown loop 'nonsense' \(available: gepa"),
        ("policy: operators.py\n", "exactly one policy class"),
        ("policy: policy.py\noperators: [operators.py:Nope]\n", "defines no Nope"),
        ("policy: policy.py\noperators: [nope]\n", r"unknown operator 'nope' \(available: debug, draft, ensemble, improve"),
        ("policy: policy.py\noperator_params: {cross: {style: bold}}\n", "has no operator 'cross'"),
        ("policy: policy.py\ntuner: nope\n", "unknown tuner 'nope'"),
        ("policy: policy.py\ngraph: nope.py\n", "climber file not found: .*nope.py"),
        ("policy: policy.py\ngraph: operators.py\n", "exactly one GraphModule subclass"),
        ("policy: policy.py\ngraph: operators.py:Cross\n", "is not a GraphModule subclass"),
        ("policy: policy.py\ngraph: nowhere.mod:X\n", "cannot import graph module 'nowhere.mod:X'"),
    ],
)
def test_a_bad_block_names_the_file_and_the_fix(tmp_path, block, message):
    root = write_climber(tmp_path / "bad")
    with pytest.raises(ClimberLoadError, match=message):
        climber = resolve_climber(yaml.safe_load(block), root)
        climber.build_loop()
        climber.operator_set()
        climber.tuner()
        climber.graph_module()


def test_unknown_reference_lists_what_exists(tmp_path):
    with pytest.raises(ClimberLoadError, match="bundled: gepa, greedy, openevolve"):
        load_climber("nope")
    with pytest.raises(ClimberLoadError, match="Unknown climber: nope .presets: gepa, greedy, openevolve"):
        resolve_climber("nope")
    with pytest.raises(ClimberLoadError, match="holds no climber.yaml"):
        load_climber(str(tmp_path))


def test_a_climber_may_not_shadow_the_contract(tmp_path):
    root = write_climber(tmp_path / "sneaky")
    (root / "prompts" / "contract_verifier.md").write_text("Anything goes.\n")
    (root / "prompts" / "improve.md").write_text("Improve it. {{made_up_token}}\n")
    problems = resolve_climber(BLOCK, root).lint_prompts()
    assert any("contract_verifier.md: harness-owned" in p for p in problems)
    assert any("improve.md" in p and "made_up_token" in p for p in problems)


def test_an_import_error_in_the_authors_file_is_reported_with_its_path(tmp_path):
    root = write_climber(tmp_path / "broken")
    (root / "policy.py").write_text("import not_a_real_module\n")
    with pytest.raises(ClimberLoadError, match="policy.py failed to import: ModuleNotFoundError"):
        resolve_climber(BLOCK, root).build_loop()


# --- run folders: what is recorded, and what older folders still load as ---


def test_a_run_folder_written_before_climbers_still_loads(tmp_path):
    """Schema v2 named a `policy`. Same thing, older word: every view keeps
    showing those runs (file store and sqlite alike — the mapping lives on
    the model)."""
    import yaml

    from hillclimb.harness.run import SearchMeta, load_search_meta

    search_dir = tmp_path / "runs" / "r" / "searches" / "gefcom-solar"
    search_dir.mkdir(parents=True)
    (search_dir / "search.yaml").write_text(yaml.safe_dump({
        "schema_version": 2, "search_id": "gefcom-solar", "run_id": "r", "problem": "p",
        "problem_id": "gefcom-solar", "agent": "claude-code", "model": "sonnet",
        "policy": "hillclimb/policies/drafts_only.py", "policy_params": {"num_drafts": 1},
        "policy_sha256": "ab" * 32, "tuner": "optuna", "tuner_params": {"seed": 3},
        "metric": "pinball", "higher_is_better": False,
        "templates_sha256": "cd" * 32, "templates_overridden": ["improve"],
    }))
    meta = load_search_meta(search_dir)
    assert meta is not None and meta.schema_version == 4
    assert meta.climber == meta.climber_ref == "hillclimb/policies/drafts_only.py"
    assert meta.climber_sha256 == "ab" * 32 and meta.hillclimb_version is None  # unknown for an old run
    # the name, the params and the tuner it recorded, as the one block 0.6 keeps
    assert meta.climber_spec == {
        "policy": "hillclimb/policies/drafts_only.py", "params": {"num_drafts": 1},
        "tuner": "optuna", "tuner_params": {"seed": 3},
    }
    # and the same record as the sqlite store holds it
    again = SearchMeta.model_validate_json(meta.model_dump_json())
    assert (again.climber, again.climber_spec) == (meta.climber, meta.climber_spec)
    # a version this build has never heard of stays invisible rather than half-read
    (search_dir / "search.yaml").write_text(yaml.safe_dump({"schema_version": 9, "search_id": "x"}))
    assert load_search_meta(search_dir) is None


def test_a_run_folder_written_by_0_5_still_loads(tmp_path):
    """Schema v3 named the climber by reference and kept its manifest, the
    user's params overlay and the user's tuner override in separate fields.
    They fold into the one block: the manifest, with the user's on top."""
    from hillclimb.harness.run import SearchMeta

    manifest = yaml.safe_load((LEGACY_SNAPSHOTS / "greedy.yaml").read_text())
    meta = SearchMeta.model_validate({
        "schema_version": 3, "search_id": "s", "run_id": "r", "problem": "p", "problem_id": "p",
        "agent": "claude-code", "model": "sonnet", "metric": "m",
        "climber": "greedy", "climber_sha256": "ab" * 32, "climber_manifest": manifest,
        "climber_params": {"num_drafts": 1}, "tuner": None, "tuner_params": {"seed": 3},
    })
    assert meta.schema_version == 4 and (meta.climber, meta.climber_ref) == ("greedy", "greedy")
    block = meta.climber_spec
    assert block["policy"] == "hillclimb.modules.policies.greedy:GreedyPolicy"
    assert block["params"]["num_drafts"] == 1 and block["params"]["ensemble_top_k"] == 3  # the user's over the manifest's
    assert block["tuner"] == "random" and block["tuner_params"] == {"seed": 3}
    assert "description" not in block and "similarity" not in block
    ClimberSpec.model_validate(block)  # it is a block
    # a 0.6 record is left as it is
    current = SearchMeta.model_validate({**meta.model_dump(), "climber_ref": None})
    assert current.climber_spec == block and current.climber_ref is None


def test_the_engine_uses_the_tuner_the_block_names(config, tmp_path):
    """`climber.tuner` reaches the harness (it silently did not for a while:
    the rig-based tune tests never went through api's wiring)."""
    from hillclimb.harness.glue import build_tuner
    from hillclimb.modules.tuners.random_search import RandomTuner

    assert isinstance(build_tuner(config), RandomTuner)  # the default
    config.climber.tuner_params = {"seed": 7}
    assert build_tuner(config).params == {"seed": 7}
    (tmp_path / "fixed.py").write_text(
        "class Fixed:\n"
        "    def ask(self, space, history, *, higher_is_better, seed):\n        return {}\n"
    )
    config.climber.tuner = str(tmp_path / "fixed.py")  # a tuner is named like any module: a file works
    tuner = build_tuner(config)
    assert (type(tuner).__name__, tuner.name, tuner.params) == ("Fixed", "fixed", {"seed": 7})
    pytest.importorskip("optuna")
    from hillclimb.config import Config

    explicit = Config.model_validate({"search": {"policy": "greedy", "tuner": "optuna"}})  # the 0.3 spelling
    assert type(build_tuner(explicit)).__name__ == "OptunaTuner"


def test_execute_search_hands_the_harness_the_climbers_tuner(task, config, tmp_path, monkeypatch):
    """End to end through api: the Harness is constructed with the tuner,
    the operators and the prompts of the search's climber snapshot."""
    import hillclimb.harness.core as harness_module
    from hillclimb import api
    from hillclimb.harness.budget import BudgetManager
    from hillclimb.harness.run import RunMeta

    seen = {}
    original = harness_module.Harness

    class Spy(original):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(harness_module, "Harness", Spy)
    agent = FakeAgent()
    agent.queue(script=ok_script(0.6), notes="d\n")
    monkeypatch.setattr("hillclimb.api.get_agent", lambda *a, **k: agent)
    config.learning.enabled = False
    config.holdout.enabled = False
    config.budget.max_evaluations = 1
    config.climber.tuner_params = {"seed": 11}
    run_dir = api.create_run(config, RunMeta(run_id="r1", name="r1", kind="problem", target="t", problem_ids=[task.problem_id]))
    search_dir = api.create_search(config, task, run_dir, "r1", 600)

    outcome = api.execute_search(config, task, search_dir, BudgetManager(600, stop_margin_s=1), log=lambda *_: None)

    assert outcome.state == "done"
    assert seen["tuner"].params == {"seed": 11}
    assert seen["operators"].names() == ("draft", "debug", "improve", "ensemble")
    assert (search_dir / "climber" / "climber.yaml").is_file()  # the snapshot the engine loaded


# --- refs recorded before the package-layout move still resolve ---

def test_modernize_maps_only_what_moved():
    from hillclimb._moved import modernize

    assert modernize("hillclimb.policies.greedy:GreedyPolicy") == "hillclimb.modules.policies.greedy:GreedyPolicy"
    assert modernize("hillclimb.similarity_scores.builtin:ApiCalls") == "hillclimb.modules.similarity.builtin:ApiCalls"
    # 0.6 moved the gepa library out of integrations/: a search recorded before resumes
    assert modernize("hillclimb.integrations" + ".gepa.loop:GepaLoop") == "hillclimb.climbers.gepa.loop:GepaLoop"
    assert modernize("hillclimb.climbers.gepa.loop:GepaLoop") == "hillclimb.climbers.gepa.loop:GepaLoop"
    assert modernize("mypkg.policies.x:Y") == "mypkg.policies.x:Y"


def test_a_manifest_with_a_pre_move_ref_still_loads(tmp_path):
    from hillclimb.modules.policies.greedy import GreedyPolicy

    root = tmp_path / "old"
    root.mkdir()
    (root / "climber.yaml").write_text("name: old\npolicy: hillclimb.policies.greedy:GreedyPolicy\n")
    loop = load_climber(str(root)).build_loop()
    assert isinstance(loop.policy, GreedyPolicy)


def _legacy_snapshot(search_dir: Path, text: str) -> Path:
    """What 0.4/0.5 left in `<search_dir>/climber/`: the climber's manifest."""
    manifest = search_dir / "climber" / "climber.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(text)
    return manifest


@pytest.mark.parametrize("name", ["greedy", "openevolve", "gepa"])
def test_a_pre_06_snapshot_of_a_bundled_climber_still_loads(tmp_path, name):
    """Every search started before 0.6 holds the bundled climber's manifest
    (frozen here as 0.5.0 shipped it). It must still resume — read as it is,
    never rewritten."""
    if name == "openevolve":
        pytest.importorskip("openevolve")
    text = (LEGACY_SNAPSHOTS / f"{name}.yaml").read_text()
    manifest = _legacy_snapshot(tmp_path, text)
    climber = load_snapshot(tmp_path, name=name)
    assert climber.name == name
    loop = climber.build_loop(log=lambda *_: None)
    if name == "gepa":
        assert climber.is_loop and type(loop).__name__ == "GepaLoop"
        assert climber.operator_set().names() == ("gepa-reflect",)
        assert climber.holdout_timing == "after"  # the manifest asked for it
        assert loop.params.max_metric_calls == 50
    else:
        assert not climber.is_loop and type(loop.policy).__name__ == {"greedy": "GreedyPolicy", "openevolve": "OpenEvolvePolicy"}[name]
        assert climber.operator_set().names() == ("draft", "debug", "improve", "ensemble")
        assert climber.operator_set().get("draft").params == {"retrieval": True}
        assert loop.policy.params["num_drafts"] == 3  # the manifest's params
    assert manifest.read_text() == text


def test_a_search_snapshot_with_a_pre_move_ref_still_resumes(tmp_path):
    """A run folder written before the package-layout move names the policy
    by the old module path, and resume loads that snapshot."""
    from hillclimb.modules.policies.greedy import GreedyPolicy

    text = (LEGACY_SNAPSHOTS / "greedy.yaml").read_text().replace(
        "hillclimb.modules.policies.greedy:", "hillclimb.policies.greedy:"
    )
    assert "hillclimb.policies.greedy:GreedyPolicy" in text  # the old spelling is what we test
    _legacy_snapshot(tmp_path, text)
    loop = load_snapshot(tmp_path, name="greedy").build_loop()
    assert isinstance(loop.policy, GreedyPolicy)


def test_a_pre_06_one_file_snapshot_still_loads(tmp_path):
    (tmp_path / "climber").mkdir()
    (tmp_path / "climber" / "drafts_only.py").write_text(OPERATORS_PY + "\n" + POLICY_PY)
    climber = load_snapshot(tmp_path)
    assert climber.name == "drafts_only" and "cross" in climber.operator_set().names()


def test_memory_is_files_and_the_old_spelling_still_loads(tmp_path):
    """`memory: knowledge-graph` named the storage after its view; it now
    reads as `files`, and nothing on disk is rewritten to say so."""
    root = write_climber(tmp_path / "old", manifest="policy: policy.py\nmemory: knowledge-graph\n")
    climber = load_climber(str(root))
    assert climber.spec.memory == "files"
    assert climber.spec.model_dump()["memory"] == "files"
    assert "memory: knowledge-graph" in (root / "climber.yaml").read_text()  # left as written
    assert ClimberSpec.model_validate({"memory": "knowledge-graph"}).memory == "files"
    one_file = write_climber(tmp_path / "solo") / "policy.py"
    assert load_climber(str(one_file)).spec.memory == "files"  # the model default


def test_a_search_snapshot_with_the_old_memory_spelling_still_resumes(config, tmp_path):
    from hillclimb.harness.glue import effective_memory

    text = (LEGACY_SNAPSHOTS / "greedy.yaml").read_text().replace("memory: files", "memory: knowledge-graph")
    assert "memory: knowledge-graph" in text
    _legacy_snapshot(tmp_path, text)
    before = tree_sha256(tmp_path / "climber")
    assert load_snapshot(tmp_path, name="greedy").spec.memory == "files"
    assert effective_memory(config, tmp_path) == "files"
    assert tree_sha256(tmp_path / "climber") == before  # resume never rewrites the snapshot


def test_a_pre_06_directory_is_the_same_climber_as_its_block(tmp_path):
    """The manifest a 0.5 directory climber held reads as the block it is —
    same modules, same identity — which is how `climber show` migrates one."""
    root = write_climber(tmp_path / "crosser")
    legacy, block = load_climber(str(root)), resolve_climber(BLOCK, root)
    assert legacy.name == "crosser" and legacy.sha256 == block.sha256
    assert legacy.operator_set().names() == block.operator_set().names() == ("draft", "cross")
    assert legacy.operator_set().get("cross").params == {"style": "bold"}
    timed = write_climber(tmp_path / "timed", manifest="policy: policy.py\nholdout_timing: after\ndescription: x\nsimilarity: [api-calls]\n")
    assert load_climber(str(timed)).holdout_timing == "after"  # carried beside the block


GRAPH_PY = """
from hillclimb.sdk import GraphModule, GraphNode, KnowledgeGraph


class Notes(GraphModule):
    name = "notes"

    def build(self, knowledge_dir, previous=None):
        return KnowledgeGraph(nodes=[GraphNode(id="note:a", type="note", label="a")])
"""


def test_a_climber_brings_its_own_graph_module(task, config, tmp_path):
    """`graph:` names the module that indexes the memory: the built-in by
    name, or a local file — which the snapshot then carries."""
    from hillclimb import api
    from hillclimb.harness.run import RunMeta

    assert load_climber("greedy").graph_module().name == "knowledge-graph"  # the default
    root = write_climber(tmp_path / "mine", manifest="policy: policy.py\ngraph: graph.py\n")
    (root / "graph.py").write_text(GRAPH_PY)
    module = resolve_climber({"policy": "policy.py", "graph": "graph.py"}, root).graph_module()
    assert module.name == "notes" and module.key.startswith("graph.py#")
    assert [n.id for n in module.build(tmp_path).nodes] == ["note:a"]
    assert resolve_climber({"policy": "policy.py", "graph": "knowledge-graph"}, root).graph_module().name == "knowledge-graph"

    config.climber = ClimberSpec.model_validate({"policy": "policy.py", "graph": "graph.py"}).anchored(root)
    run_dir = api.create_run(config, RunMeta(run_id="r1", name="r1", kind="problem", target="t", problem_ids=[task.problem_id]))
    search_dir = api.create_search(config, task, run_dir, "r1", 600)
    assert (search_dir / "climber" / "files" / "graph.py").is_file()
    snapshot = load_snapshot(search_dir)
    assert snapshot.graph_module().name == "notes"
    assert snapshot.graph_module().key == module.key  # the copy is the same builder: no rebuild of graph.json
