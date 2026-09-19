"""Climbers: a bundled name, a directory with a manifest, or one .py file —
one loader, one identity, operators and prompts scoped to the search."""

from __future__ import annotations

from pathlib import Path

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.climber import ClimberLoadError, bundled_climbers, load_climber, tree_sha256
from hillclimb.loop import PolicyLoop, SearchLoop
from hillclimb.policies.greedy import GreedyPolicy
from hillclimb.policy import Action
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
from hillclimb.sdk import Operator, Preparation, inspiration_filename

class Cross(Operator):
    name, role, needs_target = "cross", "combine", True
    def prepare(self, ctx):
        files = ", ".join(inspiration_filename(i) for i, _ in enumerate(ctx.inspirations, 1))
        return Preparation(prompt=ctx.render("cross", files=files, style=self.params.get("style", "plain")))
'''


def write_climber(root: Path, *, manifest: str | None = None) -> Path:
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
    assert loop.policy.param("num_drafts") == 1 and loop.policy.param("ensemble_top_k") == 3  # overlay, then manifest
    assert greedy.operator_set().names() == ("draft", "debug", "improve", "ensemble")
    assert greedy.operator_set().get("draft").params == {"retrieval": True}
    gepa = load_climber("gepa")
    assert gepa.is_loop and gepa.manifest.holdout_timing == "after"
    assert gepa.operator_set().names() == ("gepa-reflect",)


def test_a_directory_climber_runs_with_its_own_operator_and_prompts(task, config, tmp_path):
    climber = load_climber(str(write_climber(tmp_path / "crosser")))
    assert climber.name == "crosser" and not climber.is_loop and climber.lint_prompts() == []
    backend = FakeBackend()
    for score in (0.5, 0.7, 0.9):
        backend.queue(script=ok_script(score), notes="x\n")
    config.budget.max_evaluations = 3  # the policy holds after its cross; a hold never ends a search
    harness, journal, _ = make_harness(
        task, config, backend, operators=climber.operator_set(), prompts_dir=climber.prompts_dir,
    )

    selected = harness.execute(climber.build_loop())

    assert [r.operator for r in backend.requests] == ["draft", "draft", "cross"]
    assert selected.operator == "cross" and selected.role == "combine" and selected.val_score == 0.9
    cross_prompt = Path(selected.candidate_dir, "prompt.md").read_text()
    assert cross_prompt.startswith("Cross candidate_1.py, candidate_2.py in a bold way.")  # manifest params reached it
    assert "# Output contract" in cross_prompt  # no {{contract}} token: the harness appended it
    draft_prompt = Path(journal.get("c001").candidate_dir, "prompt.md").read_text()
    assert draft_prompt.startswith("MY OWN DRAFT PROMPT for accuracy.")  # shadows the built-in by name
    # scoped to this search: the built-in registry never heard of `cross`
    from hillclimb.operators import operator_names
    assert "cross" not in operator_names()


def test_an_operator_the_climber_did_not_list_is_refused(task, config, tmp_path):
    climber = load_climber(str(write_climber(tmp_path / "crosser")))
    backend = FakeBackend()
    backend.queue(script=ok_script(0.5), notes="x\n")
    harness, _journal, _ = make_harness(task, config, backend, operators=climber.operator_set())
    draft = harness.run(Action(operator="draft")).candidate
    refused = harness.run(Action(operator="improve", target_id=draft.candidate_id))
    assert refused.kind == "rejected" and "Unknown operator 'improve'" in refused.ticket.rejected


def test_a_one_file_climber_is_the_ten_line_story(task, config, tmp_path):
    path = tmp_path / "drafts_only.py"
    path.write_text(OPERATORS_PY + "\n" + POLICY_PY.replace("DraftsThenCross", "DraftsOnly"))
    climber = load_climber(str(path))
    assert (climber.name, climber.root, climber.is_loop) == ("drafts_only", None, False)
    # the default four, plus the Operator the file itself defines
    assert climber.operator_set().names() == ("draft", "debug", "improve", "ensemble", "cross")
    assert climber.build_loop(params={"drafts": 5}).policy.params == {"drafts": 5}
    # relative refs resolve from a base dir (the folder holding the hillclimb dir)
    assert load_climber("drafts_only.py", base_dir=tmp_path).sha256 == climber.sha256


def test_a_one_file_loop_is_recognised(tmp_path):
    path = tmp_path / "two_shots.py"
    path.write_text(
        "from hillclimb.sdk import Action, SearchLoop\n\n"
        "class TwoShots(SearchLoop):\n"
        "    name = 'two-shots'\n"
        "    def __init__(self, params=None, parallelism=1):\n"
        "        self.shots = int((params or {}).get('shots', 2)); self.parallelism = parallelism\n"
        "    def run(self, harness):\n"
        "        for _ in range(self.shots):\n"
        "            harness.run(Action(operator='draft'))\n"
    )
    climber = load_climber(str(path))
    loop = climber.build_loop(params={"shots": 3}, parallelism=4)
    assert climber.is_loop and isinstance(loop, SearchLoop) and (loop.shots, loop.parallelism) == (3, 4)


def test_identity_is_the_tree_and_ignores_caches(tmp_path):
    root = write_climber(tmp_path / "c")
    before = load_climber(str(root)).sha256
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "policy.cpython-312.pyc").write_bytes(b"junk")
    (root / ".DS_Store").write_bytes(b"junk")
    assert tree_sha256(root) == before
    (root / "prompts" / "cross.md").write_text("Cross {{files}} differently.\n")
    assert tree_sha256(root) != before  # a prompt edit is a different climber


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ("policy: policy.py\nloop: policy.py\n", "exactly one of `policy:`"),
        ("memory: none\n", "exactly one of `policy:`"),
        ("policy: policy.py\nrouting: {draft: {model: opus}}\n", "`routing` is reserved"),
        ("policy: policy.py\nmemory: sqlite\n", "memory"),
        ("policy: policy.py\nnum_drafts: 3\n", "num_drafts"),  # a typo'd top-level key, not silently ignored
        ("policy: nope.py\n", "nope.py is not a file inside the climber's directory"),
        ("policy: ../evil.py\n", "is not a file inside the climber's directory"),
        ("policy: greedy\n", "name a file"),
        ("policy: operators.py\n", "exactly one policy class"),
        ("policy: policy.py\noperators: [operators.py:Nope]\n", "defines no Nope"),
    ],
)
def test_a_bad_manifest_names_the_file_and_the_fix(tmp_path, manifest, message):
    root = write_climber(tmp_path / "bad", manifest=manifest)
    with pytest.raises(ClimberLoadError, match=message) as exc:
        climber = load_climber(str(root))
        climber.build_loop()
        climber.operator_set()
    assert "bad" in str(exc.value)


def test_unknown_reference_lists_what_exists(tmp_path):
    with pytest.raises(ClimberLoadError, match="bundled: gepa, greedy, openevolve"):
        load_climber("nope")
    with pytest.raises(ClimberLoadError, match="holds no climber.yaml"):
        load_climber(str(tmp_path))


def test_a_climber_may_not_shadow_the_contract(tmp_path):
    root = write_climber(tmp_path / "sneaky")
    (root / "prompts" / "contract_verifier.md").write_text("Anything goes.\n")
    (root / "prompts" / "improve.md").write_text("Improve it. {{made_up_token}}\n")
    problems = load_climber(str(root)).lint_prompts()
    assert any("contract_verifier.md: harness-owned" in p for p in problems)
    assert any("improve.md" in p and "made_up_token" in p for p in problems)


def test_an_import_error_in_the_authors_file_is_reported_with_its_path(tmp_path):
    root = write_climber(tmp_path / "broken")
    (root / "policy.py").write_text("import not_a_real_module\n")
    with pytest.raises(ClimberLoadError, match="policy.py failed to import: ModuleNotFoundError"):
        load_climber(str(root)).build_loop()


# --- run folders: what is recorded, and what older folders still load as ---


def test_a_run_folder_written_before_climbers_still_loads(tmp_path):
    """Schema v2 named a `policy`. Same thing, older word: every view keeps
    showing those runs (file store and sqlite alike — the mapping lives on
    the model)."""
    import yaml

    from hillclimb.run import SearchMeta, load_search_meta

    search_dir = tmp_path / "runs" / "r" / "searches" / "gefcom-solar"
    search_dir.mkdir(parents=True)
    (search_dir / "search.yaml").write_text(yaml.safe_dump({
        "schema_version": 2, "search_id": "gefcom-solar", "run_id": "r", "problem": "p",
        "problem_id": "gefcom-solar", "backend": "claude-code", "model": "sonnet",
        "policy": "hillclimb/policies/drafts_only.py", "policy_params": {"num_drafts": 1},
        "policy_sha256": "ab" * 32, "tuner": "optuna", "tuner_params": {"seed": 3},
        "metric": "pinball", "higher_is_better": False,
        "templates_sha256": "cd" * 32, "templates_overridden": ["improve"],
    }))
    meta = load_search_meta(search_dir)
    assert meta is not None and meta.schema_version == 3
    assert meta.climber == "hillclimb/policies/drafts_only.py"
    assert meta.climber_params == {"num_drafts": 1} and meta.climber_sha256 == "ab" * 32
    assert (meta.tuner, meta.tuner_params) == ("optuna", {"seed": 3})
    assert meta.climber_manifest == {} and meta.hillclimb_version is None  # unknown for an old run
    # and the same record as the sqlite store holds it
    assert SearchMeta.model_validate_json(meta.model_dump_json()).climber == meta.climber
    # a version this build has never heard of stays invisible rather than half-read
    (search_dir / "search.yaml").write_text(yaml.safe_dump({"schema_version": 9, "search_id": "x"}))
    assert load_search_meta(search_dir) is None


def test_the_engine_uses_the_tuner_the_user_or_the_manifest_names(config, tmp_path):
    """`search.tuner` reaches the harness (it silently did not for a while:
    the rig-based tune tests never went through api's wiring)."""
    from hillclimb.search_strategy import build_tuner
    from hillclimb.tuners.random_search import RandomTuner

    assert isinstance(build_tuner(config), RandomTuner)  # greedy's manifest says random
    root = tmp_path / "with-tuner"
    root.mkdir()
    (root / "climber.yaml").write_text(
        "policy: hillclimb.policies.greedy:GreedyPolicy\ntuner: random\ntuner_params: {seed: 7}\n"
    )
    config.climber.ref = str(root)
    assert build_tuner(config).params == {"seed": 7}  # the manifest's params
    config.climber.tuner_params = {"seed": 9}
    assert build_tuner(config).params == {"seed": 9}  # the user's lay over them
    pytest.importorskip("optuna")
    from hillclimb.config import Config

    explicit = Config.model_validate({"search": {"policy": str(root), "tuner": "optuna"}})
    assert type(build_tuner(explicit)).__name__ == "OptunaTuner"  # the user named one: it wins


def test_execute_search_hands_the_harness_the_climbers_tuner(task, config, tmp_path, monkeypatch):
    """End to end through api: the Harness is constructed with the tuner,
    the operators and the prompts of the search's climber snapshot."""
    import hillclimb.harness as harness_module
    from hillclimb import api
    from hillclimb.budget import BudgetManager
    from hillclimb.run import RunMeta

    seen = {}
    original = harness_module.Harness

    class Spy(original):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(harness_module, "Harness", Spy)
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="d\n")
    monkeypatch.setattr("hillclimb.api.get_backend", lambda *a, **k: backend)
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
