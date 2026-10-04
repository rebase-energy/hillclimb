"""The Python front door: compose a climber from building blocks — names,
classes, instances — and run it; the composed climber IS the block a run
config takes, whenever its classes can be found again."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import hillclimb as hc
from hillclimb.agents.fake import FakeAgent
from hillclimb.climber import ClimberLoadError, NotPortableError, load_snapshot, resolve_climber
from hillclimb.harness.run import load_search_meta
from tests.conftest import ok_script

FACADES = ("policies", "selectors", "operators", "tuners", "memory", "loops")


def test_the_namespaces_are_lazy_windows_onto_the_modules():
    code = (
        "import sys, hillclimb, hillclimb.policies, hillclimb.selectors, hillclimb.operators, "
        "hillclimb.tuners, hillclimb.memory, hillclimb.loops; "
        "print(sorted(m for m in sys.modules if m.startswith('hillclimb.modules') or m.startswith('hillclimb.harness')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]", out  # nothing is loaded until a class is used
    for facade in FACADES:
        module = getattr(hc, facade)
        for name in module.__all__:
            if name in {"MapElites", "Optuna"}:
                continue  # their classes load everywhere; only constructing them needs the extra
            # `Policy` and `Selector` are the pre-0.7 names, kept as aliases for one release
            expected = {"Policy": "OperatorPolicy", "Selector": "SelectorPolicy"}.get(name, name)
            assert getattr(module, name).__name__ == expected
    assert hc.policies.Greedy is resolve_climber("greedy").brain.target
    assert set(hc.__all__) >= {"Climber", "run", "run_spec", *FACADES}


def test_a_composed_climber_is_the_block_a_run_config_takes():
    climber = hc.Climber(
        select=hc.selectors.Best(num_drafts=3, ensemble=False),
        policy=hc.policies.Greedy(tune_budget=4),
        operators=[hc.operators.Draft(retrieval=False), hc.operators.Debug(), hc.operators.Improve],
        tuner=hc.tuners.RandomSearch(seed=7),
        memory=hc.memory.FilesMemory(max_cards=1),
        name="mine",
    )
    block = climber.to_spec().block()
    assert block == {
        "name": "mine", "operator_policy": "greedy", "params": {"tune_budget": 4},
        "selector_policy": "best", "selector_params": {"num_drafts": 3, "ensemble": False},
        "operators": [{"draft": {"retrieval": False}}, "debug", "improve"],
        "tuner": "random", "tuner_params": {"seed": 7}, "memory": "files", "memory_params": {"max_cards": 1},
    }
    # the same climber, whichever way it was written down
    assert climber.portable and resolve_climber(block).sha256 == climber.sha256
    assert climber.operator_set().get("draft").params == {"retrieval": False}
    loop = climber.build_loop()
    assert loop.policy.selector.param("num_drafts") == 3 and loop.policy is not climber.build_loop().policy  # built fresh
    # names work too, a loop instead of a policy, and a policy that brings its selector
    assert hc.Climber(loop="gepa").is_loop and hc.Climber("openevolve").spec.selector_policy == "map-elites"
    best_first = hc.Climber(policy=hc.policies.Greedy(selector=hc.selectors.Best()))
    assert best_first.to_spec().selector_policy == "best"
    # a knob the class does not have fails where it is written
    with pytest.raises(TypeError, match="Greedy has no param 'num_draft'"):
        hc.policies.Greedy(num_draft=3)
    with pytest.raises(ClimberLoadError, match="not both"):
        hc.Climber(policy="greedy", loop="gepa")


MINE_PY = '''\
import hillclimb as hc
from hillclimb.sdk import Attempt, Operator


class Shake(Operator):
    name, kind, needs_target = "shake", "refine", True

    def prepare(self, ctx):
        return Attempt(prompt="Shake it." + self.params.get("how", ""), copy_parent=True)


class DraftOnce(hc.policies.Greedy):
    """One draft, then shake the best."""
    name = "draft-once"
    DEFAULTS = {"tune_budget": 0}

    def expand_action(self, state, selection, operator="improve"):
        return super().expand_action(state, selection, operator="shake")
'''


def _import(path: Path, name: str):
    """Import a file the way a user's script would be: a module with a file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runnable(task, config, monkeypatch):
    """`hc.run` on the synthetic problem with a scripted agent."""
    agent = FakeAgent()
    monkeypatch.setattr("hillclimb.api.load_problem", lambda target, config: task)
    monkeypatch.setattr("hillclimb.api.get_agent", lambda *a, **k: agent)
    config.holdout.enabled = False
    config.learning.enabled = False
    config.budget.max_evaluations = 2
    return agent


def test_classes_from_a_file_are_written_down_as_that_file(runnable, config, tmp_path):
    """Your own policy and operator, defined in a file: the composed climber
    names them by file, so the run is recorded, snapshotted and resumable
    like any block."""
    import hillclimb.sdk  # noqa: F401 — `hc.sdk` in the user's file

    path = tmp_path / "mine.py"
    path.write_text(MINE_PY)
    mine = _import(path, "my_script_module")
    climber = hc.Climber(
        select=hc.selectors.Best(num_drafts=1, ensemble=False), policy=mine.DraftOnce,
        operators=["draft", mine.Shake(how=" Gently.")],
    )
    assert climber.to_spec().block()["operator_policy"] == f"{path.resolve()}:DraftOnce"
    assert climber.to_spec().operators == ["draft", {f"{path.resolve()}:Shake": {"how": " Gently."}}]

    for score in (0.5, 0.7):
        runnable.queue(script=ok_script(score), notes="x\n")
    outcome = hc.run("anything", climber=climber, budget="10m", config=config, log=lambda *_: None)

    assert outcome.state == "done" and outcome.selected.val_score == 0.7
    assert [r.operator for r in runnable.requests] == ["draft", "shake"]
    assert runnable.requests[1].prompt.startswith("Shake it. Gently.")
    meta = load_search_meta(outcome.search_dir)
    assert meta.climber_portable and meta.budget_s == 600 and meta.climber_sha256 == climber.sha256
    snapshot = load_snapshot(outcome.search_dir)
    assert snapshot.sha256 == climber.sha256 and (outcome.search_dir / "climber" / "files" / "mine.py").is_file()
    written = yaml.safe_load((outcome.run_dir / "spec.yaml").read_text())["problems"][0]["climber"]
    assert written == climber.to_spec().block()  # the run's own spec: the full block


def test_a_class_that_exists_only_here_still_runs_but_is_not_portable(runnable, config, tmp_path, monkeypatch):
    """A class from a notebook cell (no file to find it in again) runs in
    this process. It cannot be written down — and everything that would
    need to rebuild it elsewhere refuses, saying why."""
    namespace: dict = {}
    exec(  # noqa: S102 — a class with no source file, as a notebook cell would define it
        "import hillclimb as hc\n"
        "class DraftsOnly(hc.policies.Greedy):\n"
        "    name = 'drafts-only'\n"
        "    def propose(self, state, selection):\n"
        "        return self.draft_action(state)\n",
        namespace,
    )
    climber = hc.Climber(policy=namespace["DraftsOnly"](tune_budget=0))
    assert not climber.portable and climber.spec.operator_policy == "live:DraftsOnly"
    with pytest.raises(NotPortableError, match="live:DraftsOnly exists only in this process"):
        climber.to_spec()
    with pytest.raises(NotPortableError):
        climber.write(tmp_path / "climber.yaml")
    monkeypatch.setattr("hillclimb.agents.require_agent_clis", lambda names: None)  # not the point here
    with pytest.raises(NotPortableError):  # a detached engine could not rebuild it
        hc.api.run_fleet("anything", config=config, climber=climber)

    for score in (0.5, 0.6):
        runnable.queue(script=ok_script(score), notes="x\n")
    outcome = hc.run("anything", climber=climber, config=config, log=lambda *_: None)
    assert outcome.state == "done" and [r.operator for r in runnable.requests] == ["draft", "draft"]
    meta = load_search_meta(outcome.search_dir)
    assert meta.climber_portable is False and meta.climber == "DraftsOnly"
    assert yaml.safe_load((outcome.search_dir / "climber" / "climber.yaml").read_text())["portable"] is False
    assert (outcome.search_dir / "climber" / "live" / "DraftsOnly.py").is_file()  # for the record
    with pytest.raises(NotPortableError, match="cannot be rebuilt"):
        load_snapshot(outcome.search_dir)

    from hillclimb.cli.run import resume
    import typer

    monkeypatch.setattr("hillclimb.cli.common.load_config", lambda **kw: config.model_copy(deep=True))
    monkeypatch.setattr("hillclimb.cli.common.require_sandbox", lambda *a, **k: None)
    from hillclimb.harness.status import read_status, write_status

    # a finished search is refused before its climber is looked at: stop it
    write_status(outcome.search_dir, read_status(outcome.search_dir).model_copy(update={"state": "stopped"}))
    with pytest.raises(typer.BadParameter, match="cannot be rebuilt"):
        resume(f"{outcome.run_dir.name}/{outcome.search_dir.name}", detach=False)


def test_run_spec_runs_every_entry_here_under_one_run(runnable, config, tmp_path):
    spec = tmp_path / "run.yaml"
    spec.write_text(yaml.safe_dump({
        "climber": {"operator_policy": "greedy", "params": {"num_drafts": 1, "ensemble": False, "tune_budget": 0}},
        "problems": [
            {"target": "a", "budget": "5m", "set": ["budget.max_evaluations=1"]},
            {"target": "b", "climber": {"params": {"num_drafts": 2, "ensemble": False}}, "set": ["budget.max_evaluations=2"]},
        ],
    }))
    for score in (0.5, 0.6, 0.7):
        runnable.queue(script=ok_script(score), notes="x\n")
    first, second = hc.run_spec(spec, config=config, log=lambda *_: None)
    assert first.run_dir == second.run_dir and first.search_dir != second.search_dir
    assert (first.state, second.state) == ("done", "done")
    metas = [load_search_meta(outcome.search_dir) for outcome in (first, second)]
    assert [m.budget_s for m in metas] == [300, 3600]  # the entry's budget, else the problem's
    assert [m.climber_spec["selector_params"]["num_drafts"] for m in metas] == [1, 2]  # the spec's default, the entry's own
    assert len(runnable.requests) == 3
    rerun = yaml.safe_load((first.run_dir / "spec.yaml").read_text())["problems"]
    assert [entry["climber"]["selector_params"]["num_drafts"] for entry in rerun] == [1, 2]


def test_a_script_that_runs_at_import_is_told_to_guard_it(tmp_path):
    """Composing from classes in your own script means a search imports that
    script again; one that starts a search at import time would start it
    forever. It is stopped with the fix."""
    script = tmp_path / "script.py"
    script.write_text(
        "import hillclimb as hc\n"
        "class Mine(hc.policies.Greedy):\n    pass\n"
        "hc.run('heilbronn-11', climber=hc.Climber(policy=Mine))\n"
    )
    with pytest.raises(ClimberLoadError, match='if __name__ == "__main__"'):
        resolve_climber(str(script)).build_loop()


def test_a_written_climber_is_a_block_file(tmp_path):
    path = hc.Climber(policy="greedy", params={"num_drafts": 2}, tuner="random").write(tmp_path / "climber.yaml")
    text = yaml.safe_load(path.read_text())
    # a schedule knob written under `params` is the selector policy's: it lands as `selector_params`
    assert text == {"climber": {"operator_policy": "greedy", "selector_params": {"num_drafts": 2}, "tuner": "random", "memory": "files"}}
    assert resolve_climber(text["climber"]).build_loop().policy.selector.param("num_drafts") == 2
