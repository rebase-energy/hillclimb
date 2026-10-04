"""Public programmatic API: run_search creates the Run/Search layout and
returns a SearchOutcome (dummy agent, no agent spend)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hillclimb.api import run_search
from hillclimb.harness.run import load_run_meta, load_search_meta


@pytest.mark.slow
def test_run_search_end_to_end(config):
    config.paths.problems_dir = Path("problems")  # repo problems (circle-packing)
    logs: list[str] = []

    outcome = run_search(
        "circle-packing",
        budget_s=10,
        name="api-test",
        config=config,
        agent="dummy",
        holdout=False,
        log=logs.append,
    )

    assert outcome.state == "done"
    assert outcome.search_dir.name == "circle-packing"
    assert load_run_meta(outcome.run_dir) is not None
    assert load_search_meta(outcome.search_dir).budget_s == 10
    assert (outcome.search_dir / "journal.jsonl").exists()
    # declared floor (`baseline: 0.5`): c000 is scored but ships no files, so
    # best/ fills only once an agent candidate lands
    assert any("baseline: 0.5 (declared)" in line for line in logs)
    assert any("Search " in line for line in logs)
    assert outcome.selected is None or outcome.selected.val_score is not None


def test_runtime_packages_evaluator_kind_and_requirements_file(tmp_path):
    from hillclimb.runtime import runtime_packages

    assert runtime_packages("evaluator") == runtime_packages("csv")
    req = tmp_path / "requirements.txt"
    req.write_text("# solver deps\nnumpy\nnetworkx>=3\n")
    assert runtime_packages("evaluator", requirements_file=req) == ["numpy", "networkx>=3"]


def test_default_venv_python_keys_on_requirements_content(config, tmp_path):
    from hillclimb.api import default_venv_python

    req = tmp_path / "r.txt"
    req.write_text("numpy\n")
    first = default_venv_python(config, "evaluator", requirements=req)
    assert "problem-" in str(first)
    req.write_text("numpy\npandas\n")
    changed = default_venv_python(config, "evaluator", requirements=req)
    assert changed != first
    # identical content in a different file shares the venv
    twin = tmp_path / "r2.txt"
    twin.write_text("numpy\npandas\n")
    assert default_venv_python(config, "evaluator", requirements=twin) == changed


def test_ensure_runtime_venv_requirements_short_circuit(config, tmp_path, monkeypatch):
    from hillclimb import api

    req = tmp_path / "r.txt"
    req.write_text("numpy\n")
    python = tmp_path / "venvs" / "problem-abc" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(api, "default_venv_python", lambda *a, **k: python)

    def boom(*a, **k):
        raise AssertionError("existing venv must not trigger a build")

    monkeypatch.setattr(api.subprocess, "run", boom)
    assert api.ensure_runtime_venv(config, "evaluator", requirements=req) == python


# --- evaluator kind: dispatch + baseline ---

EVALUATE_PY = """\
import json, os, sys
sys.path.insert(0, os.getcwd())
import solution
score = float(solution.answer())
json.dump({"split": "validation", "score": score,
           "report": {"version": 1, "split": "validation", "overall": {"score": score}}},
          open("eval_result.json", "w"))
print(f"val_score: {score}")
"""


def make_evaluator_problem(tmp_path, **overrides):
    from hillclimb.problem import ProblemSpec

    problem_dir = tmp_path / "eval-problem"
    problem_dir.mkdir(exist_ok=True)
    (problem_dir / "evaluate.py").write_text(EVALUATE_PY)
    fields = dict(
        problem_id="eval-problem",
        problem_dir=problem_dir,
        data_dir=problem_dir,
        description="score the answer",
        metric_name="score",
        higher_is_better=True,
        time_budget_s=600,
        verifier_cmd=["{python}", "problem/evaluate.py"],
    )
    fields.update(overrides)
    return ProblemSpec(**fields)


def test_build_executor_dispatches_command_executor(config, tmp_path, monkeypatch):
    import sys

    from hillclimb import api
    from hillclimb.harness.executor import CommandExecutor

    monkeypatch.setattr(api, "ensure_runtime_venv", lambda *a, **k: Path(sys.executable))
    executor = api.build_executor(config, make_evaluator_problem(tmp_path), log=lambda *_: None)
    assert isinstance(executor, CommandExecutor)


def test_build_holdout_scorer_for_evaluator(config, tmp_path, monkeypatch):
    import sys

    from hillclimb import api
    from hillclimb.harness.executor import CommandHoldoutScorer

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.setattr(api, "ensure_runtime_venv", lambda *a, **k: Path(sys.executable))

    no_holdout = make_evaluator_problem(tmp_path)
    assert api.build_holdout_scorer(config, no_holdout, tmp_path / "s") is None

    # the HF-credential preflight is opt-in per problem (emflow sets it)
    with_holdout = make_evaluator_problem(
        tmp_path, holdout_cmd=["{python}", "problem/evaluate.py", "--holdout"]
    )
    scorer = api.build_holdout_scorer(config, with_holdout, tmp_path / "s")
    assert isinstance(scorer, CommandHoldoutScorer)


def test_evaluator_baseline_placeholder_and_scored(config, tmp_path):
    import sys

    from hillclimb.harness.baseline import write_baseline
    from hillclimb.harness.executor import CommandExecutor
    from hillclimb.harness.dirs import create_search_dir

    problem = make_evaluator_problem(tmp_path)
    search_dir = create_search_dir(tmp_path / "runs" / "r1", "eval-problem")
    placeholder = write_baseline(problem, search_dir)  # no baseline shipped
    assert placeholder.candidate_id == "c000"
    assert not placeholder.trials
    assert "unscored placeholder" in placeholder.summary

    baseline_file = problem.problem_dir / "baseline.py"
    baseline_file.write_text("def answer():\n    return 0.25\n")
    problem = make_evaluator_problem(tmp_path, baseline_text=baseline_file.read_text())
    search_dir = create_search_dir(tmp_path / "runs" / "r2", "eval-problem")
    from hillclimb.harness.evaluation import CandidateEvaluator

    executor = CommandExecutor(Path(sys.executable), problem.verifier_cmd)
    evaluator = CandidateEvaluator(executor=executor, problem=problem, config=config)
    scored = write_baseline(problem, search_dir, evaluator=evaluator, timeout_s=60)
    assert scored.val_score == 0.25
    assert scored.is_best
    assert (search_dir / "best" / "solution.py").exists()


def test_fleet_argv_carries_every_engine_option(tmp_path):
    from hillclimb.api import fleet_argv

    run_dir = tmp_path / "runs" / "20260909-120000-solar"
    argv = fleet_argv(
        "emflow://gefcom2014:solar", run_dir, "solar",
        budget=900, agent="dummy", model="sonnet", climber="gepa", parallel_agents=2,
        n_replicates=3, holdout=False, learning=False, seed_from=tmp_path / "seed.py",
        knowledge_context_file=tmp_path / "kc.md", overrides=["search.num_drafts=2"],
    )
    assert argv[:5] == ["emflow://gefcom2014:solar", "--run-id", run_dir.name, "--run-name", "solar"]
    assert argv[argv.index("--budget") + 1] == "900s"
    assert fleet_argv("cp", run_dir, "cp", budget="2h")[-1] == "2h"
    assert argv[argv.index("--climber") + 1] == "gepa"
    assert argv[argv.index("--parallel-agents") + 1] == "2"
    assert "--no-holdout" in argv and "--no-learning" in argv
    assert argv[argv.index("--seed-from") + 1] == str(tmp_path / "seed.py")
    assert argv[argv.index("--knowledge-context-file") + 1] == str(tmp_path / "kc.md")
    assert argv[-2:] == ["--set", "search.num_drafts=2"]
    # defaults add nothing beyond the run identity
    assert fleet_argv("circle-packing", run_dir, "cp") == ["circle-packing", "--run-id", run_dir.name, "--run-name", "cp"]


def test_a_climber_block_crosses_to_a_child_engine_as_the_first_set_pair(tmp_path):
    """A child engine gets nothing but argv. A name travels as `--climber`;
    a block as `--set climber=<json>`, ahead of the other pairs (they may
    edit its fields) — and the child ends up with exactly that block."""
    from hillclimb.api import climber_argv, fleet_argv
    from hillclimb.config import Config, parse_set_overrides

    assert climber_argv(None) == [] and climber_argv("gepa") == ["--climber", "gepa"]
    block = {"operator_policy": str(tmp_path / "mine.py"), "params": {"num_drafts": 2, "note": "a: b, c"},
             "operators": ["draft", {"improve": {"ablation": False}}], "tuner": "optuna", "memory": "none"}
    run_dir = tmp_path / "runs" / "r1"
    argv = fleet_argv("cp", run_dir, "cp", climber=block, overrides=["climber.params.num_drafts=5", "model=opus"])
    sets = [argv[i + 1] for i, word in enumerate(argv) if word == "--set"]
    assert sets[0].startswith("climber={") and sets[1:] == ["climber.params.num_drafts=5", "model=opus"]
    assert "--climber" not in argv

    child = Config()
    child.apply_overrides(parse_set_overrides(sets))  # what `hillclimb run --set ...` does in the child
    expected = Config.model_validate({"climber": {**block, "params": {**block["params"], "num_drafts": 5}}})
    assert child.climber.block() == expected.climber.block() and child.model == "opus"


def test_run_fleet_spawns_one_engine_per_search(config, monkeypatch):
    """run_fleet writes run.yaml once, builds the venv once, and starts N
    detached engines with identical argv; the handle reaps them."""
    import hillclimb.api as api

    venv_calls: list[str] = []
    monkeypatch.setattr(api, "ensure_runtime_venv", lambda cfg, kind, log=print, requirements=None: venv_calls.append(kind))
    spawned: list[tuple[int, str, list[str]]] = []

    class FakeProc:
        def __init__(self, pid):
            self.pid = pid
            self._polls = 0

        def poll(self):
            self._polls += 1
            return None if self._polls < 2 else 0

    def fake_spawn(cfg, run_dir, index, slug, argv):
        spawned.append((index, slug, argv))
        return FakeProc(1000 + index), run_dir / "logs" / f"{index:02d}-{slug}.log"

    monkeypatch.setattr(api, "spawn_search_proc", fake_spawn)

    fleet = api.run_fleet("circle-packing", config=config, parallel_searches=3, budget="1m", agent="dummy")

    assert venv_calls == ["evaluator"] or len(venv_calls) == 1
    assert load_run_meta(fleet.run_dir) is not None
    assert [index for index, _, _ in spawned] == [1, 2, 3]
    assert len({tuple(argv) for _, _, argv in spawned}) == 1
    assert spawned[0][2][1:3] == ["--run-id", fleet.run_id]
    assert fleet.wait(poll_s=0) == {1001: 0, 1002: 0, 1003: 0}
    assert fleet.alive() == []


def test_mixed_fleet_names_arms_after_policies_and_repeats_them():
    from hillclimb.api import FleetEngine, mixed_fleet

    engines = mixed_fleet(["greedy", "openevolve", "gepa"], experiment_overrides={"gepa": ["search.parallel_agents=1"]})
    assert engines == [
        FleetEngine(experiment="greedy", climber="greedy"),
        FleetEngine(experiment="openevolve", climber="openevolve"),
        FleetEngine(experiment="gepa", climber="gepa", overrides=("search.parallel_agents=1",)),
    ]
    # a repeated policy is a second experiment; repeats clone every experiment, repeat-major
    twice = mixed_fleet(["greedy", "greedy"], repeats=2)
    assert [(e.experiment, e.repeat) for e in twice] == [("greedy", 1), ("greedy-2", 1), ("greedy", 2), ("greedy-2", 2)]
    with pytest.raises(ValueError, match="unknown experiment"):
        mixed_fleet(["greedy", "gepa"], experiment_overrides={"openevolve": ["x=1"]})
    with pytest.raises(ValueError, match="at least one climber"):
        mixed_fleet([])


def test_fleet_argv_tags_an_arm(tmp_path):
    from hillclimb.api import fleet_argv

    run_dir = tmp_path / "runs" / "20260910-120000-cp"
    argv = fleet_argv("circle-packing", run_dir, "cp", study="exp", experiment="gepa", repeat=2, climber="gepa")
    assert argv[:9] == [
        "circle-packing", "--run-id", run_dir.name, "--run-name", "cp", "--study", "exp", "--experiment", "gepa",
    ]
    assert argv[argv.index("--repeat") + 1] == "2"
    assert "--repeat" not in fleet_argv("circle-packing", run_dir, "cp", study="exp", experiment="greedy")


def test_run_fleet_mixed_engines_get_their_own_policy_and_overrides(config, monkeypatch):
    """A mixed fleet spawns one engine per FleetEngine: the fleet-wide
    arguments, then the engine's policy and its overrides after the shared
    ones, tagged as an experiment of the run so `experiment report` compares them."""
    import hillclimb.api as api

    monkeypatch.setattr("hillclimb.agents.require_agent_clis", lambda names: None)  # no engine runs
    monkeypatch.setattr(api, "ensure_runtime_venv", lambda cfg, kind, log=print, requirements=None: None)
    spawned: list[tuple[int, str, list[str]]] = []

    class FakeProc:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return 0

    def fake_spawn(cfg, run_dir, index, slug, argv):
        spawned.append((index, slug, argv))
        return FakeProc(1000 + index), run_dir / "logs" / f"{index:02d}-{slug}.log"

    monkeypatch.setattr(api, "spawn_search_proc", fake_spawn)
    engines = api.mixed_fleet(
        ["greedy", "gepa"], experiment_overrides={"gepa": ["search.parallel_agents=1"]}
    )

    fleet = api.run_fleet(
        "circle-packing", config=config, parallel_searches=7, budget="1m", agent="dummy",
        parallel_agents=3, overrides=["learning.enabled=false"], engines=engines,
    )

    assert [(index, slug) for index, slug, _ in spawned] == [(1, "circle-packing-greedy"), (2, "circle-packing-gepa")]
    greedy, gepa = (argv for _, _, argv in spawned)
    for argv in (greedy, gepa):
        assert argv[1:3] == ["--run-id", fleet.run_id]
        assert argv[argv.index("--study") + 1] == fleet.run_id  # default study: the run id
        assert argv[argv.index("--parallel-agents") + 1] == "3"
        assert "--repeat" not in argv
    assert greedy[greedy.index("--experiment") + 1] == "greedy" and greedy[greedy.index("--climber") + 1] == "greedy"
    assert gepa[gepa.index("--experiment") + 1] == "gepa" and gepa[gepa.index("--climber") + 1] == "gepa"
    sets = [argv[i + 1] for i, tok in enumerate(gepa) if tok == "--set"]
    assert sets == ["learning.enabled=false", "search.parallel_agents=1"]  # the experiment's override last, so it wins
    assert [argv[i + 1] for i, tok in enumerate(greedy) if tok == "--set"] == ["learning.enabled=false"]

    api.run_fleet(
        "circle-packing", config=config, engines=api.mixed_fleet(["greedy", "openevolve"], repeats=2),
        study="three-way",
    )
    tail = spawned[2:]
    assert [slug for _, slug, _ in tail] == [
        "circle-packing-greedy-r1", "circle-packing-openevolve-r1",
        "circle-packing-greedy-r2", "circle-packing-openevolve-r2",
    ]
    assert all(argv[argv.index("--study") + 1] == "three-way" for _, _, argv in tail)
    assert tail[-1][2][tail[-1][2].index("--repeat") + 1] == "2"
    with pytest.raises(ValueError, match="at least one FleetEngine"):
        api.run_fleet("circle-packing", config=config, engines=[])
