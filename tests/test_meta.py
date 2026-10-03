"""The meta-problem kit: a problem whose solution.py is a one-file climber.

`solution_kind: climber` in problem.yaml selects the climber contract and
makes a search on it run its climber in the improver role; the verifier is
`hillclimb grade`, which runs inner searches with the candidate as
their climber and scores the gap they closed. Nothing here is documented
yet (a later launch) — these tests are the contract until then.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from hillclimb import api, meta
from hillclimb.agents.fake import FakeAgent
from hillclimb.cli import app
from hillclimb.config import Config
from hillclimb.harness.executor import verifier_env
from hillclimb.harness.run import RunMeta, SearchMeta, load_search_meta
from hillclimb.problem import load_problem
from tests.conftest import ok_script
from tests.test_parallel_search import make_searcher
from tests.test_prompt_golden import _check, collect_prompts

REPO = Path(__file__).resolve().parents[1]
META_PROBLEM = REPO / "problems" / "meta-heilbronn"
GREEDY_SOURCE = REPO / "src" / "hillclimb" / "modules" / "policies" / "greedy.py"


def _write_problem(root: Path, name: str, extra_yaml: str = "") -> Path:
    problem = root / name
    problem.mkdir(parents=True)
    (problem / "problem.yaml").write_text(
        f"problem_id: {name}\nmetric: score\nhigher_is_better: true\ndescription: description.md\n{extra_yaml}"
    )
    (problem / "description.md").write_text(name)
    verifier = problem / "verifier.sh"
    verifier.write_text('#!/bin/sh\necho 1 > "$HILLCLIMB_RESULT"\n')
    verifier.chmod(0o755)
    return problem


# --- the problem side: solution_kind ---


def test_solution_kind_defaults_to_program_with_the_verifier_contract(config, tmp_path):
    problem = _write_problem(tmp_path, "plain")
    spec = load_problem(str(problem), config)
    assert spec.solution_kind == "program"
    assert spec.contract_template == "contract_verifier"


def test_solution_kind_climber_selects_the_climber_contract(config, tmp_path):
    problem = _write_problem(tmp_path, "meta", "solution_kind: climber\n")
    spec = load_problem(str(problem), config)
    assert spec.solution_kind == "climber"
    assert spec.contract_template == "contract_climber"


def test_unknown_solution_kind_is_refused(config, tmp_path):
    problem = _write_problem(tmp_path, "odd", "solution_kind: agent\n")
    with pytest.raises(ValueError, match="solution_kind must be one of program, climber"):
        load_problem(str(problem), config)


# --- the search record: the role is derived, never declared ---


def test_search_meta_role_defaults_to_solver_for_every_existing_record():
    meta_record = SearchMeta(search_id="s", run_id="r", problem="p", problem_id="p", agent="dummy",
                             model="m", metric="score")
    assert meta_record.role == "solver"


@pytest.mark.parametrize("kind, role", [("program", "solver"), ("climber", "improver")])
def test_create_search_derives_the_role_from_the_problem(task, config, kind, role):
    task.solution_kind = kind
    run_dir = api.create_run(config, RunMeta(run_id="r1", name="r1", kind="problem", target="t",
                                             problem_ids=[task.problem_id]))
    search_dir = api.create_search(config, task, run_dir, "r1", 600)
    assert load_search_meta(search_dir).role == role


def test_watch_labels_only_the_improver(task, config):
    from contextlib import closing

    from hillclimb.harness.store import open_store
    from hillclimb.tui.watch import scan_searches

    task.solution_kind = "climber"
    run_dir = api.create_run(config, RunMeta(run_id="r1", name="r1", kind="problem", target="t",
                                             problem_ids=[task.problem_id]))
    api.create_search(config, task, run_dir, "r1", 600)
    task.solution_kind = "program"
    api.create_search(config, task, run_dir, "r1", 600)
    with closing(open_store(config)) as store:
        labels = sorted(row.climber for row in scan_searches(store, "r1"))
    assert labels == ["greedy", "greedy (improver)"]


# --- the verifier contract: what a meta verifier reads ---


def test_verifier_env_names_the_engine_interpreter(tmp_path):
    env = verifier_env(tmp_path / "python", tmp_path / "solution.py", tmp_path / "result.json", "validation")
    assert env["HILLCLIMB_ENGINE_PYTHON"] == sys.executable


def test_build_executor_pins_the_hillclimb_dir_only_when_there_is_one(config, task, tmp_path):
    config.paths.runtime_python = Path(sys.executable)
    assert "HILLCLIMB_DIR" not in api.build_executor(config, task, log=lambda *_: None).env_extra
    config.hillclimb_dir = tmp_path / "hillclimb"
    executor = api.build_executor(config, task, log=lambda *_: None)
    assert executor.env_extra["HILLCLIMB_DIR"] == str(tmp_path / "hillclimb")


# --- the score: gap closed ---


def test_gap_closed_is_direction_aware_and_clamped_below():
    assert meta.gap_closed(0.5, 0.0, 1.0, True) == pytest.approx(0.5)
    assert meta.gap_closed(110.0, 120.0, 100.0, False) == pytest.approx(0.5)
    assert meta.gap_closed(-1.0, 0.0, 1.0, True) == 0.0  # worse than the floor
    assert meta.gap_closed(1.5, 0.0, 1.0, True) == pytest.approx(1.5)  # beat the best known
    assert meta.gap_closed(None, 0.0, 1.0, True) == 0.0  # nothing scored


def test_gap_closed_with_a_degenerate_gap_reports_whether_the_floor_was_reached():
    assert meta.gap_closed(3.0, 3.0, 3.0, True) == 1.0
    assert meta.gap_closed(2.0, 3.0, 3.0, True) == 0.0


def test_aggregate_is_median_over_repeats_then_mean_over_problems():
    assert meta.aggregate({"a": [0.0, 1.0, 0.2], "b": [0.6]}) == pytest.approx((0.2 + 0.6) / 2)
    assert meta.aggregate({}) == 0.0


def test_aggregate_combines_problems_the_way_the_spec_names():
    per_problem = {"a": [0.0, 1.0, 0.2], "b": [0.6], "c": [4.0]}
    assert meta.aggregate(per_problem, meta.AGGREGATES["median"]) == pytest.approx(0.6)
    assert meta.aggregate(per_problem, meta.AGGREGATES["min"]) == pytest.approx(0.2)
    assert meta.aggregate(per_problem, meta.AGGREGATES["mean"]) == pytest.approx((0.2 + 0.6 + 4.0) / 3)


def test_solved_and_raw_scores():
    assert meta.solved(1.0, 0.0, 1.0, True) == 1.0 and meta.solved(0.9, 0.0, 1.0, True) == 0.0
    assert meta.solved(99.0, 120.0, 100.0, False) == 1.0 and meta.solved(101.0, 120.0, 100.0, False) == 0.0
    assert meta.solved(None, 0.0, 1.0, True) == 0.0
    assert meta.raw(0.7, 0.1, 1.0, True) == 0.7
    assert meta.raw(None, 0.1, 1.0, True) == 0.1  # nothing scored: the floor


def test_grade_spec_names_its_score_and_aggregate(tmp_path):
    path = tmp_path / "grade.yaml"
    path.write_text("problems: [a]\n")
    spec = meta.load_grade_spec(path)
    assert (spec.score, spec.aggregate) == ("gap-closed", "mean")  # the defaults
    assert spec.score_function() is meta.gap_closed
    path.write_text("problems: [a]\nscore: solved\naggregate: median\n")
    spec = meta.load_grade_spec(path)
    assert spec.score_function() is meta.solved and spec.aggregate_function() is meta.AGGREGATES["median"]


def test_grade_spec_takes_the_users_own_functions(tmp_path):
    (tmp_path / "mine.py").write_text(
        "def halfway(best, floor, target, higher_is_better):\n"
        "    return 1.0 if best is not None and best >= (floor + target) / 2 else 0.0\n"
        "def worst_two(scores):\n"
        "    return sum(sorted(scores)[:2]) / 2\n"
        "NOT_A_FUNCTION = 3\n"
    )
    path = tmp_path / "grade.yaml"
    path.write_text("problems: [a]\nscore: mine.py:halfway\naggregate: mine.py:worst_two\n")
    spec = meta.load_grade_spec(path)  # a file ref resolves beside the spec
    assert spec.score_function()(0.6, 0.0, 1.0, True) == 1.0
    assert spec.aggregate_function()([1.0, 0.0, 0.5]) == pytest.approx(0.25)
    assert meta.GradeSpec.model_validate({"problems": ["a"], "aggregate": "statistics:median"}).aggregate_function()(
        [1.0, 2.0, 9.0]
    ) == 2.0

    for bad, why in [
        ("nope", "not one of gap-closed, solved, raw"),
        ("mine.py:missing", "missing"),
        ("gone.py:f", "gone.py"),
        ("mine.py:NOT_A_FUNCTION", "is not a function"),
    ]:
        path.write_text(f"problems: [a]\nscore: {bad}\n")
        with pytest.raises(meta.MetaError, match=why):
            meta.load_grade_spec(path).score_function()


def test_the_holdout_split_grades_on_the_outer_holdout(tmp_path):
    path = tmp_path / "grade.yaml"
    path.write_text("problems: [a, b]\nouter_holdout:\n  - c\n  - {problem: d, target: 2.0}\nbudget: 90s\n")
    spec = meta.load_grade_spec(path)
    assert [p.problem for p in spec.problems_for("validation")] == ["a", "b"]
    assert [p.problem for p in spec.problems_for("holdout")] == ["c", "d"]
    assert spec.outer_holdout[1].target == 2.0
    assert spec.required_exec_s() == 2 * (90 + 60) and spec.required_exec_s("holdout") == 2 * (90 + 60)

    bare = meta.GradeSpec.model_validate({"problems": ["a"]})
    with pytest.raises(meta.MetaError, match="outer_holdout"):
        bare.problems_for("holdout")
    with pytest.raises(meta.MetaError, match="unknown split"):
        bare.problems_for("test")


# --- the spec ---


def test_grade_spec_accepts_bare_names_and_targets(tmp_path):
    path = tmp_path / "grade.yaml"
    path.write_text("problems:\n  - a\n  - {problem: b, target: 2.0, floor: 1.0}\nbudget: 90s\nrepeats: 2\n")
    spec = meta.load_grade_spec(path)
    assert [p.problem for p in spec.problems] == ["a", "b"]
    assert spec.problems[1].target == 2.0 and spec.problems[1].floor == 1.0
    assert spec.budget_s == 90
    assert spec.instance_key("a", 1) == "a/r1"
    assert spec.required_exec_s() == 2 * 2 * (90 + 60)


def test_grade_spec_refuses_unknown_keys_and_no_problems(tmp_path):
    path = tmp_path / "grade.yaml"
    path.write_text("problems: []\n")
    with pytest.raises(meta.MetaError, match="at least one inner problem"):
        meta.load_grade_spec(path)
    path.write_text("problems: [a]\nmodel: opus\n")
    with pytest.raises(meta.MetaError, match="model"):
        meta.load_grade_spec(path)


# --- version-one permissions: what an improver's file may import ---


def test_the_bundled_greedy_policy_passes_the_import_rule():
    assert meta.check_climber_source(GREEDY_SOURCE) == []


def test_import_rule_names_harness_thirdparty_and_relative_imports(tmp_path):
    path = tmp_path / "solution.py"
    path.write_text(
        "import os\nfrom hillclimb.sdk import Action\nfrom hillclimb.harness.core import Harness\n"
        "import numpy as np\nfrom . import helper\n"
    )
    problems = meta.check_climber_source(path)
    assert len(problems) == 3
    assert "solution.py:3: imports hillclimb.harness.core" in problems[0]
    assert "solution.py:4: imports numpy" in problems[1]
    assert "solution.py:5: relative import" in problems[2]


def test_import_rule_reports_a_syntax_error_without_importing(tmp_path):
    path = tmp_path / "solution.py"
    path.write_text("def propose(:\n")
    (problem,) = meta.check_climber_source(path)
    assert problem.startswith("solution.py:1: syntax error")


# --- the reference meta-problem ---


def test_reference_meta_problem_loads_and_its_baseline_is_the_bundled_greedy():
    config = Config()
    config.paths.problems_dir = REPO / "problems"
    spec = load_problem(str(META_PROBLEM), config)
    assert spec.solution_kind == "climber"
    assert spec.baseline_text == GREEDY_SOURCE.read_text()  # one file, byte-identical to the package's
    inner = meta.load_grade_spec(META_PROBLEM / "grade.yaml")
    assert [p.problem for p in inner.problems] == ["heilbronn-11", "heilbronn-14"]
    assert meta.check_climber_source(META_PROBLEM / "greedy.py") == []


def test_reference_meta_problem_stays_out_of_the_bundled_catalog():
    from hillclimb.demo import BUNDLED_PROBLEM_IDS

    assert "meta-heilbronn" not in BUNDLED_PROBLEM_IDS


# --- the prompt an improver reads ---


def test_improver_draft_prompt_matches_golden(task, config, tmp_path):
    task.solution_kind = "climber"
    task.contract_template = "contract_climber"
    task.contract = "solution.py is a one-file climber."
    agent = FakeAgent()
    agent.queue(script=ok_script(0.6), notes="a\n")
    searcher, _journal, search_dir = make_searcher(task, config, agent)
    searcher.run_operator("draft", None)
    prompts = collect_prompts(search_dir, tmp_path)
    (prompt,) = prompts.values()  # the draft is c000: the fixture problem has no baseline
    assert "{{contract}}" in prompt, "the improver is told where the inner contract goes"
    assert prompt.count("{{") == 1, "every other token is filled"
    _check("improver", prompts)


# --- end to end: a meta verifier run with dummy inner searches ---


def _meta_hillclimb_dir(tmp_path: Path, budget: str = "20s") -> Path:
    """A standalone hillclimb dir holding heilbronn-11 and a meta-problem over
    it, on the dummy agent and the dev interpreter (no venv build)."""
    hc = tmp_path / "folder"
    problems = hc / "problems"
    shutil.copytree(REPO / "problems" / "heilbronn-11", problems / "heilbronn-11",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(META_PROBLEM, problems / "meta-heilbronn", ignore=shutil.ignore_patterns("__pycache__"))
    (problems / "meta-heilbronn" / "grade.yaml").write_text(f"problems: [heilbronn-11]\nbudget: {budget}\n")
    (hc / "hillclimb.yaml").write_text(yaml.safe_dump({
        "agent": "dummy",
        "paths": {"runtime_python": sys.executable},
        "budget": {"exec_timeout_s": 600},
        "learning": {"enabled": False},
        "holdout": {"enabled": False},
    }))
    return hc


@pytest.mark.slow
def test_verify_scores_the_reference_meta_problem_with_inner_dummy_searches(tmp_path, monkeypatch):
    hc = _meta_hillclimb_dir(tmp_path)
    monkeypatch.setenv("HILLCLIMB_DIR", str(hc))
    monkeypatch.chdir(hc)
    result = CliRunner().invoke(app, ["verify", "meta-heilbronn"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    # the dummy agent copies the floor, so the inner search closes no gap —
    # and that is a real score, journaled through the whole verifier contract
    assert "gap-closed = 0" in result.output


@pytest.mark.slow
def test_a_climber_that_does_not_load_makes_the_meta_verifier_fail(tmp_path, monkeypatch):
    hc = _meta_hillclimb_dir(tmp_path)
    broken = tmp_path / "broken.py"
    broken.write_text("from hillclimb.sdk import Action\nclass P:\n    def propose(self, view): return None\n")
    monkeypatch.setenv("HILLCLIMB_DIR", str(hc))
    monkeypatch.chdir(hc)
    result = CliRunner().invoke(app, ["verify", "meta-heilbronn", "--solution", str(broken)])
    assert result.exit_code == 1
    assert "FAILED" in result.output


@pytest.mark.slow
def test_grade_on_the_holdout_split_runs_the_outer_holdout(tmp_path, monkeypatch):
    hc = _meta_hillclimb_dir(tmp_path)
    spec = hc / "problems" / "meta-heilbronn" / "grade.yaml"
    monkeypatch.setenv("HILLCLIMB_DIR", str(hc))
    monkeypatch.chdir(hc)
    args = ["grade", "--climber", str(META_PROBLEM / "greedy.py"), "--spec", str(spec), "--split", "holdout"]

    # no outer holdout in the spec: refused before any inner search starts
    workdir = tmp_path / "none"
    workdir.mkdir()
    result = CliRunner().invoke(app, [*args, "--workdir", str(workdir)])
    assert result.exit_code == 1 and "outer_holdout" in result.output
    assert not (workdir / "hillclimb").exists()

    spec.write_text("problems: [no-such-problem]\nouter_holdout: [heilbronn-11]\nbudget: 20s\nscore: solved\n")
    workdir, out = tmp_path / "held-out", tmp_path / "grade.json"
    workdir.mkdir()
    result = CliRunner().invoke(
        app, [*args, "--workdir", str(workdir), "--result", str(out)], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    graded = yaml.safe_load(out.read_text())
    # only the outer holdout ran (the validation problem does not even exist),
    # scored by the spec's own choice: the dummy coding agent solves nothing
    assert list(graded["instances"]) == ["heilbronn-11"]
    assert graded["score"] == 0.0


def test_evaluate_refuses_a_spec_the_exec_timeout_cannot_fit(tmp_path):
    config = Config()
    config.budget.exec_timeout_s = 100
    spec = meta.GradeSpec.model_validate({"problems": ["a"], "budget": "5m"})
    climber = tmp_path / "solution.py"
    climber.write_text("x = 1\n")
    with pytest.raises(meta.MetaError, match="budget.exec_timeout_s is 100"):
        meta.evaluate(spec, climber, config, tmp_path, log=lambda *_: None)


def test_nested_config_keeps_the_users_agent_and_isolates_the_rest(tmp_path):
    outer = Config()
    outer.agent, outer.model = "codex", "gpt-5"
    outer.paths.problems_dir = tmp_path / "problems"
    outer.store.backend = "sqlite"
    outer.learning.enabled = True
    outer.budget.stop_margin_s = 300
    spec = meta.GradeSpec.model_validate({"problems": ["a"], "budget": "60s"})
    data = meta.nested_config(outer, tmp_path / "solution.py", spec, tmp_path / "hillclimb")
    assert (data["agent"], data["model"]) == ("codex", "gpt-5")
    assert data["climber"] == str((tmp_path / "solution.py").resolve())  # a one-file climber, by its file
    assert data["paths"]["problems_dir"] == str(tmp_path / "problems")
    assert data["paths"]["runs_dir"] == str(tmp_path / "hillclimb" / "runs")
    assert data["budget"]["total_s"] == 60 and data["budget"]["stop_margin_s"] == 12
    assert data["store"] == {"backend": "files"}
    assert data["learning"]["enabled"] is False
    assert Config.model_validate(data).agent == "codex"  # it loads as a config


def test_inner_command_lays_the_trials_params_over_the_climbers(tmp_path):
    spec = meta.GradeSpec.model_validate({"problems": ["a"], "budget": "90s"})
    cmd = meta.inner_command("py", "a", spec, tmp_path / "s.py", "a-r0", {"num_drafts": 2, "temp": 0.5, "init": "grid"})
    # --no-detach: the outer verifier waits for the inner search's result
    assert cmd[:6] == ["py", "-m", "hillclimb.cli", "run", "a", "--no-detach"]
    assert cmd[6:12] == ["--budget", "90s", "--climber", str(tmp_path / "s.py"), "--name", "a-r0"]
    assert cmd[12:] == [
        "--set", "climber.params.num_drafts=2", "--set", "climber.params.temp=0.5", "--set", 'climber.params.init="grid"',
    ]
    assert meta.inner_command("py", "a", spec, tmp_path / "s.py", "a-r0")[12:] == []


def test_score_floor_measures_a_files_only_floor_once(config, tmp_path):
    """heilbronn-style problems ship a valid-by-construction floor as files;
    a search leaves it unscored, so the meta verifier scores it itself."""
    config.paths.runtime_python = Path(sys.executable)
    config.paths.problems_dir = REPO / "problems"
    problem = load_problem("heilbronn-11", config)
    floor = meta.score_floor(problem, config, tmp_path, log=lambda *_: None)
    assert floor is not None and floor > 0
    assert (tmp_path / "floor" / "candidates" / "heilbronn-11" / "submission.csv").exists()


def test_score_floor_prefers_a_declared_baseline_score(task, config, tmp_path):
    task.baseline_score = 0.25
    assert meta.score_floor(task, config, tmp_path) == 0.25


def test_inner_env_drops_the_outer_verifiers_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("HILLCLIMB_PARAMS", "/outer/params.json")
    monkeypatch.setenv("HILLCLIMB_RESULT", "/outer/result.json")
    monkeypatch.setenv("HILLCLIMB_CACHE_DIR", "/cache")
    monkeypatch.setenv("PYTHONPATH", "/cache/interface-shim/abc")
    env = meta._inner_env(tmp_path)
    assert "HILLCLIMB_PARAMS" not in env and "HILLCLIMB_RESULT" not in env
    assert "PYTHONPATH" not in env, "the outer shim would shadow hillclimb for the inner engine"
    assert env["HILLCLIMB_DIR"] == str(tmp_path)
    assert env["HILLCLIMB_CACHE_DIR"] == "/cache"


def test_meta_check_runs_the_import_rule_then_the_climber_check(tmp_path, monkeypatch, capsys):
    """`hillclimb meta check` end to end through the CLI: the reference
    improver candidate passes; a file that reaches past the sdk is refused
    before anything is imported."""
    import json

    from hillclimb.cli import main as cli_main

    (tmp_path / "hillclimb.yaml").write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    with pytest.raises(SystemExit) as exc:
        cli_main(["meta", "check", "--climber", str(GREEDY_SOURCE), "--json"])
    assert exc.value.code == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    sneaky = tmp_path / "solution.py"
    sneaky.write_text("from hillclimb.harness.journal import Journal\n" + GREEDY_SOURCE.read_text())
    with pytest.raises(SystemExit) as exc:
        cli_main(["meta", "check", "--climber", str(sneaky)])
    assert exc.value.code == 1
    assert "hillclimb.harness.journal" in capsys.readouterr().err

