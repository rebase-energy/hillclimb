"""The params.json contract: declaration rules (spaces.check_params_space),
the runtime reader (spaces.params), the engine's typed view (params.py),
and the guarantee that the runtime half is stdlib-only (it runs under the
interface shim with site-packages disabled)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hillclimb import spaces
from hillclimb.harness.params import ParamsFile, coerce, parse_space, read_candidate_space, write_inherited_params, write_trial_params
from hillclimb.spaces import ParamsError, check_params_space, describe_params, fold_defaults, params, with_values

SPACE = {
    "restarts": {"type": "int", "low": 1, "high": 64, "log": True, "default": 8},
    "step": {"type": "float", "low": 1e-4, "high": 0.1, "log": True, "default": 0.01},
    "init": {"type": "categorical", "choices": ["grid", "random"], "default": "grid"},
}


def messages(raw) -> list[str]:
    return [v.path + ": " + v.message for v in check_params_space(raw)]


class TestDeclarationRules:
    def test_valid_space_has_no_violations(self):
        assert check_params_space(SPACE) == []

    @pytest.mark.parametrize("raw, fragment", [
        ([], "not a JSON object"),
        ({}, "declares no parameters"),
        ({"1bad": {"type": "int", "low": 0, "high": 1, "default": 0}}, "not a Python identifier"),
        ({"k": 3}, "spec is not an object"),
        ({"k": {"type": "bool", "default": True}}, "unknown type"),
        ({"k": {"type": "int", "low": 0, "high": 1}}, "default: missing"),
        ({"k": {"type": "int", "low": 0, "high": 1, "default": 0, "extra": 1}}, "unknown keys"),
        ({"k": {"type": "int", "low": 0.5, "high": 1, "default": 0}}, "missing or not numeric"),
        ({"k": {"type": "float", "low": 1.0, "high": 1.0, "default": 1.0}}, "low is not below high"),
        ({"k": {"type": "float", "low": 0.0, "high": 1.0, "log": True, "default": 0.5}}, "log scale needs low > 0"),
        ({"k": {"type": "float", "low": 0.0, "high": 1.0, "step": -1, "default": 0.5}}, "not a positive number"),
        ({"k": {"type": "float", "low": 0.0, "high": 1.0, "default": 2.0}}, "outside [low, high]"),
        ({"k": {"type": "categorical", "choices": [], "default": "a"}}, "missing or empty"),
        ({"k": {"type": "categorical", "choices": ["a", 1], "default": "a"}}, "mixed types"),
        ({"k": {"type": "categorical", "choices": ["a", "a"], "default": "a"}}, "duplicate choice"),
        ({"k": {"type": "categorical", "choices": ["a"], "default": "b"}}, "not one of the choices"),
        ({"k": {"type": "categorical", "choices": ["a"], "low": 0, "default": "a"}}, "not allowed on a categorical"),
    ])
    def test_each_rule_is_located(self, raw, fragment):
        assert any(fragment in m for m in messages(raw)), messages(raw)


class TestDocuments:
    def test_with_values_and_fold_defaults(self):
        trial_doc = with_values(SPACE, {"restarts": 16})
        assert trial_doc["restarts"]["value"] == 16
        assert trial_doc["step"]["value"] == 0.01  # missing → default
        assert "value" not in SPACE["restarts"]  # input untouched
        child = fold_defaults(trial_doc, {"restarts": 16, "init": "random"})
        assert child["restarts"]["default"] == 16 and child["init"]["default"] == "random"
        assert all("value" not in spec for spec in child.values())

    def test_describe_is_sorted_and_shows_values(self):
        text = describe_params(SPACE, {"step": 0.02})
        assert text.splitlines()[0].startswith("init: one of 'grid', 'random', default 'grid'")
        assert "step: float in [0.0001, 0.1] (log), default 0.01 = 0.02" in text

    def test_coerce_keeps_declared_types(self):
        space = parse_space(SPACE)
        assert coerce(space, {"restarts": 4.0, "step": "0.5", "init": "random"}) == {
            "restarts": 4, "step": 0.5, "init": "random",
        }
        assert coerce(space, {"unknown": 1}) == {"restarts": 8, "step": 0.01, "init": "grid"}


class TestRuntimeReader:
    def test_resolution_order(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(spaces.PARAMS_ENV, raising=False)
        assert params({"k": 1}) == {"k": 1}  # no file anywhere: defaults as is
        (tmp_path / "params.json").write_text(json.dumps(with_values(SPACE, {"restarts": 2})))
        assert params({"extra": 5})["restarts"] == 2  # cwd file, defaults fill unknown names
        assert params({"extra": 5})["extra"] == 5
        other = tmp_path / "t3.json"
        other.write_text(json.dumps(with_values(SPACE, {"restarts": 3})))
        monkeypatch.setenv(spaces.PARAMS_ENV, str(other))
        assert params()["restarts"] == 3  # env beats cwd
        assert params(path=tmp_path / "params.json")["restarts"] == 2  # explicit beats env

    def test_malformed_file_raises(self, tmp_path):
        bad = tmp_path / "params.json"
        bad.write_text('{"k": {"type": "int", "low": 0, "high": 1}}')
        with pytest.raises(ParamsError, match="default"):
            params(path=bad)
        bad.write_text("{not json")
        with pytest.raises(ParamsError):
            params(path=bad)

    def test_reader_is_stdlib_only_under_the_shim(self, tmp_path):
        """`from hillclimb import spaces` inside a runtime venv sees only the
        shim: run the reader with site-packages disabled (-S) so the dev
        install cannot mask a stray import."""
        from hillclimb.runtime import ensure_interface_shim

        shim = ensure_interface_shim()
        doc = tmp_path / "params.json"
        doc.write_text(json.dumps(with_values(SPACE, {"restarts": 5, "init": "random"})))
        code = "from hillclimb import spaces; import json; print(json.dumps(spaces.params({'extra': 1})))"
        proc = subprocess.run(
            [sys.executable, "-S", "-c", code],
            env={**os.environ, "PYTHONPATH": str(shim), spaces.PARAMS_ENV: str(doc)},
            capture_output=True, text=True, check=True,
        )
        assert json.loads(proc.stdout) == {"extra": 1, "restarts": 5, "step": 0.01, "init": "random"}


class TestEngineSide:
    def test_read_candidate_space(self, tmp_path):
        assert read_candidate_space(tmp_path) == (None, None)
        (tmp_path / "params.json").write_text("{oops")
        declaration, reason = read_candidate_space(tmp_path)
        assert declaration is None and reason
        (tmp_path / "params.json").write_text(json.dumps({"k": {"type": "int", "low": 0, "high": 1}}))
        declaration, reason = read_candidate_space(tmp_path)
        assert declaration is None and "default" in reason
        (tmp_path / "params.json").write_text(json.dumps(SPACE))
        declaration, reason = read_candidate_space(tmp_path)
        assert isinstance(declaration, ParamsFile) and reason is None
        assert declaration.defaults == {"restarts": 8, "step": 0.01, "init": "grid"}

    def test_trial_and_inherited_documents(self, tmp_path):
        declaration = ParamsFile(raw=SPACE, space=parse_space(SPACE))
        trial_dir = tmp_path / "t1"
        trial_dir.mkdir()
        write_trial_params(trial_dir, declaration, {"restarts": 32})
        assert json.loads((trial_dir / "params.json").read_text())["restarts"]["value"] == 32
        child = tmp_path / "child"
        child.mkdir()
        write_inherited_params(child, declaration, {"restarts": 32})
        inherited = json.loads((child / "params.json").read_text())
        assert inherited["restarts"]["default"] == 32 and "value" not in inherited["restarts"]
        assert check_params_space(inherited) == []


class TestExecutorWiring:
    """`$HILLCLIMB_PARAMS` points the verifier at the trial's params.json
    exactly when one exists; the holdout scorer materializes the trial's
    values file and nothing else from the trial dir."""

    def test_verifier_env_sets_params_only_when_given(self, tmp_path):
        from hillclimb.harness.executor import RESULT_FILE, verifier_env

        base = (Path(sys.executable), tmp_path / "s.py", tmp_path / RESULT_FILE, "validation")
        assert "HILLCLIMB_PARAMS" not in verifier_env(*base)
        env = verifier_env(*base, params=tmp_path / "params.json")
        assert env["HILLCLIMB_PARAMS"] == str(tmp_path / "params.json")
        assert "HILLCLIMB_REPLICATE_SEED" not in env

    def test_executor_exports_the_replicate_dirs_params_file(self, tmp_path, task, config):
        from hillclimb.harness.candidate import Candidate
        from hillclimb.harness.dirs import create_candidate_dir, create_search_dir
        from hillclimb.harness.evaluation import CandidateEvaluator
        from hillclimb.harness.params import ParamsFile
        from tests.conftest import executor_for

        search_dir = create_search_dir(tmp_path / "runs" / "r", "s")
        candidate_dir = create_candidate_dir(search_dir, "c001", task.data_dir, task.problem_dir)
        (candidate_dir / "solution.py").write_text(
            "import os, shutil\n"
            "from hillclimb import spaces\n"
            'shutil.copy("data/sample_submission.csv", "submission.csv")\n'
            'print("params_env:", os.environ.get("HILLCLIMB_PARAMS", "-"))\n'
            'print(f"val_score: {spaces.params({\'restarts\': 50})[\'restarts\'] / 100}")\n'
        )
        declaration = ParamsFile(raw=SPACE, space=parse_space(SPACE))
        candidate = Candidate(candidate_id="c001", operator="draft", candidate_dir=str(candidate_dir))
        evaluator = CandidateEvaluator(executor=executor_for(task), problem=task, config=config)
        trial_, ok = evaluator.run_trial(
            candidate, candidate_dir / "solution.py", candidate_dir, 30,
            params={"restarts": 32}, params_doc=with_values(SPACE, {"restarts": 32}),
        )
        assert ok and trial_.val_score == pytest.approx(0.32)
        r0 = candidate_dir / "trials" / "t0" / "replicates" / "r0"
        assert (r0 / "params.json").exists()
        assert f"params_env: {r0 / 'params.json'}" in trial_.replicates[0].stdout_tail

        plain = Candidate(candidate_id="c001", operator="draft", candidate_dir=str(candidate_dir))
        trial_, ok = evaluator.run_trial(plain, candidate_dir / "solution.py", candidate_dir, 30, index=1)
        assert ok and "params_env: -" in trial_.replicates[0].stdout_tail  # no declaration: no env
        assert trial_.val_score == pytest.approx(0.5)  # the solution's own defaults

    def test_holdout_scorer_materializes_the_trials_values(self, tmp_path):
        from hillclimb.harness.executor import CommandHoldoutScorer, trial_params_doc
        from tests.factories import trial as mk_trial

        candidate_dir = tmp_path / "c001"
        candidate_dir.mkdir()
        (candidate_dir / "solution.py").write_text("print('hi')\n")
        (candidate_dir / "params.json").write_text(json.dumps(SPACE))
        trial_dir = candidate_dir / "trials" / "t2"
        trial_dir.mkdir(parents=True)
        (trial_dir / "exec_stdout.log").write_text("must not travel\n")
        holdout_verifier = tmp_path / "verify.py"
        holdout_verifier.write_text(
            "import json, os, pathlib\n"
            "doc = json.loads(pathlib.Path(os.environ['HILLCLIMB_PARAMS']).read_text())\n"
            "assert 'must not travel' not in pathlib.Path('exec_stdout.log').read_text()\n"
            "pathlib.Path(os.environ['HILLCLIMB_RESULT']).write_text(json.dumps({'score': doc['restarts']['value']}))\n"
        )
        scorer = CommandHoldoutScorer(
            Path(sys.executable), [sys.executable, str(holdout_verifier)],
            problem_dir=tmp_path, data_dir=tmp_path, work_root=tmp_path / "holdout-eval", timeout_s=60,
        )
        best = mk_trial(0.5, params={"restarts": 32}, index=2)
        score, error, _cpu = scorer.score(candidate_dir, best)
        assert error is None and score == 32
        eval_dir = tmp_path / "holdout-eval" / "c001" / "t2"
        assert json.loads((eval_dir / "params.json").read_text())["restarts"]["value"] == 32
        assert trial_params_doc(candidate_dir, None)["restarts"]["value"] == 8  # defaults trial
        (candidate_dir / "params.json").unlink()
        assert trial_params_doc(candidate_dir, best) is None
