"""The verifier contract end to end: CommandExecutor + CommandHoldoutScorer
against a tiny real evaluator (run with sys.executable — no venv build)."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from hillclimb.harness.executor import CommandExecutor, CommandHoldoutScorer

EVALUATE = textwrap.dedent(
    """
    import json, os, sys
    sys.path.insert(0, os.getcwd())
    import solution

    split = "holdout" if "--holdout" in sys.argv else "validation"
    score = float(solution.answer())
    if os.environ.get("HILLCLIMB_TRIAL_SEED"):
        score += 0.001 * int(os.environ["HILLCLIMB_TRIAL_SEED"])
    report = {"version": 1, "split": split, "overall": {"score": score}}
    if os.environ.get("X_SECRET_TOKEN"):
        report["saw_token"] = True
    json.dump({"split": split, "score": score, "report": report},
              open("eval_result.json", "w"))
    print(f"val_score: {score}")
    """
)


def make_problem(tmp_path: Path) -> Path:
    problem_dir = tmp_path / "problem-src"
    problem_dir.mkdir()
    (problem_dir / "evaluate.py").write_text(EVALUATE)
    return problem_dir


def make_workspace(tmp_path: Path, problem_dir: Path, solution: str = "def answer():\n    return 0.5\n") -> Path:
    candidate_dir = tmp_path / "candidates" / "c001"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "solution.py").write_text(solution)
    (candidate_dir / "problem").symlink_to(problem_dir, target_is_directory=True)
    (candidate_dir / "data").symlink_to(problem_dir, target_is_directory=True)
    return candidate_dir


COMMAND = ["{python}", "problem/evaluate.py"]


def test_happy_path_scores_and_result_json(tmp_path):
    problem_dir = make_problem(tmp_path)
    candidate_dir = make_workspace(tmp_path, problem_dir)
    executor = CommandExecutor(Path(sys.executable), COMMAND)
    result = executor.execute(candidate_dir / "solution.py", candidate_dir, timeout_s=60)
    assert result.ok, Path(result.stderr_path).read_text()[-300:]
    assert result.val_score == 0.5
    payload = json.loads((candidate_dir / "eval_result.json").read_text())
    assert payload["split"] == "validation"
    assert payload["report"]["overall"]["score"] == 0.5


def test_seed_env_and_placeholder_substitution(tmp_path):
    problem_dir = make_problem(tmp_path)
    candidate_dir = make_workspace(tmp_path, problem_dir)
    # {solution} placeholder: evaluator receives the explicit path
    (problem_dir / "evaluate.py").write_text(
        EVALUATE.replace("import solution", "import importlib.util\n"
                         "spec = importlib.util.spec_from_file_location('solution', sys.argv[1])\n"
                         "solution = importlib.util.module_from_spec(spec)\n"
                         "spec.loader.exec_module(solution)")
    )
    executor = CommandExecutor(Path(sys.executable), ["{python}", "problem/evaluate.py", "{solution}"])
    result = executor.execute(candidate_dir / "solution.py", candidate_dir, timeout_s=60, seed=3)
    assert result.ok
    assert result.val_score == 0.503  # seed propagated via HILLCLIMB_TRIAL_SEED


def test_no_result_file_is_a_contract_violation(tmp_path):
    """Exit 0 without a result file is not a zero — the printed line is a log,
    not a score (agent code shares that stream)."""
    problem_dir = make_problem(tmp_path)
    (problem_dir / "evaluate.py").write_text('print("val_score: 1.0")\n')
    candidate_dir = make_workspace(tmp_path, problem_dir)
    executor = CommandExecutor(Path(sys.executable), COMMAND)
    result = executor.execute(candidate_dir / "solution.py", candidate_dir, timeout_s=60)
    assert result.val_score is None
    assert not result.submission_ok
    assert not result.ok


def test_bare_number_result_file(tmp_path):
    """The simplest possible verifier: echo a number into $HILLCLIMB_RESULT."""
    problem_dir = make_problem(tmp_path)
    (problem_dir / "evaluate.py").write_text(
        'import os\nopen(os.environ["HILLCLIMB_RESULT"], "w").write(" 12.5\\n")\n'
    )
    candidate_dir = make_workspace(tmp_path, problem_dir)
    executor = CommandExecutor(Path(sys.executable), COMMAND)
    result = executor.execute(candidate_dir / "solution.py", candidate_dir, timeout_s=60)
    assert result.ok
    assert result.val_score == 12.5


def test_stale_result_json_scrubbed_and_crash_not_ok(tmp_path):
    problem_dir = make_problem(tmp_path)
    (problem_dir / "evaluate.py").write_text('raise RuntimeError("evaluator boom")\n')
    candidate_dir = make_workspace(tmp_path, problem_dir)
    (candidate_dir / "eval_result.json").write_text('{"split": "validation", "score": 9}')
    executor = CommandExecutor(Path(sys.executable), COMMAND)
    result = executor.execute(candidate_dir / "solution.py", candidate_dir, timeout_s=60)
    assert result.returncode != 0
    assert not result.submission_ok  # the stale file was unlinked pre-run
    assert not (candidate_dir / "eval_result.json").exists()


def test_validation_env_scrubbed(tmp_path, monkeypatch):
    monkeypatch.setenv("X_SECRET_TOKEN", "sk-123")
    problem_dir = make_problem(tmp_path)
    candidate_dir = make_workspace(tmp_path, problem_dir)
    executor = CommandExecutor(Path(sys.executable), COMMAND)
    result = executor.execute(candidate_dir / "solution.py", candidate_dir, timeout_s=60)
    assert result.ok
    payload = json.loads((candidate_dir / "eval_result.json").read_text())
    assert "saw_token" not in payload["report"]  # *_TOKEN scrubbed


def test_holdout_scorer_hidden_dir_full_env(tmp_path, monkeypatch):
    monkeypatch.setenv("X_SECRET_TOKEN", "sk-123")
    problem_dir = make_problem(tmp_path)
    candidate_dir = make_workspace(tmp_path, problem_dir)
    (candidate_dir / "candidate_1.py").write_text("# ensemble input\n")
    scorer = CommandHoldoutScorer(
        Path(sys.executable), COMMAND + ["--holdout"],
        problem_dir=problem_dir, data_dir=problem_dir,
        work_root=tmp_path / "holdout-eval", timeout_s=60,
    )
    score, error, cpu = scorer.score(candidate_dir)
    assert error is None
    assert score == 0.5
    assert cpu is not None and cpu >= 0.0  # holdout cpu rides along
    eval_dir = tmp_path / "holdout-eval" / "c001" / "t0"
    # hidden dir recreated the run layout: solution + extras + symlinks
    assert (eval_dir / "solution.py").exists()
    assert (eval_dir / "candidate_1.py").exists()
    assert (eval_dir / "problem").is_symlink() and (eval_dir / "data").is_symlink()
    payload = json.loads((eval_dir / "eval_result.json").read_text())
    assert payload["split"] == "holdout"
    assert payload["report"].get("saw_token") is True  # full env: credentials flow
    # agent-visible candidate_dir untouched by the holdout run
    assert not (candidate_dir / "eval_result.json").exists()


def test_interface_shim_importable_on_both_splits(tmp_path, monkeypatch):
    """A verifier can `from hillclimb import spaces` in venvs where hillclimb
    is not installed: the shim rides PYTHONPATH on validation AND holdout.
    (The dev interpreter has the real package, so the proof is that the
    import resolves to the shim copy — PYTHONPATH precedes site-packages.)"""
    monkeypatch.setenv("HILLCLIMB_CACHE_DIR", str(tmp_path / "cache"))
    from hillclimb.runtime import ensure_interface_shim

    shim = ensure_interface_shim()
    assert (shim / "hillclimb" / "spaces.py").exists()
    assert ensure_interface_shim() == shim  # idempotent, content-keyed

    problem_dir = make_problem(tmp_path)
    (problem_dir / "evaluate.py").write_text(textwrap.dedent(
        """
        import os, sys
        from hillclimb import spaces
        split = "holdout" if "--holdout" in sys.argv else "validation"
        payload = '{"split": "%s", "score": 1.0}' % split
        open(os.environ["HILLCLIMB_RESULT"], "w").write(payload)
        print("spaces from:", spaces.__file__)
        """
    ))
    candidate_dir = make_workspace(tmp_path, problem_dir)
    executor = CommandExecutor(Path(sys.executable), COMMAND, pythonpath=str(shim))
    result = executor.execute(candidate_dir / "solution.py", candidate_dir, timeout_s=60)
    assert result.ok, Path(result.stderr_path).read_text()[-300:]
    assert str(shim) in Path(result.stdout_path).read_text()

    scorer = CommandHoldoutScorer(
        Path(sys.executable), COMMAND + ["--holdout"],
        problem_dir=problem_dir, data_dir=problem_dir,
        work_root=tmp_path / "holdout-eval", timeout_s=60, pythonpath=str(shim),
    )
    score, error, _cpu = scorer.score(candidate_dir)
    assert error is None
    assert score == 1.0
    stdout = (tmp_path / "holdout-eval" / "c001" / "t0" / "exec_stdout.log").read_text()
    assert str(shim) in stdout


def test_holdout_scorer_failure_mapping(tmp_path):
    problem_dir = make_problem(tmp_path)
    candidate_dir = make_workspace(
        tmp_path, problem_dir, solution="def answer():\n    raise ValueError('nope')\n"
    )
    scorer = CommandHoldoutScorer(
        Path(sys.executable), COMMAND, problem_dir=problem_dir, data_dir=problem_dir,
        work_root=tmp_path / "holdout-eval", timeout_s=60,
    )
    score, error, cpu = scorer.score(candidate_dir)
    assert score is None
    assert "holdout evaluation failed" in error
    assert cpu is not None  # burned even though scoring failed

    scorer_no_score = CommandHoldoutScorer(
        Path(sys.executable), ["{python}", "-c", "print(42)"],
        problem_dir=problem_dir, data_dir=problem_dir,
        work_root=tmp_path / "holdout-eval-2", timeout_s=60,
    )
    score, error, _cpu = scorer_no_score.score(candidate_dir)
    assert score is None
    assert "produced no score" in error


def test_extra_numeric_result_keys_become_trial_metrics(tmp_path):
    """Keys next to `score` are journaled as metrics (feature dimensions for
    quality-diversity policies); non-numeric/bool/NaN values are dropped and
    `report` stays out of the metrics dict."""
    problem_dir = make_problem(tmp_path)
    (problem_dir / "evaluate.py").write_text(
        "import os, json\n"
        'json.dump({"score": 0.5, "runtime_s": 1.25, "n_params": 3, "ok": True,\n'
        '           "name": "x", "nan": float("nan"), "report": {"a": 1}},\n'
        '          open(os.environ["HILLCLIMB_RESULT"], "w"))\n'
    )
    candidate_dir = make_workspace(tmp_path, problem_dir)
    executor = CommandExecutor(Path(sys.executable), COMMAND)
    result = executor.execute(candidate_dir / "solution.py", candidate_dir, timeout_s=60)
    assert result.ok and result.val_score == 0.5
    assert result.metrics == {"runtime_s": 1.25, "n_params": 3.0}
