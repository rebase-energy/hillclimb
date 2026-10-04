"""Execution primitives: how a score is read back, and the self-reported
verifier (`run_solution.py`) that MLE-bench problems and the test fixtures
share."""

import json
import sys
from pathlib import Path

import pytest

from hillclimb.harness.executor import (
    RESULT_FILE,
    read_result,
    render_argv,
    scrubbed_env,
    verifier_env,
)
from tests.conftest import local_executor


@pytest.fixture
def executor():
    return local_executor()


def run_script(executor, tmp_path: Path, code: str, timeout: int = 30):
    script = tmp_path / "solution.py"
    script.write_text(code)
    return executor.execute(script, tmp_path, timeout)


def test_ok_script(executor, tmp_path):
    code = 'open("submission.csv", "w").write("id\\n")\nprint("val_score: 0.75")\n'
    result = run_script(executor, tmp_path, code)
    assert result.ok
    assert result.val_score == 0.75
    assert result.submission_ok
    payload = json.loads((tmp_path / RESULT_FILE).read_text())
    assert payload["source"] == "agent"  # self-reported, and labelled as such


def test_solution_written_result_wins(executor, tmp_path):
    """A solution that writes its own report keeps it (the report carries the
    breakdown the improve prompt renders)."""
    code = (
        'import json, os\n'
        'open("submission.csv", "w").write("id\\n")\n'
        'json.dump({"score": 0.9, "report": {"version": 1}},'
        ' open(os.environ["HILLCLIMB_RESULT"], "w"))\n'
    )
    result = run_script(executor, tmp_path, code)
    assert result.ok
    assert result.val_score == 0.9


def test_crash(executor, tmp_path):
    result = run_script(executor, tmp_path, 'raise RuntimeError("boom")')
    assert not result.ok
    assert result.returncode != 0
    assert "boom" in Path(result.stderr_path).read_text()


def test_hang_killed(executor, tmp_path):
    result = run_script(executor, tmp_path, "import time; time.sleep(60)", timeout=2)
    assert result.timed_out
    assert not result.ok
    assert result.duration_s < 30


def test_a_stop_kills_a_running_verifier_at_once(executor, tmp_path):
    """The engine's abort reaches the verifier: a stop does not wait for a
    slow verifier to finish (it used to wait the whole run out)."""
    import threading
    import time

    executor.abort = threading.Event()
    threading.Timer(0.5, executor.abort.set).start()
    started = time.monotonic()
    result = run_script(executor, tmp_path, "import time\ntime.sleep(30)\n", timeout=60)
    assert result.timed_out and time.monotonic() - started < 10


def two_step_executor(tmp_path: Path, scorer_code: str, **options):
    """A two-step problem's executor: the solution, then this scorer."""
    from hillclimb.harness.executor import CommandExecutor

    scorer = tmp_path / "scorer.py"
    scorer.write_text(scorer_code)
    return CommandExecutor(
        Path(sys.executable), ["{python}", "{solution}"], score_argv=["{python}", str(scorer)], **options
    )


SCORE_THE_ANSWER = (
    "import json, os, pathlib\n"
    "answer = float(pathlib.Path('answer.txt').read_text())\n"
    "pathlib.Path(os.environ['HILLCLIMB_RESULT']).write_text(json.dumps({'score': answer}))\n"
)


def test_two_steps_score_what_the_solution_wrote_and_nothing_it_claims(tmp_path):
    """The solution runs first; a result file it writes is deleted before
    the scorer runs, so only the scorer reports a score."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    executor = two_step_executor(tmp_path, SCORE_THE_ANSWER)
    claim = (
        "import os, pathlib\n"
        "pathlib.Path('answer.txt').write_text('3')\n"
        "pathlib.Path(os.environ['HILLCLIMB_RESULT']).write_text('1e9')\n"
    )
    result = run_script(executor, run_dir, claim)
    assert result.ok and result.val_score == 3.0


def test_two_steps_stop_a_solution_at_its_time_limit(tmp_path):
    """Past the problem's own limit it is a failed attempt (exit 124), not a
    run the engine's clock cut off."""
    import time

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    executor = two_step_executor(tmp_path, SCORE_THE_ANSWER, time_limit_s=1)
    started = time.monotonic()
    result = run_script(executor, run_dir, "import time\ntime.sleep(30)\n", timeout=60)
    assert result.returncode == 124 and not result.timed_out and not result.ok
    assert time.monotonic() - started < 15
    assert "time limit of 1s" in Path(result.stderr_path).read_text()


def test_no_score(executor, tmp_path):
    result = run_script(executor, tmp_path, 'open("submission.csv", "w").write("id\\n")')
    assert not result.ok
    assert result.val_score is None


def test_no_submission(executor, tmp_path):
    """--require: a solution that scores itself but ships no submission is a
    failure, not a win (MLE-bench grades the file afterwards)."""
    result = run_script(executor, tmp_path, 'print("val_score: 0.5")')
    assert not result.ok
    assert not result.submission_ok


def test_stale_result_never_counts(executor, tmp_path):
    (tmp_path / RESULT_FILE).write_text('{"score": 9.9}')
    result = run_script(executor, tmp_path, 'raise RuntimeError("boom")')
    assert not result.ok
    assert result.val_score is None
    assert not (tmp_path / RESULT_FILE).exists()


def test_solution_env_is_scrubbed(executor, tmp_path, monkeypatch):
    """Agent-authored code must never see credentials from the orchestrator."""
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("MY_SERVICE_API_KEY", "suffix-matched-secret")
    monkeypatch.setenv("HF_HOME", "/tmp/hf-cache")  # non-secret must survive
    code = (
        "import os\n"
        'leaks = [k for k in ("HF_TOKEN", "ANTHROPIC_API_KEY", "MY_SERVICE_API_KEY")'
        " if k in os.environ]\n"
        'print("leaks:", leaks)\n'
        'print("hf_home:", os.environ.get("HF_HOME"))\n'
        'open("submission.csv", "w").write("id\\n")\n'
        'print("val_score: 1.0")\n'
    )
    result = run_script(executor, tmp_path, code)
    stdout = Path(result.stdout_path).read_text()
    assert result.ok
    assert "leaks: []" in stdout
    assert "hf_home: /tmp/hf-cache" in stdout


def test_scrubbed_env_extra_overrides(monkeypatch):
    monkeypatch.setenv("SOME_PASSWORD", "x")
    env = scrubbed_env(HF_HUB_OFFLINE="1")
    assert "SOME_PASSWORD" not in env
    assert env["HF_HUB_OFFLINE"] == "1"
    assert "PATH" in env


def test_claude_oauth_token_scrubbed_but_kept_for_agent(monkeypatch):
    """Hosted subscription auth: the claude agent process must see
    CLAUDE_CODE_OAUTH_TOKEN, agent-authored code must not."""
    from hillclimb.agents.claude_code import subscription_env

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-oauth")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in scrubbed_env()  # _TOKEN suffix
    agent_env = subscription_env("subscription")
    assert agent_env.get("CLAUDE_CODE_OAUTH_TOKEN") == "sk-oauth"
    assert "ANTHROPIC_API_KEY" not in agent_env
    api_env = subscription_env("api-key")
    assert api_env.get("ANTHROPIC_API_KEY") == "sk-ant"


def test_read_result_forms(tmp_path):
    path = tmp_path / RESULT_FILE
    path.write_text('{"score": 0.5, "report": {"version": 1}}')
    score, payload = read_result(path)
    assert score == 0.5
    assert payload["report"]["version"] == 1

    path.write_text(" 12.5\n")  # bare number: the one-line verifier
    assert read_result(path) == (12.5, None)

    path.write_text("42")
    assert read_result(path)[0] == 42.0

    path.write_text('{"score": NaN}')  # nothing was graded: not a worst score
    assert read_result(path)[0] is None

    # E19: an infinite score would win every comparison and is not JSON
    for text in ('{"score": Infinity}', '{"score": -Infinity}', "1e999", "-inf"):
        path.write_text(text)
        assert read_result(path)[0] is None, text
    from hillclimb.harness.executor import result_metrics, result_problem

    path.write_text('{"score": Infinity}')
    assert "finite" in result_problem(path)
    assert result_metrics({"score": 1.0, "runtime_s": float("inf"), "n": 3}) == {"n": 3.0}

    path.write_text('{"report": {}}')
    assert read_result(path)[0] is None

    path.write_text("not a score")
    assert read_result(path) == (None, None)

    path.write_text("")
    assert read_result(path) == (None, None)

    assert read_result(tmp_path / "absent.json") == (None, None)


def test_verifier_env_and_render(tmp_path):
    env = verifier_env(Path("/venv/bin/python"), tmp_path / "solution.py",
                       tmp_path / RESULT_FILE, "holdout", seed=3)
    assert env["HILLCLIMB_SPLIT"] == "holdout"
    assert env["HILLCLIMB_TRIAL_SEED"] == "3"
    assert env["HILLCLIMB_PYTHON"] == "/venv/bin/python"
    assert "HILLCLIMB_TRIAL_SEED" not in verifier_env(
        Path(sys.executable), tmp_path / "s.py", tmp_path / RESULT_FILE, "validation"
    )

    argv = render_argv(
        ["{python}", "eval.py", "{solution}", "--out", "{result}"],
        Path("/venv/bin/python"), Path("/w/solution.py"), Path("/w/eval_result.json"),
    )
    assert argv == ["/venv/bin/python", "eval.py", "/w/solution.py",
                    "--out", "/w/eval_result.json"]


def test_pythonpath_reaches_verifier_env(tmp_path):
    """The interface shim rides PYTHONPATH into the verifier process; without
    one the env is untouched."""
    from hillclimb.harness.executor import CommandExecutor, prepend_pythonpath
    from tests.conftest import SELF_REPORT_CMD

    code = (
        "import os\n"
        'print("pythonpath:", os.environ.get("PYTHONPATH", "(unset)"))\n'
        'open("submission.csv", "w").write("id\\n")\n'
        'print("val_score: 1.0")\n'
    )
    script = tmp_path / "solution.py"
    script.write_text(code)

    with_shim = CommandExecutor(Path(sys.executable), SELF_REPORT_CMD, pythonpath="/shim")
    result = with_shim.execute(script, tmp_path, 30)
    assert "pythonpath: /shim" in Path(result.stdout_path).read_text()

    result = local_executor().execute(script, tmp_path, 30)
    first = Path(result.stdout_path).read_text().splitlines()[0]
    assert first == "pythonpath: (unset)" or "/shim" not in first  # parent value survives

    import os

    env = {"PYTHONPATH": "/existing"}
    assert prepend_pythonpath(env, "/shim")["PYTHONPATH"] == f"/shim{os.pathsep}/existing"
    assert prepend_pythonpath({"A": "b"}, None) == {"A": "b"}


def test_solution_and_agent_envs_are_single_threaded(monkeypatch):
    from hillclimb.agents.claude_code import subscription_env
    from hillclimb.harness.executor import SINGLE_THREAD_ENV

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "4")  # an explicit parent value wins
    for env in (scrubbed_env(), subscription_env()):
        assert env["OMP_NUM_THREADS"] == "1" and env["OPENBLAS_NUM_THREADS"] == "1"
        assert env["MKL_NUM_THREADS"] == "4"
    assert set(SINGLE_THREAD_ENV) >= {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"}


def test_cpu_captured(executor, tmp_path):
    """cpu_s is the child's real CPU time via os.wait4 — a busy loop burns
    at least what it spins, a trivial script close to nothing."""
    code = (
        'import time\n'
        'while time.process_time() < 0.2:\n'
        '    pass\n'
        'open("submission.csv", "w").write("id\\n")\n'
        'print("val_score: 0.5")\n'
    )
    result = run_script(executor, tmp_path, code)
    assert result.ok
    assert result.cpu_s is not None
    assert result.cpu_s >= 0.15

    trivial = run_script(executor, tmp_path, 'print("val_score: 0.5")')
    assert trivial.cpu_s is not None
    assert trivial.cpu_s < 5.0


def test_killed_child_reports_cpu(executor, tmp_path):
    """The kill path reaps through wait4 too: a timed-out run still reports
    cpu_s and keeps the negative-signal returncode — and the CPU of the
    grandchildren the kill orphans (here the solution process the verifier
    wrapper spawned, spinning until the deadline) is sampled before the
    kill, so it counts."""
    result = run_script(executor, tmp_path, "while True:\n    pass", timeout=2)
    assert result.timed_out
    assert not result.ok
    assert result.returncode is not None and result.returncode < 0
    assert result.cpu_s is not None
    assert result.cpu_s >= 0.5
