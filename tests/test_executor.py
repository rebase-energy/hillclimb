import sys
from pathlib import Path

import pytest

from hillclimb.executor import LocalExecutor, parse_val_score, scrubbed_env


@pytest.fixture
def executor() -> LocalExecutor:
    return LocalExecutor(Path(sys.executable))


def run_script(executor: LocalExecutor, tmp_path: Path, code: str, timeout: int = 30):
    script = tmp_path / "solution.py"
    script.write_text(code)
    return executor.execute(script, tmp_path, timeout)


def test_ok_script(executor, tmp_path):
    code = 'open("submission.csv", "w").write("id\\n")\nprint("val_score: 0.75")\n'
    result = run_script(executor, tmp_path, code)
    assert result.ok
    assert result.val_score == 0.75
    assert result.submission_ok


def test_verifier_scores_submission(executor, tmp_path):
    script = tmp_path / "solution.py"
    script.write_text('open("submission.csv", "w").write("id\\n")\n')
    verifier = tmp_path / "verify.py"
    verifier.write_text('print("checked")\nprint("val_score: 0.42")\n')

    result = executor.execute(script, tmp_path, timeout_s=30, verifier=verifier)

    assert result.ok
    assert result.val_score == 0.42
    assert "checked" in Path(result.stdout_path).read_text()


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


def test_no_score(executor, tmp_path):
    result = run_script(executor, tmp_path, 'open("submission.csv", "w").write("id\\n")')
    assert not result.ok
    assert result.val_score is None
    assert result.submission_ok


def test_no_submission(executor, tmp_path):
    result = run_script(executor, tmp_path, 'print("val_score: 0.5")')
    assert not result.ok
    assert not result.submission_ok


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
    from hillclimb.backends.claude_code import subscription_env

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-oauth")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in scrubbed_env()  # _TOKEN suffix
    agent_env = subscription_env("subscription")
    assert agent_env.get("CLAUDE_CODE_OAUTH_TOKEN") == "sk-oauth"
    assert "ANTHROPIC_API_KEY" not in agent_env
    api_env = subscription_env("api-key")
    assert api_env.get("ANTHROPIC_API_KEY") == "sk-ant"


def test_parse_val_score():
    assert parse_val_score("noise\nval_score: 0.5\n") == 0.5
    assert parse_val_score("val_score: 0.1\nval_score: 0.2\n") == 0.2  # last wins
    assert parse_val_score("val_score: 1e-3\n") == 0.001
    assert parse_val_score("val_score: abc\n") is None
    assert parse_val_score("score 0.5\n") is None
