"""Before a command uses a coding agent, its login is checked: a dead one is
offered a fresh login right there, never found three failed calls into a run
(`connect.ensure_agent_ready`, `cli.common.ensure_agents_ready`)."""

from __future__ import annotations

import pytest

from hillclimb import connect
from hillclimb.agents.base import AgentResult

DEAD = AgentResult(
    ok=False, error_kind="error",
    error_message="Failed to authenticate: OAuth session expired and could not be refreshed",
)
ALIVE = AgentResult(ok=True, model_id="claude-sonnet-5-5")
from tests.folder_config import write_config


@pytest.fixture
def login(tmp_path, monkeypatch):
    """The check switched back on (the suite turns it off), the machine
    cache in tmp, and every outside call recorded instead of made."""
    monkeypatch.delenv(connect.CHECKED_ENV, raising=False)
    monkeypatch.setenv("HILLCLIMB_CACHE_DIR", str(tmp_path / "cache"))
    calls: dict[str, list] = {"ping": [], "relogin": [], "ask": []}
    answers: list[AgentResult] = []

    def ping(agent, auth, model, **kw):
        calls["ping"].append(agent)
        return answers.pop(0)

    monkeypatch.setattr(connect, "ping", ping)
    monkeypatch.setattr(connect, "run_relogin", lambda target: calls["relogin"].append(target) or 0)
    monkeypatch.setattr(connect, "check", lambda target, auth: connect.Status(target, auth, "ready", "fine"))
    monkeypatch.setattr(connect, "import_credentials", lambda *a, **k: None)
    return calls, answers


def test_nothing_to_check_for_agents_hillclimb_cannot_log_in(login, monkeypatch):
    calls, _ = login
    connect.ensure_agent_ready("dummy", "subscription", "m")
    connect.ensure_agent_ready("claude-code", "api-key", "m")  # a key, not a login
    connect.ensure_agent_ready("pi", "subscription", "m")
    monkeypatch.setenv(connect.CHECKED_ENV, "1")  # an engine its launcher checked
    connect.ensure_agent_ready("claude-code", "subscription", "m")
    assert calls["ping"] == []


def test_a_live_login_is_pinged_once_an_hour(login):
    calls, answers = login
    answers.append(ALIVE)
    connect.ensure_agent_ready("claude-code", "subscription", "sonnet")
    connect.ensure_agent_ready("claude-code", "subscription", "sonnet")  # confirmed just now
    assert calls["ping"] == ["claude-code"]
    answers.append(ALIVE)
    connect.ensure_agent_ready("claude-code", "subscription", "sonnet", fresh_s=0)
    assert len(calls["ping"]) == 2  # a stamp older than fresh_s pings again


def test_a_dead_login_asks_and_logs_in_again(login):
    calls, answers = login
    answers.extend([DEAD, ALIVE])
    said = []

    def ask(question):
        calls["ask"].append(question)
        return True

    connect.ensure_agent_ready("claude-code", "subscription", "sonnet", ask=ask, say=said.append)
    assert calls["relogin"] == ["claude"] and len(calls["ping"]) == 2
    assert "Your Claude login has expired" in calls["ask"][0] and "could not be refreshed" in calls["ask"][0]
    assert said[-1] == "Claude is logged in again"


def test_codex_gets_the_same(login):
    calls, answers = login
    answers.extend([
        AgentResult(ok=False, error_kind="error", error_message="refresh token was already used"),
        ALIVE,
    ])
    connect.ensure_agent_ready("codex", "subscription", "gpt-5", ask=lambda q: True)
    assert calls["relogin"] == ["codex"]


@pytest.mark.parametrize("ask", [None, lambda question: False])
def test_no_terminal_or_no_stops_with_the_fix(login, ask):
    calls, answers = login
    answers.append(DEAD)
    with pytest.raises(connect.AgentLoginError, match="hillclimb connect claude"):
        connect.ensure_agent_ready("claude-code", "subscription", "sonnet", ask=ask)
    assert calls["relogin"] == []


def test_any_other_failure_is_left_to_the_run(login):
    """A rate limit or a network blip is not a login: the command runs and
    reports it as it always did; nobody is asked to log in."""
    calls, answers = login
    answers.append(AgentResult(ok=False, error_kind="rate_limited", error_message="usage limit reached"))
    connect.ensure_agent_ready("claude-code", "subscription", "sonnet", ask=lambda q: pytest.fail("asked"))
    assert calls["relogin"] == []


def test_run_stops_before_launching_on_a_dead_login(login, tmp_path, monkeypatch):
    """No terminal (a test, a script): `hillclimb run` stops with the fix and
    launches nothing."""
    from typer.testing import CliRunner

    from hillclimb.cli import app

    calls, answers = login
    answers.append(DEAD)
    root = tmp_path / "proj"
    root.mkdir()
    write_config(root, {"agent": "claude-code"})
    monkeypatch.chdir(root)
    launched = []
    monkeypatch.setattr("hillclimb.cli.run._run_problem_fleet", lambda *a, **k: launched.append(a))
    result = CliRunner().invoke(app, ["run", "heilbronn-11"])
    assert result.exit_code == 1
    assert "the Claude login has expired" in result.output and "hillclimb connect claude" in result.output
    assert launched == []


def test_the_launcher_marks_its_engines_as_checked(tmp_path):
    from hillclimb.api import child_launch_context
    from hillclimb.config import Config

    config = Config()
    config.hillclimb_dir = tmp_path
    _, env = child_launch_context(config)
    assert env[connect.CHECKED_ENV] == "1"


def test_the_question_quotes_only_why_the_login_died(login):
    """The claude-code agent words a dead login with its own advice around
    the reason; the question quotes the reason alone, not the advice twice."""
    calls, answers = login
    answers.extend([
        AgentResult(
            ok=False, error_kind="login_expired",
            error_message="Claude login expired — run `hillclimb connect claude`, then `hillclimb resume` "
            "(Failed to authenticate: OAuth session expired and could not be refreshed)",
        ),
        ALIVE,
    ])
    asked = []
    connect.ensure_agent_ready("claude-code", "subscription", "sonnet", ask=lambda q: asked.append(q) or True)
    assert asked == [
        "Your Claude login has expired (Failed to authenticate: OAuth session expired and could not "
        "be refreshed). Log in again now?"
    ]


def test_resume_detaches_like_run_and_its_engine_does_not(tmp_path, monkeypatch):
    """`resume` hands the search to a background engine and returns; that
    engine is itself `resume <ref> --no-detach`, or it would detach forever."""
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from hillclimb.cli import app
    from hillclimb.cli import run as run_cli

    original_spawn = run_cli._spawn_resume  # before it is patched below
    write_config(tmp_path, {"agent": "dummy"})
    monkeypatch.chdir(tmp_path)
    record = SimpleNamespace(ref="r1/s1", state="parked", meta=SimpleNamespace(agent="dummy", model="m"))
    monkeypatch.setattr(run_cli.common, "open_search", lambda config, ref: (None, record))
    monkeypatch.setattr(run_cli, "_refuse_unless_resumable", lambda store, rec: None)
    monkeypatch.setattr(run_cli, "_stop_what_the_dead_engine_left", lambda rec: None)
    spawned = []
    monkeypatch.setattr(run_cli, "_spawn_resume", lambda config, rec: spawned.append(rec.ref) or (4242, tmp_path / "log"))

    result = CliRunner().invoke(app, ["resume"])
    assert result.exit_code == 0, result.output
    assert spawned == ["r1/s1"] and "in the background: pid 4242" in result.output

    # the engine's own command line
    launched = []

    class Popen:
        def __init__(self, cmd, **kw):
            launched.append(cmd)
            self.pid = 1

    monkeypatch.setattr(run_cli.subprocess, "Popen", Popen)
    run_dir = tmp_path / "runs" / "r1"
    real = SimpleNamespace(
        ref="r1/s1", search_dir=run_dir / "searches" / "s1", meta=SimpleNamespace(search_id="s1"),
    )
    from hillclimb.config import Config

    config = Config()
    config.hillclimb_dir = tmp_path
    original_spawn(config, real)
    assert launched[0][-3:] == ["resume", "r1/s1", "--no-detach"]


def test_claude_codes_not_logged_in_reply_counts_as_a_login_failure():
    from hillclimb.agents.base import login_expired
    assert login_expired("Not logged in · Please run /login")


@pytest.fixture
def logged_out(login, monkeypatch):
    """hillclimb's own home has never logged in: a new user who skipped
    `hillclimb connect` (their own Claude Code login is not the operators')."""
    calls, answers = login
    calls["login"] = []
    state = {"logged_in": False}

    def check(target, auth):
        if state["logged_in"]:
            return connect.Status(target, auth, "ready", "fine")
        return connect.Status(target, auth, "logged-out", "not logged in", "hillclimb connect claude")

    def run_login(target, auth):
        calls["login"].append(target)
        state["logged_in"] = True
        return 0

    monkeypatch.setattr(connect, "check", check)
    monkeypatch.setattr(connect, "run_login", run_login)
    return calls, answers


def test_a_home_never_logged_in_says_run_connect_before_any_call(logged_out):
    calls, _ = logged_out
    with pytest.raises(connect.AgentLoginError, match="run `hillclimb connect claude`"):
        connect.ensure_agent_ready("claude-code", "subscription", "sonnet")
    assert calls["ping"] == [] and calls["login"] == []


def test_a_home_never_logged_in_is_offered_the_login_right_there(logged_out):
    calls, answers = logged_out
    answers.append(ALIVE)
    asked = []
    connect.ensure_agent_ready(
        "claude-code", "subscription", "sonnet", ask=lambda q: asked.append(q) or True, say=lambda _: None,
    )
    assert "not logged in yet" in asked[0] and "separate from your own login" in asked[0]
    assert calls["login"] == ["claude"] and calls["ping"] == ["claude-code"]


def test_declining_the_login_stops_with_the_connect_command(logged_out):
    calls, _ = logged_out
    with pytest.raises(connect.AgentLoginError, match="hillclimb connect"):
        connect.ensure_agent_ready("claude-code", "subscription", "sonnet", ask=lambda q: False, say=lambda _: None)
    assert calls["login"] == []
