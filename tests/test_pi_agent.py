from __future__ import annotations

import json
from contextlib import closing
import os
from pathlib import Path
import stat
import sys
import threading
import time

import pytest

import hillclimb.harness.pricing as pricing
from hillclimb.agents.base import AgentRequest
from hillclimb.agents.pi_cli import PiCliAgent, pi_env
from hillclimb.agents.base import AgentResult
from hillclimb.config import Config
from hillclimb.harness.routing import AgentPool, Router


CATALOGUE = {
    "data": [
        {
            "id": "deepseek/deepseek-v3.2",
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        }
    ]
}


@pytest.fixture(autouse=True)
def _isolated_machine(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(pricing, "_MEMO", None)
    monkeypatch.setattr(pricing, "_LAST_FETCH_FAILURE", 0.0)
    monkeypatch.setattr(pricing, "_catalogue", lambda: CATALOGUE)


STUB_OK = f"""#!{sys.executable}
import json, os, sys
prompt = sys.stdin.read()
assert prompt == "write code"
assert os.path.exists("agent.pid")
expected = os.environ.get("STUB_EXPECT_SAMPLING")
if expected == "absent":
    assert "HILLCLIMB_SAMPLING" not in os.environ
elif expected:
    assert os.environ.get("HILLCLIMB_SAMPLING") == expected
print(json.dumps({{"type": "session", "id": "pi-session-123"}}))
message = {{
    "role": "assistant",
    "provider": "openrouter",
    "model": "deepseek/deepseek-v3.2",
    "content": [{{"type": "text", "text": "done"}}],
    "usage": {{
        "input": 20, "output": 25, "cacheRead": 80, "cacheWrite": 5,
        "totalTokens": 130,
        "cost": {{"input": 0.001, "output": 0.01, "cacheRead": 0.0001,
                 "cacheWrite": 0.0014, "total": 0.0125}},
    }},
    "stopReason": "stop",
}}
print(json.dumps({{"type": "message_end", "message": message}}))
print(json.dumps({{"type": "turn_end", "message": message, "toolResults": []}}))
"""

STUB_ERROR = f"""#!{sys.executable}
import json, sys
sys.stdin.read()
print(json.dumps({{"type": "session", "id": "pi-error"}}))
print(json.dumps({{"type": "message_end", "message": {{
    "role": "assistant", "provider": "anthropic", "model": "claude-sonnet-5",
    "content": [{{"type": "text", "text": ""}}],
    "usage": {{"input": 4, "output": 0, "cacheRead": 0, "cacheWrite": 0,
              "totalTokens": 4, "cost": {{"total": 0}}}},
    "stopReason": "error",
    "errorMessage": "400 temperature is deprecated for this model",
}}}}))
print(json.dumps({{"type": "turn_end", "message": {{"role": "assistant"}}}}))
"""


def error_stub(message: str) -> str:
    return f"""#!{sys.executable}
import json, sys
sys.stdin.read()
print(json.dumps({{"type": "session", "id": "pi-error"}}))
print(json.dumps({{"type": "message_end", "message": {{
    "role": "assistant", "provider": "openrouter", "model": "x",
    "content": [], "usage": {{"input": 1, "output": 0, "cacheRead": 0,
    "cacheWrite": 0, "cost": {{"total": 0}}}}, "stopReason": "error",
    "errorMessage": {message!r}
}}}}))
"""


STUB_ZERO_COST = f"""#!{sys.executable}
import json, sys
sys.stdin.read()
print(json.dumps({{"type": "session", "id": "pi-free"}}))
message = {{"role": "assistant", "provider": "openrouter",
 "model": "deepseek/deepseek-v3.2", "content": [], "stopReason": "stop",
 "usage": {{"input": 100, "output": 25, "cacheRead": 0, "cacheWrite": 0,
             "totalTokens": 125, "cost": {{"total": 0}}}}}}
print(json.dumps({{"type": "message_end", "message": message}}))
print(json.dumps({{"type": "turn_end", "message": message, "toolResults": []}}))
"""


def make_stub(tmp_path: Path, body: str) -> str:
    stub = tmp_path / f"pi-stub-{len(list(tmp_path.glob('pi-stub-*')))}"
    stub.write_text(body)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return str(stub)


def make_request(
    tmp_path: Path,
    *,
    sampling: dict[str, float] | None = None,
    resume: str | None = None,
) -> AgentRequest:
    candidate_dir = tmp_path / "run" / "searches" / "s1" / "candidates" / "c001"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    return AgentRequest(
        operator="draft",
        prompt="write code",
        candidate_dir=candidate_dir,
        timeout_s=30,
        model="openrouter/deepseek/deepseek-v3.2",
        sampling=sampling,
        resume_session_id=resume,
    )


def raw_command(request: AgentRequest) -> list[str]:
    return json.loads((request.candidate_dir / "agent_raw.json").read_text())["cmd"]


def test_pi_success_maps_stream_usage_cost_model_and_session(tmp_path: Path, monkeypatch):
    sampling = {"temperature": 0.9, "top_p": 0.95}
    monkeypatch.setenv(
        "STUB_EXPECT_SAMPLING",
        json.dumps(sampling, sort_keys=True, separators=(",", ":")),
    )
    agent = PiCliAgent(
        pi_bin=make_stub(tmp_path, STUB_OK), auth="openrouter"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    request = make_request(tmp_path, sampling=sampling)

    result = agent.invoke(request)

    assert result.ok
    assert result.session_id == "pi-session-123"
    assert result.model_id == "openrouter/deepseek/deepseek-v3.2"
    assert result.num_turns == 1
    assert result.total_tokens == 130
    assert result.token_usage == {
        "input_tokens": 20,
        "output_tokens": 25,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 80,
    }
    assert result.cost_usd == pytest.approx(0.0125)
    assert not (request.candidate_dir / "agent.pid").exists()
    stream = (request.candidate_dir / "agent_stream.jsonl").read_text().splitlines()
    assert [json.loads(line)["type"] for line in stream] == [
        "session",
        "message_end",
        "turn_end",
    ]
    cmd = raw_command(request)
    assert cmd[:4] == [agent.pi_bin, "-p", "--mode", "json"]
    assert cmd[cmd.index("--model") + 1] == request.model.removeprefix("openrouter/")
    assert cmd[cmd.index("--provider") + 1] == "openrouter"
    assert cmd[cmd.index("--session-dir") + 1] == str(
        request.candidate_dir.parents[1] / "pi-sessions"
    )
    assert "--no-extensions" in cmd and "-e" in cmd
    assert "--no-context-files" in cmd and "--tools" in cmd


def test_pi_sampling_is_absent_when_route_has_none(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HILLCLIMB_SAMPLING", '{"temperature":99}')
    monkeypatch.setenv("STUB_EXPECT_SAMPLING", "absent")
    agent = PiCliAgent(pi_bin=make_stub(tmp_path, STUB_OK))

    assert agent.invoke(make_request(tmp_path)).ok


def test_pi_resume_and_preflight_arguments(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("STUB_EXPECT_SAMPLING", "absent")
    agent = PiCliAgent(pi_bin=make_stub(tmp_path, STUB_OK))
    request = make_request(tmp_path, resume="pi-parent")

    assert agent.invoke(request).ok
    cmd = raw_command(request)
    assert cmd[cmd.index("--fork") + 1] == "pi-parent"

    preflight_dir = tmp_path / "preflight"
    preflight = request.model_copy(
        update={"candidate_dir": preflight_dir, "resume_session_id": None}
    )
    assert agent.preflight(preflight).ok
    preflight_cmd = raw_command(preflight)
    assert "--no-tools" in preflight_cmd
    assert "--tools" not in preflight_cmd


def test_pi_reads_error_stop_reason_even_on_exit_zero(tmp_path: Path):
    agent = PiCliAgent(pi_bin=make_stub(tmp_path, STUB_ERROR))
    result = agent.invoke(make_request(tmp_path))

    assert not result.ok
    assert result.error_kind == "error"
    assert "temperature is deprecated" in result.error_message
    assert result.total_tokens == 4


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("Usage rate limit reached; try later", "rate_limited"),
        ("402 Payment Required: insufficient credits", "out_of_credits"),
    ],
)
def test_pi_classifies_provider_errors(tmp_path: Path, message: str, kind: str):
    agent = PiCliAgent(pi_bin=make_stub(tmp_path, error_stub(message)))
    result = agent.invoke(make_request(tmp_path))
    assert not result.ok
    assert result.error_kind == kind


def test_pi_openrouter_zero_cost_falls_back_to_catalogue(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    agent = PiCliAgent(
        pi_bin=make_stub(tmp_path, STUB_ZERO_COST), auth="openrouter"
    )
    result = agent.invoke(make_request(tmp_path))

    assert result.ok
    assert result.cost_usd == pytest.approx(100e-6 + 25 * 2e-6)


def test_pi_openrouter_requires_key_before_spawn(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = PiCliAgent(pi_bin="/nonexistent/pi", auth="openrouter")
    result = agent.invoke(make_request(tmp_path))
    assert not result.ok
    assert result.error_kind == "error"
    assert "OPENROUTER_API_KEY" in result.error_message


def test_pi_home_isolated_auth_copy_settings_and_models(tmp_path: Path, monkeypatch):
    source_home = tmp_path / ".pi" / "agent"
    source_home.mkdir(parents=True)
    auth = source_home / "auth.json"
    auth.write_text('{"anthropic": "oauth"}')
    models = tmp_path / "local-models.json"
    models.write_text('{"providers": {"vllm": {"models": []}}}')

    env = pi_env("subscription", models)
    home = Path(env["PI_CODING_AGENT_DIR"])

    assert home.parent == tmp_path / ".cache" / "hillclimb" / "pi-home" / "subscription"
    assert (home / "auth.json").read_text() == auth.read_text()
    assert json.loads((home / "models.json").read_text()) == json.loads(models.read_text())
    settings = json.loads((home / "settings.json").read_text())
    assert settings["quietStartup"] is True
    assert env["PI_OFFLINE"] == "1"

    auth.write_text('{"anthropic": "new"}')
    os.utime(auth, (time.time() + 10, time.time() + 10))
    pi_env("subscription", models)
    assert (home / "auth.json").read_text() == auth.read_text()


def test_pi_missing_models_file_fails_before_spawn(tmp_path: Path):
    agent = PiCliAgent(
        pi_bin="/nonexistent/pi", models_file=tmp_path / "missing.json"
    )
    result = agent.invoke(make_request(tmp_path))
    assert not result.ok
    assert "pi.models_file does not exist" in result.error_message


def test_pi_stream_is_visible_to_live_usage_and_transcript(tmp_path: Path, monkeypatch):
    from hillclimb.tui.watch import _read_stream_usage, parse_stream_line

    monkeypatch.setenv("STUB_EXPECT_SAMPLING", "absent")
    agent = PiCliAgent(pi_bin=make_stub(tmp_path, STUB_OK))
    request = make_request(tmp_path)
    assert agent.invoke(request).ok

    usage = _read_stream_usage(request.candidate_dir)
    assert sum(item["output_tokens"] for item in usage.per_turn.values()) == 25
    assert sum(usage.cost_by_turn.values()) == pytest.approx(0.0125)
    entries = [
        parse_stream_line(line)
        for line in (request.candidate_dir / "agent_stream.jsonl").read_text().splitlines()
    ]
    assert entries[0].kind == "system"
    assert entries[1].text == "done"


class _PreflightPi:
    name = "pi"

    def __init__(self, result: AgentResult | None = None):
        self.requests: list[AgentRequest] = []
        self.result = result or AgentResult(ok=True)

    def preflight(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        return self.result


def test_preflight_deduplicates_model_sampling_routes(tmp_path: Path):
    from hillclimb.api import _preflight_pi_routes

    config = Config(
        agent="pi",
        model="openrouter/deepseek/deepseek-v3.2",
        routing={
            "default": {"sampling": {"temperature": 0.2}},
            "improve": {"sampling": {"temperature": 0.9}},
        },
    )
    agent = _PreflightPi()
    pool = AgentPool()
    pool.seed("pi", "subscription", agent)

    _preflight_pi_routes(config, tmp_path, Router(config), pool, lambda *_: None)

    assert len(agent.requests) == 2
    assert {request.sampling["temperature"] for request in agent.requests} == {0.2, 0.9}
    assert all(request.prompt == "Reply with exactly pong." for request in agent.requests)


def test_preflight_surfaces_provider_rejection(tmp_path: Path):
    from hillclimb.api import _preflight_pi_routes

    config = Config(
        agent="pi",
        routing={"default": {"sampling": {"temperature": 0.9}}},
    )
    agent = _PreflightPi(
        AgentResult(
            ok=False,
            error_kind="error",
            error_message="temperature is deprecated for this model",
        )
    )
    pool = AgentPool()
    pool.seed("pi", "subscription", agent)

    with pytest.raises(RuntimeError, match="temperature is deprecated"):
        _preflight_pi_routes(config, tmp_path, Router(config), pool, lambda *_: None)


def test_preflight_checks_every_pool_model_and_auth(tmp_path):
    from hillclimb.api import _preflight_pi_routes

    config = Config(agent="pi", model="a", routing={
        "default": {"models": ["a", "b"], "sampling": {"temperature": 0.4}},
        "draft": {"model": "c"},
        "debug": {"agent_auth": "api-key"},
    })
    subscribed, billed = _PreflightPi(), _PreflightPi()
    pool = AgentPool()
    pool.seed("pi", "subscription", subscribed)
    pool.seed("pi", "api-key", billed)
    _preflight_pi_routes(config, tmp_path, Router(config), pool, lambda *_: None)
    assert {r.model for r in subscribed.requests} == {"a", "b", "c"}
    assert {r.model for r in billed.requests} == {"a", "b"}


def test_search_preflight_failure_finalizes_before_evaluation(tmp_path, monkeypatch, task, config):
    from hillclimb.api import create_search, execute_search
    from hillclimb.harness.budget import BudgetManager
    from hillclimb.harness.dirs import create_run_dir
    from hillclimb.harness.store import key_for, open_store

    config.agent = "pi"
    config.learning.enabled = False
    config.learning.skills = False
    run_dir = create_run_dir(config.paths.runs_dir, "preflight-test")
    search_dir = create_search(config, task, run_dir, "preflight-test", 60)
    agent = PiCliAgent(pi_bin=make_stub(tmp_path, STUB_ERROR))
    monkeypatch.setattr("hillclimb.api.get_agent", lambda *args, **kwargs: agent)

    def unexpected_evaluation(*args, **kwargs):
        pytest.fail("provider rejection must happen before any evaluation")

    monkeypatch.setattr("hillclimb.api.build_evaluator", unexpected_evaluation)
    with pytest.raises(RuntimeError, match="temperature is deprecated"):
        execute_search(config, task, search_dir, BudgetManager(60), log=lambda *_: None)
    with closing(open_store(config)) as store:
        assert store.read_status(key_for(search_dir)).state == "failed"
    assert not list((search_dir / "candidates").iterdir())
    assert list((search_dir / "pi-preflight").glob("*/result.json"))


@pytest.mark.parametrize("abort_call", [False, True])
def test_timeout_and_abort_keep_usage_and_clean_pid(tmp_path, abort_call):
    body = STUB_OK + '\nimport time\nsys.stdout.flush()\ntime.sleep(30)\n'
    abort = threading.Event()
    agent = PiCliAgent(pi_bin=make_stub(tmp_path, body), abort=abort)
    request = make_request(tmp_path).model_copy(update={"timeout_s": 1 if not abort_call else 30})
    timer = threading.Timer(0.5, abort.set)
    if abort_call:
        timer.start()
    try:
        result = agent.invoke(request)
    finally:
        timer.cancel()
    assert not result.ok
    assert result.error_kind == ("aborted" if abort_call else "timeout")
    assert result.total_tokens == 130
    assert result.cost_usd == pytest.approx(0.0125)
    assert not (request.candidate_dir / "agent.pid").exists()


def test_retry_success_clears_old_error_and_sums_usage(tmp_path):
    from io import StringIO
    from hillclimb.agents.pi_cli import _PiStreamReader

    reader = _PiStreamReader(StringIO(), tmp_path / "stream", "m")
    for reason, error in [("error", "429 rate limit"), ("stop", "")]:
        reader._read_message({"type": "message_end", "message": {
            "role": "assistant", "model": "m", "stopReason": reason,
            "errorMessage": error, "usage": {"input": 10, "output": 2, "cost": {"total": 0.1}},
        }})
    assert not reader.rate_limited
    assert not reader.error_message
    assert reader.usage["input_tokens"] == 20
    assert reader.cost_usd == pytest.approx(0.2)


def test_different_local_provider_files_cannot_replace_each_other(tmp_path):
    from hillclimb.agents.pi_cli import pi_home

    paths = []
    for provider in ["one", "two"]:
        path = tmp_path / f"{provider}.json"
        path.write_text(json.dumps({"providers": {provider: {"apiKey": "test"}}}))
        paths.append(pi_home("api-key", path))
    default = pi_home("api-key")
    assert len(set([default, *paths])) == 3
    assert list(json.loads((paths[0] / "models.json").read_text())["providers"]) == ["one"]
    assert not (default / "models.json").exists()


def test_pi_auth_modes_and_missing_binary(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")
    subscribed = pi_env("subscription")
    assert all(key not in subscribed for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"))
    assert pi_env("api-key")["ANTHROPIC_API_KEY"] == "test-anthropic"
    routed = pi_env("openrouter")
    assert "ANTHROPIC_API_KEY" not in routed and "OPENAI_API_KEY" not in routed
    result = PiCliAgent(pi_bin="/nonexistent/pi").invoke(make_request(tmp_path))
    assert not result.ok and "could not start pi CLI" in result.error_message
