from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys
import threading
import time

import pytest

import hillclimb.harness.pricing as pricing
from hillclimb.backends.base import OperatorRequest
from hillclimb.backends.codex_cli import CodexCliBackend

# what OpenRouter would quote for the test model: $1/M prompt, $2/M completion
CATALOGUE = {
    "data": [
        {"id": "gpt-5", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}
    ]
}


@pytest.fixture(autouse=True)
def _isolated_machine(tmp_path: Path, monkeypatch):
    """The backend copies ~/.codex/auth.json into ~/.cache and prices calls
    from OpenRouter's live catalogue; neither may touch the developer's home
    or the network from a test."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(pricing, "_MEMO", None)
    monkeypatch.setattr(pricing, "_LAST_FETCH_FAILURE", 0.0)
    monkeypatch.setattr(
        pricing, "_fetch_catalogue", lambda: (_ for _ in ()).throw(AssertionError("network"))
    )
    monkeypatch.setattr(pricing, "_catalogue", lambda: CATALOGUE)


STUB_OK = f"""#!{sys.executable}
import json, os, sys
prompt = sys.stdin.read()
assert prompt == "write code"
assert os.path.exists("agent.pid")
print(json.dumps({{"type": "thread.started", "thread_id": "thread-123"}}))
print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": "working"}}}}))
print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 25}}}}))
"""

STUB_RATE_LIMITED = f"""#!{sys.executable}
import json, sys
sys.stdin.read()
print(json.dumps({{"type": "thread.started", "thread_id": "thread-rate"}}))
print(json.dumps({{"type": "error", "message": "Usage limit reached; try later"}}))
sys.exit(1)
"""

STUB_UNSUPPORTED_MODEL = f"""#!{sys.executable}
import json, sys
sys.stdin.read()
print(json.dumps({{"type": "thread.started", "thread_id": "thread-model"}}))
payload = {{"type": "error", "status": 400, "error": {{
    "type": "invalid_request_error",
    "message": "The requested model is not supported with this account.",
}}}}
print(json.dumps({{"type": "error", "message": json.dumps(payload)}}))
sys.exit(1)
"""

STUB_OUT_OF_CREDITS = f"""#!{sys.executable}
import json, sys
sys.stdin.read()
print(json.dumps({{"type": "thread.started", "thread_id": "thread-402"}}))
print(json.dumps({{"type": "turn.failed", "error": {{"message": (
    "unexpected status 402 Payment Required: This request requires more "
    "credits, or fewer max_tokens. To increase, adjust the key's monthly limit"
)}}}}))
sys.exit(1)
"""


STUB_EMPTY = f"""#!{sys.executable}
import sys
sys.stdin.read()
"""


def make_stub(tmp_path: Path, body: str) -> str:
    stub = tmp_path / "codex-stub"
    stub.write_text(body)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return str(stub)


def make_request(tmp_path: Path, resume: str | None = None) -> OperatorRequest:
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir(exist_ok=True)
    return OperatorRequest(
        operator="draft",
        prompt="write code",
        candidate_dir=candidate_dir,
        timeout_s=30,
        model="gpt-5",
        resume_session_id=resume,
    )


def test_codex_success_parses_usage_and_normalizes_stream(tmp_path: Path):
    backend = CodexCliBackend(codex_bin=make_stub(tmp_path, STUB_OK))
    request = make_request(tmp_path)
    result = backend.invoke(request)

    assert result.ok
    assert result.session_id == "thread-123"
    assert result.num_turns == 1
    # cached_input_tokens is a subset of input_tokens, not an extra charge.
    assert result.total_tokens == 125
    # OpenRouter nests the cached subset inside input_tokens; hillclimb's keys
    # are disjoint.
    assert result.token_usage == {
        "input_tokens": 20,
        "cache_read_input_tokens": 80,
        "output_tokens": 25,
    }
    assert not (request.candidate_dir / "agent.pid").exists()
    stream = [
        json.loads(line)
        for line in (request.candidate_dir / "agent_stream.jsonl").read_text().splitlines()
    ]
    assert [message["type"] for message in stream] == ["system", "assistant", "result"]
    assert all("ts" in message for message in stream)

    raw = json.loads((request.candidate_dir / "agent_raw.json").read_text())
    assert raw["session_id"] == "thread-123"
    assert raw["cmd"][-3:] == ["--json", "--skip-git-repo-check", "-"]
    assert raw["cmd"][raw["cmd"].index("--model") + 1] == "gpt-5"


def test_codex_resume_command_uses_the_parent_thread(tmp_path: Path):
    backend = CodexCliBackend(codex_bin=make_stub(tmp_path, STUB_OK))
    request = make_request(tmp_path, resume="thread-parent")
    result = backend.invoke(request)

    assert result.ok
    cmd = json.loads((request.candidate_dir / "agent_raw.json").read_text())["cmd"]
    assert cmd[cmd.index("exec") + 1 :][-6:] == [
        "resume",
        "--all",
        "--json",
        "--skip-git-repo-check",
        "thread-parent",
        "-",
    ]


def test_codex_rate_limit_is_classified(tmp_path: Path):
    backend = CodexCliBackend(codex_bin=make_stub(tmp_path, STUB_RATE_LIMITED))
    result = backend.invoke(make_request(tmp_path))

    assert not result.ok
    assert result.error_kind == "rate_limited"
    assert "limit" in result.error_message.lower()


def test_codex_extracts_nested_server_error(tmp_path: Path):
    backend = CodexCliBackend(codex_bin=make_stub(tmp_path, STUB_UNSUPPORTED_MODEL))
    result = backend.invoke(make_request(tmp_path))

    assert not result.ok
    assert result.error_kind == "error"
    assert result.error_message == "The requested model is not supported with this account."


def test_codex_requires_a_completed_turn(tmp_path: Path):
    backend = CodexCliBackend(codex_bin=make_stub(tmp_path, STUB_EMPTY))
    result = backend.invoke(make_request(tmp_path))

    assert not result.ok
    assert result.error_kind == "error"
    assert "completed turn" in result.error_message


def test_codex_subscription_auth_ignores_inherited_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-leak")
    backend = CodexCliBackend(codex_bin=make_stub(tmp_path, STUB_OK))
    result = backend.invoke(make_request(tmp_path))
    assert result.ok

    from hillclimb.backends.codex_cli import codex_env

    assert "OPENAI_API_KEY" not in codex_env("subscription")
    assert codex_env("api-key")["OPENAI_API_KEY"] == "should-not-leak"


STUB_TURN_THEN_ERROR = f"""#!{sys.executable}
import json, sys
sys.stdin.read()
print(json.dumps({{"type": "thread.started", "thread_id": "thread-err"}}))
print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 400, "cached_input_tokens": 100, "output_tokens": 50}}}}))
print(json.dumps({{"type": "error", "message": "server exploded mid-call"}}))
sys.exit(1)
"""


def test_codex_error_path_keeps_streamed_usage(tmp_path: Path):
    """A call that completed a turn before dying burned real tokens; the
    error result must journal them, not drop them with the failure."""
    backend = CodexCliBackend(codex_bin=make_stub(tmp_path, STUB_TURN_THEN_ERROR))
    result = backend.invoke(make_request(tmp_path))

    assert not result.ok
    assert result.error_kind == "error"
    # input_tokens already includes the cached subset (see _normalized_usage)
    assert result.total_tokens == 400 + 50


def test_openrouter_auth_adds_the_provider_overrides(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    backend = CodexCliBackend(codex_bin=make_stub(tmp_path, STUB_OK), auth="openrouter")
    request = make_request(tmp_path)
    result = backend.invoke(request)

    assert result.ok
    # STUB_OK reports 100 input (80 cached) + 25 output tokens
    assert result.cost_usd == pytest.approx(20 * 1e-6 + 25 * 2e-6)
    cmd = json.loads((request.candidate_dir / "agent_raw.json").read_text())["cmd"]
    assert "model_provider=openrouter" in cmd
    provider = [arg for arg in cmd if arg.startswith("model_providers.openrouter=")]
    assert len(provider) == 1
    assert 'base_url="https://openrouter.ai/api/v1"' in provider[0]
    assert 'wire_api="responses"' in provider[0]
    assert 'env_key="OPENROUTER_API_KEY"' in provider[0]


def test_subscription_auth_has_no_provider_overrides(tmp_path: Path):
    backend = CodexCliBackend(codex_bin=make_stub(tmp_path, STUB_OK), auth="subscription")
    request = make_request(tmp_path)
    backend.invoke(request)

    cmd = json.loads((request.candidate_dir / "agent_raw.json").read_text())["cmd"]
    assert not any(arg.startswith("model_providers.") for arg in cmd)
    assert "model_provider=openrouter" not in cmd


def test_openrouter_auth_without_a_key_fails_before_spawning(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    backend = CodexCliBackend(codex_bin="/nonexistent/codex", auth="openrouter")
    result = backend.invoke(make_request(tmp_path))

    assert not result.ok
    assert result.error_kind == "error"
    assert "OPENROUTER_API_KEY" in result.error_message


def test_openrouter_auth_drops_an_inherited_openai_key(monkeypatch):
    from hillclimb.backends.codex_cli import codex_env

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-leak")
    env = codex_env("openrouter")

    assert env["OPENROUTER_API_KEY"] == "sk-or-test"
    assert "OPENAI_API_KEY" not in env


def test_codex_home_is_isolated_per_auth(tmp_path: Path, monkeypatch):
    from hillclimb.backends.codex_cli import codex_env

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    env = codex_env("openrouter")

    home = Path(env["CODEX_HOME"])
    assert home == tmp_path / ".cache" / "hillclimb" / "codex-home" / "openrouter"
    assert home.is_dir()
    assert not (home / "auth.json").exists()


def test_subscription_codex_home_gets_the_login(tmp_path: Path, monkeypatch):
    from hillclimb.backends.codex_cli import codex_env

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    real_codex = tmp_path / ".codex"
    real_codex.mkdir()
    (real_codex / "auth.json").write_text('{"token": "login"}')
    (real_codex / "config.toml").write_text('model = "gpt-5.6"\n')

    home = Path(codex_env("subscription")["CODEX_HOME"])

    assert (home / "auth.json").read_text() == '{"token": "login"}'
    assert not (home / "config.toml").exists()


def test_subscription_codex_home_refreshes_a_stale_login(tmp_path: Path, monkeypatch):
    from hillclimb.backends.codex_cli import codex_env

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    real_codex = tmp_path / ".codex"
    real_codex.mkdir()
    source = real_codex / "auth.json"
    source.write_text('{"token": "old"}')
    home = Path(codex_env("subscription")["CODEX_HOME"])

    source.write_text('{"token": "new"}')
    os.utime(source, (time.time() + 10, time.time() + 10))
    codex_env("subscription")

    assert (home / "auth.json").read_text() == '{"token": "new"}'


def test_concurrent_operators_copy_the_login_without_colliding(tmp_path: Path, monkeypatch):
    """Operators are threads of one process: a staging file named per pid
    made the second thread's rename fail with FileNotFoundError."""
    import shutil

    from hillclimb.backends.codex_cli import codex_env

    real_codex = tmp_path / ".codex"
    real_codex.mkdir()
    (real_codex / "auth.json").write_text('{"token": "login"}')

    # hold both threads after their copy, before either renames
    barrier = threading.Barrier(2, timeout=5)
    real_copy2 = shutil.copy2

    def copy_then_wait(src, dst, **kwargs):
        real_copy2(src, dst, **kwargs)
        barrier.wait()

    monkeypatch.setattr(shutil, "copy2", copy_then_wait)
    errors: list[BaseException] = []

    def worker():
        try:
            codex_env("subscription")
        except BaseException as exc:  # noqa: BLE001 — collected for the assert
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    home = tmp_path / ".cache" / "hillclimb" / "codex-home" / "subscription"
    assert (home / "auth.json").read_text() == '{"token": "login"}'
    assert [p.name for p in home.iterdir()] == ["auth.json"]  # no staging leftovers


def test_usage_mapping_un_nests_cache_kinds():
    from hillclimb.backends.codex_cli import _normalized_usage

    mapped = _normalized_usage(
        {
            "input_tokens": 1000,
            "cached_input_tokens": 800,
            "cache_write_input_tokens": 50,
            "output_tokens": 200,
            "reasoning_output_tokens": 150,
        }
    )

    assert mapped == {
        "input_tokens": 200,
        "cache_read_input_tokens": 800,
        "cache_creation_input_tokens": 50,
        "output_tokens": 200,
    }


def test_usage_mapping_survives_a_provider_without_caching():
    from hillclimb.backends.codex_cli import _normalized_usage

    assert _normalized_usage({"input_tokens": 500, "output_tokens": 30}) == {
        "input_tokens": 500,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": 30,
    }


def test_out_of_credits_is_its_own_error_kind(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    backend = CodexCliBackend(
        codex_bin=make_stub(tmp_path, STUB_OUT_OF_CREDITS), auth="openrouter"
    )
    result = backend.invoke(make_request(tmp_path))

    assert not result.ok
    # not rate_limited: credits do not come back on their own
    assert result.error_kind == "out_of_credits"
    assert "credits" in result.error_message.lower()


def test_the_provider_key_never_reaches_an_artifact(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-secret-value")
    backend = CodexCliBackend(codex_bin=make_stub(tmp_path, STUB_OK), auth="openrouter")
    request = make_request(tmp_path)
    result = backend.invoke(request)

    assert result.ok
    written = "\n".join(
        path.read_text(errors="replace")
        for path in request.candidate_dir.rglob("*")
        if path.is_file()
    )
    assert "sk-or-v1-secret-value" not in written
    assert "sk-or-v1-secret-value" not in json.dumps(result.model_dump(mode="json"))


STUB_RECOVERED_ERROR = f"""#!{sys.executable}
import json, sys
sys.stdin.read()
print(json.dumps({{"type": "thread.started", "thread_id": "thread-retry"}}))
print(json.dumps({{"type": "error", "message": (
    "Reconnecting... 1/5 (unexpected status 502 Bad Gateway: Server tool "
    "request failed, url: https://openrouter.ai/api/v1/responses)"
)}}))
print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": "done"}}}}))
print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 10, "output_tokens": 5}}}}))
"""


def test_a_transient_error_the_turn_recovered_from_is_not_fatal(tmp_path: Path):
    backend = CodexCliBackend(codex_bin=make_stub(tmp_path, STUB_RECOVERED_ERROR))
    result = backend.invoke(make_request(tmp_path))

    # codex retried the 502 itself and finished the turn; the candidate's work
    # is real and must not be thrown away
    assert result.ok
    assert result.error_kind is None
