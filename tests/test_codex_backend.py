from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys

from hillclimb.backends.base import OperatorRequest
from hillclimb.backends.codex_cli import CodexCliBackend


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
    assert result.token_usage == {"input_tokens": 100, "output_tokens": 25}
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
