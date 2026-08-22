from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from hillclimb.backends.base import OperatorRequest
from hillclimb.backends.claude_code import ClaudeCodeBackend

RESULT_LINE = json.dumps(
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "session_id": "sess-123",
        "total_cost_usd": 0.42,
        "num_turns": 7,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 8209,
            "cache_creation_input_tokens": 29297,
            "cache_read_input_tokens": 199902,
        },
        "result": "done",
    }
)

STUB_OK = f"""#!{sys.executable}
import json, os, sys
sys.stdin.read()
# the pid file must exist while we run (exposed for the heartbeat/TUI)
assert os.path.exists("agent.pid"), "agent.pid missing during call"
print(json.dumps({{"type": "system", "subtype": "init", "session_id": "sess-123"}}))
print(json.dumps({{"type": "assistant", "message": {{"content": [{{"type": "text", "text": "working"}}]}}}}))
print({RESULT_LINE!r})
"""

STUB_RATE_LIMITED = f"""#!{sys.executable}
import json, sys
sys.stdin.read()
print(json.dumps({{"type": "result", "subtype": "error", "is_error": True,
                   "session_id": "sess-rl", "result": "Usage limit reached, try later"}}))
sys.exit(1)
"""

STUB_SLEEPER = f"""#!{sys.executable}
import sys, time
sys.stdin.read()
time.sleep(30)
"""

STUB_CRASH = f"""#!{sys.executable}
import sys
sys.stdin.read()
print("garbage that is not json")
sys.stderr.write("something exploded\\n")
sys.exit(3)
"""


def make_stub(tmp_path: Path, body: str) -> str:
    stub = tmp_path / "claude-stub"
    stub.write_text(body)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return str(stub)


def make_request(tmp_path: Path, timeout_s: int = 30) -> OperatorRequest:
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    return OperatorRequest(
        operator="draft", prompt="write code", workspace=workspace, timeout_s=timeout_s
    )


def test_success_parses_result_and_streams(tmp_path: Path):
    backend = ClaudeCodeBackend(claude_bin=make_stub(tmp_path, STUB_OK))
    request = make_request(tmp_path)
    result = backend.invoke(request)

    assert result.ok
    assert result.session_id == "sess-123"
    assert result.cost_usd == 0.42
    assert result.num_turns == 7
    assert result.total_tokens == 10 + 8209 + 29297 + 199902
    stream = (request.workspace / "agent_stream.jsonl").read_text().splitlines()
    assert len(stream) == 3  # init + assistant + result, all captured
    assert json.loads(stream[-1])["type"] == "result"
    assert not (request.workspace / "agent.pid").exists()  # cleaned up
    raw = json.loads((request.workspace / "agent_raw.json").read_text())
    assert raw["result"]["session_id"] == "sess-123"


def test_rate_limit_detected_in_stream(tmp_path: Path):
    backend = ClaudeCodeBackend(claude_bin=make_stub(tmp_path, STUB_RATE_LIMITED))
    request = make_request(tmp_path)
    result = backend.invoke(request)

    assert not result.ok
    assert result.error_kind == "rate_limited"
    assert "limit" in result.error_message.lower()


def test_timeout_kills_and_cleans_pid(tmp_path: Path):
    backend = ClaudeCodeBackend(claude_bin=make_stub(tmp_path, STUB_SLEEPER))
    request = make_request(tmp_path, timeout_s=1)
    result = backend.invoke(request)

    assert not result.ok
    assert result.error_kind == "timeout"
    assert not (request.workspace / "agent.pid").exists()


def test_nonzero_exit_is_error(tmp_path: Path):
    backend = ClaudeCodeBackend(claude_bin=make_stub(tmp_path, STUB_CRASH))
    request = make_request(tmp_path)
    result = backend.invoke(request)

    assert not result.ok
    assert result.error_kind == "error"
    assert "exploded" in result.error_message


def test_zero_exit_without_result_is_error(tmp_path: Path):
    stub = f"#!{sys.executable}\nimport sys\nsys.stdin.read()\n"
    backend = ClaudeCodeBackend(claude_bin=make_stub(tmp_path, stub))
    request = make_request(tmp_path)
    result = backend.invoke(request)

    assert not result.ok
    assert result.error_kind == "error"
    assert "no result message" in result.error_message


STUB_MENTIONS_LIMITS = f"""#!{sys.executable}
import json, sys
sys.stdin.read()
# transcript that MENTIONS limits while succeeding — must not park the run
print(json.dumps({{"type": "assistant", "message": {{"content": [{{"type": "text",
    "text": "the bug was a rate limit string; also 'usage limit reached' in a comment"}}]}}}}))
print(json.dumps({{"type": "result", "subtype": "success", "is_error": False,
                   "session_id": "sess-ok", "result": "fixed the usage limit handling code"}}))
"""


def test_mentioning_rate_limits_in_transcript_is_not_rate_limited(tmp_path: Path):
    """Regression: a debug agent whose *output text* discussed limit strings
    got misclassified as rate_limited and parked a healthy run."""
    backend = ClaudeCodeBackend(claude_bin=make_stub(tmp_path, STUB_MENTIONS_LIMITS))
    request = make_request(tmp_path)
    result = backend.invoke(request)

    assert result.ok
    assert result.error_kind is None
    assert result.session_id == "sess-ok"
