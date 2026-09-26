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
print(json.dumps({{"type": "system", "subtype": "init", "session_id": "sess-123",
                   "model": "claude-sonnet-4-5-20250929"}}))
print(json.dumps({{"type": "assistant", "message": {{"content": [{{"type": "text", "text": "working"}}]}}}}))
print({RESULT_LINE!r})
"""

STUB_RATE_LIMITED = f"""#!{sys.executable}
import json, sys
sys.stdin.read()
print(json.dumps({{"type": "system", "subtype": "init", "model": "claude-opus-5"}}))
print(json.dumps({{"type": "assistant", "message": {{"model": "<synthetic>",
                   "id": "synthetic-error", "usage": {{"input_tokens": 0, "output_tokens": 0}},
                   "content": [{{"type": "text", "text": "Usage limit reached, try later"}}]}}}}))
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
    candidate_dir = tmp_path / "ws"
    candidate_dir.mkdir(exist_ok=True)
    return OperatorRequest(
        operator="draft", prompt="write code", candidate_dir=candidate_dir, timeout_s=timeout_s
    )


def test_success_parses_result_and_streams(tmp_path: Path):
    backend = ClaudeCodeBackend(claude_bin=make_stub(tmp_path, STUB_OK))
    request = make_request(tmp_path)
    result = backend.invoke(request)

    assert result.ok
    assert result.session_id == "sess-123"
    assert result.model_id == "claude-sonnet-4-5-20250929"
    assert result.cost_usd == 0.42
    assert result.num_turns == 7
    assert result.total_tokens == 10 + 8209 + 29297 + 199902
    assert result.token_usage == {
        "input_tokens": 10,
        "output_tokens": 8209,
        "cache_creation_input_tokens": 29297,
        "cache_read_input_tokens": 199902,
    }
    stream = (request.candidate_dir / "agent_stream.jsonl").read_text().splitlines()
    assert len(stream) == 3  # init + assistant + result, all captured
    assert json.loads(stream[-1])["type"] == "result"
    assert all("ts" in json.loads(line) for line in stream)  # stamped on arrival
    assert not (request.candidate_dir / "agent.pid").exists()  # cleaned up
    raw = json.loads((request.candidate_dir / "agent_raw.json").read_text())
    assert raw["result"]["session_id"] == "sess-123"


def test_rate_limit_detected_in_stream(tmp_path: Path):
    backend = ClaudeCodeBackend(claude_bin=make_stub(tmp_path, STUB_RATE_LIMITED))
    request = make_request(tmp_path)
    result = backend.invoke(request)

    assert not result.ok
    assert result.error_kind == "rate_limited"
    assert "limit" in result.error_message.lower()
    assert result.model_id == "claude-opus-5"  # synthetic error must not replace it


def test_timeout_kills_and_cleans_pid(tmp_path: Path):
    backend = ClaudeCodeBackend(claude_bin=make_stub(tmp_path, STUB_SLEEPER))
    request = make_request(tmp_path, timeout_s=1)
    result = backend.invoke(request)

    assert not result.ok
    assert result.error_kind == "timeout"
    assert not (request.candidate_dir / "agent.pid").exists()


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


def test_quota_snapshots_bracket_the_call(tmp_path: Path, monkeypatch):
    """Subscription-window utilization is captured before and after every
    subscription-auth call, so each candidate journals its before/after."""
    snaps = iter(
        [
            {"fetched_at": "t0", "five_hour": {"utilization": 10}},
            {"fetched_at": "t1", "five_hour": {"utilization": 12}},
        ]
    )
    monkeypatch.setattr("hillclimb.harness.quota.snapshot", lambda: next(snaps))
    backend = ClaudeCodeBackend(claude_bin=make_stub(tmp_path, STUB_OK))
    result = backend.invoke(make_request(tmp_path))

    assert result.ok
    assert result.quota_start["five_hour"]["utilization"] == 10
    assert result.quota_end["five_hour"]["utilization"] == 12


def test_api_key_auth_skips_quota(tmp_path: Path, monkeypatch):
    """API-key runs are not subscription-billed — no window to measure."""
    monkeypatch.setattr(
        "hillclimb.harness.quota.snapshot",
        lambda: pytest.fail("api-key auth must not fetch quota"),
    )
    backend = ClaudeCodeBackend(claude_bin=make_stub(tmp_path, STUB_OK), auth="api-key")
    result = backend.invoke(make_request(tmp_path))

    assert result.ok
    assert result.quota_start is None
    assert result.quota_end is None


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


STUB_STREAMS_THEN_HANGS = f"""#!{sys.executable}
import json, sys, time
sys.stdin.read()
# two turns stream (the second twice under one id: partial then final), then
# the call hangs — the timeout path must still report the burned tokens
print(json.dumps({{"type": "assistant", "message": {{"id": "msg_1",
    "usage": {{"input_tokens": 100, "output_tokens": 50}},
    "content": [{{"type": "text", "text": "turn one"}}]}}}}), flush=True)
print(json.dumps({{"type": "assistant", "message": {{"id": "msg_2",
    "usage": {{"input_tokens": 10, "output_tokens": 1}},
    "content": [{{"type": "text", "text": "partial"}}]}}}}), flush=True)
print(json.dumps({{"type": "assistant", "message": {{"id": "msg_2",
    "usage": {{"input_tokens": 200, "output_tokens": 80}},
    "content": [{{"type": "text", "text": "final"}}]}}}}), flush=True)
time.sleep(30)
"""


def test_timeout_still_reports_streamed_tokens(tmp_path: Path):
    """A timed-out call burned real tokens; the per-turn stream (deduped by
    message id, final write wins) is the count of record when no result
    message ever lands."""
    backend = ClaudeCodeBackend(claude_bin=make_stub(tmp_path, STUB_STREAMS_THEN_HANGS))
    request = make_request(tmp_path, timeout_s=2)
    result = backend.invoke(request)

    assert not result.ok
    assert result.error_kind == "timeout"
    assert result.total_tokens == (100 + 50) + (200 + 80)  # msg_2 deduped to its final
    assert result.token_usage == {"input_tokens": 300, "output_tokens": 130}


STUB_ERROR_RESULT_WITH_USAGE = f"""#!{sys.executable}
import json, sys
sys.stdin.read()
print(json.dumps({{"type": "result", "subtype": "error", "is_error": True,
    "session_id": "sess-err", "total_cost_usd": 0.05, "num_turns": 2,
    "usage": {{"input_tokens": 500, "output_tokens": 300}},
    "result": "the task failed"}}))
sys.exit(1)
"""


def test_error_result_keeps_usage(tmp_path: Path):
    """An error-shaped result message still carries usage — journal it."""
    backend = ClaudeCodeBackend(claude_bin=make_stub(tmp_path, STUB_ERROR_RESULT_WITH_USAGE))
    request = make_request(tmp_path)
    result = backend.invoke(request)

    assert not result.ok
    assert result.error_kind == "error"
    assert result.total_tokens == 800
    assert result.cost_usd == 0.05
    assert result.num_turns == 2
