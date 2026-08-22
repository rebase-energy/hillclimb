from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from hillclimb.backends.base import OperatorRequest, OperatorResult


def subscription_env(auth: str = "subscription") -> dict[str, str]:
    """Child env for `claude`. auth="subscription" (default) drops
    ANTHROPIC_API_KEY so calls bill the Max subscription (claude.ai login /
    CLAUDE_CODE_OAUTH_TOKEN) — an inherited API key silently takes precedence
    otherwise. auth="api-key" keeps it (headless/hosted runs with no
    subscription login)."""
    env = os.environ.copy()
    if auth != "api-key":
        env.pop("ANTHROPIC_API_KEY", None)
    return env

RATE_LIMIT_MARKERS = (
    "rate limit",
    "usage limit",
    "rate_limit",
    "usage_limit",
    "limit reached",
    "out of extra usage",
)

STREAM_FILE = "agent_stream.jsonl"
PID_FILE = "agent.pid"


def _has_rate_limit_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in RATE_LIMIT_MARKERS)


class _StreamReader(threading.Thread):
    """Drains the agent's stdout line-by-line into agent_stream.jsonl so the
    transcript is observable while the call is still running (watch TUI tails
    this file), and keeps the final `result` message for the caller.

    Rate-limit markers are deliberately NOT matched against ordinary
    transcript lines — an agent that merely *mentions* limits (or fixes code
    that does) would park the run. Only error-shaped messages count: a
    `result` with is_error, or non-JSON noise lines (CLI error banners)."""

    def __init__(self, stdout, stream_path: Path):
        super().__init__(daemon=True, name="agent-stream-reader")
        self.stdout = stdout
        self.stream_path = stream_path
        self.result_payload: dict | None = None
        self.rate_limited = False

    def run(self) -> None:
        with self.stream_path.open("w") as sink:
            for line in self.stdout:
                sink.write(line)
                sink.flush()
                try:
                    message = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    # non-JSON output from the CLI itself (error banners)
                    if _has_rate_limit_marker(line):
                        self.rate_limited = True
                    continue
                if isinstance(message, dict) and message.get("type") == "result":
                    self.result_payload = message
                    if message.get("is_error") and _has_rate_limit_marker(
                        str(message.get("result", ""))
                    ):
                        self.rate_limited = True


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()


class ClaudeCodeBackend:
    """One operator call = one headless Claude Code invocation, cwd-scoped to
    the node workspace. Auth comes from the interactive `claude` login (Max
    subscription) or CLAUDE_CODE_OAUTH_TOKEN in the environment.

    Runs with `--output-format stream-json` so the transcript lands
    incrementally in <workspace>/agent_stream.jsonl, and exposes the child
    pid in <workspace>/agent.pid while the call is in flight."""

    name = "claude-code"

    def __init__(
        self,
        claude_bin: str = "claude",
        auth: str = "subscription",
        abort: "threading.Event | None" = None,
    ):
        self.claude_bin = claude_bin
        self.auth = auth  # subscription | api-key (see subscription_env)
        self.abort = abort  # set → kill the agent and report error_kind="aborted"

    def invoke(self, request: OperatorRequest) -> OperatorResult:
        cmd = [
            self.claude_bin,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",  # required by the CLI for -p with stream-json
            "--permission-mode",
            "bypassPermissions",
            "--model",
            request.model,
        ]
        if request.resume_session_id:
            cmd += ["--resume", request.resume_session_id]

        workspace = Path(request.workspace)
        stream_path = workspace / STREAM_FILE
        pid_path = workspace / PID_FILE
        stderr_path = workspace / "agent_stderr.log"
        start = time.monotonic()
        timed_out = False
        aborted = False
        reader: _StreamReader | None = None
        proc: subprocess.Popen | None = None
        try:
            with stderr_path.open("w") as stderr_sink:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr_sink,
                    text=True,
                    cwd=request.workspace,
                    env=subscription_env(self.auth),
                    start_new_session=True,  # own process group → killable as a unit
                )
                pid_path.write_text(str(proc.pid))
                reader = _StreamReader(proc.stdout, stream_path)
                reader.start()
                try:
                    proc.stdin.write(request.prompt)
                    proc.stdin.close()
                except BrokenPipeError:
                    pass  # process died instantly; returncode tells the story
                deadline = time.monotonic() + request.timeout_s
                while proc.poll() is None:
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        if self.abort is not None and self.abort.is_set():
                            aborted = True
                            _kill_group(proc)
                            break
                        if time.monotonic() >= deadline:
                            timed_out = True
                            _kill_group(proc)
                            break
        finally:
            pid_path.unlink(missing_ok=True)
            if proc is not None and proc.poll() is None:
                # e.g. StopRequested raised by a signal handler while in wait()
                _kill_group(proc)
            if reader is not None:
                reader.join(timeout=5)

        duration = time.monotonic() - start
        payload = reader.result_payload or {}
        stderr_text = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""

        raw_path = workspace / "agent_raw.json"
        raw_path.write_text(
            json.dumps(
                {
                    "cmd": cmd,
                    "returncode": proc.returncode,
                    "stderr": stderr_text[-4000:],
                    "result": payload or None,
                }
            )
        )

        if aborted:
            return OperatorResult(
                ok=False,
                duration_s=duration,
                raw_output_path=str(raw_path),
                error_kind="aborted",
                error_message="agent call aborted (stop requested)",
            )
        if timed_out:
            return OperatorResult(
                ok=False,
                duration_s=duration,
                raw_output_path=str(raw_path),
                error_kind="timeout",
                error_message=f"agent call exceeded {request.timeout_s}s",
            )
        if reader.rate_limited or _has_rate_limit_marker(stderr_text):
            return OperatorResult(
                ok=False,
                session_id=payload.get("session_id"),
                duration_s=duration,
                raw_output_path=str(raw_path),
                error_kind="rate_limited",
                error_message=str(payload.get("result") or stderr_text)[:500],
            )
        if proc.returncode != 0 or payload.get("is_error"):
            return OperatorResult(
                ok=False,
                session_id=payload.get("session_id"),
                duration_s=duration,
                raw_output_path=str(raw_path),
                error_kind="error",
                error_message=str(payload.get("result") or stderr_text or "")[:500],
            )
        if not payload:
            return OperatorResult(
                ok=False,
                duration_s=duration,
                raw_output_path=str(raw_path),
                error_kind="error",
                error_message="agent exited 0 but emitted no result message",
            )
        usage = payload.get("usage") or {}
        token_keys = (
            "input_tokens", "output_tokens",
            "cache_creation_input_tokens", "cache_read_input_tokens",
        )
        total_tokens = sum(usage.get(k) or 0 for k in token_keys) or None
        return OperatorResult(
            ok=True,
            session_id=payload.get("session_id"),
            cost_usd=payload.get("total_cost_usd"),
            num_turns=payload.get("num_turns"),
            total_tokens=total_tokens,
            duration_s=duration,
            raw_output_path=str(raw_path),
        )
