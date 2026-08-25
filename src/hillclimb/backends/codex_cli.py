"""Codex CLI operator backend.

One Hillclimb operator call becomes one non-interactive ``codex exec`` turn
inside the candidate directory. Codex's JSONL events are normalized to the
same small stream format consumed by Hillclimb's live candidate viewer.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from hillclimb.backends.base import OperatorRequest, OperatorResult
from hillclimb.backends.claude_code import (
    PID_FILE,
    RATE_LIMIT_MARKERS,
    STREAM_FILE,
    _kill_group,
    usage_total_tokens,
)
from hillclimb.candidate import utcnow


def codex_env(auth: str = "subscription") -> dict[str, str]:
    """Build the child environment for ChatGPT-login or API-key auth."""
    from hillclimb.executor import single_threaded

    env = os.environ.copy()
    if auth != "api-key":
        # An inherited key can take precedence over the interactive login.
        env.pop("OPENAI_API_KEY", None)
    return single_threaded(env)


def _has_rate_limit_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in RATE_LIMIT_MARKERS)


def _normalized_usage(usage: dict) -> dict:
    """Responses input_tokens already includes the cached-token subset."""
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
    }


def _error_message(message: dict) -> str:
    """Extract Codex errors, including JSON embedded in an event message."""
    error = message.get("error")
    value = message.get("message")
    if value is None:
        value = error.get("message") if isinstance(error, dict) else error
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
        if isinstance(payload, dict):
            nested = payload.get("error")
            if isinstance(nested, dict) and nested.get("message"):
                return str(nested["message"])
            if payload.get("message"):
                return str(payload["message"])
    return str(value or message)


class _CodexStreamReader(threading.Thread):
    def __init__(self, stdout, stream_path: Path, model: str):
        super().__init__(daemon=True, name="codex-stream-reader")
        self.stdout = stdout
        self.stream_path = stream_path
        self.model = model
        self.session_id: str | None = None
        self.usage: dict = {}
        self.num_turns = 0
        self.completed = False
        self.started = threading.Event()
        self.rate_limited = False
        self.error_message = ""

    def _normalize(self, message: dict) -> dict:
        event_type = message.get("type")
        if event_type == "thread.started":
            self.session_id = message.get("thread_id")
            self.started.set()
            return {
                "type": "system",
                "subtype": "init",
                "session_id": self.session_id,
                "model": self.model,
            }
        if event_type == "item.completed":
            item = message.get("item") or {}
            item_type = item.get("type")
            if item_type == "agent_message":
                return {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": item.get("text", "")}]
                    },
                }
            if item_type == "command_execution":
                return {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Shell",
                                "input": {"command": item.get("command", "")},
                            }
                        ]
                    },
                }
            if item_type == "file_change":
                paths = [
                    change.get("path", "")
                    for change in item.get("changes") or []
                    if isinstance(change, dict)
                ]
                return {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Edit",
                                "input": {"file_path": ", ".join(filter(None, paths))},
                            }
                        ]
                    },
                }
        if event_type == "turn.completed":
            self.completed = True
            self.num_turns += 1
            self.usage = _normalized_usage(message.get("usage") or {})
            return {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "session_id": self.session_id,
                "num_turns": self.num_turns,
                "usage": self.usage,
            }
        if event_type in {"error", "turn.failed"}:
            self.error_message = _error_message(message)
            self.rate_limited = _has_rate_limit_marker(self.error_message)
            return {
                "type": "result",
                "subtype": "error",
                "is_error": True,
                "session_id": self.session_id,
                "num_turns": self.num_turns,
                "result": self.error_message,
            }
        # Preserve unfamiliar events for diagnostics; the TUI simply skips them.
        return message

    def run(self) -> None:
        with self.stream_path.open("w") as sink:
            for line in self.stdout:
                try:
                    message = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    sink.write(line)
                    sink.flush()
                    if _has_rate_limit_marker(line):
                        self.rate_limited = True
                    continue
                if not isinstance(message, dict):
                    continue
                normalized = self._normalize(message)
                normalized.setdefault("ts", utcnow())
                sink.write(json.dumps(normalized) + "\n")
                sink.flush()


class CodexCliBackend:
    """Run Hillclimb operators through an authenticated local Codex CLI."""

    name = "codex"

    def __init__(
        self,
        codex_bin: str = "codex",
        auth: str = "subscription",
        abort: "threading.Event | None" = None,
    ):
        self.codex_bin = codex_bin
        self.auth = auth
        self.abort = abort

    def _command(self, request: OperatorRequest) -> list[str]:
        cmd = [
            self.codex_bin,
            "--model",
            request.model,
            "--sandbox",
            "workspace-write",
            "--ask-for-approval",
            "never",
            "--cd",
            str(Path(request.candidate_dir).resolve()),
            "exec",
        ]
        if request.resume_session_id:
            cmd += [
                "resume",
                "--all",
                "--json",
                "--skip-git-repo-check",
                request.resume_session_id,
                "-",
            ]
        else:
            cmd += ["--json", "--skip-git-repo-check", "-"]
        return cmd

    def invoke(self, request: OperatorRequest) -> OperatorResult:
        cmd = self._command(request)
        candidate_dir = Path(request.candidate_dir)
        stream_path = candidate_dir / STREAM_FILE
        pid_path = candidate_dir / PID_FILE
        stderr_path = candidate_dir / "agent_stderr.log"
        raw_path = candidate_dir / "agent_raw.json"
        start = time.monotonic()
        timed_out = False
        aborted = False
        spawn_error = ""
        reader: _CodexStreamReader | None = None
        proc: subprocess.Popen | None = None

        try:
            with stderr_path.open("w") as stderr_sink:
                # Codex synchronizes shared system skills at startup. Protect
                # just that phase across independent Hillclimb search
                # processes; calls run concurrently once their thread starts.
                start_lock = Path(tempfile.gettempdir()) / "hillclimb-codex-start.lock"
                with start_lock.open("w") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    try:
                        proc = subprocess.Popen(
                            cmd,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=stderr_sink,
                            text=True,
                            cwd=candidate_dir,
                            env=codex_env(self.auth),
                            start_new_session=True,
                        )
                    except OSError as exc:
                        spawn_error = str(exc)
                    if proc is not None:
                        pid_path.write_text(str(proc.pid))
                        reader = _CodexStreamReader(proc.stdout, stream_path, request.model)
                        reader.start()
                        try:
                            proc.stdin.write(request.prompt)
                            proc.stdin.close()
                        except BrokenPipeError:
                            pass
                        deadline = time.monotonic() + request.timeout_s
                        startup_deadline = min(deadline, time.monotonic() + 15)
                        while (
                            proc.poll() is None
                            and not reader.started.wait(timeout=0.1)
                            and time.monotonic() < startup_deadline
                        ):
                            if self.abort is not None and self.abort.is_set():
                                aborted = True
                                _kill_group(proc)
                                break
                if proc is not None:
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
                _kill_group(proc)
            if reader is not None:
                reader.join(timeout=5)

        duration = time.monotonic() - start
        stderr_text = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""
        raw_path.write_text(
            json.dumps(
                {
                    "cmd": cmd,
                    "returncode": proc.returncode if proc is not None else None,
                    "stderr": stderr_text[-4000:],
                    "session_id": reader.session_id if reader is not None else None,
                }
            )
        )

        common = {"duration_s": duration, "raw_output_path": str(raw_path)}
        if spawn_error:
            return OperatorResult(
                ok=False,
                error_kind="error",
                error_message=f"could not start Codex CLI: {spawn_error}",
                **common,
            )
        if aborted:
            return OperatorResult(
                ok=False,
                error_kind="aborted",
                error_message="agent call aborted (stop requested)",
                **common,
            )
        if timed_out:
            return OperatorResult(
                ok=False,
                error_kind="timeout",
                error_message=f"agent call exceeded {request.timeout_s}s",
                **common,
            )
        assert proc is not None and reader is not None
        if reader.rate_limited or _has_rate_limit_marker(stderr_text):
            return OperatorResult(
                ok=False,
                session_id=reader.session_id,
                error_kind="rate_limited",
                error_message=(reader.error_message or stderr_text)[:500],
                **common,
            )
        if proc.returncode != 0 or reader.error_message:
            return OperatorResult(
                ok=False,
                session_id=reader.session_id,
                error_kind="error",
                error_message=(reader.error_message or stderr_text)[:500],
                **common,
            )
        if not reader.completed:
            return OperatorResult(
                ok=False,
                session_id=reader.session_id,
                error_kind="error",
                error_message="Codex exited 0 without a completed turn",
                **common,
            )
        return OperatorResult(
            ok=True,
            session_id=reader.session_id,
            num_turns=reader.num_turns,
            total_tokens=usage_total_tokens(reader.usage) or None,
            **common,
        )
