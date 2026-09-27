"""Codex CLI operator backend.

One Hillclimb operator call becomes one non-interactive ``codex exec`` turn
inside the candidate directory. Codex's JSONL events are normalized to the
same small stream format consumed by Hillclimb's live candidate viewer.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
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
    usage_total_tokens,
)
from hillclimb.harness.candidate import utcnow
from hillclimb.harness.procs import Reaper
from hillclimb.harness.pricing import cost_usd


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# codex 0.153 dropped Chat Completions, so the provider must speak the
# Responses API. OpenRouter's is stateless, which costs nothing here: codex
# replays history client-side, so `exec resume` still works.
OPENROUTER_PROVIDER = (
    'model_providers.openrouter={name="OpenRouter",'
    f'base_url="{OPENROUTER_BASE_URL}",'
    'env_key="OPENROUTER_API_KEY",wire_api="responses"}'
)


def codex_home(auth: str) -> Path:
    """A CODEX_HOME per auth mode, so a search never inherits personal codex
    settings — they change results and cost tokens. The subscription login is
    copied in because it is the credential; OpenRouter needs no file."""
    home = Path.home() / ".cache" / "hillclimb" / "codex-home" / auth
    home.mkdir(parents=True, exist_ok=True)
    if auth == "openrouter":
        return home
    source = Path.home() / ".codex" / "auth.json"
    target = home / "auth.json"
    if source.exists() and (
        not target.exists() or source.stat().st_mtime > target.stat().st_mtime
    ):
        # atomic: concurrent operators share this directory — threads of one
        # search process as much as separate processes, so the staging file
        # must be unique per call, not per pid
        fd, staging = tempfile.mkstemp(prefix=".auth.", suffix=".json", dir=home)
        os.close(fd)
        shutil.copy2(source, staging)
        os.replace(staging, target)
    return home


def codex_env(auth: str = "subscription") -> dict[str, str]:
    """Build the child environment for ChatGPT-login, API-key or OpenRouter
    auth. A missing OpenRouter key raises here, before the spawn."""
    from hillclimb.harness.executor import single_threaded

    env = os.environ.copy()
    if auth == "openrouter":
        if not env.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "backend_auth: openrouter needs OPENROUTER_API_KEY — export it "
                "or put it in a .env beside config.yaml"
            )
        # billing must not fall back to an inherited OpenAI account
        env.pop("OPENAI_API_KEY", None)
    elif auth != "api-key":
        # An inherited key can take precedence over the interactive login.
        env.pop("OPENAI_API_KEY", None)
    env["CODEX_HOME"] = str(codex_home(auth))
    return single_threaded(env)


def _has_rate_limit_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in RATE_LIMIT_MARKERS)


# What a provider says when the account cannot pay for the call. No bare
# "402": it collides with token counts in ordinary messages.
CREDIT_MARKERS = (
    "payment required",
    "requires more credits",
    "insufficient credits",
    "insufficient_quota",
)


def _has_credit_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in CREDIT_MARKERS)


def _normalized_usage(usage: dict) -> dict:
    """Responses usage onto hillclimb's disjoint token keys
    (claude_code.USAGE_TOKEN_KEYS). Upstream `input_tokens` includes the
    cached subset, so the remainder is the uncached input; cache reads are
    worth watching because codex resends an identical ~12k preamble every
    call. `reasoning_output_tokens` sits inside `output_tokens`."""
    total_input = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cached_input_tokens") or 0)
    return {
        "input_tokens": max(total_input - cache_read, 0),
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": int(usage.get("cache_write_input_tokens") or 0),
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
        self.out_of_credits = False
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
            # codex retries transient provider errors (OpenRouter 502s) and
            # emits them as it goes; a turn that finished is not a failure
            self.error_message = ""
            self.rate_limited = False
            self.out_of_credits = False
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
            self.out_of_credits = _has_credit_marker(self.error_message)
            self.rate_limited = not self.out_of_credits and _has_rate_limit_marker(
                self.error_message
            )
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
        cmd = [self.codex_bin]
        if self.auth == "openrouter":
            cmd += ["-c", OPENROUTER_PROVIDER, "-c", "model_provider=openrouter"]
        cmd += [
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
        reaper: Reaper | None = None

        try:
            child_env = codex_env(self.auth)
        except RuntimeError as exc:
            return OperatorResult(ok=False, error_kind="error", error_message=str(exc))

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
                            env=child_env,
                            start_new_session=True,
                        )
                    except OSError as exc:
                        spawn_error = str(exc)
                    if proc is not None:
                        reaper = Reaper(proc)  # reaps through wait4: the call's CPU rides along
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
                            reaper.poll() is None
                            and not reader.started.wait(timeout=0.1)
                            and time.monotonic() < startup_deadline
                        ):
                            if self.abort is not None and self.abort.is_set():
                                aborted = True
                                reaper.kill_group()
                                break
                if reaper is not None:
                    while reaper.wait(timeout=1.0) is None:
                        if self.abort is not None and self.abort.is_set():
                            aborted = True
                            reaper.kill_group()
                            break
                        if time.monotonic() >= deadline:
                            timed_out = True
                            reaper.kill_group()
                            break
        finally:
            pid_path.unlink(missing_ok=True)
            if reaper is not None and reaper.poll() is None:
                reaper.kill_group()
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

        common = {
            "duration_s": duration,
            "raw_output_path": str(raw_path),
            "cpu_s": reaper.cpu_s if reaper is not None else None,
        }
        if spawn_error:
            return OperatorResult(
                ok=False,
                error_kind="error",
                error_message=f"could not start Codex CLI: {spawn_error}",
                **common,
            )
        # `turn.completed` events carry usage as turns land, so every outcome
        # below journals the tokens the call actually burned before it died
        if reader is not None:
            common["total_tokens"] = usage_total_tokens(reader.usage) or None
            common["token_usage"] = {
                key: count for key, count in reader.usage.items() if count
            }
            if self.auth == "openrouter":
                common["cost_usd"] = cost_usd(request.model, common["token_usage"])
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
        if reader.out_of_credits or _has_credit_marker(stderr_text):
            return OperatorResult(
                ok=False,
                session_id=reader.session_id,
                num_turns=reader.num_turns,
                error_kind="out_of_credits",
                error_message=(reader.error_message or stderr_text)[:500],
                **common,
            )
        if reader.rate_limited or _has_rate_limit_marker(stderr_text):
            return OperatorResult(
                ok=False,
                session_id=reader.session_id,
                num_turns=reader.num_turns,
                error_kind="rate_limited",
                error_message=(reader.error_message or stderr_text)[:500],
                **common,
            )
        if proc.returncode != 0 or reader.error_message:
            return OperatorResult(
                ok=False,
                session_id=reader.session_id,
                num_turns=reader.num_turns,
                error_kind="error",
                error_message=(reader.error_message or stderr_text)[:500],
                **common,
            )
        if not reader.completed:
            return OperatorResult(
                ok=False,
                session_id=reader.session_id,
                num_turns=reader.num_turns,
                error_kind="error",
                error_message="Codex exited 0 without a completed turn",
                **common,
            )
        return OperatorResult(
            ok=True,
            session_id=reader.session_id,
            num_turns=reader.num_turns,
            **common,
        )
