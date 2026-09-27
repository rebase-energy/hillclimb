from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from hillclimb.harness.procs import Reaper
from hillclimb.harness import quota
from hillclimb.harness.candidate import utcnow
from hillclimb.backends.base import OperatorRequest, OperatorResult


def subscription_env(auth: str = "subscription") -> dict[str, str]:
    """Child env for `claude`. auth="subscription" (default) drops
    ANTHROPIC_API_KEY so calls bill the Max subscription (claude.ai login /
    CLAUDE_CODE_OAUTH_TOKEN) — an inherited API key silently takes precedence
    otherwise. auth="api-key" keeps it (headless/hosted runs with no
    subscription login). Single-threaded (see executor.SINGLE_THREAD_ENV):
    the agent's own experiment runs inherit it."""
    from hillclimb.harness.executor import single_threaded

    env = os.environ.copy()
    if auth != "api-key":
        env.pop("ANTHROPIC_API_KEY", None)
    return single_threaded(env)

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


def is_concrete_model_id(model_id: str | None) -> bool:
    """Whether a stream model names an inference model.

    Claude Code labels locally generated API-error messages ``<synthetic>``.
    That marker must not replace the requested/announced model in persisted
    candidate metadata or the watch UI.
    """
    return bool(model_id and model_id != "<synthetic>")


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
        # fully-qualified model that actually served the call (e.g.
        # "claude-sonnet-4-5-20250929") — the CLI resolves aliases like
        # "sonnet" internally, so the stream is the only source of truth
        self.model_id: str | None = None
        # per-turn usage deduped by message id (each turn streams twice,
        # partial then final, under one id) — the token count of record when
        # the call dies without a `result` message (timeout/abort/error)
        self.usage_by_turn: dict[str, dict] = {}

    def run(self) -> None:
        with self.stream_path.open("w") as sink:
            for line in self.stdout:
                try:
                    message = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    # non-JSON output from the CLI itself (error banners)
                    sink.write(line)
                    sink.flush()
                    if _has_rate_limit_marker(line):
                        self.rate_limited = True
                    continue
                if isinstance(message, dict):
                    # arrival time, so the watch TUI can show the transcript
                    # as a timestamped terminal log (the CLI stamps nothing)
                    message.setdefault("ts", utcnow())
                    line = json.dumps(message) + "\n"
                sink.write(line)
                sink.flush()
                if not isinstance(message, dict):
                    continue
                if message.get("type") == "system":
                    model_id = message.get("model")
                    if is_concrete_model_id(model_id):
                        self.model_id = model_id
                elif message.get("type") == "assistant":
                    body = message.get("message") or {}
                    # the turn's own model beats the init announcement
                    model_id = body.get("model")
                    if is_concrete_model_id(model_id):
                        self.model_id = model_id
                    if body.get("usage") and body.get("id"):
                        self.usage_by_turn[body["id"]] = body["usage"]
                elif message.get("type") == "result":
                    self.result_payload = message
                    if message.get("is_error") and _has_rate_limit_marker(
                        str(message.get("result", ""))
                    ):
                        self.rate_limited = True


# what counts as a token in the claude stream: the one definition the
# watch TUI's live counter and the recorded total both use
USAGE_TOKEN_KEYS = (
    "input_tokens", "output_tokens",
    "cache_creation_input_tokens", "cache_read_input_tokens",
)


def usage_total_tokens(usage: dict) -> int:
    return sum(usage.get(k) or 0 for k in USAGE_TOKEN_KEYS)


# List price per million tokens, (input, output), keyed by the model-id family
# substring — first match wins, so the more specific rows sit on top. Cache
# writes bill at 1.25x input and cache reads at 0.1x, the same multipliers
# Claude Code applies when it computes `total_cost_usd`; this table only
# stands in for that number while a call is still streaming (the stream
# carries usage per turn but no cost until the final `result` message).
MODEL_RATES_USD_PER_MTOK: tuple[tuple[str, float, float], ...] = (
    ("fable", 10.0, 50.0),
    ("mythos", 10.0, 50.0),
    ("opus", 5.0, 25.0),
    ("sonnet-4", 3.0, 15.0),
    ("sonnet", 2.0, 10.0),
    ("haiku", 1.0, 5.0),
)
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1


def estimate_cost_usd(usage: dict, model_id: str | None) -> float | None:
    """What one turn's usage costs at list price, or None when the model is
    not in the rate table (an unknown model contributes nothing rather than
    a made-up number)."""
    if not model_id:
        return None
    family = model_id.lower()
    for needle, in_rate, out_rate in MODEL_RATES_USD_PER_MTOK:
        if needle in family:
            break
    else:
        return None
    per_tok_in, per_tok_out = in_rate / 1e6, out_rate / 1e6
    return (
        (usage.get("input_tokens") or 0) * per_tok_in
        + (usage.get("cache_creation_input_tokens") or 0) * per_tok_in * CACHE_WRITE_MULTIPLIER
        + (usage.get("cache_read_input_tokens") or 0) * per_tok_in * CACHE_READ_MULTIPLIER
        + (usage.get("output_tokens") or 0) * per_tok_out
    )


def _observed_usage(reader: "_StreamReader | None", payload: dict) -> dict[str, int]:
    """Per-kind tokens the call burned, on every outcome: the final `result`
    usage when the call finished, else the per-turn sum the reader streamed
    before the call died — a failed candidate cost real tokens and must
    journal them. Zero-valued kinds are dropped."""
    if payload.get("usage"):
        turns = [payload["usage"]]
    elif reader is not None:
        turns = list(reader.usage_by_turn.values())
    else:
        turns = []
    usage = {
        key: sum(int(turn.get(key) or 0) for turn in turns) for key in USAGE_TOKEN_KEYS
    }
    return {key: count for key, count in usage.items() if count}


class ClaudeCodeBackend:
    """One operator call = one headless Claude Code invocation, cwd-scoped to
    the node candidate_dir. Auth comes from the interactive `claude` login (Max
    subscription) or CLAUDE_CODE_OAUTH_TOKEN in the environment.

    Runs with `--output-format stream-json` so the transcript lands
    incrementally in <candidate_dir>/agent_stream.jsonl, and exposes the child
    pid in <candidate_dir>/agent.pid while the call is in flight."""

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

        candidate_dir = Path(request.candidate_dir)
        stream_path = candidate_dir / STREAM_FILE
        pid_path = candidate_dir / PID_FILE
        stderr_path = candidate_dir / "agent_stderr.log"
        # window utilization before any token is burned; the end snapshot
        # follows the call so every candidate journals its before/after
        quota_start = quota.snapshot() if self.auth != "api-key" else None
        start = time.monotonic()
        timed_out = False
        aborted = False
        reader: _StreamReader | None = None
        proc: subprocess.Popen | None = None
        reaper: Reaper | None = None
        try:
            with stderr_path.open("w") as stderr_sink:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr_sink,
                    text=True,
                    cwd=request.candidate_dir,
                    env=subscription_env(self.auth),
                    start_new_session=True,  # own process group → killable as a unit
                )
                reaper = Reaper(proc)  # reaps through wait4: the call's CPU rides along
                pid_path.write_text(str(proc.pid))
                reader = _StreamReader(proc.stdout, stream_path)
                reader.start()
                try:
                    proc.stdin.write(request.prompt)
                    proc.stdin.close()
                except BrokenPipeError:
                    pass  # process died instantly; returncode tells the story
                deadline = time.monotonic() + request.timeout_s
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
                # e.g. StopRequested raised by a signal handler while in wait()
                reaper.kill_group()
            if reader is not None:
                reader.join(timeout=5)

        duration = time.monotonic() - start
        payload = reader.result_payload or {}
        stderr_text = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""

        raw_path = candidate_dir / "agent_raw.json"
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

        # every outcome journals the tokens it burned — failed calls too
        token_usage = _observed_usage(reader, payload)
        burn = {
            "token_usage": token_usage,
            "total_tokens": sum(token_usage.values()) or None,
            "quota_start": quota_start,
            "quota_end": quota.snapshot() if self.auth != "api-key" else None,
            "model_id": reader.model_id if reader else None,
            "cpu_s": reaper.cpu_s if reaper is not None else None,
        }
        if aborted:
            return OperatorResult(
                ok=False,
                **burn,
                duration_s=duration,
                raw_output_path=str(raw_path),
                error_kind="aborted",
                error_message="agent call aborted (stop requested)",
            )
        if timed_out:
            return OperatorResult(
                ok=False,
                **burn,
                duration_s=duration,
                raw_output_path=str(raw_path),
                error_kind="timeout",
                error_message=f"agent call exceeded {request.timeout_s}s",
            )
        if reader.rate_limited or _has_rate_limit_marker(stderr_text):
            return OperatorResult(
                ok=False,
                session_id=payload.get("session_id"),
                cost_usd=payload.get("total_cost_usd"),
                num_turns=payload.get("num_turns"),
                **burn,
                duration_s=duration,
                raw_output_path=str(raw_path),
                error_kind="rate_limited",
                error_message=str(payload.get("result") or stderr_text)[:500],
            )
        if proc.returncode != 0 or payload.get("is_error"):
            return OperatorResult(
                ok=False,
                session_id=payload.get("session_id"),
                cost_usd=payload.get("total_cost_usd"),
                num_turns=payload.get("num_turns"),
                **burn,
                duration_s=duration,
                raw_output_path=str(raw_path),
                error_kind="error",
                error_message=str(payload.get("result") or stderr_text or "")[:500],
            )
        if not payload:
            return OperatorResult(
                ok=False,
                **burn,
                duration_s=duration,
                raw_output_path=str(raw_path),
                error_kind="error",
                error_message="agent exited 0 but emitted no result message",
            )
        return OperatorResult(
            ok=True,
            session_id=payload.get("session_id"),
            cost_usd=payload.get("total_cost_usd"),
            num_turns=payload.get("num_turns"),
            **burn,
            duration_s=duration,
            raw_output_path=str(raw_path),
        )
