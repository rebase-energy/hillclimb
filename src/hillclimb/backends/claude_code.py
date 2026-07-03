from __future__ import annotations

import json
import os
import subprocess
import time

from hillclimb.backends.base import OperatorRequest, OperatorResult


def subscription_env() -> dict[str, str]:
    """Child env for `claude`: drop ANTHROPIC_API_KEY so calls bill the Max
    subscription (claude.ai login / CLAUDE_CODE_OAUTH_TOKEN) instead of the
    API — an inherited API key silently takes precedence otherwise."""
    env = os.environ.copy()
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


class ClaudeCodeBackend:
    """One operator call = one headless Claude Code invocation, cwd-scoped to
    the node workspace. Auth comes from the interactive `claude` login (Max
    subscription) or CLAUDE_CODE_OAUTH_TOKEN in the environment."""

    name = "claude-code"

    def __init__(self, claude_bin: str = "claude"):
        self.claude_bin = claude_bin

    def invoke(self, request: OperatorRequest) -> OperatorResult:
        cmd = [
            self.claude_bin,
            "-p",
            "--output-format",
            "json",
            "--permission-mode",
            "bypassPermissions",
            "--model",
            request.model,
        ]
        if request.resume_session_id:
            cmd += ["--resume", request.resume_session_id]
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                input=request.prompt,
                capture_output=True,
                text=True,
                cwd=request.workspace,
                timeout=request.timeout_s,
                env=subscription_env(),
            )
        except subprocess.TimeoutExpired:
            return OperatorResult(
                ok=False,
                duration_s=time.monotonic() - start,
                error_kind="timeout",
                error_message=f"agent call exceeded {request.timeout_s}s",
            )
        duration = time.monotonic() - start

        raw_path = request.workspace / "agent_raw.json"
        raw_path.write_text(
            json.dumps({"stdout": proc.stdout, "stderr": proc.stderr, "cmd": cmd})
        )

        payload: dict = {}
        try:
            payload = json.loads(proc.stdout)
        except (json.JSONDecodeError, ValueError):
            pass

        combined = (proc.stdout + proc.stderr).lower()
        if any(marker in combined for marker in RATE_LIMIT_MARKERS):
            return OperatorResult(
                ok=False,
                duration_s=duration,
                raw_output_path=str(raw_path),
                error_kind="rate_limited",
                error_message=str(payload.get("result", proc.stderr))[:500],
            )
        if proc.returncode != 0 or payload.get("is_error"):
            return OperatorResult(
                ok=False,
                session_id=payload.get("session_id"),
                duration_s=duration,
                raw_output_path=str(raw_path),
                error_kind="error",
                error_message=str(payload.get("result") or proc.stderr or "")[:500],
            )
        return OperatorResult(
            ok=True,
            session_id=payload.get("session_id"),
            cost_usd=payload.get("total_cost_usd"),
            num_turns=payload.get("num_turns"),
            duration_s=duration,
            raw_output_path=str(raw_path),
        )
