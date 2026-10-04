from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, FiniteFloat

from hillclimb.harness.sandbox import SandboxPolicy


class AgentRequest(BaseModel):
    operator: str  # the operator's name (routing and logs key on it)
    kind: str | None = None  # create | repair | refine | combine — what kind of attempt this is
    prompt: str
    candidate_dir: Path
    timeout_s: int
    model: str = "sonnet"
    sampling: dict[str, int | FiniteFloat] | None = None
    resume_session_id: str | None = None  # set within a debug chain
    # False = no internet for the agent: its web tools are off and its whole
    # process tree reaches nothing but its model provider (the sandbox's proxy)
    allow_internet: bool = True
    # the OS sandbox the coding agent runs in (`harness/sandbox.py`); None = none.
    # The coding agent adds its candidate dir and its own state to what is writable
    sandbox: SandboxPolicy | None = None
    # Claude Code plugins to run with (`--plugin-dir`), from the hillclimb
    # dir's `agent_context.claude_plugins`; the user's own never apply
    plugins: list[Path] = Field(default_factory=list)


class AgentResult(BaseModel):
    ok: bool
    session_id: str | None = None
    # fully-qualified model the coding agent actually ran (e.g.
    # "claude-sonnet-4-5-20250929"); None when the coding agent only knows the alias
    model_id: str | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    # tokens the coding agent call consumed, summed across its turns: input + output
    # + cache creation + cache reads. Cache reads dominate and are cheap, but
    # the total is the honest "how much did this call move" number.
    total_tokens: int | None = None
    # the same tokens split per kind (input_tokens, output_tokens,
    # cache_creation_input_tokens, cache_read_input_tokens); zero-valued
    # kinds are dropped, empty when nothing was observed
    token_usage: dict[str, int] = Field(default_factory=dict)
    # subscription limit-window utilization (quota.snapshot()) taken at call
    # start/end. Account-wide numbers — concurrent work moves them too, so
    # the delta is telemetry, not accounting. None when unavailable.
    quota_start: dict | None = None
    quota_end: dict | None = None
    duration_s: float = 0.0
    # CPU seconds the coding agent process tree burned on this machine — the
    # coding agent itself plus every tool it ran (tests, scripts); the model's own
    # inference happens elsewhere and is what `total_tokens` measures. None
    # where the platform cannot say.
    cpu_s: float | None = None
    raw_output_path: str | None = None
    # rate_limited | out_of_credits | aborted | timeout | error — None when ok
    error_kind: str | None = None
    error_message: str = ""


class Agent(Protocol):
    name: str

    def invoke(self, request: AgentRequest) -> AgentResult: ...


# what a coding agent says when the login on disk can no longer be used: the
# token was revoked, expired, or (codex) its single-use refresh token was
# already spent by another copy of the same credential; Claude Code says
# "OAuth token revoked" / "OAuth session expired and could not be refreshed"
LOGIN_EXPIRED_MARKERS = (
    "refresh token",
    "log out and sign in again",
    "sign in again",
    "please log in again",
    "token has expired",
    "token is expired",
    "token revoked",
    "session expired",
    "could not be refreshed",
    "invalid_grant",
    "401 unauthorized",
    "authentication_failed",
    # an operator home that never logged in (Claude Code: "Not logged in ·
    # Please run /login"): the same fix, `hillclimb connect <agent>`
    "not logged in",
    "please run /login",
)


def login_expired(message: str | None) -> bool:
    """Does a failure say the login itself is dead, rather than the model or
    the route? `login status` cannot tell: it only sees that a credential
    file exists, so a dead login reads as logged in until a call is made."""
    text = (message or "").lower()
    return any(marker in text for marker in LOGIN_EXPIRED_MARKERS)
