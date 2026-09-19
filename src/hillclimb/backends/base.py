from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, FiniteFloat


class OperatorRequest(BaseModel):
    operator: str  # draft | debug | improve
    prompt: str
    candidate_dir: Path
    timeout_s: int
    model: str = "sonnet"
    sampling: dict[str, int | FiniteFloat] | None = None
    resume_session_id: str | None = None  # set within a debug chain


class OperatorResult(BaseModel):
    ok: bool
    session_id: str | None = None
    # fully-qualified model the backend actually ran (e.g.
    # "claude-sonnet-4-5-20250929"); None when the backend only knows the alias
    model_id: str | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    # tokens the agent call consumed, summed across its turns: input + output
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
    raw_output_path: str | None = None
    # rate_limited | out_of_credits | aborted | timeout | error — None when ok
    error_kind: str | None = None
    error_message: str = ""


class OperatorBackend(Protocol):
    name: str

    def invoke(self, request: OperatorRequest) -> OperatorResult: ...
