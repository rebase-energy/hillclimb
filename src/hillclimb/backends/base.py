from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel


class OperatorRequest(BaseModel):
    operator: str  # draft | debug | improve
    prompt: str
    candidate_dir: Path
    timeout_s: int
    model: str = "sonnet"
    resume_session_id: str | None = None  # set within a debug chain


class OperatorResult(BaseModel):
    ok: bool
    session_id: str | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    # tokens the agent call consumed, summed across its turns: input + output
    # + cache creation + cache reads. Cache reads dominate and are cheap, but
    # the total is the honest "how much did this call move" number.
    total_tokens: int | None = None
    duration_s: float = 0.0
    raw_output_path: str | None = None
    # rate_limited | timeout | error — None when ok
    error_kind: str | None = None
    error_message: str = ""


class OperatorBackend(Protocol):
    name: str

    def invoke(self, request: OperatorRequest) -> OperatorResult: ...
