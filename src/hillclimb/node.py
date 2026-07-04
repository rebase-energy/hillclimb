from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

OPERATORS = ("baseline", "draft", "debug", "improve", "ensemble")
STATUSES = ("pending", "ok", "buggy", "parked", "abandoned")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class BackendInfo(BaseModel):
    name: str = ""
    session_id: str | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    agent_duration_s: float | None = None
    error_kind: str | None = None


class ExecInfo(BaseModel):
    returncode: int | None = None
    duration_s: float | None = None
    timed_out: bool = False
    stdout_tail: str = ""
    submission_ok: bool = False
    holdout_error: str | None = None  # why holdout predictions couldn't be scored


class Node(BaseModel):
    node_id: str
    parent_id: str | None = None
    operator: str  # one of OPERATORS
    status: str = "pending"  # one of STATUSES
    complexity: str | None = None  # minimal | moderate | advanced (drafts only)
    debug_depth: int = 0
    workspace: str = ""
    backend: BackendInfo = Field(default_factory=BackendInfo)
    execution: ExecInfo = Field(default_factory=ExecInfo)
    val_score: float | None = None
    holdout_score: float | None = None  # orchestrator-computed, hidden from agent
    is_best: bool = False       # best by agent-reported val_score (climbing signal)
    is_selected: bool = False   # best by holdout score (final-submission signal)
    pruned: bool = False        # user cut this lineage; status stays intact
    pruned_reason: str | None = None
    summary: str = ""
    created_at: str = Field(default_factory=utcnow)
    finished_at: str | None = None

    @property
    def is_scored(self) -> bool:
        return self.status == "ok" and self.val_score is not None
