from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

OPERATORS = ("baseline", "draft", "debug", "improve", "ensemble")
STATUSES = ("pending", "ok", "buggy", "parked", "abandoned")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class BackendInfo(BaseModel):
    """Metadata about the LLM agent call that authored the candidate's code.
    Lives on the Candidate (not a Trial): it describes creation, not execution."""

    name: str = ""
    session_id: str | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    agent_duration_s: float | None = None
    error_kind: str | None = None


class Trial(BaseModel):
    """One execution of a candidate with a fixed parameterization.

    `params`/`seed` are schema-ready for a future tuning loop; today the
    engine runs exactly one trial per candidate with empty params.
    """

    model_config = ConfigDict(extra="forbid")

    params: dict = Field(default_factory=dict)
    seed: int | None = None
    returncode: int | None = None
    duration_s: float | None = None
    timed_out: bool = False
    stdout_tail: str = ""
    submission_ok: bool = False
    holdout_error: str | None = None  # why holdout predictions couldn't be scored
    val_score: float | None = None
    holdout_score: float | None = None  # orchestrator-computed, hidden from agent
    started_at: str = Field(default_factory=utcnow)
    finished_at: str | None = None


class Candidate(BaseModel):
    """An immutable code artifact produced by an operator. Any change to the
    code is a new candidate; re-executions of the same code are new trials."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    parent_id: str | None = None
    operator: str  # one of OPERATORS
    status: str = "pending"  # one of STATUSES
    complexity: str | None = None  # minimal | moderate | advanced (drafts only)
    debug_depth: int = 0
    workspace: str = ""
    backend: BackendInfo = Field(default_factory=BackendInfo)
    trials: list[Trial] = Field(default_factory=list)
    is_best: bool = False       # best by agent-reported val_score (climbing signal)
    is_selected: bool = False   # best by holdout score (final-submission signal)
    pruned: bool = False        # user cut this lineage; status stays intact
    pruned_reason: str | None = None
    summary: str = ""
    created_at: str = Field(default_factory=utcnow)
    finished_at: str | None = None

    @property
    def last_trial(self) -> Trial | None:
        return self.trials[-1] if self.trials else None

    # Aggregate rule: the latest trial speaks for the candidate. With one
    # trial per candidate this is exact; a future tuning loop changes the
    # rule (best/mean over trials) here and nowhere else.
    @property
    def val_score(self) -> float | None:
        return self.trials[-1].val_score if self.trials else None

    @property
    def holdout_score(self) -> float | None:
        return self.trials[-1].holdout_score if self.trials else None

    @property
    def is_scored(self) -> bool:
        return self.status == "ok" and self.val_score is not None
