from __future__ import annotations

from datetime import datetime, timezone
from statistics import median

from pydantic import BaseModel, ConfigDict, Field, model_validator

OPERATORS = ("baseline", "draft", "debug", "improve", "ensemble")
STATUSES = ("pending", "ok", "buggy", "parked", "abandoned")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class BackendInfo(BaseModel):
    """Metadata about the LLM agent call that authored the candidate's code.
    Lives on the Candidate (not a Trial): it describes creation, not execution."""

    name: str = ""
    model: str | None = None  # requested model/route alias (bandit arm on replay)
    # fully-qualified model that served the call, from the agent stream
    # (e.g. "claude-sonnet-4-5-20250929"); None on old journals and backends
    # that only know the alias
    model_id: str | None = None
    session_id: str | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    total_tokens: int | None = None
    # total_tokens split per kind (input_tokens, output_tokens,
    # cache_creation_input_tokens, cache_read_input_tokens); empty on
    # journals predating the field or when nothing was observed
    token_usage: dict[str, int] = Field(default_factory=dict)
    # subscription limit-window utilization (%) snapshotted at agent-call
    # start/end (see quota.py). Account-wide — parallel operators and other
    # sessions move it too, so the delta is telemetry, not accounting.
    quota_start: dict | None = None
    quota_end: dict | None = None
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
    # verifier CPU seconds (user+system, via os.wait4); None on journals
    # predating the field or platforms without wait4 — consumers fall back to
    # duration_s there (verifier envs are single-threaded, wall ≈ cpu)
    cpu_s: float | None = None
    timed_out: bool = False
    stdout_tail: str = ""
    submission_ok: bool = False
    holdout_error: str | None = None  # why holdout predictions couldn't be scored
    # CPU seconds of this trial's holdout run (set even when it errored —
    # the cost was paid); None where holdout didn't run or predates the field
    holdout_cpu_s: float | None = None
    val_score: float | None = None
    holdout_score: float | None = None  # orchestrator-computed, hidden from agent
    # compact VALIDATION-split breakdown from eval_result.json (never holdout —
    # that would leak the selection signal into prompts); consumers read the
    # first trial's report (trial 0 in multi-trial mode, no averaging)
    report: dict | None = None
    # auxiliary numeric measurements the verifier wrote next to `score`
    # (feature dimensions for quality-diversity policies); never a score
    metrics: dict[str, float] = Field(default_factory=dict)
    # per-instance validation scores from the verifier's reserved `instances`
    # key: same metric/direction as val_score, stable keys across a search;
    # empty when the verifier emits none (old journals replay unchanged)
    instance_scores: dict[str, float] = Field(default_factory=dict)
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
    candidate_dir: str = ""
    backend: BackendInfo = Field(default_factory=BackendInfo)
    trials: list[Trial] = Field(default_factory=list)
    is_best: bool = False       # best by agent-reported val_score (climbing signal)
    is_selected: bool = False   # best by holdout score (final-submission signal)
    # opaque annotation from the search policy that proposed this candidate
    # (e.g. a MAP-Elites cell); the engine never reads it
    policy_meta: dict = Field(default_factory=dict)
    pruned: bool = False        # user cut this lineage; status stays intact
    pruned_reason: str | None = None
    summary: str = ""
    created_at: str = Field(default_factory=utcnow)
    finished_at: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_workspace_key(cls, data):
        """Journals written before the rename carry `workspace`; map it onto
        `candidate_dir` so old runs replay (extra="forbid" would reject it)."""
        if isinstance(data, dict) and "workspace" in data:
            data = dict(data)
            data.setdefault("candidate_dir", data.pop("workspace"))
            data.pop("workspace", None)
        return data

    @property
    def last_trial(self) -> Trial | None:
        return self.trials[-1] if self.trials else None

    # Aggregate rule: MEDIAN val over scored trials is the climbing signal.
    # With one trial (the default) it is that trial's score; with several it
    # resists the single slow run or unlucky seed that a mean would carry
    # straight into the search's ranking. Holdout is evaluated once per
    # candidate, so the last non-None value is the candidate's holdout score.
    @property
    def val_score(self) -> float | None:
        return median(self.trial_scores) if self.trial_scores else None

    @property
    def metrics(self) -> dict[str, float]:
        """Per-key MEDIAN of the scored trials' auxiliary metrics — the same
        aggregate rule as val_score, so a policy binning on them sees the
        candidate, not one noisy run."""
        pooled: dict[str, list[float]] = {}
        for trial in self.trials:
            if trial.val_score is None:
                continue
            for key, value in trial.metrics.items():
                pooled.setdefault(key, []).append(value)
        return {key: median(values) for key, values in pooled.items()}

    @property
    def instance_scores(self) -> dict[str, float]:
        """Per-key MEDIAN of the scored trials' per-instance scores — the
        same aggregate rule as val_score, so an engine's per-instance
        frontier sees the candidate, not one noisy run."""
        pooled: dict[str, list[float]] = {}
        for trial in self.trials:
            if trial.val_score is None:
                continue
            for key, value in trial.instance_scores.items():
                pooled.setdefault(key, []).append(value)
        return {key: median(values) for key, values in pooled.items()}

    @property
    def trial_scores(self) -> list[float]:
        return [t.val_score for t in self.trials if t.val_score is not None]

    @property
    def trial_spread(self) -> float | None:
        """Median absolute deviation of this candidate's trial scores — how
        much the same code moves between identical evaluations. None until
        two trials have scored."""
        scores = self.trial_scores
        if len(scores) < 2:
            return None
        centre = median(scores)
        return median([abs(value - centre) for value in scores])

    @property
    def holdout_score(self) -> float | None:
        for trial in reversed(self.trials):
            if trial.holdout_score is not None:
                return trial.holdout_score
        return None

    @property
    def is_scored(self) -> bool:
        return self.status == "ok" and self.val_score is not None
