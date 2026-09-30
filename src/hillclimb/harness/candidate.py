from __future__ import annotations

from datetime import datetime, timezone
from statistics import median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

STATUSES = ("pending", "passing", "failing", "buggy", "parked", "abandoned")


def source_hash(source: str) -> str:
    """Stable identity of a solution's text: sha256 over newline-normalized
    source, so a CRLF checkout or a trailing blank line is the same solution."""
    import hashlib

    normalized = "\n".join(source.splitlines()).strip() + "\n"
    return hashlib.sha256(normalized.encode()).hexdigest()


# Trial fields the hidden split produces (Candidate.holdout_blind strips them)
HOLDOUT_FIELDS = ("holdout_score", "holdout_error", "holdout_cpu_s")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentInfo(BaseModel):
    """Metadata about the LLM agent call that authored the candidate's code.
    Lives on the Candidate (not a Trial): it describes creation, not execution."""

    name: str = ""
    model: str | None = None  # requested model/route alias (bandit arm on replay)
    sampling: dict[str, int | float] | None = None
    # fully-qualified model that served the call, from the agent stream
    # (e.g. "claude-sonnet-4-5-20250929"); None on old journals and agents
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
    # start/end (see quota.py). Account-wide — parallel agents and other
    # sessions move it too, so the delta is telemetry, not accounting.
    quota_start: dict | None = None
    quota_end: dict | None = None
    agent_duration_s: float | None = None
    # local CPU seconds of the agent call (see AgentResult.cpu_s); None
    # on journals predating the field
    cpu_s: float | None = None
    error_kind: str | None = None


class Replicate(BaseModel):
    """One seeded execution of a trial's parameter set — the leaf of
    Candidate (code) → Trial (params) → Replicate (run). Replicates of one
    trial differ only by seed; their spread is the search's noise floor."""

    model_config = ConfigDict(extra="forbid")

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
    val_score: float | None = None
    # compact VALIDATION-split breakdown from eval_result.json (never holdout —
    # that would leak the selection signal into prompts); consumers read the
    # first replicate's report (r0, no averaging)
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


class UnitTestResult(BaseModel):
    """One execution of the run-frozen unit-test suite for a Trial."""

    model_config = ConfigDict(extra="forbid")

    passed: bool = False
    returncode: int | None = None
    duration_s: float = 0.0
    cpu_s: float | None = None
    timed_out: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""


def _per_key_median(dicts) -> dict[str, float]:
    pooled: dict[str, list[float]] = {}
    for entries in dicts:
        for key, value in entries.items():
            pooled.setdefault(key, []).append(value)
    return {key: median(values) for key, values in pooled.items()}


class Trial(BaseModel):
    """One parameter set of a candidate, executed as `search.n_replicates`
    seeded runs. `params` is empty for candidates that declare no tunable
    parameters (today's default: one trial per candidate). Holdout is scored
    once per trial (with its params), never per replicate. `is_best` is
    stamped by `Candidate.stamp_best_trial` — the only place score direction
    enters the candidate aggregate."""

    model_config = ConfigDict(extra="forbid")

    index: int = 0
    params: dict = Field(default_factory=dict)
    replicates: list[Replicate] = Field(default_factory=list)
    # None is the backward-compatible value for historical trials and for
    # problems without a unit-test gate.
    verdict: Literal["passing", "failing", "buggy"] | None = None
    unit_tests: UnitTestResult | None = None
    is_best: bool = False
    holdout_score: float | None = None  # orchestrator-computed, hidden from agent
    holdout_error: str | None = None  # why holdout predictions couldn't be scored
    # CPU seconds of this trial's holdout run (set even when it errored —
    # the cost was paid); None where holdout didn't run or predates the field
    holdout_cpu_s: float | None = None
    started_at: str = Field(default_factory=utcnow)
    finished_at: str | None = None

    @property
    def last_replicate(self) -> Replicate | None:
        return self.replicates[-1] if self.replicates else None

    @property
    def replicate_scores(self) -> list[float]:
        return [r.val_score for r in self.replicates if r.val_score is not None]

    # Aggregate rule: MEDIAN val over scored replicates is the trial's score.
    # With one replicate (the default) it is that run's score; with several
    # it resists the single slow run or unlucky seed that a mean would carry
    # straight into the search's ranking.
    @property
    def val_score(self) -> float | None:
        return median(self.replicate_scores) if self.replicate_scores else None

    @property
    def replicate_spread(self) -> float | None:
        """Median absolute deviation of this trial's replicate scores — how
        much the same code and params move between identical evaluations.
        None until two replicates have scored."""
        scores = self.replicate_scores
        if len(scores) < 2:
            return None
        centre = median(scores)
        return median([abs(value - centre) for value in scores])

    @property
    def metrics(self) -> dict[str, float]:
        """Per-key MEDIAN of the scored replicates' auxiliary metrics."""
        return _per_key_median(r.metrics for r in self.replicates if r.val_score is not None)

    @property
    def instance_scores(self) -> dict[str, float]:
        """Per-key MEDIAN of the scored replicates' per-instance scores."""
        return _per_key_median(
            r.instance_scores for r in self.replicates if r.val_score is not None
        )

    @property
    def report(self) -> dict | None:
        """First replicate carrying one (r0 in multi-replicate mode, matching
        the r0 artifact hoist)."""
        return next((r.report for r in self.replicates if r.report), None)

    @property
    def submission_ok(self) -> bool:
        """r0's verdict — r0's artifacts are what the candidate root holds."""
        return bool(self.replicates) and self.replicates[0].submission_ok


class Candidate(BaseModel):
    """An immutable code artifact produced by an operator. Any change to the
    code is a new candidate; a new parameter set for the same code is a new
    trial; a re-execution of the same code and params is a new replicate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    parent_id: str | None = None
    operator: str  # a registered operator's name, or a harness-native one (baseline | seed)
    # what the operator's candidates ARE (create | repair | refine | combine |
    # baseline | seed): views colour by it and the journal walks chains by it,
    # so a climber's own operators need no change anywhere else. Stamped by
    # the harness; records that predate it get their operator's role on load.
    role: str | None = None
    # `source_hash` of the solution that was scored, stamped by the harness:
    # how a loop recognises a text it has already paid to evaluate
    solution_sha256: str | None = None
    status: str = "pending"  # one of STATUSES
    # the action's per-attempt knobs as proposed (`Action.args`), e.g. a
    # draft's {"complexity": "minimal"}; bulk `Action.payload` is never here
    args: dict = Field(default_factory=dict)
    debug_depth: int = 0
    candidate_dir: str = ""
    agent: AgentInfo = Field(default_factory=AgentInfo)   # journals before the rename say `backend`
    trials: list[Trial] = Field(default_factory=list)
    # the candidate declared a valid params.json (tune actions may target it);
    # params_error carries why a declaration was rejected (scored on defaults)
    tunable: bool = False
    params_error: str | None = None
    is_best: bool = False       # best by agent-reported val_score (climbing signal)
    is_selected: bool = False   # best by holdout score (final-submission signal)
    # opaque annotation from the search policy that proposed this candidate
    # (e.g. a MAP-Elites cell); the engine never reads it
    climber_meta: dict = Field(default_factory=dict)
    pruned: bool = False        # user cut this lineage; status stays intact
    pruned_reason: str | None = None
    summary: str = ""
    created_at: str = Field(default_factory=utcnow)
    finished_at: str | None = None

    @model_validator(mode="after")
    def _backfill_role(self) -> Candidate:
        if self.role is None:
            from hillclimb.modules.operators import role_of

            self.role = role_of(self.operator)
        return self

    @model_validator(mode="before")
    @classmethod
    def _legacy_agent_key(cls, data):
        """Journals from before the rename record the agent call as `backend`."""
        if isinstance(data, dict) and "backend" in data:
            data = dict(data)
            data.setdefault("agent", data.pop("backend"))
        return data

    @model_validator(mode="before")
    @classmethod
    def _legacy_policy_meta_key(cls, data):
        """Journals written before 0.6 record what a climber notes on a
        candidate as "policy_meta"."""
        if isinstance(data, dict) and "policy_meta" in data:
            data = dict(data)
            data.setdefault("climber_meta", data.pop("policy_meta"))
        return data

    @model_validator(mode="before")
    @classmethod
    def _legacy_workspace_key(cls, data):
        """Journals written before the rename carry `workspace`; map it onto
        `candidate_dir` so old runs replay (extra="forbid" would reject it)."""
        if isinstance(data, dict) and "workspace" in data:
            data = dict(data)
            data.setdefault("candidate_dir", data.pop("workspace"))
            data.pop("workspace", None)
        if isinstance(data, dict) and data.get("status") == "ok":
            data = dict(data)
            data["status"] = "passing"
        if isinstance(data, dict) and "complexity" in data:
            # pre-`args` records (and callers) carry the draft cue as its own field
            data = dict(data)
            complexity = data.pop("complexity")
            if complexity is not None:
                data["args"] = {"complexity": complexity, **(data.get("args") or {})}
        return data

    @model_validator(mode="before")
    @classmethod
    def _legacy_flat_trials(cls, data):
        """Journals written before the Trial (params) → Replicate (seed) split
        carry `trials` as a flat list of seeded executions (each with
        `seed`/`val_score` and empty `params`). Fold them into ONE trial whose
        replicates they are; holdout fields lift from the last execution that
        carried them (the old "last non-None wins" rule). Idempotent: entries
        that already have `replicates` are the new shape and pass through
        (Trial instances too — they are not dicts)."""
        if not isinstance(data, dict) or not isinstance(data.get("trials"), list):
            return data
        entries = data["trials"]
        old = [t for t in entries if isinstance(t, dict) and "replicates" not in t]
        if not old:
            return data
        new = [t for t in entries if not (isinstance(t, dict) and "replicates" not in t)]
        replicates: list[dict] = []
        holdout: dict = {}
        params: dict = {}
        for entry in old:
            entry = dict(entry)
            params = entry.pop("params", None) or params
            for key in HOLDOUT_FIELDS:
                value = entry.pop(key, None)
                if value is not None:
                    holdout[key] = value
            replicates.append(entry)
        folded: dict = {"index": 0, "params": params, "replicates": replicates, **holdout}
        if replicates and "started_at" in replicates[0]:
            folded["started_at"] = replicates[0]["started_at"]
        finished = [r["finished_at"] for r in replicates if r.get("finished_at")]
        if finished:
            folded["finished_at"] = finished[-1]
        if not new:
            folded["is_best"] = True  # the only trial there is
        data = dict(data)
        data["trials"] = [folded, *new]
        return data

    @property
    def last_trial(self) -> Trial | None:
        return self.trials[-1] if self.trials else None

    @property
    def last_replicate(self) -> Replicate | None:
        last = self.last_trial
        return last.last_replicate if last is not None else None

    def stamp_best_trial(self, higher_is_better: bool) -> Trial | None:
        """Mark the trial the candidate is scored by (max/min median over
        replicates; ties keep the earliest). Engines call this after every
        trial lands; the flag is journaled so every reader sees the same
        aggregate without knowing the direction."""
        scored = [
            t for t in self.trials
            if t.val_score is not None and t.verdict not in ("failing", "buggy")
        ]
        for trial in self.trials:
            trial.is_best = False
        if not scored:
            return None
        best = scored[0]
        for trial in scored[1:]:
            if (trial.val_score > best.val_score) if higher_is_better else (trial.val_score < best.val_score):
                best = trial
        best.is_best = True
        return best

    # Aggregate rule: the candidate is scored by its BEST trial (the parameter
    # set the search would ship), each trial by the MEDIAN of its replicates.
    # Metrics, reports and holdout follow the best trial too — pooling across
    # parameter sets would describe no run that actually happened.
    @property
    def best_trial(self) -> Trial | None:
        scored = [
            t for t in self.trials
            if t.val_score is not None and t.verdict not in ("failing", "buggy")
        ]
        if not scored:
            return None
        flagged = [t for t in scored if t.is_best]
        return flagged[-1] if flagged else scored[-1]  # unstamped: the only/last scored trial

    @property
    def val_score(self) -> float | None:
        best = self.best_trial
        return best.val_score if best is not None else None

    @property
    def metrics(self) -> dict[str, float]:
        best = self.best_trial
        return dict(best.metrics) if best is not None else {}

    @property
    def instance_scores(self) -> dict[str, float]:
        best = self.best_trial
        return dict(best.instance_scores) if best is not None else {}

    @property
    def report(self) -> dict | None:
        best = self.best_trial
        return best.report if best is not None else None

    @property
    def replicate_spread(self) -> float | None:
        best = self.best_trial
        return best.replicate_spread if best is not None else None

    @property
    def replicate_spreads(self) -> list[float]:
        """Every trial's replicate spread — the search's evidence about its
        own noise (spread ACROSS parameter sets is signal, not noise)."""
        return [
            t.replicate_spread for t in self.trials
            if t.verdict not in ("failing", "buggy") and t.replicate_spread is not None
        ]

    @property
    def complexity(self) -> str | None:
        """The draft complexity cue, when the attempt carried one."""
        return self.args.get("complexity")

    @property
    def holdout_score(self) -> float | None:
        best = self.best_trial
        return best.holdout_score if best is not None else None

    def holdout_blind(self) -> Candidate:
        """A copy with everything the hidden split produced removed: the
        trials' holdout fields and `is_selected` (one bit of the same
        signal). What a search policy is handed (`journal.JournalView`) —
        a process that may be optimized must never see what it must not
        optimize. A trial that FAILED holdout keeps its not-passing verdict:
        that is an execution failure, not a score."""
        blank = dict.fromkeys(HOLDOUT_FIELDS)
        return self.model_copy(
            update={
                "is_selected": False,
                "trials": [trial.model_copy(update=blank) for trial in self.trials],
            }
        )

    @property
    def is_scored(self) -> bool:
        return self.status == "passing" and self.val_score is not None
