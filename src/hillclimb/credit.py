"""Credit assignment: claims earn a track record from search outcomes.

The distill pass (claims.py) authors a confidence; this module makes claims
answer for it. When a search's draft prompt carried distilled claims, the
search's outcome becomes a reward shared by every injected claim — coarse
per-search attribution by design (claims reach the draft operator only, and
disentangling their individual influence is not worth the machinery yet).

Rewards mirror the routing bandit's scale (bandit.py): 1.0 when the search
beat the best previously recorded score on the SAME problem (falling back to
its own baseline candidate when it is the first search on the problem), 0.25
for a scored search that set no record, 0.0 when nothing scored.

Concurrency rule: suite searches are separate processes with no file
locking, so credit is never a mutable registry — each search writes ONE
event file under knowledge/credit/ (atomic, git-versioned, replayable), and
the graph builder folds all events into per-claim track records at rebuild
time. A claim's `adjusted_confidence` shrinks toward its measured mean
reward as injections accumulate; chronically failing claims retire (become
superseded) once the record is conclusive.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from hillclimb.candidate import utcnow

CREDIT_SCHEMA_VERSION = 1
CREDIT_DIRNAME = "credit"

REWARD_OK_NO_GAIN = 0.25  # same scale as bandit.candidate_reward
# Beta-style smoothing: the authored confidence acts as PRIOR_WEIGHT
# pseudo-observations, so one bad search doesn't kill a claim
PRIOR_WEIGHT = 2.0
# retirement is conservative on purpose: a wrongly retired good claim costs
# more than a lingering bad one, which adjusted ranking already buries
RETIRE_MIN_INJECTIONS = 3
RETIRE_THRESHOLD = 0.15


class CreditEvent(BaseModel):
    schema_version: int = CREDIT_SCHEMA_VERSION
    run_ref: str
    problem_id: str
    family: str = ""
    claim_ids: list[str] = Field(default_factory=list)
    reward: float = 0.0
    basis: str = "none-scored"  # prior-best | own-baseline | none-scored
    observed_at: str = Field(default_factory=utcnow)


class Track(BaseModel):
    injections: int = 0
    reward_sum: float = 0.0
    last_observed_at: str = ""

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.injections if self.injections else 0.0


def search_reward(
    journal, problem, prior_cards: list, *, selection: str = "rank-blend"
) -> tuple[float, str]:
    """(reward, basis) for a finished search. `prior_cards` must already
    exclude this search's own card."""
    direction = 1 if problem.lower_is_better else -1
    selected = journal.selected_candidate(problem.lower_is_better, selection)
    val = selected.val_score if selected is not None else None
    if val is None:
        return 0.0, "none-scored"
    priors = [
        c.selected_val
        for c in prior_cards
        if c.problem_id == problem.problem_id and c.selected_val is not None
    ]
    if priors:
        best_prior = min(priors, key=lambda v: direction * v)
        improved = direction * val < direction * best_prior
        return (1.0 if improved else REWARD_OK_NO_GAIN), "prior-best"
    baseline = journal.candidates.get("c000")
    baseline_val = baseline.val_score if baseline is not None else None
    if baseline_val is not None:
        improved = direction * val < direction * baseline_val
        return (1.0 if improved else REWARD_OK_NO_GAIN), "own-baseline"
    # first search on the problem with an unscored baseline: producing any
    # scored solution sets the bar
    return 1.0, "own-baseline"


INJECTED_CLAIMS_FILENAME = "injected_claims.json"


def record_injected_claims(search_dir: Path, claim_ids: list[str]) -> None:
    """Search-dir sidecar written at injection time (crash-safe: exists as
    soon as the prompt context does). A resumed search retrieves afresh, so
    ids are unioned — credit covers everything any draft was shown."""
    import json

    path = search_dir / INJECTED_CLAIMS_FILENAME
    known = set(read_injected_claims(search_dir))
    merged = sorted(known | set(claim_ids))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"claim_ids": merged, "recorded_at": utcnow()}, indent=1))
    tmp.replace(path)


def read_injected_claims(search_dir: Path) -> list[str]:
    import json

    path = search_dir / INJECTED_CLAIMS_FILENAME
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text()) or {}
        return [str(cid) for cid in data.get("claim_ids", [])]
    except Exception:  # noqa: BLE001
        return []


def credit_event_path(knowledge_dir: Path, run_ref: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", run_ref).strip("-") or "run"
    return knowledge_dir / CREDIT_DIRNAME / f"{slug}.yaml"


def write_credit_event(knowledge_dir: Path, event: CreditEvent) -> Path:
    """One file per search (idempotent by run_ref) — no locks needed across
    concurrent suite processes."""
    path = credit_event_path(knowledge_dir, event.run_ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(event.model_dump(exclude_none=True), sort_keys=False))
    tmp.replace(path)
    return path


def load_credit_events(knowledge_dir: Path) -> list[CreditEvent]:
    directory = knowledge_dir / CREDIT_DIRNAME
    if not directory.exists():
        return []
    events: list[CreditEvent] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text()) or {}
            if data.get("schema_version") != CREDIT_SCHEMA_VERSION:
                continue
            events.append(CreditEvent.model_validate(data))
        except Exception:  # noqa: BLE001 — a corrupt event must never block a rebuild
            continue
    return events


def fold_track(events: list[CreditEvent]) -> dict[str, Track]:
    """Per-claim accumulation over all credit events, replay-style."""
    tracks: dict[str, Track] = {}
    for event in sorted(events, key=lambda e: e.observed_at):
        for claim_id in event.claim_ids:
            track = tracks.setdefault(claim_id, Track())
            track.injections += 1
            track.reward_sum += event.reward
            track.last_observed_at = event.observed_at
    return tracks


def adjusted_confidence(base: float, track: Track) -> float:
    """The authored confidence counts as PRIOR_WEIGHT pseudo-observations;
    measured rewards take over as the record grows."""
    return (base * PRIOR_WEIGHT + track.reward_sum) / (PRIOR_WEIGHT + track.injections)


def should_retire(adjusted: float, track: Track) -> bool:
    return track.injections >= RETIRE_MIN_INJECTIONS and adjusted < RETIRE_THRESHOLD
