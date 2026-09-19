"""Cross-search learning: knowledge cards distilled from finished searches.

Every completed search writes a compact, human-readable card — what won, what
failed, what it cost — into the candidate_dir's `hillclimb/knowledge/` folder
(git-versionable: the repo accumulates learning). New searches on the same
problem or problem family retrieve recent cards and inject a "prior
experience" section into draft prompts, so agents start from what already
worked instead of rediscovering it.

Extraction is purely mechanical (journal + notes.md + import scanning) — no
model calls. The agents themselves wrote the summaries; this module just
routes them forward in time.

Live sharing (CORAL-style) routes them sideways as well: a running search
republishes its card into the run-scoped `runs/<run-id>/knowledge/` dir after
every executed candidate, and concurrent sibling searches in the same run
poll those cards when building operator prompts — the solar search's feature
trick reaches the wind search's next operator mid-run, not next run.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from hillclimb.candidate import utcnow
from hillclimb.direction import legacy_direction_key
from hillclimb.claims import Claim

SCHEMA_VERSION = 1
CARD_FILENAME = "knowledge_card.yaml"

# stdlib-ish / harness modules that say nothing about the approach
_UNINFORMATIVE_IMPORTS = {
    "os", "sys", "re", "json", "math", "time", "pathlib", "typing", "shutil",
    "warnings", "datetime", "collections", "itertools", "functools", "io",
    "random", "dataclasses", "abc", "copy", "pickle", "subprocess", "argparse",
    "__future__",
    "emflow", "pandas", "numpy",  # present in ~every solution; carry no signal
}


class ApproachNote(BaseModel):
    candidate_id: str
    operator: str
    val_score: float | None = None
    holdout_score: float | None = None
    complexity: str | None = None
    summary: str = ""
    libraries: list[str] = Field(default_factory=list)


class OperatorStat(BaseModel):
    attempts: int = 0
    ok: int = 0
    best_val: float | None = None


class KnowledgeCard(BaseModel):
    schema_version: int = SCHEMA_VERSION
    problem_id: str
    family: str
    target: str = ""
    metric: str = ""
    higher_is_better: bool = True

    @model_validator(mode="before")
    @classmethod
    def _legacy_direction_key(cls, data):
        return legacy_direction_key(data)
    run_ref: str = ""
    finished_at: str = Field(default_factory=utcnow)
    budget_s: int = 0
    cost_usd: float = 0.0
    n_candidates: int = 0
    n_ok: int = 0
    n_failing: int = 0
    n_buggy: int = 0
    selected_val: float | None = None
    selected_holdout: float | None = None
    selected_operator: str | None = None
    operator_stats: dict[str, OperatorStat] = Field(default_factory=dict)
    top_approaches: list[ApproachNote] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    # semantic layer (claims.py): typed claims distilled by the post-search
    # LLM pass. Additive — cards without it load unchanged.
    claims: list[Claim] = Field(default_factory=list)


def problem_family(problem_id: str, target: str = "") -> str:
    """Grouping key for retrieval: emflow suite package (gefcom2014 from
    emflow://gefcom2014:solar), else the problem id itself."""
    if target.startswith("emflow://"):
        rest = target.removeprefix("emflow://")
        return rest.split(":", 1)[0]
    if "-" in problem_id and problem_id.rsplit("-", 1)[-1] in {
        "solar", "wind", "load", "price", "demand",
    }:
        return problem_id.rsplit("-", 1)[0]
    return problem_id


def extract_libraries(solution: Path) -> list[str]:
    """Top-level imported modules of a solution — a factual approach
    fingerprint (lightgbm vs torch says more than any adjective)."""
    if not solution.exists():
        return []
    try:
        tree = ast.parse(solution.read_text())
    except SyntaxError:
        return []
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return sorted(modules - _UNINFORMATIVE_IMPORTS)


def _failure_phrase(candidate) -> str | None:
    trial = candidate.last_trial
    replicate = candidate.last_replicate
    if trial is None or replicate is None:
        return None
    if trial.unit_tests is not None:
        tests = trial.unit_tests
        if tests.timed_out:
            return "unit tests timed out"
        if not tests.passed:
            tail = (tests.stderr_tail or tests.stdout_tail).strip()
            match = re.findall(r"([A-Za-z_]*(?:Error|Exception)[^\n]{0,80})", tail)
            if match:
                return f"unit tests: {match[-1].strip()}"
            return f"unit tests failed (exit {tests.returncode})"
    if replicate.timed_out:
        return "execution timed out"
    if trial.holdout_error:
        flat = " ".join(trial.holdout_error.split())
        return f"holdout contract violation: {flat[:80]}"
    tail = (replicate.stdout_tail or "").strip()
    match = re.findall(r"([A-Za-z_]*(?:Error|Exception)[^\n]{0,80})", tail)
    if match:
        return match[-1].strip()
    if replicate.returncode not in (0, None):
        return f"exited {replicate.returncode}"
    return None


def distill_card(
    journal,
    *,
    problem,
    run_ref: str,
    target: str = "",
    budget_s: int = 0,
    cost_usd: float = 0.0,
    selection: str = "rank-blend",
    top_n: int = 3,
) -> KnowledgeCard:
    candidates = [c for c in journal.candidates.values() if c.operator != "baseline"]
    stats: dict[str, OperatorStat] = {}
    direction = -1 if problem.higher_is_better else 1
    for c in candidates:
        stat = stats.setdefault(c.operator, OperatorStat())
        stat.attempts += 1
        if c.status == "passing":
            stat.ok += 1
            if c.val_score is not None and (
                stat.best_val is None or direction * c.val_score < direction * stat.best_val
            ):
                stat.best_val = c.val_score

    scored = sorted(
        (c for c in candidates if c.status == "passing" and c.val_score is not None),
        key=lambda c: direction * c.val_score,
    )
    seen_summaries: set[str] = set()
    top: list[ApproachNote] = []
    for c in scored:
        key = (c.summary or c.candidate_id).strip().lower()
        if key in seen_summaries:
            continue
        seen_summaries.add(key)
        top.append(
            ApproachNote(
                candidate_id=c.candidate_id,
                operator=c.operator,
                val_score=c.val_score,
                holdout_score=c.holdout_score,
                complexity=c.complexity,
                summary=(c.summary or "").strip()[:240],
                libraries=extract_libraries(Path(c.candidate_dir) / "solution.py"),
            )
        )
        if len(top) >= top_n:
            break

    failures = Counter(
        phrase for c in candidates if c.status in ("failing", "buggy") and (phrase := _failure_phrase(c))
    )
    selected = journal.selected_candidate(problem.higher_is_better, selection)
    return KnowledgeCard(
        problem_id=problem.problem_id,
        family=problem_family(problem.problem_id, target),
        target=target,
        metric=problem.metric_name,
        higher_is_better=problem.higher_is_better,
        run_ref=run_ref,
        budget_s=budget_s,
        cost_usd=round(cost_usd, 4),
        n_candidates=len(candidates),
        n_ok=sum(1 for c in candidates if c.status == "passing"),
        n_failing=sum(1 for c in candidates if c.status == "failing"),
        n_buggy=sum(1 for c in candidates if c.status == "buggy"),
        selected_val=selected.val_score if selected else None,
        selected_holdout=selected.holdout_score if selected else None,
        selected_operator=selected.operator if selected else None,
        operator_stats=stats,
        top_approaches=top,
        failure_modes=[f"{phrase} (x{count})" for phrase, count in failures.most_common(3)],
    )


def write_card(knowledge_dir: Path, card: KnowledgeCard) -> Path:
    """knowledge/<family>/<problem_id>--<run-ref>.yaml — one card per search."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", card.run_ref).strip("-") or "run"
    path = knowledge_dir / card.family / f"{card.problem_id}--{slug}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(card.model_dump(exclude_none=True), sort_keys=False))
    return path


def load_cards(
    knowledge_dir: Path,
    *,
    problem_id: str,
    family: str,
    limit: int = 10,
) -> list[KnowledgeCard]:
    """Cards for this problem (preferred) and its family, newest first."""
    if not knowledge_dir.exists():
        return []
    cards: list[KnowledgeCard] = []
    for path in (knowledge_dir / family).glob("*.yaml") if (knowledge_dir / family).exists() else []:
        try:
            data = yaml.safe_load(path.read_text()) or {}
            if data.get("schema_version") != SCHEMA_VERSION:
                continue
            cards.append(KnowledgeCard.model_validate(data))
        except Exception:  # noqa: BLE001 — a corrupt card must never block a search
            continue
    # quality-aware order: same problem first, then searches that actually
    # produced a non-baseline winner with real approaches, then recency —
    # a broken run must never outrank a podium run just by being newer
    cards.sort(
        key=lambda c: (
            c.problem_id == problem_id,
            c.selected_operator not in (None, "baseline") and bool(c.top_approaches),
            c.n_ok,
            c.finished_at,
        ),
        reverse=True,
    )
    return cards[:limit]


def render_prior_experience(cards: list[KnowledgeCard], *, max_cards: int = 3) -> str:
    """The prompt section: what worked, what to avoid, in the agents' own
    words. Compact by construction — a few hundred tokens, not a memoir."""
    if not cards:
        return ""
    lines: list[str] = []
    for card in cards[:max_cards]:
        header = f"- Past search on {card.problem_id} ({card.n_candidates} candidates"
        if card.selected_val is not None:
            header += (
                f", best {card.metric} {card.selected_val:.5g}, "
                f"winner: {card.selected_operator or '?'}"
            )
        header += "):"
        lines.append(header)
        for approach in card.top_approaches[:2]:
            libs = f" [{', '.join(approach.libraries[:4])}]" if approach.libraries else ""
            score = f" ({approach.val_score:.5g})" if approach.val_score is not None else ""
            lines.append(f"    - {approach.summary or approach.candidate_id}{libs}{score}")
        if card.failure_modes:
            lines.append(f"    - failure modes seen: {'; '.join(card.failure_modes[:2])}")
    body = "\n".join(lines)
    return (
        "Summaries of what worked in PREVIOUS searches on this problem/family "
        "(scores use the same metric and split conventions):\n" + body + "\n\n"
        "Use these as a head start — prefer refining a proven approach over "
        "rediscovering it, but do not copy an approach that is already listed "
        "under drafts below."
    )


LIVE_DIRNAME = "knowledge"


def live_card_path(run_dir: Path, search_id: str) -> Path:
    return run_dir / LIVE_DIRNAME / f"live--{search_id}.yaml"


def write_live_card(run_dir: Path, card: KnowledgeCard, search_id: str) -> Path:
    """Run-scoped live card: a snapshot of what a still-running search has
    learned so far, republished after every executed candidate so concurrent
    sibling searches in the same run can read it mid-flight. One file per
    search (the engine stays the single writer of its own card); the write is
    atomic because siblings may read at any moment."""
    path = live_card_path(run_dir, search_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(card.model_dump(exclude_none=True), sort_keys=False))
    tmp.replace(path)
    return path


def load_live_cards(
    run_dir: Path,
    *,
    exclude_search_id: str = "",
    family: str = "",
) -> list[KnowledgeCard]:
    """Live cards published by the run's other searches, most useful first:
    same problem family, then cards with real approaches, then progress."""
    directory = run_dir / LIVE_DIRNAME
    if not directory.exists():
        return []
    cards: list[KnowledgeCard] = []
    for path in sorted(directory.glob("live--*.yaml")):
        if exclude_search_id and path.name == f"live--{exclude_search_id}.yaml":
            continue
        try:
            data = yaml.safe_load(path.read_text()) or {}
            if data.get("schema_version") != SCHEMA_VERSION:
                continue
            cards.append(KnowledgeCard.model_validate(data))
        except Exception:  # noqa: BLE001 — a corrupt card must never block an operator
            continue
    cards.sort(
        key=lambda c: (c.family == family, bool(c.top_approaches), c.n_ok, c.finished_at),
        reverse=True,
    )
    return cards


def render_live_experience(cards: list[KnowledgeCard], *, max_cards: int = 3) -> str:
    """Prompt section for discoveries from sibling searches running
    CONCURRENTLY in this run. Their scores come from other problems (own
    metric and splits), so approaches transfer — numbers do not; approach
    lines therefore carry libraries and summaries, not scores."""
    if not cards:
        return ""
    lines: list[str] = []
    for card in cards[:max_cards]:
        header = f"- Concurrent search on {card.problem_id} ({card.n_ok} scored candidates so far"
        if card.selected_val is not None:
            header += f", best {card.metric or 'score'} {card.selected_val:.5g}"
        header += "):"
        lines.append(header)
        for approach in card.top_approaches[:2]:
            libs = f" [{', '.join(approach.libraries[:4])}]" if approach.libraries else ""
            lines.append(f"    - {approach.summary or approach.candidate_id}{libs}")
        if card.failure_modes:
            lines.append(f"    - failure modes seen: {'; '.join(card.failure_modes[:2])}")
    body = "\n".join(lines)
    return (
        "Discoveries from sibling searches running CONCURRENTLY on related "
        "problems (their scores use different data and metrics — borrow the "
        "approaches, not the numbers):\n" + body
    )


def complexity_offset(cards: list[KnowledgeCard]) -> int:
    """Opt-in policy bias: if past winners were never 'minimal', start the
    draft complexity schedule one step up. Conservative by design — the
    offset is 0 or 1, never more."""
    winner_complexities = [
        a.complexity
        for card in cards
        for a in card.top_approaches[:1]
        if a.complexity
    ]
    if winner_complexities and all(c != "minimal" for c in winner_complexities):
        return 1
    return 0
