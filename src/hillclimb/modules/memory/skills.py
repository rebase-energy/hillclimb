"""Procedural memory: winning solutions harvested as reference scaffolds.

Claims say what helped; skills ARE what helped. After a search whose selected
candidate is a real winner (non-baseline, scored), its `solution.py` is
copied verbatim into `knowledge/skills/<family>--<run-ref>/` with a metadata
card — no model calls. The next search on the family (or, failing that, a
concept-sibling problem) gets the best skill copied into its FIRST draft's
candidate dir as `reference_solution.py`, with a prompt cue to adapt rather than
resubmit. Later drafts stay reference-free so the search still explores.

Kept deliberately small: at most SKILLS_PER_FAMILY skills per family,
best-score-first (direction-aware) — a library of proven scaffolds, not an
archive.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from hillclimb.candidate import utcnow
from hillclimb.direction import better
from hillclimb.direction import legacy_direction_key
from hillclimb.modules.memory.knowledge import extract_libraries

SKILLS_DIRNAME = "skills"
SKILLS_PER_FAMILY = 2
SKILL_CODE_FILENAME = "skill.py"
SKILL_META_FILENAME = "skill.yaml"
SKILL_SCHEMA_VERSION = 1


class Skill(BaseModel):
    schema_version: int = SKILL_SCHEMA_VERSION
    problem_id: str
    family: str
    concepts: list[str] = Field(default_factory=list)
    metric: str = ""
    higher_is_better: bool = True

    @model_validator(mode="before")
    @classmethod
    def _legacy_direction_key(cls, data):
        return legacy_direction_key(data)
    score: float | None = None
    holdout: float | None = None
    libraries: list[str] = Field(default_factory=list)
    summary: str = ""
    run_ref: str = ""
    created_at: str = Field(default_factory=utcnow)


def _skill_dir(knowledge_dir: Path, family: str, run_ref: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", run_ref).strip("-") or "run"
    return knowledge_dir / SKILLS_DIRNAME / f"{family}--{slug}"


def load_skills(knowledge_dir: Path, *, family: str = "") -> list[tuple[Skill, Path]]:
    directory = knowledge_dir / SKILLS_DIRNAME
    if not directory.exists():
        return []
    skills: list[tuple[Skill, Path]] = []
    for skill_dir in sorted(directory.iterdir()):
        meta = skill_dir / SKILL_META_FILENAME
        if not meta.is_file() or not (skill_dir / SKILL_CODE_FILENAME).is_file():
            continue
        try:
            data = yaml.safe_load(meta.read_text()) or {}
            if data.get("schema_version") != SKILL_SCHEMA_VERSION:
                continue
            skill = Skill.model_validate(data)
        except Exception:  # noqa: BLE001 — a corrupt skill must never block anything
            continue
        if family and skill.family != family:
            continue
        skills.append((skill, skill_dir))
    return skills


def harvest_skill(
    journal,
    *,
    problem,
    card,
    knowledge_dir: Path,
    selection: str,
    log,
) -> Path | None:
    """Post-search harvest (mechanical). Quality gate: only a scored,
    non-baseline winner enters the library; within a family the library
    keeps the SKILLS_PER_FAMILY best by validation score."""
    selected = journal.selected_candidate(problem.higher_is_better, selection)
    if selected is None or selected.operator == "baseline" or selected.val_score is None:
        return None
    solution = Path(selected.candidate_dir) / "solution.py"
    if not solution.exists():
        return None
    existing = load_skills(knowledge_dir, family=card.family)
    scored = [(s, d) for s, d in existing if s.score is not None]
    if len(scored) >= SKILLS_PER_FAMILY:
        # direction-normalized score: smaller is better, so the worst is max
        worst_skill, worst_dir = max(
            scored,
            key=lambda pair: (-1 if problem.higher_is_better else 1) * pair[0].score,
        )
        if not better(selected.val_score, worst_skill.score, problem.higher_is_better):
            return None
        shutil.rmtree(worst_dir, ignore_errors=True)
    from hillclimb.modules.memory.claims import problem_concepts

    kind = "emflow" if card.target.startswith("emflow://") else "csv"
    skill = Skill(
        problem_id=card.problem_id,
        family=card.family,
        concepts=problem_concepts(kind, card.metric),
        metric=card.metric,
        higher_is_better=card.higher_is_better,
        score=selected.val_score,
        holdout=selected.holdout_score,
        libraries=extract_libraries(solution),
        summary=(selected.summary or "").strip()[:240],
        run_ref=card.run_ref,
    )
    skill_dir = _skill_dir(knowledge_dir, card.family, card.run_ref)
    skill_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(solution, skill_dir / SKILL_CODE_FILENAME)
    (skill_dir / SKILL_META_FILENAME).write_text(
        yaml.safe_dump(skill.model_dump(exclude_none=True), sort_keys=False)
    )
    return skill_dir


def select_skill(
    knowledge_dir: Path,
    *,
    family: str,
    concepts: list[str],
    higher_is_better: bool,
) -> tuple[Skill, Path] | None:
    """Same-family skills compete on score (comparable metric); across
    families scores don't compare, so concept-overlapping skills fall back
    to recency."""
    same_family = [
        (s, d) for s, d in load_skills(knowledge_dir, family=family) if s.score is not None
    ]
    if same_family:
        return min(
            same_family,
            key=lambda pair: (-1 if higher_is_better else 1) * pair[0].score,
        )
    wanted = set(concepts)
    siblings = [
        (s, d) for s, d in load_skills(knowledge_dir)
        if s.family != family and wanted & set(s.concepts)
    ]
    if siblings:
        return max(siblings, key=lambda pair: pair[0].created_at)
    return None
