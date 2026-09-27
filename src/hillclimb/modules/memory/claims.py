"""Semantic memory: typed claims distilled from finished searches.

The statistical layer (knowledge.py) records what happened; this module
records what it means. After a search finishes, one cheap agent pass reads a
digest of the journal plus the winning solution and writes typed claims —
"histgradientboosting helps on spaceship-titanic under a 600s budget" — each
carrying a canonical entity slug, a confidence, and provenance (candidate
ids), so downstream consumers (graph.py, prompt retrieval) can rank and
trace them.

Entities are canonicalized against a per-hillclimb-dir registry
(knowledge/entities.yaml) and auto-categorized into a small curated concept
ontology (knowledge/concepts.yaml) by CLOSED-SET classification: the agent
picks concepts from the ontology and may only PROPOSE additions (flagged
`proposed: true` for the user to promote), which keeps the taxonomy from
fragmenting into near-duplicates.

Everything stays file-based and git-versionable. Claims are immutable once
written into their card; supersession between claims is computed at
graph-build time (graph.py), never by rewriting old cards.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field, field_validator

from hillclimb.agents.base import OperatorRequest
from hillclimb.harness.candidate import utcnow
from hillclimb.config import Config
from hillclimb.prompts.render import render
from hillclimb.harness.routing import Router

if TYPE_CHECKING:
    from hillclimb.harness.journal import Journal
    from hillclimb.modules.memory.knowledge import KnowledgeCard

CLAIMS_FILENAME = "claims.yaml"
ENTITIES_FILENAME = "entities.yaml"
CONCEPTS_FILENAME = "concepts.yaml"
REGISTRY_SCHEMA_VERSION = 1

CLAIM_RELATIONS = ("helps", "hurts", "outperforms", "fails_with", "requires", "no_effect")
ENTITY_KINDS = ("technique", "library", "model_family", "feature", "practice")

# the distill pass is cheap summarization work; route it to the small model
# unless the user's routing block says otherwise
DEFAULT_DISTILL_MODEL = "haiku"


class Claim(BaseModel):
    claim_id: str = ""
    subject: str  # canonical entity slug
    relation: str
    object: str = ""  # entity slug or short free text; empty for unary claims
    scope: dict = Field(default_factory=dict)  # family / problem_id / budget_s
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)  # candidate ids
    observed_at: str = Field(default_factory=utcnow)
    superseded_by: str | None = None

    @field_validator("relation")
    @classmethod
    def _known_relation(cls, value: str) -> str:
        if value not in CLAIM_RELATIONS:
            raise ValueError(f"unknown relation {value!r}")
        return value


class Entity(BaseModel):
    slug: str
    kind: str = "technique"
    aliases: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    first_seen: str = Field(default_factory=utcnow)


class Concept(BaseModel):
    slug: str
    dimension: str
    parent: str | None = None
    description: str = ""
    proposed: bool = False


# The curated starting ontology. Small on purpose: closed-set classification
# stays reliable only while the choice list fits comfortably in a prompt, and
# the user promotes proposed additions by flipping `proposed: false`.
SEED_CONCEPTS: tuple[Concept, ...] = (
    Concept(slug="tabular", dimension="modality", description="Row/column structured data"),
    Concept(slug="time-series", dimension="modality", description="Temporally ordered observations"),
    Concept(slug="text", dimension="modality", description="Natural-language data"),
    Concept(slug="image", dimension="modality", description="Pixel data"),
    Concept(slug="classification", dimension="task", description="Predict a discrete label"),
    Concept(slug="regression", dimension="task", description="Predict a continuous value"),
    Concept(slug="forecasting", dimension="task", description="Predict future values of a series"),
    Concept(slug="decision-trees", dimension="model-family", description="Tree and gradient-boosted-tree models"),
    Concept(slug="neural-networks", dimension="model-family", description="Deep learning models"),
    Concept(slug="linear-models", dimension="model-family", description="Linear/logistic regression and kin"),
    Concept(slug="ensemble", dimension="model-family", description="Combinations of multiple models"),
    Concept(slug="nearest-neighbors", dimension="model-family", description="Instance/similarity-based models"),
    Concept(slug="probabilistic", dimension="model-family", description="Quantile/distributional/Bayesian models"),
    Concept(slug="feature-engineering", dimension="stage", description="Constructing or transforming features"),
    Concept(slug="preprocessing", dimension="stage", description="Cleaning, imputation, encoding, scaling"),
    Concept(slug="validation", dimension="stage", description="Split design and evaluation hygiene"),
    Concept(slug="tuning", dimension="stage", description="Hyperparameter search"),
    Concept(slug="post-processing", dimension="stage", description="Calibration, clipping, blending of outputs"),
)

_CLASSIFICATION_METRICS = ("accuracy", "auc", "auroc", "log-loss", "logloss", "f1")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9._]+", "-", name.strip().lower()).strip("-")


def make_claim_id(subject: str, relation: str, obj: str, scope: dict, run_ref: str) -> str:
    scope_key = ",".join(f"{k}={scope[k]}" for k in sorted(scope))
    raw = "|".join((subject, relation, obj, scope_key, run_ref))
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def claim_key(claim: Claim) -> tuple[str, str, str, str]:
    """Identity for supersession/dedup across searches: the newest claim with
    this key supersedes older ones (computed at graph-build time)."""
    return (claim.subject, claim.relation, claim.object, str(claim.scope.get("family", "")))


def problem_concepts(kind: str, metric: str) -> list[str]:
    """Deterministic concepts for a problem node — no model call needed."""
    concepts: list[str] = []
    if kind == "emflow":
        concepts += ["time-series", "forecasting"]
    elif kind == "csv":
        concepts.append("tabular")
    metric_l = (metric or "").lower()
    if any(m in metric_l for m in _CLASSIFICATION_METRICS):
        concepts.append("classification")
    elif metric_l and "forecasting" not in concepts:
        concepts.append("regression")
    return concepts


# --- registries (knowledge/entities.yaml, knowledge/concepts.yaml) ---


def _load_registry(path: Path, key: str, model: type[BaseModel]) -> list:
    """Corrupt entries are skipped, never fatal — same contract as cards."""
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:  # noqa: BLE001
        return []
    items = []
    for raw in data.get(key) or []:
        try:
            items.append(model.model_validate(raw))
        except Exception:  # noqa: BLE001
            continue
    return items


def _save_registry(path: Path, key: str, items: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        key: [item.model_dump(exclude_none=True) for item in items],
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False))
    tmp.replace(path)


def load_entities(knowledge_dir: Path) -> list[Entity]:
    return _load_registry(knowledge_dir / ENTITIES_FILENAME, "entities", Entity)


def save_entities(knowledge_dir: Path, entities: list[Entity]) -> None:
    _save_registry(knowledge_dir / ENTITIES_FILENAME, "entities", entities)


def load_concepts(knowledge_dir: Path) -> list[Concept]:
    return _load_registry(knowledge_dir / CONCEPTS_FILENAME, "concepts", Concept)


def ensure_concepts(knowledge_dir: Path) -> list[Concept]:
    """Seed the ontology on first use; afterwards the file is the user's."""
    concepts = load_concepts(knowledge_dir)
    if not concepts:
        concepts = list(SEED_CONCEPTS)
        _save_registry(knowledge_dir / CONCEPTS_FILENAME, "concepts", concepts)
    return concepts


def save_concepts(knowledge_dir: Path, concepts: list[Concept]) -> None:
    _save_registry(knowledge_dir / CONCEPTS_FILENAME, "concepts", concepts)


CONSOLIDATED_FILENAME = "consolidated.yaml"


def load_consolidated_claims(knowledge_dir: Path) -> list[Claim]:
    """Generalized claims written by the consolidation pass — claims whose
    scope is a concept rather than a family."""
    return _load_registry(knowledge_dir / CONSOLIDATED_FILENAME, "claims", Claim)


def save_consolidated_claims(knowledge_dir: Path, claims: list[Claim]) -> None:
    """Dedup by claim_id (deterministic for a given generalization), newest
    observed_at wins — re-running consolidation is idempotent."""
    by_id: dict[str, Claim] = {}
    for claim in sorted(claims, key=lambda c: c.observed_at):
        by_id[claim.claim_id] = claim
    _save_registry(
        knowledge_dir / CONSOLIDATED_FILENAME, "claims",
        sorted(by_id.values(), key=lambda c: c.claim_id),
    )


def merge_entities(
    existing: list[Entity], new: list[Entity], known_concepts: set[str]
) -> list[Entity]:
    """Alias-aware dedup: a new entity whose slug OR any alias matches an
    existing entity (case-insensitive) merges into it; otherwise it is
    appended. Concept assignments are filtered to the known ontology so a
    hallucinated concept never enters the registry."""
    merged = [entity.model_copy(deep=True) for entity in existing]
    by_name: dict[str, Entity] = {}
    for entity in merged:
        by_name[entity.slug.lower()] = entity
        for alias in entity.aliases:
            by_name.setdefault(alias.lower(), entity)
    for candidate in new:
        slug = slugify(candidate.slug)
        if not slug:
            continue
        names = [slug] + [a.lower() for a in candidate.aliases]
        target = next((by_name[n] for n in names if n in by_name), None)
        concepts = [c for c in candidate.concepts if c in known_concepts]
        if target is None:
            entity = Entity(
                slug=slug,
                kind=candidate.kind if candidate.kind in ENTITY_KINDS else "technique",
                aliases=sorted({a for a in candidate.aliases if a.lower() != slug}),
                concepts=sorted(set(concepts)),
                first_seen=candidate.first_seen,
            )
            merged.append(entity)
            by_name[slug] = entity
            for alias in entity.aliases:
                by_name.setdefault(alias.lower(), entity)
        else:
            target.aliases = sorted(
                set(target.aliases) | {a for a in candidate.aliases if a.lower() != target.slug.lower()}
            )
            target.concepts = sorted(set(target.concepts) | set(concepts))
            for name in names:
                by_name.setdefault(name, target)
    return merged


def merge_concepts(existing: list[Concept], proposed: list[Concept]) -> list[Concept]:
    """Proposed concepts are appended (flagged) only when the slug is new;
    the agent never mutates the curated set."""
    merged = list(existing)
    known = {c.slug for c in existing}
    for concept in proposed:
        slug = slugify(concept.slug)
        if not slug or slug in known:
            continue
        merged.append(
            Concept(
                slug=slug,
                dimension=concept.dimension or "other",
                parent=concept.parent,
                description=concept.description,
                proposed=True,
            )
        )
        known.add(slug)
    return merged


# --- the extraction pass ---


def parse_claims_file(
    path: Path,
    *,
    run_ref: str,
    family: str,
    problem_id: str,
    budget_s: int,
) -> tuple[list[Claim], list[Entity], list[Concept]]:
    """Validate the agent-written claims.yaml. Per-entry: one malformed claim
    drops that claim, not the file. A missing/unparseable file yields empty
    results — the search must never fail on a bad distill pass."""
    if not path.exists():
        return [], [], []
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:  # noqa: BLE001
        return [], [], []
    if not isinstance(data, dict):
        return [], [], []
    claims: list[Claim] = []
    for raw in data.get("claims") or []:
        try:
            claim = Claim.model_validate(raw)
        except Exception:  # noqa: BLE001
            continue
        claim.subject = slugify(claim.subject)
        claim.object = claim.object.strip()
        if not claim.subject:
            continue
        claim.scope = {
            "family": family,
            "problem_id": problem_id,
            "budget_s": budget_s,
            **{k: v for k, v in (claim.scope or {}).items() if k not in ("family", "problem_id", "budget_s")},
        }
        claim.claim_id = make_claim_id(
            claim.subject, claim.relation, claim.object, claim.scope, run_ref
        )
        claims.append(claim)
    entities = []
    for raw in data.get("entities") or []:
        try:
            entities.append(Entity.model_validate(raw))
        except Exception:  # noqa: BLE001
            continue
    proposed = []
    for raw in data.get("proposed_concepts") or []:
        try:
            proposed.append(Concept.model_validate(raw))
        except Exception:  # noqa: BLE001
            continue
    return claims, entities, proposed


def _clean(text: str, limit: int) -> str:
    return " ".join((text or "").split())[:limit]


def _search_digest(journal: Journal, problem, card: KnowledgeCard, max_lines: int = 40) -> str:
    """Compact factual record of the search for the distill prompt: per-
    candidate outcomes plus the card's aggregates."""
    lines: list[str] = []
    for name, stat in card.operator_stats.items():
        best = f" best_val={stat.best_val:.5g}" if stat.best_val is not None else ""
        lines.append(f"- operator {name}: {stat.ok}/{stat.attempts} ok{best}")
    candidates = [c for c in journal.candidates.values() if c.operator != "baseline"]
    for c in candidates[:max_lines]:
        score = f" val={c.val_score:.5g}" if c.val_score is not None else ""
        summary = _clean(c.summary, 160)
        lines.append(f"- {c.candidate_id} {c.operator} {c.status}{score}: {summary}")
    if len(candidates) > max_lines:
        lines.append(f"- … {len(candidates) - max_lines} more candidates omitted")
    for failure in card.failure_modes:
        lines.append(f"- failure mode: {failure}")
    return "\n".join(lines)


def _solution_excerpt(journal: Journal, problem, config: Config) -> str:
    selected = journal.selected_candidate(problem.higher_is_better, config.holdout.selection)
    if selected is None:
        return "(no selected candidate)"
    parts: list[str] = [f"Selected candidate {selected.candidate_id} ({selected.operator}):"]
    candidate_dir = Path(selected.candidate_dir)
    solution = candidate_dir / "solution.py"
    if solution.exists():
        parts.append("```python\n" + solution.read_text()[:4000] + "\n```")
    notes = candidate_dir / "notes.md"
    if notes.exists():
        parts.append("Agent notes:\n" + notes.read_text()[:2000])
    return "\n".join(parts)


def _concepts_block(concepts: list[Concept]) -> str:
    lines = [
        f"- {c.slug} ({c.dimension}): {c.description}" + (" [proposed]" if c.proposed else "")
        for c in concepts
    ]
    return "\n".join(lines) or "(empty)"


def _entities_block(entities: list[Entity], limit: int = 80) -> str:
    lines = []
    for entity in entities[:limit]:
        aliases = f" (aka {', '.join(entity.aliases[:3])})" if entity.aliases else ""
        lines.append(f"- {entity.slug} [{entity.kind}]{aliases}")
    return "\n".join(lines) or "(none yet)"


def resolve_pass_route(
    config: Config, operator: str, default_model: str
) -> tuple[str, str, str, dict[str, int | float] | None]:
    """(agent, model, auth, sampling) for a knowledge pass.
    Falls through the normal routing layers, but when neither the operator
    key nor `default` pins a model the global scalar is overridden by the
    pass's own default — these passes don't need the search's operator
    model."""
    route = Router(config).resolve(operator)
    explicit = any(
        key in config.routing and (config.routing[key].model or config.routing[key].models)
        for key in (operator, "default")
    )
    model = route.model if explicit or route.agent != "claude-code" else default_model
    return route.agent, model, route.agent_auth, route.sampling


def invoke_knowledge_agent(
    config: Config,
    *,
    operator: str,
    prompt: str,
    work_dir: Path,
    timeout_s: int,
    default_model: str,
):
    """One headless agent call for a knowledge pass; the agent communicates
    by writing files into `work_dir` (prompt.md stays there for
    inspection). Resolved through the api seam tests patch."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "prompt.md").write_text(prompt)
    agent_name, model, auth, sampling = resolve_pass_route(
        config, operator, default_model
    )
    # resolve through the api namespace — the seam tests patch to keep every
    # agent call fake; lazy import avoids the module cycle
    from hillclimb.api import get_agent

    agent = get_agent(
        agent_name, auth=auth, pi_models_file=config.pi.models_file
    )
    return agent.invoke(
        OperatorRequest(
            operator=operator, prompt=prompt, candidate_dir=work_dir,
            timeout_s=timeout_s, model=model, sampling=sampling,
        )
    )


def _distill(
    *,
    card: KnowledgeCard,
    digest: str,
    excerpt: str,
    work_dir: Path,
    knowledge_dir: Path,
    config: Config,
    log,
) -> list[Claim]:
    concepts = ensure_concepts(knowledge_dir)
    entities = load_entities(knowledge_dir)
    direction = "higher is better" if card.higher_is_better else "lower is better"
    prompt = render(
        "distill",
        problem_id=card.problem_id,
        family=card.family,
        metric=card.metric or "unknown",
        direction=direction,
        budget_s=card.budget_s,
        search_digest=digest,
        solution_excerpt=excerpt,
        concepts_block=_concepts_block(concepts),
        entities_block=_entities_block(entities),
        relations=" | ".join(CLAIM_RELATIONS),
        claims_filename=CLAIMS_FILENAME,
    )
    result = invoke_knowledge_agent(
        config,
        operator="distill",
        prompt=prompt,
        work_dir=work_dir,
        timeout_s=config.learning.claims_timeout_s,
        default_model=DEFAULT_DISTILL_MODEL,
    )
    if not result.ok:
        log(f"learning: distill agent failed ({result.error_kind}): {result.error_message}")
        return []
    claims, new_entities, proposed = parse_claims_file(
        work_dir / CLAIMS_FILENAME,
        run_ref=card.run_ref,
        family=card.family,
        problem_id=card.problem_id,
        budget_s=card.budget_s,
    )
    return absorb_parsed(knowledge_dir, claims, new_entities, proposed, log)


def absorb_parsed(
    knowledge_dir: Path,
    claims: list[Claim],
    new_entities: list[Entity],
    proposed: list[Concept],
    log,
) -> list[Claim]:
    """Fold one parsed claims file into the registries: merge proposed
    concepts and entities, then keep only claims whose subject is a
    registered entity — an unregistered subject can't be linked in the
    graph. The shared tail of every distill pass (searches and papers)."""
    concepts = ensure_concepts(knowledge_dir)
    if proposed:
        concepts = merge_concepts(concepts, proposed)
        save_concepts(knowledge_dir, concepts)
    if new_entities:
        known = {c.slug for c in concepts}
        save_entities(knowledge_dir, merge_entities(load_entities(knowledge_dir), new_entities, known))
    registered = {e.slug for e in load_entities(knowledge_dir)}
    kept = [c for c in claims if c.subject in registered]
    if len(kept) < len(claims):
        log(f"learning: dropped {len(claims) - len(kept)} claim(s) with unregistered subjects")
    return kept


def distill_claims(
    journal: Journal,
    *,
    problem,
    card: KnowledgeCard,
    search_dir: Path,
    knowledge_dir: Path,
    config: Config,
    log,
) -> list[Claim]:
    """The post-search LLM pass: journal + winning solution -> typed claims.
    Best effort by contract — callers treat [] as 'nothing learned'."""
    claims = _distill(
        card=card,
        digest=_search_digest(journal, problem, card),
        excerpt=_solution_excerpt(journal, problem, config),
        work_dir=search_dir / "distill",
        knowledge_dir=knowledge_dir,
        config=config,
        log=log,
    )
    return backdate_claims(claims, journal)


def backdate_claims(claims: list[Claim], journal: Journal) -> list[Claim]:
    """Stamp each claim with the moment its evidence existed: the finish of
    the last candidate it cites, instead of the distill pass's "now". The
    knowledge graph slices time by these stamps, so with them the graph's
    timeline steps candidate by candidate rather than in one clump per
    search. Claims without recognisable evidence keep their stamp."""
    finished = {
        c.candidate_id: (c.finished_at or c.created_at)
        for c in journal.candidates.values()
        if c.finished_at or c.created_at
    }
    for claim in claims:
        stamps = [finished[e] for e in claim.evidence if e in finished]
        if stamps:
            claim.observed_at = max(stamps)
    return claims


def distill_claims_from_card(
    card: KnowledgeCard,
    *,
    work_dir: Path,
    knowledge_dir: Path,
    config: Config,
    log,
) -> list[Claim]:
    """Backfill variant for cards whose search artifacts are gone: the card
    itself is the only evidence, so claims lean on its aggregates."""
    digest = yaml.safe_dump(
        card.model_dump(exclude_none=True, exclude={"claims"}), sort_keys=False
    )
    return _distill(
        card=card,
        digest=digest,
        excerpt="(search artifacts unavailable — card data only)",
        work_dir=work_dir,
        knowledge_dir=knowledge_dir,
        config=config,
        log=log,
    )


def render_claims(claims: list[Claim], *, max_claims: int = 8) -> str:
    """Prompt section for retrieval (graph-ranked upstream): one line per
    claim, provenance kept implicit — the numbers already went through the
    same metric conventions."""
    if not claims:
        return ""
    lines = []
    for claim in claims[:max_claims]:
        obj = f" {claim.object}" if claim.object else ""
        where = claim.scope.get("problem_id") or claim.scope.get("family") or "prior searches"
        lines.append(
            f"- {claim.subject} {claim.relation}{obj} (on {where}, "
            f"confidence {claim.confidence:.0%})"
        )
    return (
        "Distilled claims from prior searches (treat as strong hints, not "
        "ground truth — re-verify anything load-bearing):\n" + "\n".join(lines)
    )
