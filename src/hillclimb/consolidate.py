"""The sleep phase: consolidate accumulated claims into generalized
knowledge and per-concept playbooks.

Two stages, run manually via `hillclimb knowledge consolidate` (LLM spend
and playbook rewrites are deliberate, reviewable events — every playbook
change is a git diff):

1. **Mechanical generalization (no model calls).** A claim asserted
   independently in >= GENERALIZE_MIN_FAMILIES distinct families whose
   subject shares a concept gets lifted up the hierarchy: one synthesized
   claim scoped `{concept: <slug>}` with the evidence union and the mean of
   the sources' *measured* confidences (adjusted track record when one
   exists, authored otherwise). Deterministic claim ids make re-runs
   idempotent; results live in `knowledge/consolidated.yaml`.

2. **Playbook rewrite (one agent call per qualifying concept).** Concepts
   with >= PLAYBOOK_MIN_CLAIMS live claims get a compact prose playbook
   (`knowledge/playbooks/<concept>.md`, YAML frontmatter recording the
   source claim ids). Draft prompts inject the playbook INSTEAD of the raw
   claim list — and credit for the search outcome flows to the playbook's
   source claims, so the learning loop stays closed through the rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from hillclimb.candidate import utcnow
from hillclimb.claims import (
    Claim,
    invoke_knowledge_agent,
    load_consolidated_claims,
    make_claim_id,
    save_consolidated_claims,
)
from hillclimb.config import Config
from hillclimb.graph import GraphNode, KnowledgeGraph, node_to_claim
from hillclimb.prompts.render import render

PLAYBOOKS_DIRNAME = "playbooks"
GENERALIZE_MIN_FAMILIES = 2
PLAYBOOK_MIN_CLAIMS = 3
PLAYBOOK_MAX_CHARS = 4000
DEFAULT_CONSOLIDATE_MODEL = "sonnet"  # synthesis, not extraction
CONSOLIDATE_TIMEOUT_S = 600


@dataclass
class Playbook:
    concept: str
    body: str
    built_at: str = ""
    source_claims: list[str] = field(default_factory=list)
    source_searches: list[str] = field(default_factory=list)


def _live_claim_nodes(graph: KnowledgeGraph) -> list[GraphNode]:
    """Claims still believed (not superseded/retired) and not themselves
    consolidation products — those never re-generalize."""
    return [
        n for n in graph.nodes
        if n.type == "claim"
        and n.superseded_at is None
        and "concept" not in (n.data.get("scope") or {})
    ]


def _measured_confidence(node: GraphNode) -> float:
    track = node.data.get("track") or {}
    return float(track.get("adjusted_confidence", node.data.get("confidence", 0.5)))


def generalize_claims(graph: KnowledgeGraph, *, now: str = "") -> list[Claim]:
    """Stage 1: lift multi-family claims up the concept hierarchy."""
    now = now or utcnow()
    groups: dict[tuple[str, str, str], list[GraphNode]] = {}
    for node in _live_claim_nodes(graph):
        key = (node.data.get("subject", ""), node.data.get("relation", ""),
               node.data.get("object", ""))
        if key[0]:
            groups.setdefault(key, []).append(node)
    generalized: list[Claim] = []
    for (subject, relation, obj), nodes in sorted(groups.items()):
        families = {str((n.data.get("scope") or {}).get("family", "")) for n in nodes}
        families.discard("")
        if len(families) < GENERALIZE_MIN_FAMILIES:
            continue
        concepts = sorted(set().union(*(n.concepts for n in nodes)))
        if not concepts:
            continue
        primary = concepts[0]
        evidence = sorted(set().union(*(set(n.data.get("evidence") or []) for n in nodes)))
        confidence = sum(_measured_confidence(n) for n in nodes) / len(nodes)
        sources = sorted(n.id.removeprefix("claim:") for n in nodes)
        scope = {"concept": primary, "families": sorted(families), "sources": sources}
        generalized.append(Claim(
            claim_id=make_claim_id(subject, relation, obj, {"concept": primary}, "consolidated"),
            subject=subject,
            relation=relation,
            object=obj,
            scope=scope,
            confidence=round(min(confidence, 1.0), 3),
            evidence=evidence,
            observed_at=now,
        ))
    return generalized


# --- playbooks ---


def playbook_path(knowledge_dir: Path, concept: str) -> Path:
    return knowledge_dir / PLAYBOOKS_DIRNAME / f"{concept}.md"


def install_playbook(knowledge_dir: Path, playbook: Playbook) -> Path:
    path = playbook_path(knowledge_dir, playbook.concept)
    path.parent.mkdir(parents=True, exist_ok=True)
    front = yaml.safe_dump(
        {
            "concept": playbook.concept,
            "built_at": playbook.built_at or utcnow(),
            "source_claims": playbook.source_claims,
            "source_searches": playbook.source_searches,
        },
        sort_keys=False,
    )
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(f"---\n{front}---\n\n{playbook.body.strip()}\n")
    tmp.replace(path)
    return path


def _parse_playbook(path: Path) -> Playbook | None:
    try:
        text = path.read_text()
        if not text.startswith("---"):
            return Playbook(concept=path.stem, body=text.strip())
        _, front, body = text.split("---", 2)
        meta = yaml.safe_load(front) or {}
        return Playbook(
            concept=str(meta.get("concept", path.stem)),
            body=body.strip(),
            built_at=str(meta.get("built_at", "")),
            source_claims=[str(c) for c in meta.get("source_claims", [])],
            source_searches=[str(s) for s in meta.get("source_searches", [])],
        )
    except Exception:  # noqa: BLE001 — a corrupt playbook must never block a search
        return None


def load_playbooks(knowledge_dir: Path, concepts: list[str] | None = None) -> list[Playbook]:
    directory = knowledge_dir / PLAYBOOKS_DIRNAME
    if not directory.exists():
        return []
    playbooks = []
    for path in sorted(directory.glob("*.md")):
        playbook = _parse_playbook(path)
        if playbook is None or not playbook.body:
            continue
        if concepts is not None and playbook.concept not in concepts:
            continue
        playbooks.append(playbook)
    return playbooks


def render_playbooks(playbooks: list[Playbook]) -> str:
    if not playbooks:
        return ""
    parts = [
        f"## Playbook: {p.concept}\n\n{p.body}"
        for p in playbooks
    ]
    return (
        "Consolidated playbooks distilled from ALL previous searches on this "
        "kind of problem (follow unless the data contradicts them):\n\n"
        + "\n\n".join(parts)
    )


def _concept_digest(concept: str, nodes: list[GraphNode]) -> str:
    lines = []
    for node in nodes:
        claim = node_to_claim(node)
        track = node.data.get("track") or {}
        record = (
            f" [measured: {track['adjusted_confidence']:.2f} over "
            f"{track['injections']} search(es)]"
            if track else ""
        )
        obj = f" {claim.object}" if claim.object else ""
        where = claim.scope.get("family") or claim.scope.get("concept") or "?"
        lines.append(
            f"- {claim.subject} {claim.relation}{obj} (on {where}, "
            f"authored confidence {claim.confidence:.2f}{record})"
        )
    return "\n".join(lines)


def _playbook_concepts(graph: KnowledgeGraph) -> dict[str, list[GraphNode]]:
    """Concept -> live claim nodes (family-scoped AND generalized) that
    mention it."""
    by_concept: dict[str, list[GraphNode]] = {}
    for node in graph.nodes:
        if node.type != "claim" or node.superseded_at is not None:
            continue
        for concept in node.concepts:
            by_concept.setdefault(concept, []).append(node)
        scope_concept = (node.data.get("scope") or {}).get("concept")
        if scope_concept and node not in by_concept.get(scope_concept, []):
            by_concept.setdefault(scope_concept, []).append(node)
    return {
        concept: nodes
        for concept, nodes in by_concept.items()
        if len(nodes) >= PLAYBOOK_MIN_CLAIMS
    }


def write_concept_playbook(
    knowledge_dir: Path,
    graph: KnowledgeGraph,
    concept: str,
    nodes: list[GraphNode],
    config: Config,
    log,
) -> Path | None:
    """One agent call: claims digest in, prose playbook out. Best effort."""
    prompt = render(
        "consolidate",
        concept=concept,
        n_claims=len(nodes),
        claims_digest=_concept_digest(concept, nodes),
        max_chars=PLAYBOOK_MAX_CHARS,
    )
    workspace = knowledge_dir / ".consolidate" / concept
    result = invoke_knowledge_agent(
        config,
        operator="consolidate",
        prompt=prompt,
        workspace=workspace,
        timeout_s=CONSOLIDATE_TIMEOUT_S,
        default_model=DEFAULT_CONSOLIDATE_MODEL,
    )
    if not result.ok:
        log(f"consolidate: playbook agent failed for {concept} ({result.error_kind})")
        return None
    body_path = workspace / "playbook.md"
    if not body_path.exists() or not body_path.read_text().strip():
        log(f"consolidate: agent wrote no playbook for {concept}")
        return None
    body = body_path.read_text().strip()[:PLAYBOOK_MAX_CHARS]
    node_ids = {n.id for n in nodes}
    sources = sorted(n.id.removeprefix("claim:") for n in nodes)
    searches = sorted({
        edge.dst.removeprefix("search:")
        for edge in graph.edges
        if edge.type == "derived_from" and edge.src in node_ids
    })
    return install_playbook(knowledge_dir, Playbook(
        concept=concept, body=body, built_at=utcnow(),
        source_claims=sources, source_searches=searches,
    ))


def consolidate(knowledge_dir: Path, config: Config, log, *, dry_run: bool = False) -> dict:
    """The full sleep phase. Returns a summary dict for the CLI."""
    from hillclimb.graph import load_or_build_graph, rebuild_graph

    graph = load_or_build_graph(knowledge_dir)
    generalized = generalize_claims(graph)
    if generalized and not dry_run:
        save_consolidated_claims(knowledge_dir, load_consolidated_claims(knowledge_dir) + generalized)
        graph = rebuild_graph(knowledge_dir)  # generalized claims join the graph
    concepts = _playbook_concepts(graph)
    playbooks_written: list[str] = []
    if not dry_run:
        for concept, nodes in sorted(concepts.items()):
            path = write_concept_playbook(knowledge_dir, graph, concept, nodes, config, log)
            if path is not None:
                playbooks_written.append(str(path))
                log(f"consolidate: playbook for {concept} -> {path}")
        if playbooks_written:
            rebuild_graph(knowledge_dir)
    return {
        "generalized": [c.claim_id for c in generalized],
        "playbook_concepts": sorted(concepts),
        "playbooks_written": playbooks_written,
    }
