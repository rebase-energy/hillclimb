"""The knowledge graph index: a derived, rebuildable view over the knowledge
directory (cards + claims + entity/concept registries).

`graph.json` is never a source of truth — like journal replay, `build_graph`
is a deterministic fold over the YAML files, so the index can be deleted and
rebuilt at any time (`hillclimb knowledge rebuild`). Every node and edge
carries `first_seen` (and, where a newer claim displaces an older one,
`superseded_at`), which makes any historical view a pure filter: `graph_at(t)`
returns the graph as it was known at time t. Timestamps are ISO-8601 UTC
strings and compare lexicographically — no datetime parsing anywhere.

Node positions are computed here at build time (networkx spring layout,
seeded, with previously placed nodes pinned so the map stays spatially stable
as knowledge grows) and cached in the index; viewers only read them. Without
networkx installed (engine-only installs) positions stay None and viewers
fall back to a deterministic placement.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from hillclimb.candidate import utcnow
from hillclimb.claims import (
    Claim,
    claim_key,
    load_concepts,
    load_entities,
    problem_concepts,
    slugify,
)

GRAPH_SCHEMA_VERSION = 1
GRAPH_FILENAME = "graph.json"
LAYOUT_SEED = 42

NODE_KINDS = (
    "problem", "family", "search", "operator", "concept", "claim",
    # entity kinds pass through from the registry:
    "technique", "library", "model_family", "feature", "practice",
)
EDGE_KINDS = (
    "ran_on",        # search -> problem
    "belongs_to",    # problem -> family
    "used",          # search -> entity/operator
    "about",         # claim -> subject/object entity
    "derived_from",  # claim -> search (provenance)
    "applies_to",    # claim -> problem/family (scope)
    "supersedes",    # newer claim -> older claim
    "has_concept",   # entity/problem -> concept
    "is_a",          # concept -> parent concept
)

# concept nodes are timeless: first_seen="" sorts before every ISO timestamp,
# so they are visible at any scrubbed time


class GraphNode(BaseModel):
    id: str
    type: str
    label: str = ""
    concepts: list[str] = Field(default_factory=list)
    pos: tuple[float, float] | None = None
    first_seen: str = ""
    superseded_at: str | None = None
    data: dict = Field(default_factory=dict)


class GraphEdge(BaseModel):
    src: str
    dst: str
    type: str
    first_seen: str = ""
    superseded_at: str | None = None
    weight: float = 1.0


class KnowledgeGraph(BaseModel):
    schema_version: int = GRAPH_SCHEMA_VERSION
    built_at: str = Field(default_factory=utcnow)
    events: list[str] = Field(default_factory=list)  # sorted search-finish times
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)

    def node_map(self) -> dict[str, GraphNode]:
        return {node.id: node for node in self.nodes}


def graph_path(knowledge_dir: Path) -> Path:
    return knowledge_dir / GRAPH_FILENAME


def load_all_cards(knowledge_dir: Path) -> list:
    """Every card in the knowledge dir (family subdirs), oldest first.
    Corrupt or version-mismatched cards are skipped, as everywhere."""
    from hillclimb.knowledge import SCHEMA_VERSION, KnowledgeCard

    cards = []
    if not knowledge_dir.exists():
        return cards
    for path in sorted(knowledge_dir.glob("*/*.yaml")):
        try:
            data = yaml.safe_load(path.read_text()) or {}
            if data.get("schema_version") != SCHEMA_VERSION:
                continue
            cards.append(KnowledgeCard.model_validate(data))
        except Exception:  # noqa: BLE001
            continue
    cards.sort(key=lambda c: c.finished_at)
    return cards


def compute_supersessions(claims: list[Claim]) -> dict[str, tuple[str, str]]:
    """claim_id -> (superseded_at, superseded_by). Two rules, both purely
    temporal so old cards never need rewriting:
    - a newer claim with the same identity key displaces the older one;
    - opposing outcome relations (helps/hurts/no_effect) about the same
      subject+object+family contradict — the newest belief wins."""
    superseded: dict[str, tuple[str, str]] = {}
    by_key: dict[tuple, list[Claim]] = {}
    opposing: dict[tuple, list[Claim]] = {}
    for claim in claims:
        by_key.setdefault(claim_key(claim), []).append(claim)
        if claim.relation in ("helps", "hurts", "no_effect"):
            key = (claim.subject, claim.object, str(claim.scope.get("family", "")))
            opposing.setdefault(key, []).append(claim)
    for group in list(by_key.values()) + list(opposing.values()):
        group.sort(key=lambda c: c.observed_at)
        newest = group[-1]
        for older in group[:-1]:
            if older.claim_id != newest.claim_id and older.claim_id not in superseded:
                superseded[older.claim_id] = (newest.observed_at, newest.claim_id)
    return superseded


def _spring_positions(nodes: list[str], edges: list[tuple[str, str]],
                      previous: dict[str, tuple[float, float]]) -> dict:
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(sorted(nodes))
    graph.add_edges_from(sorted(edges))
    pinned = {n: previous[n] for n in graph.nodes if n in previous}
    # new nodes start at the centroid of already-placed neighbors so they
    # settle near their cluster instead of teleporting across the map
    pos = dict(pinned)
    for node in graph.nodes:
        if node in pos:
            continue
        placed = [pos[n] for n in graph.neighbors(node) if n in pinned]
        if placed:
            pos[node] = (
                sum(p[0] for p in placed) / len(placed),
                sum(p[1] for p in placed) / len(placed),
            )
    layout = nx.spring_layout(
        graph,
        pos=pos or None,
        fixed=list(pinned) or None,
        seed=LAYOUT_SEED,
    )
    return {node: (round(float(x), 4), round(float(y), 4)) for node, (x, y) in layout.items()}


def compute_layout(
    nodes: list[GraphNode], edges: list[GraphEdge],
    previous: KnowledgeGraph | None,
) -> dict[str, tuple[float, float]]:
    """Pinned incremental spring layout; {} (pos stays None) when networkx
    is not installed — the index is still fully usable, only unplaced."""
    prior = {}
    if previous is not None:
        prior = {n.id: n.pos for n in previous.nodes if n.pos is not None}
    try:
        return _spring_positions(
            [n.id for n in nodes], [(e.src, e.dst) for e in edges], prior
        )
    except ModuleNotFoundError:
        return {}


def build_graph(knowledge_dir: Path, previous: KnowledgeGraph | None = None) -> KnowledgeGraph:
    """Deterministic fold: same knowledge dir -> same graph (modulo built_at).
    Structural nodes/edges come straight from cards; semantic ones from
    claims; membership from the registries."""
    cards = load_all_cards(knowledge_dir)
    entities = load_entities(knowledge_dir)
    concepts = load_concepts(knowledge_dir)
    entity_by_slug = {e.slug: e for e in entities}
    alias_to_slug = {e.slug.lower(): e.slug for e in entities}
    for entity in entities:
        for alias in entity.aliases:
            alias_to_slug.setdefault(alias.lower(), entity.slug)

    nodes: dict[str, GraphNode] = {}
    edges: dict[tuple[str, str, str], GraphEdge] = {}

    def add_node(node: GraphNode) -> None:
        existing = nodes.get(node.id)
        if existing is None:
            nodes[node.id] = node
        else:
            if node.first_seen and (not existing.first_seen or node.first_seen < existing.first_seen):
                existing.first_seen = node.first_seen
            existing.data.update(node.data)
            existing.concepts = sorted(set(existing.concepts) | set(node.concepts))

    def add_edge(src: str, dst: str, kind: str, first_seen: str = "", weight: float = 1.0) -> None:
        key = (src, dst, kind)
        existing = edges.get(key)
        if existing is None:
            edges[key] = GraphEdge(src=src, dst=dst, type=kind, first_seen=first_seen, weight=weight)
        else:
            if first_seen and (not existing.first_seen or first_seen < existing.first_seen):
                existing.first_seen = first_seen
            existing.weight += weight

    def entity_id(name: str) -> str:
        slug = alias_to_slug.get(name.lower(), slugify(name))
        return f"entity:{slug}"

    for concept in concepts:
        add_node(GraphNode(
            id=f"concept:{concept.slug}", type="concept", label=concept.slug,
            data={"dimension": concept.dimension, "proposed": concept.proposed},
        ))
        if concept.parent:
            add_edge(f"concept:{concept.slug}", f"concept:{concept.parent}", "is_a")

    for entity in entities:
        add_node(GraphNode(
            id=f"entity:{entity.slug}", type=entity.kind, label=entity.slug,
            concepts=entity.concepts, first_seen=entity.first_seen,
            data={"aliases": entity.aliases},
        ))
        for concept in entity.concepts:
            add_edge(f"entity:{entity.slug}", f"concept:{concept}", "has_concept",
                     first_seen=entity.first_seen)

    all_claims: list[Claim] = []
    for card in cards:
        t = card.finished_at
        kind_guess = "emflow" if card.target.startswith("emflow://") else "csv"
        p_concepts = problem_concepts(kind_guess, card.metric)
        problem_id = f"problem:{card.problem_id}"
        family_id = f"family:{card.family}"
        search_id = f"search:{card.run_ref}"
        add_node(GraphNode(
            id=family_id, type="family", label=card.family, first_seen=t,
        ))
        add_node(GraphNode(
            id=problem_id, type="problem", label=card.problem_id,
            concepts=p_concepts, first_seen=t,
            data={"metric": card.metric, "lower_is_better": card.lower_is_better},
        ))
        for concept in p_concepts:
            add_edge(problem_id, f"concept:{concept}", "has_concept", first_seen=t)
        add_node(GraphNode(
            id=search_id, type="search", label=card.run_ref, first_seen=t,
            data={
                "problem_id": card.problem_id, "run_ref": card.run_ref,
                "finished_at": t, "best_val": card.selected_val,
                "metric": card.metric, "n_candidates": card.n_candidates,
                "selected_operator": card.selected_operator,
            },
        ))
        add_edge(search_id, problem_id, "ran_on", first_seen=t)
        add_edge(problem_id, family_id, "belongs_to", first_seen=t)
        for name, stat in card.operator_stats.items():
            operator_id = f"operator:{name}"
            add_node(GraphNode(id=operator_id, type="operator", label=name, first_seen=t))
            add_edge(search_id, operator_id, "used", first_seen=t, weight=stat.attempts)
        for approach in card.top_approaches:
            for library in approach.libraries:
                lib_id = entity_id(library)
                if lib_id not in nodes:
                    add_node(GraphNode(
                        id=lib_id, type="library", label=lib_id.removeprefix("entity:"),
                        first_seen=t,
                    ))
                add_edge(search_id, lib_id, "used", first_seen=t)
        for claim in card.claims:
            all_claims.append(claim)
            claim_id = f"claim:{claim.claim_id}"
            subject = entity_by_slug.get(claim.subject)
            label = f"{claim.subject} {claim.relation}" + (f" {claim.object}" if claim.object else "")
            add_node(GraphNode(
                id=claim_id, type="claim", label=label,
                concepts=subject.concepts if subject else [],
                first_seen=claim.observed_at,
                data={
                    "relation": claim.relation, "confidence": claim.confidence,
                    "evidence": claim.evidence, "scope": claim.scope,
                },
            ))
            add_edge(claim_id, f"entity:{claim.subject}", "about", first_seen=claim.observed_at)
            object_slug = alias_to_slug.get(claim.object.lower())
            if object_slug:
                add_edge(claim_id, f"entity:{object_slug}", "about", first_seen=claim.observed_at)
            add_edge(claim_id, search_id, "derived_from", first_seen=claim.observed_at)
            scope_problem = claim.scope.get("problem_id", "")
            scope_target = (
                f"problem:{scope_problem}" if f"problem:{scope_problem}" in nodes
                else family_id
            )
            add_edge(claim_id, scope_target, "applies_to", first_seen=claim.observed_at)

    for claim_id, (at, by) in compute_supersessions(all_claims).items():
        node = nodes.get(f"claim:{claim_id}")
        if node is not None:
            node.superseded_at = at
            node.data["superseded_by"] = by
            add_edge(f"claim:{by}", f"claim:{claim_id}", "supersedes", first_seen=at)

    # drop edges whose endpoints never materialized (e.g. a claim subject
    # missing from the registry — shouldn't happen, but stay robust)
    edge_list = [e for e in edges.values() if e.src in nodes and e.dst in nodes]
    node_list = sorted(nodes.values(), key=lambda n: n.id)
    edge_list.sort(key=lambda e: (e.src, e.dst, e.type))
    positions = compute_layout(node_list, edge_list, previous)
    for node in node_list:
        node.pos = positions.get(node.id)
    return KnowledgeGraph(
        events=sorted({card.finished_at for card in cards}),
        nodes=node_list,
        edges=edge_list,
    )


def graph_at(graph: KnowledgeGraph, t: str | None) -> KnowledgeGraph:
    """The graph as known at time t (None = now). Pure filter: elements born
    after t vanish, elements superseded by t vanish, edges lose any endpoint
    that vanished."""
    if t is None:
        return graph
    nodes = [
        n for n in graph.nodes
        if n.first_seen <= t and (n.superseded_at is None or n.superseded_at > t)
    ]
    ids = {n.id for n in nodes}
    edges = [
        e for e in graph.edges
        if e.src in ids and e.dst in ids
        and e.first_seen <= t and (e.superseded_at is None or e.superseded_at > t)
    ]
    return KnowledgeGraph(
        schema_version=graph.schema_version, built_at=graph.built_at,
        events=[event for event in graph.events if event <= t],
        nodes=nodes, edges=edges,
    )


def load_graph(path: Path) -> KnowledgeGraph | None:
    if not path.exists():
        return None
    try:
        graph = KnowledgeGraph.model_validate_json(path.read_text())
    except Exception:  # noqa: BLE001
        return None
    return graph if graph.schema_version == GRAPH_SCHEMA_VERSION else None


def write_graph(path: Path, graph: KnowledgeGraph) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(graph.model_dump_json(indent=1))
    tmp.replace(path)
    return path


def _knowledge_mtime(knowledge_dir: Path) -> float:
    """Newest mtime among the graph's YAML inputs (cards + registries)."""
    latest = 0.0
    for pattern in ("*/*.yaml", "*.yaml"):
        for path in knowledge_dir.glob(pattern):
            latest = max(latest, path.stat().st_mtime)
    return latest


def load_or_build_graph(knowledge_dir: Path, *, force: bool = False) -> KnowledgeGraph:
    """The one entry point viewers/retrieval should use: reuses graph.json
    when it is newer than every YAML input, else rebuilds (pinning the
    previous layout) and rewrites it."""
    path = graph_path(knowledge_dir)
    previous = load_graph(path)
    if (
        not force
        and previous is not None
        and path.exists()
        and path.stat().st_mtime >= _knowledge_mtime(knowledge_dir)
    ):
        return previous
    graph = build_graph(knowledge_dir, previous)
    write_graph(path, graph)
    return graph


def rebuild_graph(knowledge_dir: Path) -> KnowledgeGraph:
    return load_or_build_graph(knowledge_dir, force=True)


def graph_stats(graph: KnowledgeGraph) -> str:
    from collections import Counter

    node_kinds = Counter(n.type for n in graph.nodes)
    edge_kinds = Counter(e.type for e in graph.edges)
    lines = [
        f"{len(graph.nodes)} nodes, {len(graph.edges)} edges, "
        f"{len(graph.events)} search event(s), built {graph.built_at}",
        "nodes: " + ", ".join(f"{k}={v}" for k, v in sorted(node_kinds.items())),
        "edges: " + ", ".join(f"{k}={v}" for k, v in sorted(edge_kinds.items())),
    ]
    superseded = sum(1 for n in graph.nodes if n.superseded_at is not None)
    if superseded:
        lines.append(f"{superseded} superseded claim(s)")
    return "\n".join(lines)
