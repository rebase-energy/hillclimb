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

This is the built-in graph module (registry name `knowledge-graph`, the
default `graph:` of every climber): `KnowledgeGraphBuilder` at the bottom
wraps `build_graph` in the contract that `modules/memory/base.py` holds, so
a climber may bring its own.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from hillclimb.harness.candidate import utcnow
from hillclimb.modules.memory.base import (  # the models live with the contract; re-exported here
    DEFAULT_GRAPH,
    GRAPH_SCHEMA_VERSION,
    GraphEdge,
    GraphModule,
    GraphNode,
    KnowledgeGraph,
)
from hillclimb.modules.memory.claims import (
    Claim,
    claim_key,
    load_concepts,
    load_entities,
    problem_concepts,
    slugify,
)

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
    "generalizes",   # consolidated claim -> the family claims it lifted
)

# concept nodes are timeless: first_seen="" sorts before every ISO timestamp,
# so they are visible at any scrubbed time


def graph_path(knowledge_dir: Path) -> Path:
    return knowledge_dir / GRAPH_FILENAME


def load_all_cards(knowledge_dir: Path) -> list:
    """Every card in the knowledge dir (family subdirs), oldest first.
    Corrupt or version-mismatched cards are skipped, as everywhere."""
    from hillclimb.modules.memory.knowledge import SCHEMA_VERSION, KnowledgeCard

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
                      previous: dict[str, tuple], dim: int = 2) -> dict:
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
            pos[node] = tuple(
                sum(p[k] for p in placed) / len(placed) for k in range(dim)
            )
    layout = nx.spring_layout(
        graph,
        pos=pos or None,
        fixed=list(pinned) or None,
        seed=LAYOUT_SEED,
        dim=dim,
    )
    return {node: tuple(round(float(c), 4) for c in xy) for node, xy in layout.items()}


def compute_layout(
    nodes: list[GraphNode], edges: list[GraphEdge],
    previous: KnowledgeGraph | None,
    *, dim: int = 2,
) -> dict[str, tuple]:
    """Pinned incremental spring layout; {} (pos stays None) when networkx
    is not installed — the index is still fully usable, only unplaced.
    dim=2 pins against previous pos, dim=3 against previous pos3."""
    prior = {}
    if previous is not None:
        if dim == 3:
            prior = {n.id: n.pos3 for n in previous.nodes if n.pos3 is not None}
        else:
            prior = {n.id: n.pos for n in previous.nodes if n.pos is not None}
    try:
        return _spring_positions(
            [n.id for n in nodes], [(e.src, e.dst) for e in edges], prior, dim=dim
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
            data={"metric": card.metric, "higher_is_better": card.higher_is_better},
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
                    "subject": claim.subject, "relation": claim.relation,
                    "object": claim.object, "confidence": claim.confidence,
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

    # paper-derived claims: a paper is a knowledge source like a search —
    # its claims join the same supersession/retrieval/credit economy, wired
    # to a `paper:` node instead of a `search:` node. A scoped paper also
    # materializes its family/problem anchors so the linkage is inspectable
    # BEFORE any search has run on that problem.
    from hillclimb.modules.memory.papers import load_papers

    for paper in load_papers(knowledge_dir):
        t = paper.added_at
        paper_node_id = f"paper:{paper.slug}"
        add_node(GraphNode(
            id=paper_node_id, type="paper", label=paper.title or paper.slug, first_seen=t,
            data={
                "file": paper.file, "family": paper.family,
                "problem_id": paper.problem_id, "n_claims": len(paper.claims),
            },
        ))
        paper_family_id = f"family:{paper.family}" if paper.family else ""
        paper_problem_id = f"problem:{paper.problem_id}" if paper.problem_id else ""
        if paper_family_id:
            add_node(GraphNode(id=paper_family_id, type="family", label=paper.family, first_seen=t))
        if paper_problem_id:
            add_node(GraphNode(
                id=paper_problem_id, type="problem", label=paper.problem_id, first_seen=t,
            ))
            if paper_family_id:
                add_edge(paper_problem_id, paper_family_id, "belongs_to", first_seen=t)
        anchor = paper_problem_id or paper_family_id
        if anchor:
            add_edge(paper_node_id, anchor, "applies_to", first_seen=t)
        for claim in paper.claims:
            all_claims.append(claim)
            claim_id = f"claim:{claim.claim_id}"
            subject = entity_by_slug.get(claim.subject)
            label = f"{claim.subject} {claim.relation}" + (f" {claim.object}" if claim.object else "")
            add_node(GraphNode(
                id=claim_id, type="claim", label=label,
                concepts=subject.concepts if subject else [],
                first_seen=claim.observed_at,
                data={
                    "subject": claim.subject, "relation": claim.relation,
                    "object": claim.object, "confidence": claim.confidence,
                    "evidence": claim.evidence, "scope": claim.scope,
                    "paper": paper.slug,
                },
            ))
            add_edge(claim_id, f"entity:{claim.subject}", "about", first_seen=claim.observed_at)
            object_slug = alias_to_slug.get(claim.object.lower())
            if object_slug:
                add_edge(claim_id, f"entity:{object_slug}", "about", first_seen=claim.observed_at)
            add_edge(claim_id, paper_node_id, "derived_from", first_seen=claim.observed_at)
            if anchor:
                add_edge(claim_id, anchor, "applies_to", first_seen=claim.observed_at)

    # generalized (consolidated) claims: scoped to a concept, linked down to
    # the family-scoped claims they were lifted from
    from hillclimb.modules.memory.claims import load_consolidated_claims

    for claim in load_consolidated_claims(knowledge_dir):
        all_claims.append(claim)
        claim_id = f"claim:{claim.claim_id}"
        subject = entity_by_slug.get(claim.subject)
        label = f"{claim.subject} {claim.relation}" + (f" {claim.object}" if claim.object else "")
        concept = str(claim.scope.get("concept", ""))
        add_node(GraphNode(
            id=claim_id, type="claim", label=label,
            concepts=sorted(set((subject.concepts if subject else [])) | ({concept} if concept else set())),
            first_seen=claim.observed_at,
            data={
                "subject": claim.subject, "relation": claim.relation,
                "object": claim.object, "confidence": claim.confidence,
                "evidence": claim.evidence, "scope": claim.scope,
                "consolidated": True,
            },
        ))
        if f"entity:{claim.subject}" in nodes:
            add_edge(claim_id, f"entity:{claim.subject}", "about", first_seen=claim.observed_at)
        if concept and f"concept:{concept}" in nodes:
            add_edge(claim_id, f"concept:{concept}", "applies_to", first_seen=claim.observed_at)
        for source in claim.scope.get("sources", []):
            if f"claim:{source}" in nodes:
                add_edge(claim_id, f"claim:{source}", "generalizes", first_seen=claim.observed_at)

    for claim_id, (at, by) in compute_supersessions(all_claims).items():
        node = nodes.get(f"claim:{claim_id}")
        if node is not None:
            node.superseded_at = at
            node.data["superseded_by"] = by
            add_edge(f"claim:{by}", f"claim:{claim_id}", "supersedes", first_seen=at)

    # fold credit events into per-claim track records: adjusted confidence
    # drives retrieval; a conclusively bad record retires the claim through
    # the same supersession machinery (visible in history, gone from now)
    from hillclimb.modules.memory.credit import (
        adjusted_confidence,
        fold_track,
        load_credit_events,
        should_retire,
    )

    for claim_id, track in fold_track(load_credit_events(knowledge_dir)).items():
        node = nodes.get(f"claim:{claim_id}")
        if node is None:
            continue
        adjusted = adjusted_confidence(float(node.data.get("confidence", 0.5)), track)
        node.data["track"] = {
            "injections": track.injections,
            "mean_reward": round(track.mean_reward, 3),
            "adjusted_confidence": round(adjusted, 3),
        }
        if node.superseded_at is None and should_retire(adjusted, track):
            node.superseded_at = track.last_observed_at
            node.data["retired"] = "track record"

    # drop edges whose endpoints never materialized (e.g. a claim subject
    # missing from the registry — shouldn't happen, but stay robust)
    edge_list = [e for e in edges.values() if e.src in nodes and e.dst in nodes]
    node_list = sorted(nodes.values(), key=lambda n: n.id)
    edge_list.sort(key=lambda e: (e.src, e.dst, e.type))
    positions = compute_layout(node_list, edge_list, previous)
    positions3 = compute_layout(node_list, edge_list, previous, dim=3)
    for node in node_list:
        node.pos = positions.get(node.id)
        node.pos3 = positions3.get(node.id)
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
        schema_version=graph.schema_version, builder=graph.builder, built_at=graph.built_at,
        events=[event for event in graph.events if event <= t],
        nodes=nodes, edges=edges,
    )


def change_events(graph: KnowledgeGraph) -> list[str]:
    """Every distinct moment the graph changed: a node or edge appearing
    (first_seen) or a claim being displaced (superseded_at). The fine
    timeline for the graph view's scrubber — with claims backdated to their
    evidencing candidate (claims.backdate_claims), one tick per candidate
    that taught us something."""
    stamps = {n.first_seen for n in graph.nodes} | {e.first_seen for e in graph.edges}
    stamps |= {n.superseded_at for n in graph.nodes if n.superseded_at}
    stamps |= {e.superseded_at for e in graph.edges if e.superseded_at}
    stamps.discard("")
    return sorted(stamps)


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


def retrieve_claims(
    graph: KnowledgeGraph,
    *,
    family: str,
    problem_id: str,
    concepts: list[str],
    limit: int = 8,
) -> list[GraphNode]:
    """The graph walk behind prompt retrieval: live (non-superseded) claims
    scoped to this family/problem first, then cross-family claims that share
    a concept with the problem — the concept layer is what lets a lesson from
    one tabular competition reach another. Ranked by scope match, then
    confidence weighted by evidence volume, then recency."""
    wanted = set(concepts)
    ranked: list[tuple[bool, float, str, GraphNode]] = []
    for node in graph.nodes:
        if node.type != "claim" or node.superseded_at is not None:
            continue
        scope = node.data.get("scope") or {}
        same_family = scope.get("family") == family or scope.get("problem_id") == problem_id
        if not same_family and not (wanted & set(node.concepts)):
            continue
        evidence = node.data.get("evidence") or []
        track = node.data.get("track") or {}
        # a measured track record beats the authored guess
        confidence = float(track.get("adjusted_confidence", node.data.get("confidence", 0.5)))
        score = confidence * (1 + 0.2 * min(len(evidence), 3))
        ranked.append((same_family, score, node.first_seen, node))
    ranked.sort(key=lambda r: (r[0], r[1], r[2]), reverse=True)
    return [r[3] for r in ranked[:limit]]


def node_to_claim(node: GraphNode) -> Claim:
    """Rehydrate a Claim from its graph node (for rendering)."""
    return Claim(
        claim_id=node.id.removeprefix("claim:"),
        subject=node.data.get("subject") or node.label.split(" ")[0],
        relation=node.data.get("relation", "helps"),
        object=node.data.get("object", ""),
        scope=node.data.get("scope") or {},
        confidence=float(node.data.get("confidence", 0.5)),
        evidence=list(node.data.get("evidence") or []),
        observed_at=node.first_seen or utcnow(),
    )


def fuzzy_match(nodes: list[GraphNode], query: str, limit: int = 20) -> list[GraphNode]:
    """Subsequence scorer: all query chars must appear in order; contiguity
    and prefix matches score higher."""
    query = query.strip().lower()
    if not query:
        return []
    scored: list[tuple[float, str, GraphNode]] = []
    for node in nodes:
        haystack = f"{node.label} {node.id}".lower()
        score = _subsequence_score(haystack, query)
        if score > 0:
            scored.append((score, node.id, node))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [node for _, _, node in scored[:limit]]


def _subsequence_score(haystack: str, query: str) -> float:
    index = 0
    score = 0.0
    streak = 0
    for char in query:
        found = haystack.find(char, index)
        if found < 0:
            return 0.0
        streak = streak + 1 if found == index else 1
        score += streak
        if found == 0:
            score += 2  # prefix bonus
        index = found + 1
    return score / (1 + len(haystack) / 40)


def query_graph(
    graph: KnowledgeGraph, terms: str, *, family: str = "", limit: int = 5
) -> list[dict]:
    """The read-only lookup behind `hillclimb knowledge query` — built for
    operator agents consulting memory mid-search, so it answers in facts:
    matching nodes, the live claims about them (with measured track records
    and retirement status), and the searches that used them."""
    node_map = graph.node_map()
    matches = fuzzy_match(graph.nodes, terms, limit=limit)
    # entities lead (they aggregate their claims), claims follow, plumbing
    # nodes (searches/operators) last — stable within a tier by fuzzy score
    tier = {"technique": 0, "library": 0, "model_family": 0, "feature": 0,
            "practice": 0, "problem": 1, "family": 1, "concept": 2, "claim": 3}
    matches.sort(key=lambda n: tier.get(n.type, 4))
    hits: list[dict] = []
    for node in matches:
        entry: dict = {
            "id": node.id,
            "type": node.type,
            "label": node.label,
            "concepts": list(node.concepts),
        }
        if node.superseded_at is not None:
            entry["superseded_at"] = node.superseded_at
        if node.type == "claim":
            entry["claim"] = _claim_summary(node)
        claims = []
        for edge in graph.edges:
            if edge.type == "about" and edge.dst == node.id and edge.src in node_map:
                claim_node = node_map[edge.src]
                scope = claim_node.data.get("scope") or {}
                if family and scope.get("family") not in ("", None, family) and "concept" not in scope:
                    continue
                claims.append(_claim_summary(claim_node))
        if claims:
            entry["claims"] = claims
        searches = sorted({
            node_map[e.src].label
            for e in graph.edges
            if e.type == "used" and e.dst == node.id and e.src in node_map
        })
        if searches:
            entry["used_by_searches"] = searches[:5]
        hits.append(entry)
    return hits


def _claim_summary(node: GraphNode) -> dict:
    summary = {
        "statement": node.label,
        "confidence": node.data.get("confidence"),
        "scope": node.data.get("scope") or {},
    }
    if node.data.get("track"):
        summary["track"] = node.data["track"]
    if node.superseded_at is not None:
        summary["superseded"] = True
        if node.data.get("retired"):
            summary["retired_by"] = node.data["retired"]
    return summary


def render_query_hits(hits: list[dict]) -> str:
    if not hits:
        return "no matches in the knowledge graph"
    lines: list[str] = []
    for hit in hits:
        concepts = f"  [{', '.join(hit['concepts'])}]" if hit.get("concepts") else ""
        dead = "  (superseded)" if hit.get("superseded_at") else ""
        lines.append(f"{hit['label']} ({hit['type']}){concepts}{dead}")
        for claim in [hit["claim"]] if "claim" in hit else hit.get("claims", []):
            track = claim.get("track")
            record = (
                f", measured {track['adjusted_confidence']} over {track['injections']} search(es)"
                if track else ""
            )
            status = " [RETIRED]" if claim.get("superseded") else ""
            where = claim["scope"].get("family") or claim["scope"].get("concept") or "?"
            lines.append(
                f"  - {claim['statement']} (on {where}, "
                f"confidence {claim.get('confidence')}{record}){status}"
            )
        if hit.get("used_by_searches"):
            lines.append(f"  used by: {', '.join(hit['used_by_searches'])}")
    return "\n".join(lines)


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
    tracked = [n for n in graph.nodes if n.type == "claim" and "track" in n.data]
    if tracked:
        retired = sum(1 for n in tracked if n.data.get("retired"))
        lines.append(f"{len(tracked)} claim(s) with a track record, {retired} retired by record")
    return "\n".join(lines)
