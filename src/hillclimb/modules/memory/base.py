"""The memory contracts: the memory kind a climber names, and the graph
module that turns the knowledge directory into a graph, picks the claims a
search is shown, and answers `hillclimb knowledge query`.

Memory is the YAML under `hillclimb/knowledge/` — cards per finished search,
the claims written into them, the entity and concept registries, credit,
playbooks, papers, skills. A climber names it `memory: files | none`
(`knowledge-graph` is the pre-0.4 spelling of `files` and still loads).

The knowledge graph (`knowledge/graph.json`) is never a source of truth: a
derived, rebuildable index over those files. `graph:` names the GraphModule
that builds and traverses it — the registry name `knowledge-graph` (the
built-in, in graph.py), a `file.py[:Class]` next to the climber's manifest,
or `package.module:Class`. The contract has three parts over one data model:

1. `build(knowledge_dir, previous)` — a deterministic fold over the YAML
   files into a KnowledgeGraph; `previous` is the last index, for pinning
   layout positions so the map stays spatially stable.
2. `retrieve(graph, *, family, problem_id, concepts, limit)` — the claim
   nodes a new search is shown as prior experience.
3. `query(graph, terms, *, family, limit)` — JSON-serializable hits for the
   read-only lookup operator agents call mid-search.

`retrieve` and `query` default to the built-in claim walk, so a module that
only changes the graph's structure has nothing to write but `build` — as
long as it keeps the CLAIM-NODE CONVENTION the harness reads claims by:

- a claim is a node with `type == "claim"` and id `claim:<claim_id>`; the id
  after the prefix is the credit key (whoever is quoted in a prompt answers
  for the outcome);
- `node.data` carries `subject`, `relation`, `object`, `scope`, `confidence`
  and `evidence` (candidate ids), optionally `track` (a measured record);
- `first_seen` and `superseded_at` are ISO-8601 UTC strings (or `""` /
  None), compared lexicographically by the time scrubber — never parsed;
- a live claim has `superseded_at is None`.

Query hits are plain dicts that JSON-serialize and carry at least `id`,
`type` and `label`. Consolidation (`hillclimb knowledge consolidate`) walks
claim nodes by the same convention and is not part of the contract yet; a
graph without claim nodes consolidates to nothing.

This module imports only the standard library and pydantic, so a climber's
graph module can take the models from `hillclimb.sdk` without pulling the
harness in.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, BeforeValidator, Field

# --- the memory kind ---------------------------------------------------------

MEMORY_KINDS = ("files", "none")
MEMORY_ALIASES = {"knowledge-graph": "files"}  # the pre-0.4 spelling


def normalize_memory(value: Any) -> Any:
    """Map an old spelling to the current one; everything else is left to
    the Literal, so an unknown kind is still refused by name."""
    return MEMORY_ALIASES.get(value, value) if isinstance(value, str) else value


MemoryKind = Annotated[Literal["files", "none"], BeforeValidator(normalize_memory)]


# --- the graph data model ----------------------------------------------------

# v2: nodes gained pos3 (3D spring layout for the plotui viewer). The version
# check in graph.load_graph makes a stale v1 graph.json rebuild on first load.
GRAPH_SCHEMA_VERSION = 2
DEFAULT_GRAPH = "knowledge-graph"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class GraphNode(BaseModel):
    id: str
    type: str
    label: str = ""
    concepts: list[str] = Field(default_factory=list)
    pos: tuple[float, float] | None = None
    # 3D twin of pos, used by the plotui graph viewer. pos stays 2D for
    # external consumers of graph.json (hillclimb-go renders from it).
    pos3: tuple[float, float, float] | None = None
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
    # the key of the GraphModule that built this index (a registry name, a
    # `file.py#<sha>` or a `module:Class`); an index built by another module
    # is stale however fresh its mtime
    builder: str = DEFAULT_GRAPH
    built_at: str = Field(default_factory=_utcnow)
    events: list[str] = Field(default_factory=list)  # sorted search-finish times
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)

    def node_map(self) -> dict[str, GraphNode]:
        return {node.id: node for node in self.nodes}


# --- the contract ------------------------------------------------------------


class GraphModule:
    """Subclass, set `name`, implement `build`; override `retrieve` and
    `query` when the built-in claim walk does not fit your graph."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    version: ClassVar[str] = "1"

    # set by the resolver: the registry name, `file.py#<sha12>` for a file
    # module (so editing the file invalidates graph.json), or `module:Class`
    key: str = ""

    def build(self, knowledge_dir: Path, previous: KnowledgeGraph | None = None) -> KnowledgeGraph:
        raise NotImplementedError(f"{type(self).__name__} must implement build(knowledge_dir, previous)")

    def retrieve(
        self,
        graph: KnowledgeGraph,
        *,
        family: str,
        problem_id: str,
        concepts: list[str],
        limit: int = 8,
    ) -> list[GraphNode]:
        """The claim nodes a search is shown: the built-in walk — live claims
        scoped to the family or problem first, then cross-family claims that
        share a concept, ranked by scope, confidence and recency."""
        from hillclimb.modules.memory.graph import retrieve_claims  # graph.py imports this module

        return retrieve_claims(graph, family=family, problem_id=problem_id, concepts=concepts, limit=limit)

    def query(self, graph: KnowledgeGraph, terms: str, *, family: str = "", limit: int = 5) -> list[dict]:
        """The read-only lookup: the built-in fuzzy match over labels and
        ids, each hit with the live claims about it and the searches that
        used it."""
        from hillclimb.modules.memory.graph import query_graph

        return query_graph(graph, terms, family=family, limit=limit)
