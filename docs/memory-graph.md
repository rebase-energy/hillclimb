# Memory graph: design

Status: draft (2026-07-10). Branch: `memory-graph`.

The next big push: turn the knowledge hillclimb accumulates across searches
into a temporal knowledge graph — semantically distilled, queryable for
retrieval, and explorable as an interactive (zoom/pan/click) graph inside the
watch TUI.

## Where we are

Knowledge today (`src/hillclimb/knowledge.py`, cards under
`hillclimb/knowledge/<family>/`) is purely statistical: operator stats, top
approaches with libraries, failure modes. Retrieval is "most recent N cards by
problem family" rendered as a prose block into operator prompts. There is no
semantic layer (no claims like "technique X helps on family Y"), so a graph
built from current cards would just be stars of searches around problems.

## Design decision: file-based, Graphiti as reference

We evaluated adopting a memory framework (Mem0, Graphiti, Letta, Cognee,
LangMem, ReMe). **Decision: build file-based; do not adopt.** Graphiti
(github.com/getzep/graphiti, Apache 2.0) is the closest conceptual match —
episodes ≙ searches, entity/edge extraction with provenance ≙ claims with
candidate evidence, temporal validity windows ≙ supersession — but adoption
costs too much here:

- Its only real backends are servers (Neo4j default; embedded Kuzu is
  deprecated, FalkorDB "Lite" needs py3.12+ and Redis). Memory would leave
  git-versioned YAML for an undiffable DB, and break the hillclimb-go twin
  consuming a plain `graph.json`.
- It inverts our "journal is truth, everything else is deterministic replay"
  rule: the graph becomes a second stateful store mutated by non-reproducible
  LLM calls.
- Its extraction drives its own LLM client per episode → API-billed spend
  (operators run on subscription via headless claude; cards currently record
  `cost_usd: 0`).
- Its prompts are tuned for open-world conversational data; most of our graph
  is already structured in the journal and needs zero LLM to extract.

**Do steal:** Graphiti's extraction/dedup prompt designs (Apache 2.0), its
edge-invalidation semantics, and keep our claims schema close enough to its
episode/entity/edge model that a one-way YAML→Graphiti sync stays a weekend
project if retrieval ever needs hybrid (semantic+BM25+graph) search.

## Architecture: four layers

### 1. Claims — the semantic memory layer (the real work)

Extend `_distill_knowledge` with an LLM pass (headless claude, like operators)
over the journal + best candidates that emits typed claims:

```yaml
claims:
  - subject: histgradientboosting        # canonical entity slug
    relation: outperforms
    object: lightgbm
    scope: {family: spaceship-titanic, budget_s: 600}
    polarity: positive
    confidence: 0.8
    evidence: [c030, c031]               # candidate ids → journal provenance
    observed_at: "2026-07-09T20:39:16Z"
```

- Claims live in the card (or a sibling `claims:` block) — YAML, append-only,
  git-diffable.
- **Entity canonicalization**: `hillclimb/knowledge/entities.yaml` slug
  registry with alias lists ("GBDT"/"gradient boosting" → one node); new
  entities go through an LLM merge-or-create step against existing slugs.
  This is the hard problem — dedup quality decides whether the graph
  converges or fragments. Our entity space is small and closed-ish
  (techniques/libraries/models/families, low hundreds), so registry + one
  merge prompt should suffice where Graphiti needs embedding similarity.
- **Temporal truth**: never delete a claim. Contradictions get
  `contradicts`/`superseded_by` edges with timestamps.

### 2. Graph index — derived, rebuildable

`hillclimb/knowledge/graph.json` (versioned schema), built deterministically
from cards + claims, exactly like journal replay — never a second source of
truth. Nodes: problem, family, search, technique, library, model, operator,
claim. Edges: ran_on, used, improved, failed_with, supports, contradicts,
derived_from. Much of this needs no LLM (operator stats, libraries, scores
are already structured). Rebuild lazily on watch startup (mtime check) plus
`hillclimb knowledge rebuild`. Layout positions are cached in the index so
the graph is spatially stable across sessions. Plain JSON keeps hillclimb-go
able to render the same graph.

### 3. Graph-aware retrieval — the measurable win

Replace "last 3 cards by family" with a graph walk: new problem → family node
→ neighboring claims ranked by confidence × recency × evidence count, plus
cross-family claims that generalize. Injected as the prior-experience block.
Evaluate with/without on holdout — this layer is where memory earns its keep.

### 4. TUI graph — interactive visualization

New screen in `hillclimb watch` (`g` key), Textual ≥1.0 (mouse support is
native):

- Custom canvas widget: edges/nodes in braille subpixels (2×4 per cell, the
  textual-plotext trick), labels as text above a zoom threshold.
- Layout: networkx spring_layout computed at rebuild, cached in graph.json.
- Zoom = scroll / `+`/`-` (scale on graph→cell mapping); pan = drag/arrows;
  click = nearest-node hit test → right-hand detail panel (claims, evidence,
  linked searches; Enter jumps to the search's journal view).
- Readable ceiling is a few hundred visible nodes → filters (node type,
  family, time window) and fuzzy search are core features, not polish.
- Escape hatch later: `hillclimb knowledge graph --web` serving Sigma.js/d3
  from the same graph.json. TUI-first.

## Build order

1. Claims + entity extraction in the distiller (schema first — everything
   hangs off it)
2. Graph index builder + `hillclimb knowledge` CLI queries
3. Graph-aware retrieval into operator prompts (measurable on holdout)
4. Textual graph widget

Risks: #1 entity dedup quality (the make-or-break), #4 braille canvas
rendering (fun, contained). Everything else is plumbing already proven by
journal replay.
