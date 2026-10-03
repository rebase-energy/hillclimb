"""The exploration tree of one search — pure data, no Textual.

A search's journal is a forest: the baseline and the drafts are roots, every
debug/improve/ensemble candidate hangs off the candidate it was built from.
`build_tree` turns that into a drawable tidy tree (y = depth, x = subtree
order) and classifies every node by its *fate* — what the search did with
it — so the topology of the exploration reads at a glance: which lineages
were expanded, which were scored and then left, where the coding agents failed,
what the user pruned.

`treeview.py` renders this through plotui; `chart.py --detail` reuses the
same nodes/edges/fates with time and score as the axes. Both stay
replay-only readers of the journal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from hillclimb.harness.candidate import Candidate
from hillclimb.harness.direction import better

# What happened to a candidate after it landed. Order is the legend order.
FATES = ("expanded", "best", "discontinued", "failed", "pruned")
FATE_HELP = {
    "expanded": "children were built on it",
    "best": "the current best — the next improve hangs here",
    "discontinued": "scored, never built on",
    "failed": "buggy / abandoned / parked",
    "pruned": "cut by the user",
}
# Status values that count as "did not produce a scorable solution".
FAILED_STATUSES = frozenset({"failing", "buggy", "abandoned", "parked"})


@dataclass(frozen=True)
class TreeNode:
    id: str
    x: float            # subtree-order column (leaves 0, 1, 2…; parents centred)
    depth: int          # 0 for roots (baseline, drafts)
    operator: str
    status: str
    fate: str           # one of FATES, or "pending"
    score: float | None
    n_children: int
    summary: str
    created_at: str
    finished_at: str | None
    parent_id: str | None
    pruned: bool = False


@dataclass(frozen=True)
class TreeEdge:
    src: str
    dst: str
    kind: str = "parent"  # parent | ensemble-input
    on_path: bool = False  # both ends on the accepted lineage


@dataclass(frozen=True)
class SearchTree:
    nodes: tuple[TreeNode, ...]
    edges: tuple[TreeEdge, ...]
    accepted: tuple[str, ...]  # ids that were best-so-far when they landed, in order

    @property
    def depth(self) -> int:
        return max((n.depth for n in self.nodes), default=-1) + 1

    @property
    def best_id(self) -> str | None:
        return self.accepted[-1] if self.accepted else None

    def node(self, node_id: str) -> TreeNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def fate_counts(self) -> dict[str, int]:
        counts = {fate: 0 for fate in FATES}
        for node in self.nodes:
            if node.fate in counts:
                counts[node.fate] += 1
        return counts


def _when(candidate: Candidate) -> str:
    return candidate.finished_at or candidate.created_at


def accepted_lineage(candidates: list[Candidate], higher_is_better: bool) -> list[str]:
    """Ids that were the best-so-far the moment they landed (replay order by
    finish time) — the same staircase `chart` draws, as candidate ids."""
    best: float | None = None
    accepted: list[str] = []
    for cand in sorted(candidates, key=_when):
        score = cand.val_score
        if score is None or cand.status != "passing" or cand.pruned:
            continue
        if best is None or better(score, best, higher_is_better):
            best = score
            accepted.append(cand.candidate_id)
    return accepted


def candidates_until(candidates: list[Candidate], until: str | None) -> list[Candidate]:
    """The journal as it stood at `until` (an ISO stamp): candidates created
    by then, with results that had not landed yet shown as still pending."""
    if until is None:
        return list(candidates)
    view = []
    for cand in candidates:
        if cand.created_at > until:
            continue
        if cand.finished_at and cand.finished_at > until:
            cand = cand.model_copy(update={"status": "pending", "trials": [], "finished_at": None})
        view.append(cand)
    return view


def tree_events(candidates: list[Candidate]) -> list[str]:
    """Scrubber ticks: one per landed result, oldest first."""
    return sorted({c.finished_at for c in candidates if c.finished_at})


def build_tree(
    candidates: list[Candidate], higher_is_better: bool = True, *, layout: SearchTree | None = None
) -> SearchTree:
    """Tidy-tree layout plus fates. Children keep creation order left to
    right; a parent sits over the centre of its children; roots (baseline,
    drafts) line up at depth 0 in creation order. Ensemble inputs beyond the
    parent become extra `ensemble-input` edges (journaled in
    `climber_meta.inspiration_ids`).

    `layout` pins node positions to another tree's (by id): a scrubbed-back
    view keeps every node exactly where the live tree draws it, so stepping
    through time only adds and removes nodes. Nodes the layout does not know
    are placed by the normal rule."""
    by_id = {c.candidate_id: c for c in candidates}
    ordered = sorted(candidates, key=lambda c: (c.created_at, c.candidate_id))
    children: dict[str | None, list[Candidate]] = {}
    for cand in ordered:
        parent = cand.parent_id if cand.parent_id in by_id else None
        children.setdefault(parent, []).append(cand)

    x: dict[str, float] = {}
    depth: dict[str, int] = {}
    cursor = 0.0
    # iterative DFS: (candidate, depth, entered) — leaves take the next
    # column, a parent takes its children's centre once they are placed
    stack = [(root, 0, False) for root in reversed(children.get(None, []))]
    while stack:
        cand, d, entered = stack.pop()
        depth[cand.candidate_id] = d
        kids = children.get(cand.candidate_id, [])
        if entered or not kids:
            if kids:
                x[cand.candidate_id] = sum(x[k.candidate_id] for k in kids) / len(kids)
            else:
                x[cand.candidate_id] = cursor
                cursor += 1.0
            continue
        stack.append((cand, d, True))
        stack.extend((k, d + 1, False) for k in reversed(kids))

    if layout is not None:
        for node in layout.nodes:
            if node.id in x:
                x[node.id], depth[node.id] = node.x, node.depth

    accepted = accepted_lineage(candidates, higher_is_better)
    on_path = set(accepted)
    best_id = accepted[-1] if accepted else None
    nodes = []
    for cand in ordered:
        kids = children.get(cand.candidate_id, [])
        if cand.status == "pending":
            fate = "pending"
        elif cand.pruned:
            fate = "pruned"
        elif cand.status in FAILED_STATUSES or cand.val_score is None:
            fate = "failed"
        elif cand.candidate_id == best_id:
            fate = "best"
        elif kids:
            fate = "expanded"
        else:
            fate = "discontinued"
        nodes.append(TreeNode(
            id=cand.candidate_id,
            x=x[cand.candidate_id],
            depth=depth[cand.candidate_id],
            operator=cand.operator,
            status=cand.status,
            fate=fate,
            score=cand.val_score,
            n_children=len(kids),
            summary=cand.summary,
            created_at=cand.created_at,
            finished_at=cand.finished_at,
            parent_id=cand.parent_id if cand.parent_id in by_id else None,
            pruned=cand.pruned,
        ))
    edges = []
    for cand in ordered:
        parent = cand.parent_id if cand.parent_id in by_id else None
        if parent is not None:
            edges.append(TreeEdge(parent, cand.candidate_id, "parent",
                                  on_path=parent in on_path and cand.candidate_id in on_path))
        for source in cand.climber_meta.get("inspiration_ids", []) or []:
            if source != parent and source in by_id:
                edges.append(TreeEdge(source, cand.candidate_id, "ensemble-input"))
    return SearchTree(nodes=tuple(nodes), edges=tuple(edges), accepted=tuple(accepted))


def minutes_since(stamp: str | None, origin: str | None) -> float | None:
    """Minutes from `origin` to `stamp`; None when either is missing/unparseable."""
    try:
        if not stamp or not origin:
            return None
        return max(0.0, (datetime.fromisoformat(stamp) - datetime.fromisoformat(origin)).total_seconds() / 60.0)
    except ValueError:
        return None
