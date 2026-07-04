from __future__ import annotations

import json
from pathlib import Path

from hillclimb.node import Node


def _ranks(values: list[float], lower_is_better: bool) -> list[float]:
    """Average-tie ranks, 1 = best."""
    order = sorted(values, reverse=not lower_is_better)
    return [
        (order.index(v) + 1 + len(order) - 1 - order[::-1].index(v) + 1) / 2 for v in values
    ]


class Journal:
    """Append-only JSONL journal of search nodes.

    Events: `node_created` (node enters the tree, status=pending) and
    `node_result` (terminal state for the node). The in-memory view is the
    replay of all events; the file is never rewritten, which is what makes
    `resume` and `status` safe against crashes mid-run.
    """

    def __init__(self, path: Path):
        self.path = path
        self.nodes: dict[str, Node] = {}
        if path.exists():
            self._replay()

    def _replay(self) -> None:
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record.pop("event", None)
            node = Node.model_validate(record)
            self.nodes[node.node_id] = node

    def _append(self, event: str, node: Node) -> None:
        record = {"event": event, **node.model_dump()}
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self.nodes[node.node_id] = node.model_copy(deep=True)

    def node_created(self, node: Node) -> None:
        self._append("node_created", node)

    def node_result(self, node: Node) -> None:
        self._append("node_result", node)

    # --- queries ---

    def get(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def next_node_id(self) -> str:
        return f"n{len(self.nodes):03d}"

    def children(self, node_id: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.parent_id == node_id]

    def drafts(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.operator == "draft"]

    def scored_nodes(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.is_scored]

    def best_node(self, lower_is_better: bool) -> Node | None:
        scored = self.scored_nodes()
        if not scored:
            return None
        return min(scored, key=lambda n: n.val_score if lower_is_better else -n.val_score)

    def selected_node(self, lower_is_better: bool, mode: str = "rank-blend") -> Node | None:
        """Node whose submission ships. Falls back to val_score when no node
        has a holdout score (holdout disabled / legacy runs).

        rank-blend (default): min(val_rank + holdout_rank) — robust when either
        signal is unreliable: an overfit val score is vetoed by its holdout
        rank, a noisy holdout outlier is vetoed by its val rank. `holdout` and
        `val` select by a single signal.
        """
        with_holdout = [n for n in self.scored_nodes() if n.holdout_score is not None]
        if not with_holdout or mode == "val":
            return self.best_node(lower_is_better)
        if mode == "holdout":
            return min(
                with_holdout,
                key=lambda n: n.holdout_score if lower_is_better else -n.holdout_score,
            )
        val_rank = _ranks([n.val_score for n in with_holdout], lower_is_better)
        hold_rank = _ranks([n.holdout_score for n in with_holdout], lower_is_better)
        direction = 1 if lower_is_better else -1
        return min(
            zip(with_holdout, val_rank, hold_rank),
            key=lambda t: (t[1] + t[2], direction * t[0].holdout_score, direction * t[0].val_score),
        )[0]

    def pending_nodes(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.status == "pending"]

    def debug_chain(self, node_id: str) -> list[Node]:
        """The failed node being repaired plus every debug attempt so far,
        oldest first (the context a DEBUG operator needs)."""
        chain = [self.get(node_id)]
        while chain[0].operator == "debug" and chain[0].parent_id:
            chain.insert(0, self.get(chain[0].parent_id))
        return chain

    def siblings(self, node_id: str) -> list[Node]:
        node = self.get(node_id)
        return [
            n
            for n in self.nodes.values()
            if n.parent_id == node.parent_id and n.node_id != node_id
        ]
