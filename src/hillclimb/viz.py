from __future__ import annotations

from pathlib import Path

import pydot

from hillclimb.journal import Journal

STATUS_FILL = {
    "ok": "#c8e6c9",        # green: executed and scored
    "buggy": "#ffcdd2",     # red: crashed or violated the contract
    "abandoned": "#e0e0e0", # gray: agent call failed / nothing to evaluate
    "parked": "#ffe0b2",    # orange: interrupted by rate limit
    "pending": "#ffffff",
}
BEST_FILL = "#fff59d"       # gold: the run's final best node
PRUNED_FILL = "#eeeeee"     # light gray + dashed border: user cut this lineage
EDGE_STYLE = {
    "debug": "dashed",
    "improve": "solid",
    "draft": "solid",
    "baseline": "solid",
    "ensemble": "bold",
}


def _label(node, lower_is_better: bool) -> str:
    parts = [f"{node.node_id}  [{node.operator}"]
    if node.complexity:
        parts[0] += f"/{node.complexity}"
    parts[0] += "]"
    if node.val_score is not None:
        line = f"val = {node.val_score:.5g}"
        if node.holdout_score is not None:
            line += f" / hold = {node.holdout_score:.5g}"
        parts.append(line)
    elif node.status != "ok":
        parts.append(node.status)
    summary = (node.summary or "").strip()
    if summary:
        words = summary.split()
        lines, line = [], ""
        for w in words:
            if len(line) + len(w) > 28 and line:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
            if len(lines) >= 3:
                lines[-1] += " …"
                break
        else:
            lines.append(line)
        parts.extend(lines)
    return "\n".join(parts)


def build_tree(journal: Journal, lower_is_better: bool, title: str = "") -> pydot.Dot:
    """Render the journal as an exploration tree.

    Reading the graph: green = scored, red = failed attempt, gray = abandoned,
    gold = final best. A leaf that is neither best nor gold is a pruned line
    of exploration — greedy search moved elsewhere; dashed edges are debug
    repairs, dotted-border nodes were never executed.
    """
    graph = pydot.Dot(
        graph_type="digraph",
        rankdir="TB",
        label=title,
        labelloc="t",
        fontsize=16,
        fontname="Helvetica",
    )
    selected = journal.selected_node(lower_is_better)
    best_id = selected.node_id if selected else None

    for node in journal.nodes.values():
        fill = BEST_FILL if node.node_id == best_id else STATUS_FILL.get(node.status, "#ffffff")
        style = "rounded,filled"
        if node.pruned:
            fill = PRUNED_FILL
            style = "rounded,filled,dashed"
        graph.add_node(
            pydot.Node(
                node.node_id,
                label=_label(node, lower_is_better),
                shape="box",
                style=style,
                fillcolor=fill,
                fontname="Helvetica",
                fontsize=10,
                penwidth=2.5 if node.node_id == best_id else 1,
            )
        )
        if node.parent_id and node.parent_id in journal.nodes:
            graph.add_edge(
                pydot.Edge(
                    node.parent_id,
                    node.node_id,
                    style=EDGE_STYLE.get(node.operator, "solid"),
                    color={"debug": "#c62828", "ensemble": "#6a1b9a"}.get(node.operator, "#455a64"),
                    label=node.operator if node.operator in ("debug", "ensemble") else "",
                    fontsize=8,
                    fontname="Helvetica",
                )
            )

    legend = pydot.Cluster(graph_name="legend", label="legend", fontsize=10, fontname="Helvetica")
    for name, color in (("scored", STATUS_FILL["ok"]), ("failed", STATUS_FILL["buggy"]),
                        ("abandoned", STATUS_FILL["abandoned"]), ("selected", BEST_FILL)):
        legend.add_node(pydot.Node(f"legend_{name}", label=name, shape="box",
                                   style="rounded,filled", fillcolor=color,
                                   fontsize=9, fontname="Helvetica"))
    graph.add_subgraph(legend)
    return graph


def render_tree(journal: Journal, lower_is_better: bool, out_path: Path, title: str = "") -> Path:
    graph = build_tree(journal, lower_is_better, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lstrip(".") or "png"
    graph.write(str(out_path), format=suffix)
    return out_path
