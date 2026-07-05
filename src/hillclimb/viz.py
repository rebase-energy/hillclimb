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
BEST_FILL = "#fff59d"       # gold: the search's final best candidate
PRUNED_FILL = "#eeeeee"     # light gray + dashed border: user cut this lineage
EDGE_STYLE = {
    "debug": "dashed",
    "improve": "solid",
    "draft": "solid",
    "baseline": "solid",
    "ensemble": "bold",
}


def _label(candidate, lower_is_better: bool) -> str:
    parts = [f"{candidate.candidate_id}  [{candidate.operator}"]
    if candidate.complexity:
        parts[0] += f"/{candidate.complexity}"
    parts[0] += "]"
    if candidate.val_score is not None:
        line = f"val = {candidate.val_score:.5g}"
        if candidate.holdout_score is not None:
            line += f" / hold = {candidate.holdout_score:.5g}"
        parts.append(line)
    elif candidate.status != "ok":
        parts.append(candidate.status)
    summary = (candidate.summary or "").strip()
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
    repairs, dotted-border candidates were never executed.
    """
    graph = pydot.Dot(
        graph_type="digraph",
        rankdir="TB",
        label=title,
        labelloc="t",
        fontsize=16,
        fontname="Helvetica",
    )
    selected = journal.selected_candidate(lower_is_better)
    best_id = selected.candidate_id if selected else None

    for candidate in journal.candidates.values():
        fill = BEST_FILL if candidate.candidate_id == best_id else STATUS_FILL.get(candidate.status, "#ffffff")
        style = "rounded,filled"
        if candidate.pruned:
            fill = PRUNED_FILL
            style = "rounded,filled,dashed"
        graph.add_node(
            pydot.Node(
                candidate.candidate_id,
                label=_label(candidate, lower_is_better),
                shape="box",
                style=style,
                fillcolor=fill,
                fontname="Helvetica",
                fontsize=10,
                penwidth=2.5 if candidate.candidate_id == best_id else 1,
            )
        )
        if candidate.parent_id and candidate.parent_id in journal.candidates:
            graph.add_edge(
                pydot.Edge(
                    candidate.parent_id,
                    candidate.candidate_id,
                    style=EDGE_STYLE.get(candidate.operator, "solid"),
                    color={"debug": "#c62828", "ensemble": "#6a1b9a"}.get(candidate.operator, "#455a64"),
                    label=candidate.operator if candidate.operator in ("debug", "ensemble") else "",
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
