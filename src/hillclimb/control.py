from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from hillclimb.journal import Journal
from hillclimb.node import utcnow
from hillclimb.status import effective_state

CONTROL_DIR = "control"
ACTIONS = ("stop", "prune")


class ControlCommand(BaseModel):
    action: str  # one of ACTIONS
    node_id: str | None = None  # prune only
    reason: str = ""
    source: str = "cli"  # cli | tui | agent
    requested_at: str = Field(default_factory=utcnow)


def write_command(run_dir: Path, cmd: ControlCommand) -> Path:
    """Queue a command for the engine. Atomic so the engine never reads a
    half-written file."""
    control_dir = run_dir / CONTROL_DIR
    control_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")
    name = f"{stamp}-{cmd.action}" + (f"-{cmd.node_id}" if cmd.node_id else "") + ".json"
    tmp = control_dir / (name + ".tmp")
    tmp.write_text(cmd.model_dump_json(indent=2))
    path = control_dir / name
    os.replace(tmp, path)
    return path


def read_commands(run_dir: Path) -> list[tuple[Path, ControlCommand]]:
    """Queued commands oldest-first; unparseable files are dropped."""
    control_dir = run_dir / CONTROL_DIR
    if not control_dir.exists():
        return []
    commands = []
    for path in sorted(control_dir.glob("*.json")):
        try:
            commands.append((path, ControlCommand.model_validate_json(path.read_text())))
        except ValueError:
            path.unlink(missing_ok=True)
    return commands


def clear_stale_stops(run_dir: Path) -> None:
    """At engine startup, any queued stop predates this engine and must not
    instantly kill the fresh run. Queued prunes stay: the first control poll
    applies them."""
    for path, cmd in read_commands(run_dir):
        if cmd.action == "stop":
            path.unlink(missing_ok=True)


def apply_prune(journal: Journal, node_id: str, reason: str = "", source: str = "cli") -> list[str]:
    """Mark a node and its whole subtree pruned. Returns the node ids newly
    pruned (empty if everything was already pruned). Raises ValueError for an
    unknown node or the baseline (it backs the fallback submission)."""
    if node_id not in journal.nodes:
        raise ValueError(f"No node {node_id!r} in this run")
    root = journal.get(node_id)
    if root.operator == "baseline":
        raise ValueError("Cannot prune the baseline node; it backs the fallback submission")
    pruned = []
    for node in [root, *journal.descendants(node_id)]:
        if node.pruned:
            continue
        node.pruned = True
        node.pruned_reason = reason or f"pruned via {node_id}"
        node.is_selected = False
        journal.node_result(node)
        pruned.append(node.node_id)
    if pruned:
        journal.control_event("prune", node_id=node_id, pruned=pruned, reason=reason, source=source)
    return pruned


def resync_best(run_dir: Path, journal: Journal, lower_is_better: bool, selection_mode: str) -> str | None:
    """Repoint best/ at the current selection (used after prune). Falls back
    to the baseline submission when no scored node remains. Returns the newly
    selected node id, or None on baseline fallback."""
    selected = journal.selected_node(lower_is_better, selection_mode)
    best_dir = run_dir / "best"
    if selected is None:
        baseline = journal.nodes.get("n000")
        if baseline is not None:
            submission = Path(baseline.workspace) / "submission.csv"
            if submission.exists():
                shutil.copy(submission, best_dir / "submission.csv")
        (best_dir / "solution.py").unlink(missing_ok=True)
        return None
    src = Path(selected.workspace)
    if (src / "submission.csv").exists():
        shutil.copy(src / "submission.csv", best_dir / "submission.csv")
    if (src / "solution.py").exists():
        shutil.copy(src / "solution.py", best_dir / "solution.py")
    if not selected.is_selected:
        selected.is_selected = True
        journal.node_result(selected)
    return selected.node_id


def request_prune(
    run_dir: Path,
    node_id: str,
    lower_is_better: bool,
    selection_mode: str,
    reason: str = "",
    source: str = "cli",
) -> str:
    """Single entry point for every surface (CLI, TUI): queue the prune when
    the engine is running, apply it directly otherwise. Returns a short
    human-readable outcome."""
    if effective_state(run_dir) == "running":
        write_command(run_dir, ControlCommand(action="prune", node_id=node_id, reason=reason, source=source))
        return f"queued: engine will prune {node_id} before its next operator"
    journal = Journal(run_dir / "journal.jsonl")
    pruned = apply_prune(journal, node_id, reason, source)
    if not pruned:
        return f"{node_id} (and its subtree) was already pruned"
    selected = resync_best(run_dir, journal, lower_is_better, selection_mode)
    outcome = f"pruned {', '.join(pruned)}"
    outcome += f"; best/ now {selected}" if selected else "; best/ reverted to baseline"
    return outcome


def request_stop(run_dir: Path, source: str = "cli") -> str | None:
    """Queue a graceful stop if the engine is running; returns an outcome
    message, or None when there is nothing to stop."""
    if effective_state(run_dir) != "running":
        return None
    write_command(run_dir, ControlCommand(action="stop", source=source))
    return "stop queued: engine parks after the current operator finishes"
