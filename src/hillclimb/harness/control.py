from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from pydantic import BaseModel, Field

from hillclimb.harness.candidate import utcnow
from hillclimb.harness.dirs import PARAMS_FILE, trial_dir
from hillclimb.harness.journal import Journal
from hillclimb.harness.oscompat import replace_file
from hillclimb.harness.status import derive_state

if TYPE_CHECKING:
    from hillclimb.harness.store import DataStore, SearchKey

CONTROL_DIR = "control"
ACTIONS = ("stop", "prune")


class ControlCommand(BaseModel):
    action: str  # one of ACTIONS
    candidate_id: str | None = None  # prune only
    reason: str = ""
    source: str = "cli"  # cli | tui | agent
    # stop only: False aborts in-flight operators now (their candidates are
    # journaled abandoned); True lets them finish and be scored first. A
    # record written before the field existed was a graceful stop.
    graceful: bool = True
    requested_at: str = Field(default_factory=utcnow)


def write_command(search_dir: Path, cmd: ControlCommand) -> Path:
    """Queue a command for the engine. Atomic so the engine never reads a
    half-written file."""
    control_dir = search_dir / CONTROL_DIR
    control_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")
    name = f"{stamp}-{cmd.action}" + (f"-{cmd.candidate_id}" if cmd.candidate_id else "") + ".json"
    tmp = control_dir / (name + ".tmp")
    tmp.write_text(cmd.model_dump_json(indent=2))
    path = control_dir / name
    replace_file(tmp, path)
    return path


def read_commands(search_dir: Path) -> list[tuple[Path, ControlCommand]]:
    """Queued commands oldest-first; unparseable files are dropped."""
    control_dir = search_dir / CONTROL_DIR
    if not control_dir.exists():
        return []
    commands = []
    for path in sorted(control_dir.glob("*.json")):
        try:
            commands.append((path, ControlCommand.model_validate_json(path.read_text())))
        except ValueError:
            path.unlink(missing_ok=True)
    return commands


def drain_commands_dir(search_dir: Path) -> list[ControlCommand]:
    """Queued commands oldest-first, each unlinked before it is returned —
    consumed before applied, so a crash mid-apply never replays a command."""
    commands = []
    for path, cmd in read_commands(search_dir):
        path.unlink(missing_ok=True)
        commands.append(cmd)
    return commands


def clear_stale_stops_dir(search_dir: Path) -> None:
    """At engine startup, any queued stop predates this engine and must not
    instantly kill the fresh search. Queued prunes stay: the first control
    poll applies them."""
    for path, cmd in read_commands(search_dir):
        if cmd.action == "stop":
            path.unlink(missing_ok=True)


clear_stale_stops = clear_stale_stops_dir


def apply_prune(journal: Journal, candidate_id: str, reason: str = "", source: str = "cli") -> list[str]:
    """Mark a candidate and its whole subtree pruned. Returns the candidate
    ids newly pruned (empty if everything was already pruned). Raises
    ValueError for an unknown candidate or the baseline (it backs the
    fallback submission)."""
    if candidate_id not in journal.candidates:
        raise ValueError(f"No candidate {candidate_id!r} in this search")
    root = journal.get(candidate_id)
    if root.operator == "baseline":
        raise ValueError("Cannot prune the baseline candidate; it backs the fallback submission")
    pruned = []
    for candidate in [root, *journal.descendants(candidate_id)]:
        if candidate.pruned:
            continue
        candidate.pruned = True
        candidate.pruned_reason = reason or f"pruned via {candidate_id}"
        candidate.is_selected = False
        journal.candidate_result(candidate)
        pruned.append(candidate.candidate_id)
    if pruned:
        journal.control_event(
            "prune", candidate_id=candidate_id, pruned=pruned, reason=reason, source=source
        )
    return pruned


def resync_best(
    search_dir: Path,
    journal: Journal,
    higher_is_better: bool,
    selection_mode: str,
    output_artifacts: Sequence[str] = ("submission.csv",),
) -> str | None:
    """Repoint best/ at the current selection (used after prune). Falls back
    to the baseline submission when no scored candidate remains. Returns the
    newly selected candidate id, or None on baseline fallback."""
    selected = journal.selected_candidate(higher_is_better, selection_mode)
    best_dir = search_dir / "best"
    # Clear declared outputs first: a newly selected candidate that lacks an
    # artifact must never inherit the previous winner's file.
    for name in output_artifacts:
        (best_dir / name).unlink(missing_ok=True)
    (best_dir / PARAMS_FILE).unlink(missing_ok=True)
    if selected is None:
        baseline = journal.candidates.get("c000")
        if baseline is not None:
            for name in output_artifacts:
                artifact = Path(baseline.candidate_dir) / name
                if artifact.exists():
                    shutil.copy(artifact, best_dir / name)
        (best_dir / "solution.py").unlink(missing_ok=True)
        return None
    src = Path(selected.candidate_dir)
    for name in output_artifacts:
        if (src / name).exists():
            shutil.copy(src / name, best_dir / name)
    if (src / "solution.py").exists():
        shutil.copy(src / "solution.py", best_dir / "solution.py")
    # the shipped parameter values are the best trial's (its trial dir's
    # params.json), never the candidate root's declaration of defaults
    best = selected.best_trial
    if best is not None:
        params = trial_dir(src, best.index) / PARAMS_FILE
        if params.exists():
            shutil.copy(params, best_dir / PARAMS_FILE)
    if not selected.is_selected:
        selected.is_selected = True
        journal.candidate_result(selected)
    return selected.candidate_id


def request_prune(
    store: DataStore,
    key: SearchKey,
    candidate_id: str,
    higher_is_better: bool,
    selection_mode: str,
    reason: str = "",
    source: str = "cli",
) -> str:
    """Single entry point for every surface (CLI, TUI): queue the prune when
    the engine is running, apply it directly otherwise (no engine means no
    second writer). Returns a short human-readable outcome."""
    if derive_state(store.read_status(key)) == "running":
        store.enqueue_command(
            key, ControlCommand(action="prune", candidate_id=candidate_id, reason=reason, source=source)
        )
        return f"queued: engine will prune {candidate_id} before its next operator"
    journal = Journal(store.journal(key))
    pruned = apply_prune(journal, candidate_id, reason, source)
    if not pruned:
        return f"{candidate_id} (and its subtree) was already pruned"
    record = store.search(key)
    output_artifacts = record.meta.output_artifacts if record is not None else ["submission.csv"]
    selected = resync_best(
        store.search_dir(key), journal, higher_is_better, selection_mode, output_artifacts
    )
    outcome = f"pruned {', '.join(pruned)}"
    outcome += f"; best/ now {selected}" if selected else "; best/ reverted to baseline"
    return outcome


def request_stop(
    store: DataStore, key: SearchKey, source: str = "cli", *, graceful: bool = False
) -> str | None:
    """Queue a stop if the engine is running; returns an outcome message, or
    None when there is nothing to stop. By default the engine aborts its
    in-flight operators within about a second (their candidates are journaled
    abandoned; the search stays resumable); `graceful` lets them finish and be
    scored first."""
    if derive_state(store.read_status(key)) != "running":
        return None
    store.enqueue_command(key, ControlCommand(action="stop", source=source, graceful=graceful))
    if graceful:
        return "stop queued: engine parks once the operators in flight finish"
    return "stop queued: engine aborts the operators in flight and parks"
