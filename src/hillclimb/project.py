"""Workspace discovery and machine-scoped directories.

A hillclimb *workspace* is any directory containing a `hillclimb/` folder
with a `config.yaml` inside — all hillclimb data (config, problems, specs,
runs) lives in that one folder, so it never mingles with the rest of the
repo. Commands find the workspace by upward search from the CWD, like git.

Machine-scoped state (shared runtime venvs, the emflow problem cache, the
agent-concurrency semaphore) lives under XDG-style user directories, shared
by every workspace on the machine.
"""

from __future__ import annotations

import os
from pathlib import Path

MARKER_DIR = "hillclimb"
MARKER_FILE = "config.yaml"


class WorkspaceNotFound(Exception):
    def __init__(self, start: Path):
        super().__init__(
            f"No hillclimb workspace found from {start} upward.\n"
            f"Run `hillclimb init` to create one (makes ./{MARKER_DIR}/{MARKER_FILE}), "
            "or set HILLCLIMB_WORKSPACE to an existing workspace root."
        )
        self.start = start


def marker_path(root: Path) -> Path:
    return root / MARKER_DIR / MARKER_FILE


def find_workspace_root(start: Path | None = None) -> Path | None:
    """Workspace root for `start` (default CWD): the nearest ancestor holding
    `hillclimb/config.yaml`. Running from inside the hillclimb/ folder itself
    also resolves. `HILLCLIMB_WORKSPACE` pins the root and skips the search."""
    pinned = os.environ.get("HILLCLIMB_WORKSPACE")
    if pinned:
        return Path(pinned).expanduser().resolve()
    current = (start or Path.cwd()).resolve()
    for ancestor in (current, *current.parents):
        if marker_path(ancestor).is_file():
            return ancestor
        if ancestor.name == MARKER_DIR and (ancestor / MARKER_FILE).is_file():
            return ancestor.parent
    return None


def require_workspace_root(start: Path | None = None) -> Path:
    root = find_workspace_root(start)
    if root is None:
        raise WorkspaceNotFound((start or Path.cwd()).resolve())
    return root


def machine_cache_dir() -> Path:
    """Machine-scoped cache root (venvs/, emflow-problems/, agent-slots/).
    `HILLCLIMB_CACHE_DIR` overrides (containers); else XDG_CACHE_HOME|~/.cache.
    Same convention on macOS, like uv."""
    override = os.environ.get("HILLCLIMB_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "hillclimb"


def user_config_path() -> Path:
    """User-level defaults, lowest-precedence config file."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "hillclimb" / "config.yaml"
