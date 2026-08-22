"""Finding the hillclimb dir, and the machine-scoped directories beside it.

The **hillclimb dir** is a folder named `hillclimb/` with a `config.yaml`
inside — config, problems, specs and runs all live in that one folder, so
hillclimb data never mingles with the rest of a repo. Commands find it by
upward search from the CWD, like git finding `.git/`.

Machine-scoped state (shared runtime venvs, the emflow problem cache, the
agent-concurrency semaphore) lives under XDG-style user directories, shared
by every hillclimb dir on the machine.
"""

from __future__ import annotations

import os
from pathlib import Path

MARKER_DIR = "hillclimb"
MARKER_FILE = "config.yaml"


class HillclimbDirNotFound(Exception):
    def __init__(self, start: Path):
        super().__init__(
            f"No hillclimb/ dir found from {start} upward.\n"
            f"Run `hillclimb init` to create one (makes ./{MARKER_DIR}/{MARKER_FILE}), "
            "or set HILLCLIMB_DIR to an existing one."
        )
        self.start = start


def find_hillclimb_dir(start: Path | None = None) -> Path | None:
    """The nearest `hillclimb/` dir at or above `start` (default CWD).

    Standing inside the hillclimb/ folder itself also resolves. `HILLCLIMB_DIR`
    pins it and skips the search; `HILLCLIMB_WORKSPACE` is the pre-rename name
    for the folder's *parent* and is still honored.
    """
    pinned = os.environ.get("HILLCLIMB_DIR")
    if pinned:
        return Path(pinned).expanduser().resolve()
    legacy = os.environ.get("HILLCLIMB_WORKSPACE")
    if legacy:
        return Path(legacy).expanduser().resolve() / MARKER_DIR
    current = (start or Path.cwd()).resolve()
    for ancestor in (current, *current.parents):
        if (ancestor / MARKER_DIR / MARKER_FILE).is_file():
            return ancestor / MARKER_DIR
        if ancestor.name == MARKER_DIR and (ancestor / MARKER_FILE).is_file():
            return ancestor
    return None


def require_hillclimb_dir(start: Path | None = None) -> Path:
    found = find_hillclimb_dir(start)
    if found is None:
        raise HillclimbDirNotFound((start or Path.cwd()).resolve())
    return found


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
