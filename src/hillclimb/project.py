"""Finding the hillclimb dir, and the machine-scoped directories beside it.

The **hillclimb dir** is any folder holding a `hillclimb.yaml` — config,
problems/ and runs/ live right beside it (`hillclimb init` sets up the
current folder; `hillclimb init DIR` another). Commands find it by upward
search from the CWD, like git finding `.git/`.

Machine-scoped state (shared runtime venvs, the emflow problem cache, the
coding-agent-concurrency semaphore) lives under XDG-style user directories, shared
by every hillclimb dir on the machine.
"""

from __future__ import annotations

import os
from pathlib import Path

MARKER_FILE = "hillclimb.yaml"


class HillclimbDirNotFound(Exception):
    def __init__(self, start: Path):
        super().__init__(
            f"No {MARKER_FILE} found from {start} upward.\n"
            f"Run `hillclimb init` to make this folder a hillclimb dir (writes ./{MARKER_FILE}), "
            "or set HILLCLIMB_DIR to an existing one."
        )
        self.start = start


def find_hillclimb_dir(start: Path | None = None) -> Path | None:
    """The nearest folder at or above `start` (default CWD) holding a
    `hillclimb.yaml`. `HILLCLIMB_DIR` pins it and skips the search."""
    pinned = os.environ.get("HILLCLIMB_DIR")
    if pinned:
        return Path(pinned).expanduser().resolve()
    current = (start or Path.cwd()).resolve()
    for ancestor in (current, *current.parents):
        if (ancestor / MARKER_FILE).is_file():
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


def user_env_path() -> Path:
    """The user-level `.env` beside the user config: provider keys that
    apply to every folder, read under a folder's own `.env`."""
    return user_config_path().with_name(".env")
