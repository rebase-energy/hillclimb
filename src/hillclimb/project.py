"""Finding the hillclimb dir, and the machine-scoped directories beside it.

The **hillclimb dir** is any folder holding a `hillclimb.yaml` — config,
problems/ and runs/ live right beside it (`hillclimb init` sets up the
current folder; `hillclimb init DIR` another). Commands look for it in the
folder they run in — never above it: a folder up the tree that happens to
hold one (a hillclimb checkout beside your project, say) is not yours.

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
            f"No {MARKER_FILE} in {start}.\n"
            f"Run hillclimb from the root of a hillclimb dir (the folder holding {MARKER_FILE}), "
            f"run `hillclimb init` to make this folder one (writes ./{MARKER_FILE}), "
            "or set HILLCLIMB_DIR to an existing one."
        )
        self.start = start


# a project whose root already has folders of hillclimb's names keeps
# hillclimb in a subfolder of this name (`hillclimb init` picks it then)
SUBFOLDER = "hillclimb"
# in every folder hillclimb creates: what `hillclimb reset` may delete. A
# folder of the same name without it is the user's, and stays
OWNED_MARKER = ".hillclimb"
_OWNED_NOTE = "Created by hillclimb: `hillclimb reset` deletes this folder.\n"


def find_hillclimb_dir(start: Path | None = None) -> Path | None:
    """`start` (default CWD) when it holds a `hillclimb.yaml`, else its
    `hillclimb/` subfolder when that does (a project that keeps hillclimb
    apart, run from the project's root). No upward search: from anywhere
    else there is no hillclimb dir — a folder up the tree that holds one, or
    a sibling named `hillclimb`, is not this folder's. `HILLCLIMB_DIR` pins
    it outright (engine children run with it set)."""
    pinned = os.environ.get("HILLCLIMB_DIR")
    if pinned:
        return Path(pinned).expanduser().resolve()
    current = (start or Path.cwd()).resolve()
    if (current / MARKER_FILE).is_file():
        return current
    if (current / SUBFOLDER / MARKER_FILE).is_file():
        return current / SUBFOLDER
    return None


def ensure_owned_dir(path: Path) -> Path:
    """Create `path` as a folder of hillclimb's, marked so `reset` may delete
    it. A folder that exists already is left as it is, never claimed."""
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
        (path / OWNED_MARKER).write_text(_OWNED_NOTE)
    return path


def is_owned_dir(path: Path) -> bool:
    """Did hillclimb create this folder (it carries the marker)?"""
    return (Path(path) / OWNED_MARKER).is_file()


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
