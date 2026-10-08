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
# the run defaults, in runs/ beside the runs they launch (committed with them)
RUNS_CONFIG = "config.yaml"


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
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            # another engine of the same run made it between the check and
            # the mkdir (--parallel-searches in a fresh dir): it is theirs
            return path
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


INIT_CONFIG = """\
# hillclimb config — this file marks the hillclimb dir (problems/, climbers/ and
# runs/ sit beside it); run hillclimb from this folder. HOW runs climb (climber,
# coding agent, model, budget, concurrency) is in runs/config.yaml, the run
# defaults. Precedence: CLI flags > runs/config.yaml > this file >
# ~/.config/hillclimb/config.yaml > built-in defaults.
__STORE__
# agent_auth: subscription        # how the coding agent logs in (`hillclimb connect` sets it)
# allow_internet_for_agents: true # may the coding agents use the web?

# concurrency:
#   machine_max_agents: 8  # cap across every search on this machine (default min(8, cores-2))

# holdout:
#   top_k: 5             # holdout scored only for top-k-by-val candidates
"""


# The run defaults `init` writes to runs/config.yaml: how the runs launched
# here climb unless a flag or a run spec says otherwise. Committed with the
# runs, so the record shows what they defaulted to. The budget is left unset
# on purpose: a run without one asks before it climbs on the default.
RUNS_CONFIG_TEMPLATE = """\
# hillclimb run defaults — how the runs launched in this folder climb, unless a
# flag (`hillclimb run --budget 30m`) or a run spec says otherwise. Each run
# still records what it used in runs/<run-id>/spec.yaml.

# climber: climbers/greedy/policy.py   # HOW to climb: a climber `hillclimb climber get greedy` fetched from the
#                                      # catalog (greedy | openevolve | gepa — the whole climber as Python you can
#                                      # edit), or your own .py. A run spec's own `climber:` replaces it
# climber:                 # ...or the whole block (`hillclimb climber show climbers/greedy` prints one to edit)
#   selector_policy: climbers/greedy/policy.py:Best    # which candidate to build on next: a file:Class or package.module:Class
#   operator_policy: climbers/greedy/policy.py:Greedy  # which operator to use on it
#   selector_params: {num_drafts: 3}   # overrides of the classes' DEFAULTS (`params:` for the operator policy's)
#   operators: [draft, debug, improve, ensemble]
#   tuner: random          # random | optuna (parameter tuning of candidates that declare params.json)
#   memory: files          # files | none

model: sonnet
# agent: claude-code       # the coding agent (a personal default can live in ~/.config/hillclimb/config.yaml)
# routing:                 # another agent or model for one operator
#   draft: {model: opus}

# budget:
#   total_s: 1800          # the wall clock of a run (unset: `hillclimb run` asks first)
#   deadline: graceful     # `hard` aborts in-flight operators when total_s runs out
#   max_evaluations: 0     # verifier trials the climber may spend (0 = unlimited)
#   max_cost_usd: 0        # coding-agent spend that parks the search (0 = no ceiling)

# concurrency:              # see docs.hillclimb.sh/parallelism
#   parallel_agents: 1     # coding agents (attempts in flight) per search
#   parallel_replicates: 0 # a trial's seeded runs at once: 0 = all; 1 when the metric measures the machine (time!)
#   solution_cpus: 1       # cores each run of a solution may use ($HILLCLIMB_CPUS)

# learning:
#   enabled: true          # knowledge cards in knowledge/ inform new searches

# similarity:              # `hillclimb similarity scores`: name (or my_score.py) -> params
#   scores:
#     solution-card: {card_model: anthropic/claude-haiku-4.5, embedding_model: voyageai/voyage-4}
#     api-calls: {}
"""


# What `hillclimb init` adds to the hillclimb dir's .gitignore. The RECORD
# of every run is committed — run.yaml, spec.yaml, each search's search.yaml,
# journal, status, knowledge card, climber snapshot, and the best solution —
# so `git log` explains every run and `hillclimb chart` works on a fresh
# clone. The BULK is not: candidates (coding agent streams, replicate outputs,
# runtime data), engine logs, the control queue, the rest of best/ (a
# submission can be large), the sqlite store and the derived knowledge graph.
# Keys never are. Leading slashes anchor each rule at the hillclimb dir.
INIT_GITIGNORE = (
    "# hillclimb: the record of every run is committed, its bulk is not",
    "/.env",
    "/runs/*/logs/",
    "/runs/*/searches/*/candidates/",
    "/runs/*/searches/*/control/",
    "/runs/*/searches/*/best/*",
    "!/runs/*/searches/*/best/solution.py",
    "!/runs/*/searches/*/best/params.json",
    "/store.sqlite*",
    "/knowledge/graph.json",
)

# The folders `init` creates beside hillclimb.yaml.
SCAFFOLD_DIRS = ("problems", "runs", "climbers")
# what hillclimb writes into beside hillclimb.yaml: a folder of one of these
# names that is not hillclimb's means hillclimb goes in a subfolder instead
OWNED_DIR_NAMES = ("problems", "runs", "knowledge", "climbers")


def scaffold_blockers(folder: Path) -> list[Path]:
    """What stops `folder` from becoming a hillclimb dir: a folder of one of
    hillclimb's names (problems/, runs/, knowledge/, climbers/) that is
    already there and not hillclimb's (a project's own). Empty when the
    folder is free or already a hillclimb dir."""
    if (folder / MARKER_FILE).exists():
        return []
    return [folder / sub for sub in OWNED_DIR_NAMES if (folder / sub).exists() and not is_owned_dir(folder / sub)]


def scaffold_target(folder: Path) -> tuple[Path, list[Path]]:
    """Where a hillclimb dir for `folder` goes: the folder itself, or its
    `hillclimb/` subfolder when the folder already has folders of hillclimb's
    names of its own (returned as the second item, for the message)."""
    blockers = scaffold_blockers(folder)
    return (folder / SUBFOLDER, blockers) if blockers else (folder, [])


DATASTORES = ("files", "sqlite")

# the store block INIT_CONFIG carries: commented out for the default, live
# when `hillclimb init --datastore sqlite` chose the database
_STORE_BLOCK = {
    "files": (
        "\n# store:                   # where the records of runs live (candidates/ and best/ stay in runs/):\n"
        "#   backend: files         # files: yaml/jsonl under runs/ (default) | sqlite: one store.sqlite\n"
    ),
    "sqlite": (
        "\nstore:                     # where the records of runs live (candidates/ and best/ stay in runs/):\n"
        "  backend: sqlite          # one store.sqlite beside this file | files: yaml/jsonl under runs/\n"
    ),
}


def init_config(datastore: str = "files") -> str:
    """The hillclimb.yaml `init` writes, with the record store it was asked for."""
    return INIT_CONFIG.replace("__STORE__", _STORE_BLOCK[datastore])


def scaffold_hillclimb_dir(folder: Path, datastore: str = "files") -> Path:
    """Make `folder` (created if missing) a hillclimb dir: hillclimb.yaml,
    empty problems/ and climbers/ beside it, runs/ holding the run defaults
    (runs/config.yaml), and the gitignore rules that keep
    run artifacts and keys out of git while the record of every run goes in
    (`INIT_GITIGNORE`). No problem is added: picking one (`hillclimb problem
    get`) is the user's first real choice. Idempotent on the folder layout;
    never overwrites an existing config, only adds ignore rules that are
    missing. Callers check `scaffold_blockers` first."""
    if folder.name == SUBFOLDER and not folder.exists():
        ensure_owned_dir(folder)  # a hillclimb/ subfolder of hillclimb's: `reset` removes it whole
    folder.mkdir(parents=True, exist_ok=True)
    for sub in SCAFFOLD_DIRS:
        ensure_owned_dir(folder / sub)  # marked: `reset` may delete it
        (folder / sub / ".gitkeep").touch()
    if not (folder / MARKER_FILE).exists():
        (folder / MARKER_FILE).write_text(init_config(datastore))
    if not (folder / "runs" / RUNS_CONFIG).exists():
        (folder / "runs" / RUNS_CONFIG).write_text(RUNS_CONFIG_TEMPLATE)
    gitignore = folder / ".gitignore"
    existing_ignore = gitignore.read_text() if gitignore.exists() else ""
    present = existing_ignore.splitlines()
    missing = [line for line in INIT_GITIGNORE if line not in present]
    if missing == [INIT_GITIGNORE[0]]:  # every rule is there, only the heading is not
        missing = []
    if missing:
        gitignore.write_text(
            existing_ignore.rstrip("\n")
            + ("\n" if existing_ignore else "")
            + "\n".join(missing)
            + "\n"
        )
    return folder
