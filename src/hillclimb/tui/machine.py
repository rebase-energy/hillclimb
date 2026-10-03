"""One snapshot of every hillclimb engine on this machine — the data `hillclimb
ps` and `hillclimb top` both draw.

Machine-wide, not tied to the folder a command runs in: each engine's
hillclimb dir comes from its `HILLCLIMB_DIR`, and its search records are read
through THAT dir's store (`Config.load(start=…)`). Process roles come from
`harness.orphans.classify`. Pure: no Textual, no rich, so the plain CLI can
import it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from hillclimb.harness.orphans import Engine, Proc, _environ_dir, classify, is_engine, process_table
from hillclimb.harness.status import live_remaining_s

_RUN_ID_RE = re.compile(r"--run-id\s+(\S+)")
# a hillclimb command doing compute in the user's own terminal: `verify`,
# `grade`, or `run --no-detach` / `resume` in the foreground — the console
# script or `python -m hillclimb.cli`. Engines (`is_engine`) are matched first.
# Anchored on the program itself — `[python] …/hillclimb verify` or `python -m
# hillclimb.cli verify` — so a shell whose command line merely mentions it
# (`sh -c "hillclimb verify … &"`) is not a job.
_JOB_RE = re.compile(
    r"^(?:\S*python[\d.]*\s+(?:-m\s+hillclimb\.cli|\S*/hillclimb)|\S*/hillclimb|hillclimb)"
    r"\s+(verify|grade|run|resume)\b(.*)$"
)

# what each role reads as in the process table, in `role` sort order
PROCESS_ROLES = ("agent", "tool", "mcp", "verifier", "solution", "child")


def parse_etime(text: str) -> int:
    """Seconds from a `ps -o etime` field: `[[dd-]hh:]mm:ss`."""
    days = 0
    if "-" in text:
        day_part, text = text.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            return 0
    seconds = 0
    for part in text.split(":"):
        try:
            seconds = seconds * 60 + int(part)
        except ValueError:
            return 0
    return days * 86400 + seconds


@dataclass
class ProcRow:
    pid: int
    role: str
    depth: int  # below the engine: 0 = its direct child
    cpu: float
    rss_mb: float
    up_s: int
    up: str
    command: str
    guide: str = ""  # tree-branch prefix (├─ └─ │), empty for the engine's children


@dataclass
class SearchInfo:
    ref: str  # <run-id>/<search-id>
    search_dir: Path
    problem: str
    climber: str
    state: str
    budget_left_s: float | None
    best: float | None
    higher_is_better: bool
    candidates: int
    passing: int
    buggy: int
    in_flight: list[str] = field(default_factory=list)  # "c012 improve (agent)"


@dataclass
class EngineRow:
    pid: int
    pgid: int
    hillclimb_dir: Path | None
    orphan: bool
    up_s: int
    up: str
    argv: str  # what follows `hillclimb.cli` on its command line
    procs: list[ProcRow]
    searches: list[SearchInfo]
    cpu: float = 0.0  # the engine and every descendant
    rss_mb: float = 0.0
    own_cpu: float = 0.0  # the engine process alone
    own_rss_mb: float = 0.0
    # `engine` (a search's background process), else the foreground command
    # doing compute in a terminal: `verify`, `grade`, `run`
    kind: str = "engine"

    @property
    def agents(self) -> int:
        return sum(1 for p in self.procs if p.role == "agent")

    @property
    def search(self) -> SearchInfo | None:
        return self.searches[0] if self.searches else None

    def as_engine(self) -> Engine:
        return Engine(pid=self.pid, pgid=self.pgid, hillclimb_dir=self.hillclimb_dir)


@dataclass
class MachineRow:
    cores: int
    load: tuple[float, float, float] | None
    mem_total_mb: float | None
    agents: int  # live coding agent processes across every engine
    agent_slots: int  # the machine cap (concurrency.machine_max_agents)
    cpu: float  # every engine tree summed, % of one core
    rss_mb: float


def process_tree(engine_pid: int, table: dict[int, Proc], root_guides: bool = False) -> list[ProcRow]:
    """The engine's descendants in pre-order (a parent above its children),
    each with its role and depth. `root_guides` draws the engine's own
    children as branches too (`ps` puts the engine above them; `top` lists
    them flush)."""
    children: dict[int, list[Proc]] = {}
    for proc in table.values():
        children.setdefault(proc.ppid, []).append(proc)
    rows: list[ProcRow] = []

    def visit(pid: int, depth: int, parent_role: str | None, prefix: str) -> None:
        kids = sorted(children.get(pid, []), key=lambda p: p.pid)
        for index, proc in enumerate(kids):
            last = index == len(kids) - 1
            role = classify(proc.command, parent_role)
            rows.append(
                ProcRow(
                    pid=proc.pid, role=role, depth=depth, cpu=proc.cpu, rss_mb=proc.rss_mb,
                    up_s=parse_etime(proc.elapsed), up=proc.elapsed, command=proc.command,
                    guide=prefix + ("└─ " if last else "├─ ") if depth or root_guides else "",
                )
            )
            below = prefix + ("   " if last else "│  ") if depth or root_guides else ""
            visit(proc.pid, depth + 1, role, below)

    visit(engine_pid, 0, None, "")
    return rows


def machine_memory_mb() -> float | None:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (ValueError, OSError, AttributeError):
        return None


def machine_load() -> tuple[float, float, float] | None:
    try:
        return os.getloadavg()
    except (OSError, AttributeError):
        return None


class SearchReader:
    """Reads the searches an engine runs through its own hillclimb dir's
    store. Configs and stores are opened once per dir; a dir that cannot be
    read (deleted, broken yaml) yields no searches, never an error."""

    def __init__(self) -> None:
        self._stores: dict[Path, tuple[object, object] | None] = {}

    def config_store(self, hillclimb_dir: Path):
        if hillclimb_dir not in self._stores:
            try:
                from hillclimb.config import Config
                from hillclimb.harness.store import open_store

                config = Config.load(start=hillclimb_dir)
                self._stores[hillclimb_dir] = (config, open_store(config))
            except Exception:
                self._stores[hillclimb_dir] = None
        return self._stores[hillclimb_dir]

    def searches(self, engine_pid: int, hillclimb_dir: Path | None, argv: str) -> list[SearchInfo]:
        if hillclimb_dir is None or not hillclimb_dir.exists():
            return []
        pair = self.config_store(hillclimb_dir)
        if pair is None:
            return []
        _, store = pair
        match = _RUN_ID_RE.search(argv)
        try:
            records = store.searches(run_id=match.group(1)) if match else store.searches()
        except Exception:
            return []
        found = []
        for record in records:
            try:
                status = store.read_status(record.key)
            except Exception:
                continue
            if status is None or status.pid != engine_pid:
                continue
            found.append(search_info(record, status))
        return found


def search_info(record, status) -> SearchInfo:
    """What `top` shows about one search, from its status record alone (no
    journal replay: this runs every second for every engine)."""
    from hillclimb.climber import climber_label

    meta, state = record.meta, record.state
    return SearchInfo(
        ref=record.ref,
        search_dir=record.search_dir,
        problem=meta.problem_id,
        climber=climber_label(meta.climber),
        state=state,
        budget_left_s=live_remaining_s(status, state) if status.budget.total_s else None,
        best=status.best.val_score if status.best else None,
        higher_is_better=bool(meta.higher_is_better),
        candidates=status.candidates.total,
        passing=status.candidates.passing,
        buggy=status.candidates.buggy,
        in_flight=[f"{c.candidate_id} {c.operator} ({c.phase})" for c in status.current],
    )


def job_kind(command: str) -> str | None:
    """What a process is to `ps`/`top`: `engine`, a foreground `verify` /
    `grade` / `run` (`run` also for `resume`), or None — not hillclimb
    compute (`ps`, `watch`, `chart` … read, they do not compute)."""
    if is_engine(command):
        return "engine"
    match = _JOB_RE.search(command)
    if match is None:
        return None
    return "run" if match.group(1) == "resume" else match.group(1)


def job_argv(command: str) -> str:
    """What follows `hillclimb` / `hillclimb.cli` on a job's command line:
    `run heilbronn-11 --run-id …`, `verify heilbronn-11 --solution …`."""
    match = _JOB_RE.search(command)
    if match is not None:
        return (match.group(1) + match.group(2)).strip()
    return command.split("hillclimb.cli", 1)[-1].strip()


def process_cwd(pid: int) -> Path | None:
    """A process's working directory, or None where it cannot be read."""
    import subprocess
    import sys

    try:
        if sys.platform == "linux":
            return Path(os.readlink(f"/proc/{pid}/cwd"))
        if sys.platform == "win32":
            import psutil

            return Path(psutil.Process(pid).cwd())
        out = subprocess.run(
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if line.startswith("n"):
            return Path(line[1:])
    return None


def job_dir(pid: int) -> Path | None:
    """The hillclimb dir a job works in: its pinned `HILLCLIMB_DIR` (every
    engine has one), else the one its working directory sits in — a
    command in the user's terminal found it by the same rule — its cwd or
    that folder's `hillclimb/` subfolder, no upward search."""
    pinned = _environ_dir(pid)
    if pinned is not None:
        return pinned
    from hillclimb.project import MARKER_FILE, SUBFOLDER

    cwd = process_cwd(pid)
    if cwd is None:
        return None
    for folder in (cwd, cwd / SUBFOLDER):
        if (folder / MARKER_FILE).is_file():
            return folder
    return None


def scan(
    table: dict[int, Proc] | None = None,
    reader: SearchReader | None = None,
    env_dirs: dict[int, Path | None] | None = None,
    agent_slots: int | None = None,
    *,
    read_searches: bool = True,
    root_guides: bool = False,
) -> tuple[MachineRow, list[EngineRow]]:
    """One snapshot of the machine: every engine, and every hillclimb command
    computing in a terminal (`verify`, `grade`, a foreground `run`), each
    with its process tree. A job inside another one's tree (a meta-problem's
    `grade` under its engine) is shown there, not twice. `env_dirs` caches
    each job's hillclimb dir across calls (reading it costs a `ps -E` per
    job); pids that are gone are dropped from it. `read_searches=False`
    skips the search records (`ps` is about compute; `top` shows the
    searches)."""
    if table is None:
        table = process_table()
    if env_dirs is None:
        env_dirs = {}
    reader = reader or SearchReader()
    kinds = {proc.pid: kind for proc in table.values() if (kind := job_kind(proc.command))}

    def inside_another_job(proc: Proc) -> bool:
        seen = {proc.pid}
        parent = table.get(proc.ppid)
        while parent is not None and parent.pid not in seen:
            if parent.pid in kinds:
                return True
            seen.add(parent.pid)
            parent = table.get(parent.ppid)
        return False

    engines: list[EngineRow] = []
    for proc in sorted(table.values(), key=lambda p: p.pid):
        kind = kinds.get(proc.pid)
        if kind is None or inside_another_job(proc):
            continue
        if proc.pid not in env_dirs:
            env_dirs[proc.pid] = _environ_dir(proc.pid) if kind == "engine" else job_dir(proc.pid)
        hillclimb_dir = env_dirs[proc.pid]
        argv = job_argv(proc.command) if kind != "engine" else proc.command.split("hillclimb.cli", 1)[-1].strip()
        procs = process_tree(proc.pid, table, root_guides=root_guides)
        engines.append(
            EngineRow(
                pid=proc.pid,
                pgid=proc.pgid,
                hillclimb_dir=hillclimb_dir,
                orphan=hillclimb_dir is not None and not hillclimb_dir.exists(),
                up_s=parse_etime(proc.elapsed),
                up=proc.elapsed,
                argv=argv,
                procs=procs,
                searches=reader.searches(proc.pid, hillclimb_dir, argv) if read_searches and kind == "engine" else [],
                cpu=proc.cpu + sum(p.cpu for p in procs),
                rss_mb=proc.rss_mb + sum(p.rss_mb for p in procs),
                own_cpu=proc.cpu,
                own_rss_mb=proc.rss_mb,
                kind=kind,
            )
        )
    for pid in [pid for pid in env_dirs if pid not in table]:
        del env_dirs[pid]
    if agent_slots is None:
        from hillclimb.config import default_machine_max_agents

        agent_slots = default_machine_max_agents()
    machine = MachineRow(
        cores=os.cpu_count() or 1,
        load=machine_load(),
        mem_total_mb=machine_memory_mb(),
        agents=sum(e.agents for e in engines),
        agent_slots=agent_slots,
        cpu=sum(e.cpu for e in engines),
        rss_mb=sum(e.rss_mb for e in engines),
    )
    return machine, engines


# engine sort keys, in the order `o` cycles them: (label, key, biggest first)
ENGINE_SORTS: list[tuple[str, object, bool]] = [
    ("started", lambda e: e.pid, False),
    ("cpu", lambda e: e.cpu, True),
    ("memory", lambda e: e.rss_mb, True),
    ("uptime", lambda e: e.up_s, True),
    ("budget left", lambda e: (e.search.budget_left_s if e.search and e.search.budget_left_s is not None else -1), False),
    ("candidates", lambda e: e.search.candidates if e.search else -1, True),
    ("problem", lambda e: (e.search.problem if e.search else "~", e.pid), False),
    ("folder", lambda e: (str(e.hillclimb_dir or "~"), e.pid), False),
]

PROC_SORTS: list[tuple[str, object, bool]] = [
    ("tree", None, False),
    ("cpu", lambda p: p.cpu, True),
    ("memory", lambda p: p.rss_mb, True),
    ("uptime", lambda p: p.up_s, True),
    ("role", lambda p: (PROCESS_ROLES.index(p.role), p.pid), False),
]


def sort_rows(rows: list, sorts: list[tuple[str, object, bool]], index: int, reverse: bool) -> list:
    _, key, descending = sorts[index % len(sorts)]
    if key is None:  # the order rows already come in
        return list(reversed(rows)) if reverse else list(rows)
    return sorted(rows, key=key, reverse=descending != reverse)


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    whole = max(0, int(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def format_mem(mb: float) -> str:
    return f"{mb / 1024:.1f}G" if mb >= 1024 else f"{mb:.0f}M"


def short_dir(path: Path | None) -> str:
    if path is None:
        return "?"
    home = Path.home()
    try:
        return "~/" + str(path.relative_to(home))
    except ValueError:
        return str(path)


def machine_line(machine: MachineRow, engines: int) -> str:
    """The header line: the machine at a glance."""
    parts = [f"{engines} engine{'s' if engines != 1 else ''}"]
    slots = f"{machine.agents}/{machine.agent_slots}" if machine.agent_slots else str(machine.agents)
    parts.append(f"coding agents {slots}")
    parts.append(f"cpu {machine.cpu:.0f}% of {machine.cores * 100}%")
    if machine.load is not None:
        parts.append("load " + " ".join(f"{x:.2f}" for x in machine.load))
    mem = format_mem(machine.rss_mb)
    if machine.mem_total_mb:
        mem += f" of {format_mem(machine.mem_total_mb)}"
    parts.append(f"mem {mem}")
    return "   ".join(parts)


_MODEL_RE = re.compile(r"--model\s+(\S+)")
# a candidate's files: the candidate id is what tells two runs apart
_CANDIDATE_PATH_RE = re.compile(
    r"\S*/runs/[^/\s]+/searches/[^/\s]+/candidates/|\S*/hillclimb-verify-[^/\s]+/candidates/"
)
_VENV_PYTHON_RE = re.compile(r"\S*/\.cache/hillclimb/venvs/[^/\s]+/bin/(python\S*)")


def _eval_body(command: str) -> str | None:
    """What a coding agent's Bash tool actually ran: Claude Code wraps every
    call as `zsh -c source <snapshot> && … && eval '<command>' …`. The first
    line of that command, or None when the shape is not recognized."""
    import shlex

    index = command.find(" eval ")
    if index < 0:
        return None
    try:
        tokens = shlex.split(command[index + len(" eval "):])
    except ValueError:
        return None
    if not tokens:
        return None
    body = tokens[0].replace("\\012", "\n").strip()
    first, _, rest = body.partition("\n")
    return essential_commands(first) + (" …" if rest.strip() else "")


# setup a shell line does before the command that matters
_SETUP_WORDS = {"export", "unset", "cd", "set", "setopt", "source", ".", "true", ":", "pushd", "popd"}
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# `2>&1`, `>/dev/null`, `&>/dev/null`, `< /dev/null` — plumbing, not the
# command; a redirection to a real file (`> solution.py`) or a heredoc says
# what the command does and stays
_REDIRECT_RE = re.compile(r"(?:^|\s)(?:\d*>&\d+|(?:\d*>>?\|?|&>>?|<)\s*/dev/null)(?=\s|$)")


def essential_commands(line: str) -> str:
    """The commands of one shell line that do something: `export A=1 B=2;
    unset C; ./problem/verifier.sh 2>&1 | tail -5` → `./problem/verifier.sh |
    tail -5`. Setup statements (export, unset, cd, …) go, so do leading
    `VAR=value` assignments and redirections; pipes stay. A line that is all
    setup is returned as it was."""
    import shlex

    text = _REDIRECT_RE.sub(" ", line)
    try:
        lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return line
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in (";", "&&", "||", "&"):
            segments.append([])
        else:
            segments[-1].append(token)
    kept = []
    for words in segments:
        while words and _ASSIGNMENT_RE.match(words[0]):
            words = words[1:]
        if not words or words[0] in _SETUP_WORDS:
            continue
        # quote only what needs it to read as one word
        kept.append(" ".join(shlex.quote(w) if any(c.isspace() for c in w) else w for w in words))
    return " ; ".join(kept) if kept else line


def short_command(command: str, role: str, hillclimb_dir: Path | None = None) -> str:
    """A command line cut down to what tells it apart: the coding agent and its
    model, the command a tool shell evals, a candidate's files from its id
    (`c003/trials/t0/…`), other paths relative to the hillclimb dir (else
    `~`), a runtime venv's python as plain `python`."""
    if role == "agent":
        words = command.split()
        name = next((Path(w).name for w in words if Path(w).name in ("claude", "codex", "pi")), words[0])
        model = _MODEL_RE.search(command)
        return f"{name} · {model.group(1)}" if model else name
    if role == "tool" and "shell-snapshots" in command:
        body = _eval_body(command)
        # the evaled command gets the same path shortening as any other
        return f"$ {_short_paths(body, hillclimb_dir)}" if body else "$ (shell)"
    return _short_paths(command, hillclimb_dir)


def _short_paths(text: str, hillclimb_dir: Path | None) -> str:
    # `ps` shows a newline inside an argument as a literal \012
    text = _VENV_PYTHON_RE.sub(r"\1", text).replace("\\012", " ↵ ")
    text = _CANDIDATE_PATH_RE.sub("", text)
    if hillclimb_dir is not None:
        text = text.replace(f"{hillclimb_dir}/", "")
    return text.replace(f"{Path.home()}/", "~/")


def engine_target(argv: str) -> str:
    """What an engine works on, from its own argv: `run problems/heilbronn-11
    …` → `heilbronn-11`, a run spec → its stem, `resume <ref>` → the ref."""
    words = argv.split()
    if len(words) < 2 or words[0] not in ("run", "resume", "verify", "grade"):
        return words[0] if words else "?"
    target = words[1]
    if words[0] == "resume" or "://" in target:
        return target
    name = Path(target.rstrip("/")).name
    return Path(name).stem if name.endswith((".yaml", ".yml")) else name
