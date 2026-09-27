"""What every command shares: config loading, search lookup, budget parsing, the
hillclimb-dir scaffold and the CLI's voice. Command modules call these as
`common.<name>()` so a test that patches `hillclimb.cli.common` reaches them all."""

from __future__ import annotations

import re
from pathlib import Path

import typer

from hillclimb.config import Config
from hillclimb.harness.store import DataStore, SearchRecord, open_store, resolve_search

_CONSOLE = None
_ERR_CONSOLE = None

# The CLI's voice. Every sentence hillclimb speaks goes through `say` (or
# `warn`/`fail` for the two stderr registers) with rich markup on the
# theme's styles, so a reader tells hillclimb's words from a verifier's
# stdout or an agent's log at a glance:
#
#   [head]   the lead of a line — the verdict, the thing that happened
#            ("Deleted", "Search r1/alpha", "No hillclimb dir here.")
#   [path]   paths, search refs, candidate ids, file names, env var names
#   [cmd]    a command the reader could type
#   [note]   the explanation beside it, in dim ink
#   [ok] [warn] [bad]   verdicts: green, yellow, bold red
#
# Every dynamic value is wrapped in `_m()` so a `[` in a path or an id is
# not read as markup. Data the reader pipes elsewhere — `--json` output,
# diffs, file bodies, a candidate's stdout tail — is NOT the voice: it
# stays on `typer.echo`, byte-exact, with no styling.
THEME = {
    "cmd": "bold cyan", "path": "cyan", "note": "dim", "head": "bold",
    "ok": "green", "warn": "yellow", "bad": "bold red",
}


def _make_console(stderr: bool, width: int | None = None):
    from rich.console import Console
    from rich.theme import Theme

    # emoji off: a `:tag:` inside a path or an agent's summary must print
    # as written. No color when the stream is not a terminal (a pipe, a
    # test), exactly like the banner.
    return Console(stderr=stderr, highlight=False, emoji=False, theme=Theme(THEME), width=width)


def _console():
    """One rich console for the CLI's own messages, on the theme's palette:
    commands bold cyan, paths cyan, explanations dim."""
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = _make_console(stderr=False)
    return _CONSOLE


def _err_console():
    """The same voice on stderr, for `warn` and `fail`."""
    global _ERR_CONSOLE
    if _ERR_CONSOLE is None:
        _ERR_CONSOLE = _make_console(stderr=True)
    return _ERR_CONSOLE


def say(text: str = "", *, err: bool = False) -> None:
    """Print with rich markup: [cmd]hillclimb init[/], [path]…[/], [note]…[/], [head]…[/].
    `err=True` speaks on stderr."""
    (_err_console() if err else _console()).print(text, soft_wrap=True)


def warn(text: str) -> None:
    """A warning on stderr: the whole line in the warn colour, markup inside honoured."""
    say(f"[warn]{text}[/]", err=True)


def fail(text: str) -> None:
    """An error on stderr: the whole line in the bad colour, markup inside honoured.
    The caller raises `typer.Exit(1)` itself, so a command can say several things first."""
    say(f"[bad]{text}[/]", err=True)


def _m(text) -> str:
    """Escape a value (a path, an id) for rich markup."""
    from rich.markup import escape

    return escape(str(text))


_CLOCK = re.compile(r"^(\s*)(\[[^\]]*\bleft\])(.*)$")
_PREFIX = re.compile(r"^(\s*)([a-z][a-z ]{0,23}[a-z]):(\s.*|$)")
_CANDIDATE = re.compile(r"\b(c\d{3,})\b")
_TROUBLE = re.compile(r"\b(failed|crashed|refused|abandoned|TIMEOUT|buggy|out of credits)\b")


def engine_line(line: str) -> str:
    """An engine log line in the CLI's voice, as markup. The engine speaks
    in a few stable shapes and this marks them without parsing anything
    else: a leading `[N minutes left]` clock goes dim, a `word: rest` lead
    (`learning:`, `new selection:`, `baseline written:`) goes bold,
    candidate ids go in the path colour, and a line reporting trouble
    (`failed`, `crashed`, `refused`, `TIMEOUT`) is warned as a whole. The
    text itself is escaped first, so a `[` in an agent's summary is never
    read as markup."""
    text = _m(line)
    if _TROUBLE.search(line):
        return f"[warn]{text}[/]"
    lead = ""
    if m := _CLOCK.match(text):
        lead, text = f"{m.group(1)}[note]{m.group(2)}[/]", m.group(3)
    elif m := _PREFIX.match(text):
        lead, text = f"{m.group(1)}[head]{m.group(2)}:[/]", m.group(3)
    text = _CANDIDATE.sub(r"[path]\1[/]", text)
    return lead + text


CLOCK_GUTTER = 13  # `1:05:00 left` fits; the message column starts after it


def split_engine_line(line: str) -> tuple[str, str]:
    """`(clock, message)` for the log's two columns: the `m:ss left` text of
    a leading `[… left]` clock (empty for a line without one) and the rest
    as [`engine_line`] markup. A line under a clock keeps its own indent,
    so `  new selection:` still nests under the operator that made it."""
    m = _CLOCK.match(line)
    if not m:
        return "", engine_line(line)
    clock = m.group(2)[1:-1]  # without the brackets: the gutter is the frame
    return clock, engine_line(m.group(1) + m.group(3).lstrip(" ") if m.group(3) else "")


def engine_log(line: str) -> None:
    """The `log=` callback for a foreground engine. On a terminal each line
    is two columns — the clock, right-aligned and dim in a gutter of
    `CLOCK_GUTTER` cells, then the message — so a message that wraps folds
    under itself and the clocks stand in one column down the log. Off a
    terminal (a pipe, a test) the line prints whole, marked up."""
    console = _console()
    if not console.is_terminal:
        say(engine_line(line))
        return
    from rich.table import Table

    clock, message = split_engine_line(line)
    grid = Table.grid(padding=(0, 2, 0, 0))
    grid.add_column(width=CLOCK_GUTTER, justify="right", no_wrap=True, style="note")
    grid.add_column(ratio=1, overflow="fold")  # a long path folds, never an ellipsis
    grid.add_row(clock, message)
    console.print(grid)


def table(columns, rows, *, err: bool = False) -> None:
    """A listing in the CLI's voice: bold headings over a single rule, no
    borders, cells in whatever markup the caller gives (already escaped
    with `_m`). `columns` are `(heading, style | None)` pairs — the style
    colours a whole column, e.g. `("problem", "path")`; a heading may be
    `""` for a marker column. One helper so `connect`, `problem list` and
    every later listing look like one tool."""
    from rich import box
    from rich.table import Table

    grid = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="head", show_edge=False)
    for heading, style in columns:
        # a cell wraps inside its own column, and a long word (a path, an
        # id) folds rather than being cut with an ellipsis
        grid.add_column(heading, style=style, overflow="fold")
    for row in rows:
        grid.add_row(*row)
    console = _err_console() if err else _console()
    # A terminal has a width to fit, so long cells wrap there. A pipe (a
    # test, `| grep`) does not: rich would still squeeze the table into 80
    # columns and crop cells with an ellipsis, so there the table is printed
    # through a console as wide as its natural width and every cell prints
    # whole (`Console.print(width=…)` cannot widen, only narrow).
    if not console.is_terminal:
        from rich.measure import Measurement

        natural = Measurement.get(console, console.options.update(max_width=10_000), grid).maximum
        if natural > console.width:
            console = _make_console(err, width=natural)
    console.print(grid)


def legend(rows, indent: int = 2) -> None:
    """Aligned `key — note` rows: the key in the path color, the note dim."""
    width = max(len(key) for key, _ in rows)
    for key, note in rows:
        say(f"{' ' * indent}[path]{_m(key):<{width}}[/]  [note]— {_m(note)}[/]")


def next_steps(rows) -> None:
    """`Next:` then one command per line with a dim note beside it."""
    width = max(len(cmd) for cmd, _ in rows)
    for index, (cmd, note) in enumerate(rows):
        lead = "[head]Next:[/]" if index == 0 else "     "
        say(f"{lead} [cmd]{_m(cmd):<{width}}[/]  [note]{_m(note)}[/]")


def load_config(*, raise_not_found: bool = False, **overrides) -> Config:
    """Config.load with the no-hillclimb-dir hint rendered for the CLI
    (or re-raised, for commands that have a fallback)."""
    from hillclimb.project import HillclimbDirNotFound

    try:
        return Config.load(**overrides)
    except HillclimbDirNotFound as exc:
        if raise_not_found:
            raise
        say_no_hillclimb_dir(exc)
        raise typer.Exit(1) from exc


def say_no_hillclimb_dir(exc) -> None:
    """The no-hillclimb-dir hint in the CLI's voice, on stderr: the same
    words as the exception's, with the path, the command and the env var
    marked up."""
    from hillclimb.project import MARKER_DIR, MARKER_FILE

    say(f"[head]No hillclimb/ dir found[/] from [path]{_m(exc.start)}[/] upward.", err=True)
    say(
        f"Run [cmd]hillclimb init[/] to create one [note](makes ./{MARKER_DIR}/{MARKER_FILE})[/], "
        "or set [path]HILLCLIMB_DIR[/] to an existing one.",
        err=True,
    )


INIT_CONFIG = """\
# hillclimb config — this file marks the hillclimb dir; commands work from
# any subdirectory below it. Precedence: CLI flags > this file >
# ~/.config/hillclimb/config.yaml > built-in defaults.

model: sonnet
# agent: claude-code

# climber: greedy          # HOW to climb: greedy | openevolve | gepa | hillclimb/climbers/<name>
# climber:                 # ...or with your overrides on the climber's own params
#   ref: greedy
#   params: {num_drafts: 3}
#   tuner: random          # random | optuna (parameter tuning of candidates that declare params.json)

# budget:
#   total_s: 7200
#   deadline: graceful     # `hard` aborts in-flight operators when total_s runs out
#   max_evaluations: 0     # verifier trials the climber may spend (0 = unlimited)

# evaluation:
#   n_replicates: 1        # seeded runs per trial (median is the trial's score)
#   replicate_mode: parallel # `serial` when the metric measures the machine (time!)
#   noise_k: 0             # require gains > k x the measured noise floor
#   min_improvement: 0     # ...or an absolute floor, in metric units

# concurrency:
#   parallel_agents: 1   # >1 runs concurrent agents
#   machine_max_agents: 8  # cap across every search on this machine (default min(8, cores-2))

# holdout:
#   enabled: true
#   top_k: 5             # holdout scored only for top-k-by-val candidates

# similarity:            # `hillclimb similarity scores`: name (or my_score.py) -> params
#   scores:
#     solution-card: {card_model: anthropic/claude-haiku-4.5, embedding_model: voyageai/voyage-4}
#     api-calls: {}

# learning:
#   enabled: true        # knowledge cards in hillclimb/knowledge/ inform new searches
#   max_cards: 3
#   complexity_prior: false
#   live: true           # concurrent searches in one run share discoveries mid-flight

# report:
#   enabled: true        # inject eval breakdowns (per-zone/horizon/quantile) into improve prompts
"""


INIT_PROBLEM_YAML = """\
problem_id: example
metric: score
higher_is_better: true
description: description.md
time_budget_s: 900
# verifier: verifier.sh   # the default; a problem IS its verifier
# holdout: true           # engine also runs `verifier.sh --holdout`
# unit_tests:             # optional frozen correctness gate, run once per trial
#   root: tests
#   command: ["{python}", "-m", "pytest", "-q", "{tests}"]
# baseline: baseline.py   # scored at t=0 as the floor to beat (or a number, e.g. 0.5)
# requirements: requirements.txt
# interface: interface.py  # optional machine-checked I/O declaration (hillclimb spaces)
"""


INIT_PROBLEM_DESCRIPTION = """\
# Example problem

Replace this with what the solution has to do, what data it gets, and how it
is judged. The agent reads this file verbatim.

The toy objective below: write `solution.py` that prints a number. Bigger wins.
"""


INIT_PROBLEM_VERIFIER = """\
#!/usr/bin/env bash
# A problem is defined by this file. hillclimb runs it in the candidate's
# working directory (./solution.py, ./problem/ and ./data/ are present) and
# reads one thing back: the score.
#
#   exit 0                -> the candidate is valid
#   $HILLCLIMB_RESULT     -> where the score goes: a bare number, or
#                            {"score": <float>, "report": {...}}
#
# Also available: $HILLCLIMB_PYTHON (the managed venv interpreter — use it
# instead of bare `python`), $HILLCLIMB_SOLUTION, $HILLCLIMB_SPLIT,
# $HILLCLIMB_REPLICATE_SEED. `--holdout` is passed when scoring the hidden split.
set -euo pipefail

"$HILLCLIMB_PYTHON" "$HILLCLIMB_SOLUTION" > solution_out.txt

# Score whatever the solution produced. Do the real checking here: a verifier
# that cannot fail is a verifier the search will learn to cheat.
tail -n 1 solution_out.txt > "$HILLCLIMB_RESULT"
"""


# What `hillclimb init` adds to the folder's .gitignore. The RECORD of every
# run is committed — run.yaml, spec.yaml, each search's search.yaml, journal,
# status, knowledge card, climber snapshot, and the best solution — so `git
# log` explains every run and `hillclimb chart` works on a fresh clone. The
# BULK is not: candidates (agent streams, replicate outputs, runtime data),
# engine logs, the control queue, the rest of best/ (a submission can be
# large), the sqlite store and the derived knowledge graph. Keys never are.
INIT_GITIGNORE = (
    "# hillclimb: the record of every run is committed, its bulk is not",
    "hillclimb/.env",
    "hillclimb/runs/*/logs/",
    "hillclimb/runs/*/searches/*/candidates/",
    "hillclimb/runs/*/searches/*/control/",
    "hillclimb/runs/*/searches/*/best/*",
    "!hillclimb/runs/*/searches/*/best/solution.py",
    "!hillclimb/runs/*/searches/*/best/params.json",
    "hillclimb/store.sqlite*",
    "hillclimb/knowledge/graph.json",
)


def scaffold_hillclimb_dir(root: Path, *, example: bool = True) -> Path:
    """Create `<root>/hillclimb/` with config, the example problem, and the
    gitignore rules that keep run artifacts and keys out of git while the
    record of every run goes in (`INIT_GITIGNORE`). `example=False` (what
    `problem get` does when it has to create the dir) leaves out the example
    problem: the user asked for one bundled problem, not a scaffold.
    Idempotent on the folder layout; never overwrites an existing config,
    only adds ignore rules that are missing."""
    from hillclimb.project import MARKER_DIR, MARKER_FILE

    folder = root / MARKER_DIR
    for sub in ("problems", "runs"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
        (folder / sub / ".gitkeep").touch()
    if not (folder / MARKER_FILE).exists():
        (folder / MARKER_FILE).write_text(INIT_CONFIG)
    if example:
        example_dir = folder / "problems" / "example"
        example_dir.mkdir(parents=True, exist_ok=True)
        (example_dir / "problem.yaml").write_text(INIT_PROBLEM_YAML)
        (example_dir / "description.md").write_text(INIT_PROBLEM_DESCRIPTION)
        (example_dir / "verifier.sh").write_text(INIT_PROBLEM_VERIFIER)
        (example_dir / "verifier.sh").chmod(0o755)
    gitignore = root / ".gitignore"
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


def parse_budget(value: str) -> int:
    match = re.fullmatch(r"(\d+)\s*([hms]?)", value.strip())
    if not match:
        raise typer.BadParameter(f"Cannot parse budget {value!r} (use e.g. 2h, 30m, 3600s)")
    amount, unit = int(match.group(1)), match.group(2)
    return amount * {"h": 3600, "m": 60, "s": 1, "": 1}[unit]


def open_search(config: Config, ref: str | None) -> tuple[DataStore, SearchRecord]:
    """The configured store and the search `ref` names in it, with the
    lookup message as a usage error."""
    store = open_store(config)
    try:
        return store, resolve_search(store, ref)
    except LookupError as exc:
        raise typer.BadParameter(str(exc)) from exc


def resolve_search_dir(config: Config, ref: str | None) -> Path:
    return open_search(config, ref)[1].search_dir


def _spec_provenance(config: Config, suite_path: Path) -> str:
    """Spec path recorded in run.yaml, relative to the folder holding the
    hillclimb dir (absolute if the spec lives outside it)."""
    if config.hillclimb_dir is not None:
        try:
            return str(suite_path.relative_to(config.hillclimb_dir.parent))
        except ValueError:
            pass
    return str(suite_path)


def _parse_arm_set(pairs: list[str]) -> dict[str, list[str]]:
    """`ARM:KEY=VALUE` strings (the `--arm-set` flag) → arm name -> its
    `--set` pairs, validated the way `--set` is."""
    out: dict[str, list[str]] = {}
    for item in pairs:
        arm, sep, pair = item.partition(":")
        if not sep or not arm.strip() or "=" not in pair:
            raise typer.BadParameter(f"--arm-set expects ARM:KEY=VALUE, got {item!r}")
        _parse_set([pair])
        out.setdefault(arm.strip(), []).append(pair)
    return out


def _parse_set(pairs: list[str]) -> dict:
    from hillclimb.config import parse_set_overrides

    try:
        return parse_set_overrides(pairs)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
