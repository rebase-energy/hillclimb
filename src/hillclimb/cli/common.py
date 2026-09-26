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


def _console():
    """One rich console for the CLI's own messages, on the theme's palette:
    commands bold cyan, paths cyan, explanations dim. No color when stdout
    is not a terminal (a pipe, a test), exactly like the banner."""
    global _CONSOLE
    if _CONSOLE is None:
        from rich.console import Console
        from rich.theme import Theme

        _CONSOLE = Console(highlight=False, theme=Theme({
            "cmd": "bold cyan", "path": "cyan", "note": "dim", "head": "bold",
            "ok": "green", "warn": "yellow", "bad": "bold red",
        }))
    return _CONSOLE


def say(text: str = "") -> None:
    """Print with rich markup: [cmd]hillclimb init[/], [path]…[/], [note]…[/], [head]…[/]."""
    _console().print(text, soft_wrap=True)


def _m(text) -> str:
    """Escape a value (a path, an id) for rich markup."""
    from rich.markup import escape

    return escape(str(text))


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
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


INIT_CONFIG = """\
# hillclimb config — this file marks the hillclimb dir; commands work from
# any subdirectory below it. Precedence: CLI flags > this file >
# ~/.config/hillclimb/config.yaml > built-in defaults.

model: sonnet
# backend: claude-code

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
#   parallel_operators: 1   # >1 runs concurrent operators
#   machine_max_operators: 8  # cap across every search on this machine (default min(8, cores-2))

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


INIT_SPEC_EXAMPLE = """\
# Example run spec — committed run parameters (`hillclimb run hillclimb/specs/example.yaml`).
# Single search:
#   target: emflow://gefcom2014:solar
#   model: opus
#   budget: 2h
# Or several, with per-search parameters:
# problems:
#   - target: emflow://gefcom2014:solar
#     model: opus
#     budget: 2h
#   - target: emflow://gefcom2014:wind
#     budget: 1h
"""


def scaffold_hillclimb_dir(root: Path, *, example: bool = True) -> Path:
    """Create `<root>/hillclimb/` with config, the example problem, an
    example spec, and gitignore entries for runs/ and the .env that carries
    provider keys. `example=False` (what `problem get` does when it has to
    create the dir) leaves out the example problem and spec: the user asked
    for one bundled problem, not a scaffold. Idempotent on the folder
    layout; never overwrites an existing config."""
    from hillclimb.project import MARKER_DIR, MARKER_FILE

    folder = root / MARKER_DIR
    for sub in ("problems", "specs", "runs"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
        (folder / sub / ".gitkeep").touch()
    if not (folder / MARKER_FILE).exists():
        (folder / MARKER_FILE).write_text(INIT_CONFIG)
    if example:
        (folder / "specs" / "example.yaml").write_text(INIT_SPEC_EXAMPLE)
        example_dir = folder / "problems" / "example"
        example_dir.mkdir(parents=True, exist_ok=True)
        (example_dir / "problem.yaml").write_text(INIT_PROBLEM_YAML)
        (example_dir / "description.md").write_text(INIT_PROBLEM_DESCRIPTION)
        (example_dir / "verifier.sh").write_text(INIT_PROBLEM_VERIFIER)
        (example_dir / "verifier.sh").chmod(0o755)
    gitignore = root / ".gitignore"
    existing_ignore = gitignore.read_text() if gitignore.exists() else ""
    present = existing_ignore.splitlines()
    missing = [
        line
        for line in (f"{MARKER_DIR}/runs/", f"{MARKER_DIR}/.env")
        if line not in present
    ]
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
