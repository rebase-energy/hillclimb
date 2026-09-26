"""Every `--help` screen, pinned byte for byte.

The CLI is a package of command modules registered on import; a module that
`cli/__init__.py` forgets to import drops its commands silently. These goldens
make that loud in both directions: every command path has a golden, and every
golden has a command path. `HILLCLIMB_UPDATE_GOLDEN=1 uv run pytest
tests/test_cli_help.py` regenerates them — review the diff.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import typer.main
from typer.testing import CliRunner

from hillclimb.cli import app

GOLDEN = Path(__file__).parent / "golden" / "help"
UPDATE = os.environ.get("HILLCLIMB_UPDATE_GOLDEN") == "1"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _paths() -> list[tuple[str, ...]]:
    """Every command path in the click tree, hidden ones included."""
    root = typer.main.get_command(app)
    found: list[tuple[str, ...]] = []

    def walk(command, path: tuple[str, ...]) -> None:
        found.append(path)
        names = command.list_commands(None) if hasattr(command, "commands") else []
        for name in names:
            walk(command.get_command(None, name), (*path, name))

    walk(root, ())
    return found


def _render(path: tuple[str, ...]) -> str:
    result = CliRunner().invoke(app, [*path, "--help"], env={"COLUMNS": "120"})
    assert result.exit_code == 0, (path, result.output)
    return ANSI.sub("", result.output)


def _golden_file(path: tuple[str, ...]) -> Path:
    return GOLDEN / (("-".join(path) if path else "root") + ".txt")


pytestmark = pytest.mark.skipif(
    "TERMINAL_WIDTH" in os.environ, reason="typer pins its width from TERMINAL_WIDTH at import"
)


@pytest.mark.parametrize("path", _paths(), ids=lambda p: " ".join(p) or "root")
def test_help_screen_matches_golden(path):
    rendered = _render(path)
    file = _golden_file(path)
    if UPDATE:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(rendered)
    assert file.is_file(), f"no golden for `hillclimb {' '.join(path)}` — HILLCLIMB_UPDATE_GOLDEN=1 writes it"
    assert rendered == file.read_text()


def test_every_golden_is_a_live_command():
    """An orphaned golden means a command disappeared — most likely a command
    module the cli package no longer imports."""
    live = {_golden_file(path).name for path in _paths()}
    orphans = sorted(p.name for p in GOLDEN.glob("*.txt") if p.name not in live)
    assert not orphans, orphans


def test_python_dash_m_hillclimb_cli_runs():
    """Engine children, resume and the knowledge query the harness advertises
    all start as `python -m hillclimb.cli`; the suite otherwise only asserts
    that argv, never runs it."""
    subprocess.run([sys.executable, "-m", "hillclimb.cli", "--help"], check=True,
                   capture_output=True, timeout=120)
