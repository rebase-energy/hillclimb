"""`hillclimb ps` drawn as nvidia-smi-style boxes: fits the terminal, never wraps."""

from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from hillclimb.harness.orphans import process_table
from hillclimb.tui.machine import SearchReader, scan, short_command
from hillclimb.tui.psview import render_ps


def _listing(engines: int, mcp: int = 6) -> str:
    rows = []
    for n in range(engines):
        e = 100 * (n + 1)
        agent = e + 1
        rows.append(f"{e} 1 {e} 0.5 40000 05:00 /venv/bin/python3 -m hillclimb.cli run heilbronn-11 --run-id r{e}")
        rows.append(f"{agent} {e} {agent} 1.0 200000 04:00 claude -p --verbose --model sonnet")
        for k in range(mcp):
            rows.append(f"{e + 10 + k} {agent} {agent} 0.0 7000 03:59 npm exec some-mcp-server-{k}")
        rows.append(
            f"{e + 50} {agent} {agent} 0.0 1000 00:30 /bin/zsh -c source /x/shell-snapshots/s.sh "
            "&& eval 'python solution.py' < /dev/null"
        )
        rows.append(f"{e + 51} {e + 50} {agent} 99.0 30000 00:30 python solution.py")
    return "\n".join(rows)


def _render(listing: str, width: int, height: int | None = None) -> list[str]:
    table = process_table(listing)
    engines = sorted(p.pid for p in table.values() if "hillclimb.cli" in p.command)
    machine, rows = scan(
        table=table, reader=SearchReader(), env_dirs={pid: Path("/nowhere") for pid in engines}, agent_slots=8,
        read_searches=False, root_guides=True,
    )
    console = Console(width=width, file=StringIO(), color_system=None)
    frame = render_ps(
        machine, rows, width, max_height=height, footer=" ctrl+c to quit" if height else None,
        now=datetime(2026, 10, 3, 15, 0, 0),
    )
    console.print(frame)
    return console.file.getvalue().rstrip("\n").splitlines()


@pytest.mark.parametrize("width", [80, 100, 120, 160])
def test_every_line_fits_the_terminal(width):
    lines = _render(_listing(2), width)
    assert lines[0].startswith("╭─ hillclimb ps") and lines[0].endswith("╮")
    assert all(len(line) <= width for line in lines)
    assert all(line[0] in "╭│╰" for line in lines)  # everything is inside a box


def test_one_box_one_table_compute_only():
    lines = _render(_listing(2), 120)
    text = "\n".join(lines)
    assert sum(line.startswith("╭") for line in lines) == 1  # one box
    assert text.count("   pid ") == 1  # one table header
    # each engine heads its own tree, its problem and folder on its row
    assert text.count(" engine ") == 2 and text.count("heilbronn-11  ·  ") == 2
    assert text.count("orphan, dir deleted  ·  /nowhere") == 2  # /nowhere does not exist
    assert "└─ claude · sonnet" in text
    # search progress belongs to `watch`
    for word in ("state", "best", "candidates", "budget"):
        assert word not in text


def test_narrow_terminals_drop_columns_before_cutting_commands():
    wide = "\n".join(_render(_listing(1), 160))
    narrow = "\n".join(_render(_listing(1), 80))
    assert " up " in wide and " up " not in narrow
    assert "$ python solution.py" in narrow  # the command a tool shell ran, readable at 80


@pytest.mark.parametrize("height", [30, 24])
def test_watch_frame_stays_on_one_screen(height):
    lines = _render(_listing(3), 80, height=height)
    assert len(lines) <= height
    text = "\n".join(lines)
    # every engine keeps its own row and its coding agent; what did not fit
    # is counted, not dropped silently
    assert text.count(" engine ") == 3
    assert text.count("claude · sonnet") == 3
    assert "MCP server processes" in text or "more · hillclimb top" in text


def test_without_a_height_every_process_is_listed():
    text = "\n".join(_render(_listing(3), 120))
    assert text.count("npm exec some-mcp-server-") == 18
    assert "more · hillclimb top" not in text


def test_short_command():
    root = Path("/u/proj")
    assert short_command("claude -p --output-format stream-json --model sonnet", "agent") == "claude · sonnet"
    assert short_command("/opt/bin/codex exec --json", "agent") == "codex"
    shell = (
        "/bin/zsh -c source /u/.claude/shell-snapshots/s.sh 2>/dev/null || true "
        "&& eval 'cat > solution.py <<'\"'\"'E'\"'\"'\\012import numpy\\012' < /dev/null && pwd -P"
    )
    assert short_command(shell, "tool") == "$ cat > solution.py <<E …"
    assert short_command("/bin/zsh -c source /x/shell-snapshots/s.sh", "tool") == "$ (shell)"
    assert short_command("bash /u/proj/problems/p/verifier.sh", "verifier", root) == "bash problems/p/verifier.sh"
    run = (
        "/u/.cache/hillclimb/venvs/csv-7eed/bin/python "
        "/u/proj/runs/r1/searches/p/candidates/c003/trials/t0/replicates/r0/solution.py"
    )
    assert short_command(run, "solution", root) == "python c003/trials/t0/replicates/r0/solution.py"
    assert short_command("python -c \\012import os", "tool") == "python -c  ↵ import os"
    # the command a tool shell evals is shortened like any other
    venv_shell = (
        "/bin/zsh -c source /x/shell-snapshots/s.sh && "
        "eval '/u/.cache/hillclimb/venvs/csv-7eed/bin/python /u/proj/solution.py' < /dev/null"
    )
    assert short_command(venv_shell, "tool", root) == "$ python solution.py"


def test_machine_cpu_is_counted_in_cores():
    """`ps` counts 100% per core: one busy core on ten is 1.0 of 10, not "100%"."""
    from hillclimb.tui.machine import MachineRow
    from hillclimb.tui.psview import summary_line

    machine = MachineRow(
        cores=10, load=None, mem_total_mb=None, agents=1, agent_slots=8, cpu=99.9, rss_mb=300.0
    )
    text = summary_line(machine, 1).plain
    assert "cpu 1.0 of 10 cores busy" in text and "%" not in text


@pytest.mark.parametrize(
    "line, shown",
    [
        (
            "export HILLCLIMB_PYTHON=/u/py PYTHONPATH=/u/shim; unset HILLCLIMB_PARAMS; "
            "./problem/verifier.sh 2>&1 | tail -5",
            "./problem/verifier.sh | tail -5",
        ),
        ("cd /u/proj && timeout 200 python solution.py > out.txt 2>&1", "timeout 200 python solution.py > out.txt"),
        ("HILLCLIMB_PARAMS= python -c 'import x; print(1)' >/dev/null 2>&1", "python -c 'import x; print(1)'"),
        ("cat > solution.py <<'E'", "cat > solution.py <<E"),  # writing a file IS the command
        ("ls -la && cat f.txt | head -20 </dev/null", "ls -la ; cat f.txt | head -20"),
        ("export A=1", "export A=1"),  # all setup: shown as it was
    ],
)
def test_essential_commands_keep_what_does_something(line, shown):
    from hillclimb.tui.machine import essential_commands

    assert essential_commands(line) == shown


def test_job_kind_tells_compute_from_mentions():
    from hillclimb.tui.machine import job_kind

    assert job_kind("/venv/bin/python3 -m hillclimb.cli run heilbronn-11 --run-id r1") == "engine"
    assert job_kind("/u/.local/share/uv/tools/hillclimb/bin/python3 /u/.local/bin/hillclimb verify heilbronn-11") == "verify"
    assert job_kind("/venv/bin/hillclimb verify heilbronn-11 --solution s.py") == "verify"
    assert job_kind("python -m hillclimb.cli grade --split holdout") == "grade"
    assert job_kind("/venv/bin/python3 /venv/bin/hillclimb run heilbronn-11 --no-detach") == "run"
    assert job_kind("/venv/bin/hillclimb resume latest") == "run"
    # reading, not computing
    assert job_kind("/venv/bin/python3 /venv/bin/hillclimb ps --watch") is None
    assert job_kind("/venv/bin/python3 /venv/bin/hillclimb watch") is None
    # a shell that merely mentions it is not the job
    assert job_kind("/bin/zsh -c (/venv/bin/hillclimb verify heilbronn-11 > out &)") is None


def test_ps_lists_a_verify_running_in_a_terminal():
    """Compute hillclimb started is in `ps`, engine or not; a job nested in
    another one's tree (a meta-problem's grade) is shown there, not twice."""
    listing = (
        "300 1 300 0.0 30000 00:06 /u/.local/bin/python3 /u/.local/bin/hillclimb verify heilbronn-11 --solution solution.py\n"
        "301 300 300 0.0 2000 00:06 bash /proj/problems/heilbronn-11/verifier.sh\n"
        "302 301 300 99.0 90000 00:06 /u/.cache/hillclimb/venvs/csv-1/bin/python /tmp/hillclimb-verify-ab/candidates/v0/solution.py\n"
        "400 1 400 0.5 40000 05:00 /venv/bin/python3 -m hillclimb.cli run meta-heilbronn --run-id r1\n"
        "401 400 400 0.0 2000 00:30 bash /proj/problems/meta-heilbronn/verifier.sh\n"
        "402 401 400 1.0 40000 00:29 /venv/bin/python3 -m hillclimb.cli grade --split val\n"
    )
    machine, rows = scan(
        table=process_table(listing), reader=SearchReader(),
        env_dirs={300: Path("/proj"), 400: Path("/proj")}, agent_slots=8,
        read_searches=False, root_guides=True,
    )
    assert [(r.pid, r.kind) for r in rows] == [(300, "verify"), (400, "engine")]  # 402 lives under 400
    assert machine.cpu == pytest.approx(0.0 + 0.0 + 99.0 + 0.5 + 0.0 + 1.0)
    console = Console(width=110, file=StringIO(), color_system=None)
    console.print(render_ps(machine, rows, 110))
    text = console.file.getvalue()
    assert "engines 1  ·  verify 1" in text
    assert " verify " in text and "heilbronn-11  ·  " in text
    assert "python v0/solution.py" in text  # the temp candidate path, shortened
