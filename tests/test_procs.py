"""Process accounting (`harness/procs.py`): CPU read at reaping, the
descendants a group kill would orphan, and the agent's CPU reaching the
journal."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from hillclimb.agents.fake import FakeAgent
from hillclimb.harness.procs import Reaper, descendant_cpu_s, parse_ps_time
from hillclimb.modules.policies.base import Action
from tests.conftest import ok_script
from tests.harness_factory import make_harness

BURN = "import time; t = time.process_time()\nwhile time.process_time() - t < {secs}: pass\n"


def test_parse_ps_time_reads_every_format():
    assert parse_ps_time("0:00.41") == pytest.approx(0.41)   # macOS mm:ss.ss
    assert parse_ps_time("00:01:30") == 90.0                  # Linux hh:mm:ss
    assert parse_ps_time("1-02:00:00") == 86400 + 7200        # Linux dd-hh:mm:ss
    assert parse_ps_time("garbage") == 0.0


def _burning_shell(tmp_path):
    """A shell whose background grandchild burns 0.4 s of CPU and then
    sleeps — alive, and unwaited-for, when the group is killed."""
    script = tmp_path / "burn.py"
    script.write_text(BURN.format(secs=0.4) + "time.sleep(30)\n")
    return subprocess.Popen(
        ["bash", "-c", f"{sys.executable} {script} & sleep 30"],
        start_new_session=True,
    )


def test_descendant_cpu_is_read_from_ps_while_they_live(tmp_path):
    """A grandchild that burned CPU and is still running shows up under the
    shell that started it — the sum a group kill would otherwise lose."""
    shell = _burning_shell(tmp_path)
    try:
        time.sleep(1.0)
        assert descendant_cpu_s(shell.pid) >= 0.3
    finally:
        Reaper(shell).kill_group()


def test_kill_group_keeps_the_orphans_cpu(tmp_path):
    """Killing at a timeout reports the CPU of the descendants the kill
    orphaned, not just the leader's own."""
    shell = _burning_shell(tmp_path)
    reaper = Reaper(shell)
    assert reaper.wait(timeout=1.0) is None  # still running
    reaper.kill_group()
    assert shell.returncode is not None and shell.returncode < 0
    assert reaper.cpu_s is not None and reaper.cpu_s >= 0.3


def test_reaper_reads_cpu_at_a_normal_exit():
    proc = subprocess.Popen([sys.executable, "-c", BURN.format(secs=0.3)])
    reaper = Reaper(proc)
    assert reaper.wait(timeout=30) == 0
    assert reaper.cpu_s is not None and reaper.cpu_s >= 0.2
    assert reaper.poll() == 0 and reaper.cpu_s < 1.0  # a second poll does not re-count


def test_agent_cpu_reaches_the_journal(task, config):
    """The agent's cpu_s lands on the candidate's agent record, next to
    its tokens, so the chart's cost fold can add it to the verifier's."""
    agent = FakeAgent()
    agent.queue(script=ok_script(0.5), notes="d\n", result={"total_tokens": 10, "cpu_s": 12.5})
    harness, journal, _ = make_harness(task, config, agent)
    harness.run(Action(operator="draft"))
    drafted = [c for c in journal.candidates.values() if c.operator == "draft"]
    assert len(drafted) == 1
    assert drafted[0].agent.cpu_s == 12.5
    assert drafted[0].agent.total_tokens == 10


def test_the_ledger_lists_children_while_they_live(tmp_path):
    """An engine killed with SIGKILL takes none of its children with it: the
    ledger in the search dir is how resume/stop/kill find them."""
    import json
    import subprocess

    from hillclimb.harness import procs

    ledger = tmp_path / procs.CHILDREN_FILE
    procs.track_children(ledger)
    try:
        proc = subprocess.Popen(["sleep", "5"], start_new_session=True)
        reaper = procs.Reaper(proc)
        [entry] = json.loads(ledger.read_text())
        assert entry["pid"] == proc.pid and entry["program"] == "sleep"
        reaper.kill_group()
        assert not ledger.exists()  # reaped: off the ledger
    finally:
        procs.track_children(None)


def test_what_a_dead_engine_left_running_is_stopped(tmp_path):
    """A child still leading its own process group is stopped with its
    group; a pid that no longer leads one (recycled) is left alone."""
    import json
    import os
    import subprocess

    from hillclimb.harness import procs
    from hillclimb.harness.orphans import stop_orphaned_children

    leader = subprocess.Popen(["sleep", "30"], start_new_session=True)
    entries = [
        {"pid": leader.pid, "program": "sleep", "started": 0},
        {"pid": os.getpid(), "program": "python", "started": 0},  # alive, but no group leader of its own
    ]
    (tmp_path / procs.CHILDREN_FILE).write_text(json.dumps(entries))
    stopped = stop_orphaned_children(tmp_path, grace_s=2)
    assert [entry["pid"] for entry in stopped] == [leader.pid]
    assert leader.wait(timeout=5) is not None
    assert not (tmp_path / procs.CHILDREN_FILE).exists()


def test_a_sandbox_wrapper_is_named_by_what_it_runs():
    from types import SimpleNamespace

    from hillclimb.harness.procs import _program

    assert _program(SimpleNamespace(args=["sandbox-exec", "-p", "(version 1)", "/usr/local/bin/claude", "-p"])) == "claude"
    assert _program(SimpleNamespace(args=["bwrap", "--ro-bind", "/", "/", "--", "bash", "verifier.sh"])) == "bash"
    assert _program(SimpleNamespace(args=["codex", "exec"])) == "codex"
