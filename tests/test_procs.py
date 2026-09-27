"""Process accounting (`harness/procs.py`): CPU read at reaping, the
descendants a group kill would orphan, and the agent's CPU reaching the
journal."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from hillclimb.backends.fake import FakeBackend
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
    """The backend's cpu_s lands on the candidate's backend record, next to
    its tokens, so the chart's cost fold can add it to the verifier's."""
    backend = FakeBackend()
    backend.queue(script=ok_script(0.5), notes="d\n", result={"total_tokens": 10, "cpu_s": 12.5})
    harness, journal, _ = make_harness(task, config, backend)
    harness.run(Action(operator="draft"))
    drafted = [c for c in journal.candidates.values() if c.operator == "draft"]
    assert len(drafted) == 1
    assert drafted[0].backend.cpu_s == 12.5
    assert drafted[0].backend.total_tokens == 10
