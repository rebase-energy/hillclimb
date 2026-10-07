"""The CPU allotment of a solution run (`concurrency.solution_cpus`): what
the run's env says, what the coding agent is told, and how a run that used
more is flagged."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from hillclimb.config import Config, ConcurrencyConfig
from hillclimb.harness.candidate import Candidate, Replicate, Trial
from hillclimb.harness.executor import CPUS_ENV, CommandExecutor, scrubbed_env, single_threaded
from tests.conftest import SELF_REPORT_CMD


def test_env_carries_the_allotment(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "2")  # an explicit parent value still wins
    env = scrubbed_env(4)
    assert env[CPUS_ENV] == "4"
    assert env["OMP_NUM_THREADS"] == "4" and env["OPENBLAS_NUM_THREADS"] == "4"
    assert env["MKL_NUM_THREADS"] == "2"
    assert scrubbed_env()[CPUS_ENV] == "1"  # the default: one core


def test_the_harness_owns_hillclimb_cpus(monkeypatch):
    """An inherited $HILLCLIMB_CPUS never overrides the configured allotment."""
    monkeypatch.setenv(CPUS_ENV, "16")
    assert single_threaded({CPUS_ENV: "16"}, 2)[CPUS_ENV] == "2"
    assert scrubbed_env(3)[CPUS_ENV] == "3"


def test_coding_agent_env_carries_the_allotment(monkeypatch):
    """The coding agent's own test runs see what the real run gets."""
    from hillclimb.agents.claude_code import subscription_env

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    env = subscription_env("subscription", cpus=3)
    assert env[CPUS_ENV] == "3" and env["OMP_NUM_THREADS"] == "3"
    assert subscription_env()[CPUS_ENV] == "1"


def test_verifier_run_gets_and_records_the_allotment(tmp_path):
    executor = CommandExecutor(Path(sys.executable), SELF_REPORT_CMD, cpus=3)
    script = tmp_path / "solution.py"
    script.write_text(
        "import os\n"
        'print("cpus:", os.environ["HILLCLIMB_CPUS"])\n'
        'open("submission.csv", "w").write("id\\n")\n'
        'print("val_score: 1.0")\n'
    )
    result = executor.execute(script, tmp_path, 30)
    assert result.ok
    assert "cpus: 3" in Path(result.stdout_path).read_text()
    assert result.cpus == 3


def test_solution_cpus_setting():
    assert ConcurrencyConfig().solution_cpus == 1
    with pytest.raises(ValidationError):
        ConcurrencyConfig(solution_cpus=0)
    config = Config()
    config.apply_overrides({"concurrency.solution_cpus": 4})
    assert config.concurrency.solution_cpus == 4


@pytest.mark.parametrize(
    "cpu_s, duration_s, cpus, flagged",
    [
        (70.0, 10.0, 1, True),     # seven cores busy on a one-core allotment
        (9.5, 10.0, 1, False),     # one core, as given
        (70.0, 10.0, 8, False),    # seven of eight: within the allotment
        (14.0, 2.0, 1, False),     # too short to judge
        (70.0, 10.0, None, True),  # a journal predating the field: one core
        (None, 10.0, 1, False),    # no CPU accounting on this platform
    ],
)
def test_oversubscribed(cpu_s, duration_s, cpus, flagged):
    replicate = Replicate(cpu_s=cpu_s, duration_s=duration_s, cpus=cpus)
    assert replicate.oversubscribed is flagged


def test_candidate_cpu_overuse_is_its_worst_run():
    runs = [
        Replicate(cpu_s=10.0, duration_s=10.0, cpus=1),
        Replicate(cpu_s=40.0, duration_s=10.0, cpus=1),
        Replicate(cpu_s=70.0, duration_s=10.0, cpus=1),
    ]
    candidate = Candidate(
        candidate_id="c001", operator="draft", candidate_dir="/tmp/c001",
        trials=[Trial(index=0, replicates=runs[:2]), Trial(index=1, replicates=runs[2:])],
    )
    assert candidate.cpu_overuse is runs[2]
    calm = Candidate(
        candidate_id="c002", operator="draft", candidate_dir="/tmp/c002",
        trials=[Trial(index=0, replicates=runs[:1])],
    )
    assert calm.cpu_overuse is None


def test_watch_mark():
    from hillclimb.tui.watch import cpu_mark

    assert cpu_mark(Replicate(cpu_s=69.0, duration_s=10.0, cpus=1)) == "cpu 6.9/1"
    assert cpu_mark(Replicate(cpu_s=69.0, duration_s=10.0)) == "cpu 6.9/1"


def test_contract_states_the_allotment():
    from hillclimb.harness.core import _cores
    from hillclimb.prompts.render import render

    assert _cores(1) == "1 CPU core" and _cores(4) == "4 CPU cores"
    text = render("contract_verifier", solution_cpus=_cores(4))
    assert "gets 4 CPU cores; `HILLCLIMB_CPUS` holds the number" in text
    assert "never from `os.cpu_count()`" in text


def test_launch_warns_when_the_machine_is_oversubscribed(monkeypatch):
    from hillclimb.cli import run as run_cli

    said = []
    monkeypatch.setattr(run_cli, "warn", said.append)
    monkeypatch.setattr("os.cpu_count", lambda: 10)
    config = Config()
    config.concurrency.parallel_agents = 3
    run_cli._warn_oversubscribed(config, {"concurrency.solution_cpus": 2}, searches=2)
    assert said and "12 cores; this machine has 10" in said[0]
    said.clear()
    run_cli._warn_oversubscribed(config, {}, searches=2)  # 6 runs x 1 core fit
    assert not said
