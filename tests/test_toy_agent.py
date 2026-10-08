"""The scripted `toy` agent and the seam for registering an agent in Python."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

import hillclimb as hc
from hillclimb import agents, api
from hillclimb.agents import AgentRequest, AgentResult, get_agent, register_agent
from hillclimb.agents.toy import ToyAgent, _point
from hillclimb.harness.journal import Journal
from hillclimb.sdk import inspiration_filename

PROBLEM = Path("problems/fitness-landscape")


@pytest.fixture
def registry(monkeypatch):
    """Registrations made in a test do not outlive it."""
    monkeypatch.setattr(agents, "_AGENTS", dict(agents._AGENTS))


@pytest.fixture
def lean_runtime(monkeypatch):
    """The problem asks for its own lean venv; the dev interpreter already has
    numpy and pandas, so no test builds one."""
    monkeypatch.setattr("hillclimb.api.ensure_runtime_venv", lambda *a, **k: Path(sys.executable))


def candidate_dir(tmp_path: Path, name: str = "c001", *, landscape: bool = True) -> Path:
    path = tmp_path / name
    (path / "problem").mkdir(parents=True)
    if landscape:
        shutil.copy(PROBLEM / "landscape.py", path / "problem" / "landscape.py")
    return path


def request(path: Path, kind: str | None = "create", operator: str = "draft", prompt: str = "") -> AgentRequest:
    return AgentRequest(operator=operator, kind=kind, prompt=prompt, candidate_dir=path, timeout_s=10)


def stand_at(path: Path, x: float, y: float, name: str = "solution.py") -> None:
    (path / name).write_text(f'"""a point."""\nX, Y = {x!r}, {y!r}\n')


def test_create_writes_a_point_inside_the_domain_with_two_knobs(tmp_path):
    path = candidate_dir(tmp_path)
    result = ToyAgent().invoke(request(path))
    assert result.ok and result.cost_usd == 0.0 and result.total_tokens == 0
    x, y = _point(path / "solution.py")
    assert -5.0 <= x <= 5.0 and -5.0 <= y <= 5.0
    assert set(yaml.safe_load((path / "params.json").read_text())) == {"dx", "dy"}
    assert (path / "notes.md").read_text().startswith("toy: ")


def test_the_same_candidate_gets_the_same_point_and_another_seed_a_different_one(tmp_path):
    first, again, other = (candidate_dir(tmp_path / n) for n in ("a", "b", "c"))
    ToyAgent().invoke(request(first))
    ToyAgent().invoke(request(again))
    ToyAgent(seed=1).invoke(request(other))
    assert _point(first / "solution.py") == _point(again / "solution.py") != _point(other / "solution.py")


def test_refine_steps_from_the_parent_and_the_prompt_sets_the_step(tmp_path):
    near, far = candidate_dir(tmp_path / "near"), candidate_dir(tmp_path / "far")
    for path in (near, far):
        stand_at(path, 1.0, 1.0)
    ToyAgent().invoke(request(near, "refine", "improve", prompt="Improve it.\ntoy: step=0.001\n"))
    ToyAgent().invoke(request(far, "refine", "improve", prompt="Improve it.\ntoy: step=2\n"))

    def moved(path: Path) -> float:
        x, y = _point(path / "solution.py")
        return abs(x - 1.0) + abs(y - 1.0)

    assert 0 < moved(near) < 0.02 < moved(far)


def test_refine_starts_from_where_a_tuned_parent_ended_up(tmp_path):
    path = candidate_dir(tmp_path)
    stand_at(path, 1.0, 1.0)
    # what the harness writes for the child of a tunable parent: its best values as defaults
    (path / "params.json").write_text(
        '{"dx": {"type": "float", "low": -1, "high": 1, "default": 0.5},'
        ' "dy": {"type": "float", "low": -1, "high": 1, "default": -0.5}}'
    )
    assert _point(path / "solution.py", path / "params.json") == (1.5, 0.5)
    ToyAgent().invoke(request(path, "refine", "improve", prompt="toy: step=0.0001"))
    x, y = _point(path / "solution.py")
    assert abs(x - 1.5) < 0.01 and abs(y - 0.5) < 0.01
    fresh = yaml.safe_load((path / "params.json").read_text())
    assert fresh["dx"]["default"] == fresh["dy"]["default"] == 0.0


def test_combine_is_the_mean_and_repair_pulls_the_parent_inside(tmp_path):
    blend = candidate_dir(tmp_path / "blend")
    stand_at(blend, 0.0, 0.0)
    stand_at(blend, 2.0, 4.0, inspiration_filename(1))
    stand_at(blend, 4.0, -1.0, inspiration_filename(2))
    ToyAgent().invoke(request(blend, "combine", "ensemble"))
    assert _point(blend / "solution.py") == (2.0, 1.0)

    outside = candidate_dir(tmp_path / "outside")
    stand_at(outside, 9.0, -9.0)
    ToyAgent().invoke(request(outside, "repair", "debug"))
    assert _point(outside / "solution.py") == (5.0, -5.0)


def test_the_toy_refuses_what_it_cannot_do(tmp_path):
    distill = ToyAgent().invoke(request(candidate_dir(tmp_path / "a"), kind=None, operator="distill"))
    assert not distill.ok and "distill" in distill.error_message
    elsewhere = ToyAgent().invoke(request(candidate_dir(tmp_path / "b", landscape=False)))
    assert not elsewhere.ok and "fitness-landscape" in elsewhere.error_message


def test_register_agent_adds_a_name_and_guards_the_built_ins(registry):
    class Mine:
        name = "mine"

        def invoke(self, request):
            return AgentResult(ok=True)

    register_agent("mine", Mine)
    assert isinstance(get_agent("mine"), Mine) and "mine" in agents.agent_names()
    assert not agents.is_builtin("mine") and agents.is_builtin("toy")
    with pytest.raises(ValueError, match="already registered"):
        register_agent("mine", Mine)
    register_agent("mine", Mine, replace=True)
    with pytest.raises(ValueError, match="built-in"):
        register_agent("dummy", Mine, replace=True)
    with pytest.raises(TypeError):
        register_agent("other", Mine())
    with pytest.raises(ValueError, match="only in the process that registered it"):
        get_agent("nobody")
    assert hc.register_agent is register_agent


def test_a_fleet_refuses_an_agent_only_this_process_knows(registry, config):
    register_agent("mine", ToyAgent)
    with pytest.raises(ValueError, match="detached engine cannot know it"):
        api.run_fleet("fitness-landscape", config=config, agent="mine")


def test_the_terrain_is_in_the_catalog():
    from hillclimb.catalog import PROBLEM_IDS, problem_path

    assert "fitness-landscape" in PROBLEM_IDS and problem_path("fitness-landscape") == PROBLEM.resolve()


def test_a_toy_search_climbs_and_its_spec_reruns_it(config, lean_runtime):
    outcome = hc.run(
        "fitness-landscape", agent="toy", max_evaluations=6, learning=False, holdout=False,
        config=config, log=lambda *_: None,
    )
    assert outcome.state == "done"
    journal = Journal(outcome.search_dir / "journal.jsonl")
    scored = [c for c in journal.candidates.values() if c.operator != "baseline"]
    assert scored and all(c.status == "passing" and set(c.metrics) >= {"x", "y"} for c in scored)
    assert len({c.val_score for c in scored}) > 1  # the scores move
    assert sum(len(c.trials) for c in scored) == 6  # the cap ended it, not the clock
    assert outcome.selected.val_score == max(c.val_score for c in journal.candidates.values())
    entry = yaml.safe_load((outcome.run_dir / "spec.yaml").read_text())["problems"][0]
    assert entry["agent"] == "toy"
    assert entry["set"] == ["learning.enabled=false", "budget.max_evaluations=6"]


def test_a_search_from_python_says_where_to_watch_and_reports_every_candidate(config, lean_runtime):
    """A Python search names the live views once, at the start (`hints=False`
    leaves that out), and every finished candidate gets its own line: its
    status, its score and how long it took, not only the new bests."""
    lines: list[str] = []
    hc.run(
        "fitness-landscape", agent="toy", max_evaluations=3, learning=False, holdout=False,
        config=config, log=lines.append,
    )
    assert lines[1] == api.FOLLOW_HINT
    landed = [line for line in lines if line.startswith("  c0") and " passing val=" in line]
    started = [line for line in lines if "] draft" in line or "] improve" in line]
    assert started and len(landed) == len(started)

    quiet: list[str] = []
    hc.run(
        "fitness-landscape", agent="toy", max_evaluations=1, learning=False, holdout=False,
        config=config, log=quiet.append, hints=False,
    )
    assert api.FOLLOW_HINT not in quiet


def test_landed_line_names_why_a_candidate_has_no_score():
    from hillclimb.harness.candidate import Candidate
    from hillclimb.harness.core import landed_line

    common = dict(candidate_id="c004", operator="improve", created_at="2026-10-03T10:00:00+00:00")
    abandoned = Candidate(**common, status="abandoned", summary="coding agent call failed (timeout)\nmore",
                          finished_at="2026-10-03T10:01:32+00:00")
    assert landed_line(abandoned) == "  c004 abandoned (1m32s) — coding agent call failed (timeout)"
    from hillclimb.harness.candidate import Replicate, Trial

    crashed = Candidate(**common, status="buggy", summary="the agent's notes", finished_at="2026-10-03T10:00:05+00:00",
                        trials=[Trial(index=0, verdict="buggy", replicates=[Replicate(returncode=1)])])
    assert landed_line(crashed) == "  c004 buggy (5s) — verifier exited 1"
    slow = Candidate(**common, status="buggy", finished_at="2026-10-03T10:00:12+00:00",
                     trials=[Trial(index=0, verdict="buggy",
                                   replicates=[Replicate(returncode=-9, timed_out=True, duration_s=10.2)])])
    assert landed_line(slow) == "  c004 buggy (12s) — timed out after 10s"
    passing = Candidate(**common, status="passing", finished_at="2026-10-03T10:00:09+00:00")
    passing.trials = []
    assert landed_line(passing).startswith("  c004 passing")


def test_a_registered_agent_runs_a_search_in_this_process(registry, config, lean_runtime):
    class Corner:
        """Always stands in the same corner."""

        name = "corner"

        def invoke(self, request):
            (request.candidate_dir / "solution.py").write_text(
                'open("submission.csv", "w").write("x,y\\n4.0,4.0\\n")\n'
            )
            return AgentResult(ok=True)

    hc.register_agent("corner", Corner)
    outcome = hc.run(
        "fitness-landscape", agent="corner", max_evaluations=2, learning=False, holdout=False,
        config=config, log=lambda *_: None,
    )
    assert outcome.state == "done"
    journal = Journal(outcome.search_dir / "journal.jsonl")
    assert [c.metrics for c in journal.candidates.values() if c.operator == "draft"] == [{"x": 4.0, "y": 4.0}] * 2


def test_ctrl_c_stops_a_python_script_instead_of_moving_on(registry, config, lean_runtime):
    """S9: from Python a Ctrl-C (SIGINT to the main thread) ends the search
    as `stopped`, resumable, and then goes on as a KeyboardInterrupt, so the
    next search of a script never starts."""
    import os
    import signal
    import threading
    import time

    from hillclimb.harness.journal import Journal
    from hillclimb.harness.status import read_status

    class Slow:
        name = "slow"

        def invoke(self, request):
            time.sleep(0.3)
            stand_at(Path(request.candidate_dir), 0.5, 0.5)
            return AgentResult(ok=True, session_id="s")

    register_agent("slow", Slow)
    lines: list[str] = []
    timer = threading.Timer(2.0, lambda: os.kill(os.getpid(), signal.SIGINT))
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            hc.run("fitness-landscape", agent="slow", budget="2m", learning=False,
                   holdout=False, config=config, log=lines.append)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 60
    [search_dir] = list(config.paths.runs_dir.glob("*/searches/*"))
    assert read_status(search_dir).state == "stopped"
    assert any("Resume with: hillclimb resume" in line for line in lines)
    assert Journal(search_dir / "journal.jsonl").candidates  # the record is kept


def test_an_agent_that_keeps_crashing_fails_the_search_instead_of_looping(registry, config, lean_runtime):
    """E36: a worker that dies (here an agent raising KeyboardInterrupt in its
    thread) used to abandon its candidate and let the search try again until
    the budget ran out; three crashes in a row now fail it, with the error."""
    import time

    from hillclimb.harness.status import read_status

    class Crashing:
        name = "crashing"

        def invoke(self, request):
            raise KeyboardInterrupt

    register_agent("crashing", Crashing)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="3 operators crashed in a row; last: KeyboardInterrupt"):
        hc.run("fitness-landscape", agent="crashing", budget="5m", learning=False,
               holdout=False, config=config, log=lambda *_: None)
    assert time.monotonic() - started < 60
    [search_dir] = list(config.paths.runs_dir.glob("*/searches/*"))
    assert read_status(search_dir).state == "failed"
