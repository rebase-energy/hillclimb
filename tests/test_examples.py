"""Every script in examples/ runs, on the scripted toy agent, in a fresh hillclimb dir."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import hillclimb.agents
from hillclimb import register_agent
from hillclimb.agents import AgentResult
from hillclimb.demo import install_demo_problem

EXAMPLES = sorted(Path("examples").glob("*.py"))
EVALUATIONS = 4


@pytest.fixture
def fresh_dir(tmp_path, monkeypatch) -> Path:
    """What `hillclimb init` + `hillclimb problem get fitness-landscape` leave:
    the examples find it the way a user's script would, through the folder."""
    (tmp_path / "hillclimb.yaml").write_text("model: sonnet\n")
    install_demo_problem(tmp_path / "problems", "fitness-landscape")
    monkeypatch.setenv("HILLCLIMB_DIR", str(tmp_path))
    # the problem's lean venv is the dev interpreter here: no test builds one
    monkeypatch.setattr("hillclimb.api.ensure_runtime_venv", lambda *a, **k: Path(sys.executable))
    return tmp_path


def _import(path: Path):
    name = f"example_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # a class's module must be findable when its file is written as a ref
    spec.loader.exec_module(module)
    return module


class Digits:
    """A scripted stand-in for nano_climb: each attempt writes one more digit of pi."""

    name = "digits"

    def __init__(self):
        self.n = 2

    def invoke(self, request):
        self.n += 1
        digits = "3.14159265358979323846"[: self.n]
        (request.candidate_dir / "solution.py").write_text(f'open("pi.txt", "w").write("{digits}")\n')
        return AgentResult(ok=True)


@pytest.fixture(autouse=True)
def _scripted_agents(monkeypatch):
    monkeypatch.setattr("hillclimb.agents._AGENTS", dict(hillclimb.agents._AGENTS))
    register_agent("digits", Digits)


def test_there_are_examples():
    assert len(EXAMPLES) >= 7


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(path, fresh_dir, capsys):
    source = path.read_text()
    # a search re-imports the file a climber's classes live in: an unguarded call would run it again
    assert 'if __name__ == "__main__":\n    main(' in source, f"{path.name} must guard its entry point"
    module = _import(path)
    # the scripts default to Claude Code; the suite runs them on a scripted agent
    agent = {"define_a_problem": "bisector", "custom_agent": "grid", "nano_climb": "digits"}.get(path.stem, "toy")
    try:
        result = module.main(evaluations=EVALUATIONS, agent=agent)
    finally:
        sys.modules.pop(module.__name__, None)
    outcomes = list(result.values()) if isinstance(result, dict) else [result]
    assert outcomes
    for outcome in outcomes:
        assert outcome.state in ("done", "stopped"), outcome.error
        assert outcome.search_dir.is_relative_to(fresh_dir / "runs")
        assert outcome.spend.evaluations == EVALUATIONS and outcome.best is not None
    assert capsys.readouterr().out.strip()  # an example shows what it did
    assert not (fresh_dir / "knowledge").exists()  # and leaves the folder's knowledge alone
