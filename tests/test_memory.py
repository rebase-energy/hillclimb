"""Memory as a module: a climber's block names it (`memory`, `memory_params`)
like any other slot; the harness walks it through one search — retrieve,
live, publish, record — and the user keeps the switch."""

from __future__ import annotations

import pytest

from hillclimb.agents.fake import FakeAgent
from hillclimb.climber import ClimberLoadError, resolve_climber
from hillclimb.config import Config, parse_set_overrides
from hillclimb.harness.glue import build_memory, effective_memory
from hillclimb.modules.memory.base import Memory, MemoryEnv, Retrieved
from hillclimb.modules.memory.files import FilesMemory, NoMemory, get_memory
from tests.conftest import ok_script

NOTEBOOK_PY = '''\
from pathlib import Path

from hillclimb.sdk import Memory, Retrieved


class Notebook(Memory):
    """One text file of notes per folder; says what happened to it in `log.txt`."""
    DEFAULTS = {"where": "notes.txt", "shout": False}

    def _log(self, line):
        with (Path(self.param("where")).parent / "log.txt").open("a") as f:
            f.write(line + "\\n")

    def retrieve(self, *, context=None):
        notes = Path(self.param("where"))
        text = notes.read_text() if notes.exists() else ""
        self._log(f"retrieve for {self.env.problem.problem_id}")
        return Retrieved(text=text.upper() if self.param("shout") else text, priors={"complexity_start": 1, "nope": 9})

    def live(self):
        return "A sibling is trying annealing."

    def publish(self, journal, *, budget_s, cost_usd):
        self._log(f"publish {len(journal.candidates)}")

    def record(self, journal, *, budget_s, cost_usd):
        best = max((c.val_score for c in journal.scored_candidates()), default=None)
        Path(self.param("where")).write_text(f"Last time the best was {best}.\\n")
        self._log("record")
'''


def test_the_block_names_the_memory_and_sets_its_behaviour():
    default = resolve_climber("greedy").memory()
    assert isinstance(default, FilesMemory) and default.param("max_cards") == 3 and default.param("claims") is True
    tuned = resolve_climber({"memory_params": {"max_cards": 1, "claims": False, "skills": False}}).memory()
    assert (tuned.param("max_cards"), tuned.agent_passes()) == (1, ())  # no distill pass to route
    assert default.agent_passes() == ("distill",)
    assert isinstance(resolve_climber({"memory": "none"}).memory(), NoMemory)
    # a setting the memory does not have is an error before anything runs
    with pytest.raises(ClimberLoadError, match="memory files has no setting .'max_card'. .it has: .*max_cards"):
        resolve_climber({"memory_params": {"max_card": 1}}).memory()
    with pytest.raises(ClimberLoadError, match="unknown memory 'sqlite' .available: files, none"):
        resolve_climber({"memory": "sqlite"}).memory()


def test_the_user_keeps_the_switch():
    config = Config()
    assert effective_memory(config) == "files" and isinstance(build_memory(config), FilesMemory)
    config.learning.enabled = False  # `learning.enabled: false`, `--no-learning`
    assert effective_memory(config) == "none" and isinstance(build_memory(config), NoMemory)
    config = Config.model_validate({"climber": {"memory": "none"}})
    assert effective_memory(config) == "none"  # ...and a climber may simply not use one


def test_the_0_5_learning_settings_are_the_memorys_params():
    """How memory behaves was the user's `learning:` block; it is the
    climber's `memory_params` now. Old configs and `--set` keys still load."""
    config = Config.model_validate({"learning": {"max_cards": 5, "claims": False, "tool": False, "dir": "kb"}})
    assert config.climber.memory_params == {"max_cards": 5, "claims": False}
    assert (config.learning.tool, str(config.learning.dir)) == (False, "kb")  # the user's stay where they were
    config = Config()
    config.apply_overrides(parse_set_overrides(["learning.skills=false", "learning.complexity_prior=true"]))
    assert config.climber.memory_params == {"skills": False, "complexity_prior": True}
    assert build_memory(config).param("skills") is False


def test_a_climber_brings_its_own_memory(task, config, tmp_path, monkeypatch):
    """`memory: notebook.py`: the harness walks it through the search. What
    it retrieves reaches the draft prompt, its priors the policy, its live
    section every prompt; it hears of every result and records at the end.
    The file travels in the snapshot, its settings are `memory_params`."""
    from hillclimb import api
    from hillclimb.harness.budget import BudgetManager
    from hillclimb.harness.run import RunMeta, load_search_meta

    (tmp_path / "notebook.py").write_text(NOTEBOOK_PY)
    notes = tmp_path / "notes.txt"
    notes.write_text("Last time the best was 0.4.\n")
    config.apply_overrides({"climber": {
        "operator_policy": "greedy", "params": {"num_drafts": 1, "tune_budget": 0, "ensemble": False},
        "memory": str(tmp_path / "notebook.py"), "memory_params": {"where": str(notes), "shout": True},
    }})
    agent = FakeAgent()
    agent.queue(script=ok_script(0.6), notes="d\n")
    agent.queue(script=ok_script(0.7), notes="i\n")
    monkeypatch.setattr("hillclimb.api.get_agent", lambda *a, **k: agent)
    config.holdout.enabled = False
    config.budget.max_evaluations = 2
    run_dir = api.create_run(config, RunMeta(run_id="r1", name="r1", kind="problem", target="t", problem_ids=[task.problem_id]))
    search_dir = api.create_search(config, task, run_dir, "r1", 600)
    assert (search_dir / "climber" / "files" / "notebook.py").is_file()
    meta = load_search_meta(search_dir)
    assert meta.learning_enabled and meta.climber_spec["memory_params"] == {"where": str(notes), "shout": True}

    outcome = api.execute_search(config, task, search_dir, BudgetManager(600, stop_margin_s=1), log=lambda *_: None)

    assert outcome.state == "done"
    draft, improve = (r.prompt for r in agent.requests)
    assert "LAST TIME THE BEST WAS 0.4." in draft  # retrieved, shouted: memory_params reached it
    assert "A sibling is trying annealing." in draft and "A sibling is trying annealing." in improve
    assert load_search_meta(search_dir).memory_priors == {"complexity_start": 1, "nope": 9}
    assert "moderate" in draft.lower() or "complexity" in draft.lower()  # the learned offset moved the cue
    assert notes.read_text() == "Last time the best was 0.7.\n"
    log = (tmp_path / "log.txt").read_text().splitlines()
    assert log[0] == f"retrieve for {task.problem_id}" and log[-1] == "record"
    assert sum(1 for line in log if line.startswith("publish")) >= 2  # after every committed result


def test_no_memory_still_leaves_the_search_its_own_card(task, config, tmp_path, monkeypatch):
    from hillclimb import api
    from hillclimb.harness.budget import BudgetManager
    from hillclimb.harness.run import RunMeta, load_search_meta
    from hillclimb.modules.memory.knowledge import CARD_FILENAME

    config.apply_overrides({"climber": {"memory": "none"}})
    config.learning.dir = tmp_path / "knowledge"
    agent = FakeAgent()
    agent.queue(script=ok_script(0.6), notes="d\n")
    monkeypatch.setattr("hillclimb.api.get_agent", lambda *a, **k: agent)
    config.holdout.enabled = False
    config.budget.max_evaluations = 1
    run_dir = api.create_run(config, RunMeta(run_id="r1", name="r1", kind="problem", target="t", problem_ids=[task.problem_id]))
    search_dir = api.create_search(config, task, run_dir, "r1", 600)
    assert load_search_meta(search_dir).learning_enabled is False
    lines = []
    api.execute_search(config, task, search_dir, BudgetManager(600, stop_margin_s=1), log=lines.append)
    assert any("memory: none" in line for line in lines)
    assert (search_dir / CARD_FILENAME).is_file()  # its own record, beside its artifacts
    assert not (tmp_path / "knowledge").exists()  # nothing kept for other searches
    assert [r.operator for r in agent.requests] == ["draft"]  # and no distill pass


def test_the_base_memory_does_nothing(tmp_path):
    memory = Memory()
    memory.bind(MemoryEnv(config=Config(), problem=None, search_dir=tmp_path))
    assert memory.retrieve() == Retrieved() and memory.retrieve(context="given").text == "given"
    assert memory.live() == "" and memory.graph_module() is None and memory.agent_passes() == ()
    memory.publish(None, budget_s=1, cost_usd=0.0)
    memory.record(None, budget_s=1, cost_usd=0.0)
    assert get_memory("none").enabled is False
