"""Skills and standing instructions for operators: a global layer and the
hillclimb dir's own, rendered for whichever coding agent runs, and skills an
operator creates added to the local layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from hillclimb.config import Config
from hillclimb.harness import agent_context
from tests.catalog_fixture import pin
from tests.folder_config import write_config


def write_skill(root: Path, name: str, description: str, body: str = "Do the thing.\n") -> Path:
    folder = root / "skills" / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}")
    return folder


@pytest.fixture
def layers(tmp_path, monkeypatch):
    """A global layer under a stand-in user config dir, and a hillclimb dir."""
    user = tmp_path / "user"
    monkeypatch.setattr("hillclimb.project.user_config_path", lambda: user / "config.yaml")
    hillclimb_dir = tmp_path / "energy"
    hillclimb_dir.mkdir()
    config = Config(hillclimb_dir=hillclimb_dir)
    pin(config)  # the engine ships no climber
    glob, local = user / "agent", hillclimb_dir / "agent"
    write_skill(glob, "profile-first", "Use when code is slow.")
    write_skill(glob, "pinball-loss", "global version")
    write_skill(local, "pinball-loss", "Use when scoring quantiles.")
    write_skill(local, "dst-gaps", "Use when timestamps cross a DST change.")
    (glob / "AGENTS.md").write_text("Write deterministic code.\n")
    (local / "AGENTS.md").write_text("These are energy time series.\n")
    return config, glob, local


def test_local_skills_override_global_ones_and_instructions_join(layers):
    config, _, _ = layers
    found = agent_context.skills(config)
    assert sorted(found) == ["dst-gaps", "pinball-loss", "profile-first"]
    assert found["pinball-loss"].layer == "local"
    assert found["pinball-loss"].description == "Use when scoring quantiles."
    assert found["profile-first"].layer == "global"
    assert agent_context.instructions(config) == "Write deterministic code.\n\nThese are energy time series."


def test_a_folder_can_leave_the_global_layer_out_or_skip_a_skill(layers):
    config, _, _ = layers
    config.agent_context.skip = ["dst-gaps"]
    assert sorted(agent_context.skills(config)) == ["pinball-loss", "profile-first"]
    config.agent_context.include_global = False
    assert sorted(agent_context.skills(config)) == ["pinball-loss"]
    assert agent_context.instructions(config) == "These are energy time series."


@pytest.mark.parametrize(
    ("agent", "skills_dir", "instructions_file"),
    [("claude-code", ".claude/skills", "CLAUDE.md"), ("codex", ".agents/skills", "AGENTS.md"), ("pi", ".agents/skills", "AGENTS.md")],
)
def test_the_layers_are_rendered_where_each_agent_reads_a_projects_own(layers, tmp_path, agent, skills_dir, instructions_file):
    config, _, _ = layers
    candidate = tmp_path / "c001"
    candidate.mkdir()
    given = agent_context.render(config, agent, candidate)
    assert [(s["name"], s["layer"], s["origin"]) for s in given["skills"]] == [
        ("dst-gaps", "local", "manual"), ("pinball-loss", "local", "manual"), ("profile-first", "global", "manual"),
    ]
    assert all(len(s["sha256"]) == 64 for s in given["skills"]) and len(given["instructions_sha256"]) == 64
    assert "quantiles" in (candidate / skills_dir / "pinball-loss" / "SKILL.md").read_text()
    assert (candidate / instructions_file).read_text().startswith("Write deterministic code.")


def test_an_agent_without_a_layout_gets_nothing(layers, tmp_path):
    config, _, _ = layers
    candidate = tmp_path / "c001"
    candidate.mkdir()
    assert agent_context.render(config, "dummy", candidate) == {}
    assert list(candidate.iterdir()) == []


def test_a_skill_an_operator_creates_is_added_to_the_local_layer(layers, tmp_path):
    """New skills land in agent/skills/ of the hillclimb dir; the rendered
    ones and a name that exists already are left alone; a plugin's skill in
    the operator home is moved, so the next family starts clean."""
    config, _, local = layers
    candidate = tmp_path / "c001"
    candidate.mkdir()
    agent_context.render(config, "claude-code", candidate)
    write_skill(candidate / ".claude", "solar-clear-sky", "Use for solar features.")
    (candidate / ".claude" / "skills" / "pinball-loss" / "SKILL.md").write_text("changed by the agent")
    home = tmp_path / "claude-home"
    write_skill(home, "from-a-plugin", "Written by a plugin.")
    logged = []

    made = {"run": "r1", "search": "gefcom-solar", "candidate": "c004", "agent": "claude-code", "model": "sonnet"}
    added = agent_context.harvest(config, candidate, homes=(home,), created_by=made, log=logged.append)

    assert added == ["solar-clear-sky", "from-a-plugin"]
    found = agent_context.skills(config)
    assert found["solar-clear-sky"].origin == "generated"
    record = found["solar-clear-sky"].record
    assert record["created_by"] == {**made, "found_in": ".claude/skills"}
    assert record["sha256"] == found["solar-clear-sky"].sha256 and record["created_at"]
    assert found["from-a-plugin"].record["created_by"]["found_in"] == "operator home"
    assert found["dst-gaps"].origin == "manual"  # a person's: no record
    # rendered again, the record stays hillclimb's: the agent never sees it
    agent_context.render(config, "codex", tmp_path / "c002")
    assert not (tmp_path / "c002" / ".agents" / "skills" / "solar-clear-sky" / "skill.yaml").exists()
    assert (local / "skills" / "solar-clear-sky" / "SKILL.md").is_file()
    assert (local / "skills" / "from-a-plugin" / "SKILL.md").is_file()
    assert "quantiles" in (local / "skills" / "pinball-loss" / "SKILL.md").read_text()  # untouched
    assert not (home / "skills" / "from-a-plugin").exists()
    assert logged and "solar-clear-sky" in logged[0]
    assert agent_context.harvest(config, candidate, homes=(home,)) == []  # once only


def test_without_a_hillclimb_dir_nothing_is_harvested(tmp_path, monkeypatch):
    monkeypatch.setattr("hillclimb.project.user_config_path", lambda: tmp_path / "user" / "config.yaml")
    candidate = tmp_path / "c001"
    write_skill(candidate / ".agents", "orphan", "x")
    assert agent_context.harvest(Config(), candidate) == []


def test_skills_lists_and_deletes(layers, monkeypatch):
    from typer.testing import CliRunner

    from hillclimb import cli

    config, _, local = layers
    monkeypatch.setenv("HILLCLIMB_DIR", str(config.hillclimb_dir))
    write_config(config.hillclimb_dir, {"model": "sonnet"})
    result = CliRunner().invoke(cli.app, ["skills"])
    assert result.exit_code == 0, result.output
    assert "dst-gaps" in result.output and "profile-first" in result.output and "local" in result.output
    result = CliRunner().invoke(cli.app, ["skills", "delete", "dst-gaps"])
    assert result.exit_code == 0, result.output
    assert not (local / "skills" / "dst-gaps").exists()
    result = CliRunner().invoke(cli.app, ["skills", "delete", "nope"])
    assert result.exit_code == 1


def test_an_edited_generated_skill_says_so_and_keep_makes_it_manual(layers, tmp_path):
    config, _, local = layers
    candidate = tmp_path / "c001"
    write_skill(candidate / ".agents", "wind-ramps", "Use for ramp events.")
    agent_context.harvest(config, candidate, created_by={"candidate": "c001"})
    path = local / "skills" / "wind-ramps" / "SKILL.md"
    path.write_text(path.read_text() + "Also check hub height.\n")
    skill = agent_context.skills(config)["wind-ramps"]
    assert skill.origin == "generated, edited"
    agent_context.keep(skill)
    kept = agent_context.skills(config)["wind-ramps"]
    assert kept.origin == "manual"
    assert kept.record["created_by"] == {"candidate": "c001", "found_in": ".agents/skills"} and kept.record["kept_at"]


def test_the_journal_records_what_each_candidate_was_given_and_added(layers, tmp_path, monkeypatch):
    """A search with a scripted codex-shaped agent: the candidate's journal
    entry carries the skills it was given (name, layer, origin, hash) and
    the skill it wrote, which is now in the local layer as generated."""
    import sys

    import hillclimb as hc
    from hillclimb import agents
    from hillclimb.agents import AgentResult, register_agent
    from hillclimb.harness.journal import Journal

    config, _, local = layers
    config.paths.runs_dir = tmp_path / "runs"
    config.paths.problems_dir = Path("problems")
    config.paths.runtime_python = Path(sys.executable)
    monkeypatch.setattr(agents, "_AGENTS", dict(agents._AGENTS))
    monkeypatch.setattr("hillclimb.api.ensure_runtime_venv", lambda *a, **k: Path(sys.executable))
    monkeypatch.setattr(agent_context, "LAYOUT", {**agent_context.LAYOUT, "scripted": (".agents/skills", "AGENTS.md")})

    class Scripted:
        name = "scripted"

        def invoke(self, request):
            folder = Path(request.candidate_dir)
            (folder / "solution.py").write_text('"""a point."""\nX, Y = 0.5, 0.5\n')
            write_skill(folder / ".agents", "grid-first", "Use when a landscape is unknown.")
            return AgentResult(ok=True, session_id="s")

    register_agent("scripted", Scripted)
    outcome = hc.run(
        "fitness-landscape", agent="scripted", max_evaluations=1, learning=False, holdout=False,
        config=config, log=lambda *_: None,
    )
    journal = Journal(outcome.search_dir / "journal.jsonl")
    drafted = next(c for c in journal.candidates.values() if c.operator != "baseline")
    assert [s["name"] for s in drafted.agent_context["skills"]] == ["dst-gaps", "pinball-loss", "profile-first"]
    assert drafted.agent_context["skills_added"] == ["grid-first"]
    made = agent_context.skills(config)["grid-first"].record["created_by"]
    assert made["candidate"] == drafted.candidate_id and made["agent"] == "scripted"
    assert made["run"] == outcome.run_dir.name
