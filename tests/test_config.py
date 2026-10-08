import os
from pathlib import Path

import pytest

from hillclimb.config import Config
from tests.catalog_fixture import GEPA, GREEDY, OPENEVOLVE, block, class_ref, pin


def test_defaults_load():
    config = Config.load()
    assert config.agent == "claude-code"
    assert config.climber is None  # the engine ships no climber: the folder names one (`hillclimb climber get greedy`)


def test_overrides():
    config = Config.load(agent="dummy", model="opus", **{"budget.total_s": 60})
    assert config.agent == "dummy"
    assert config.model == "opus"
    assert config.budget.total_s == 60


def test_none_overrides_ignored():
    config = Config.load(agent=None)
    assert config.agent == "claude-code"


def test_missing_file_uses_defaults(tmp_path: Path):
    config = Config.load(path=tmp_path / "nope.yaml")
    assert config.climber is None and config.concurrency.parallel_agents == 1


def test_policy_and_routing_defaults():
    config = Config()
    assert config.climber is None
    assert config.routing == {}


def test_routing_block_round_trip(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        f"""
routing:
  draft: {{agent: claude-code, model: opus-4.8}}
  improve: {{model: haiku}}
search:
  policy: {GREEDY}
  policy_params: {{beam: 3}}
"""
    )
    config = Config.load(path=cfg_file)
    assert config.routing["draft"].agent == "claude-code"
    assert config.routing["draft"].model == "opus-4.8"
    assert config.routing["improve"].agent is None  # inherits global agent
    assert config.routing["improve"].model == "haiku"
    assert config.climber.operator_policy == class_ref("greedy", "Greedy")
    assert config.climber.params == {"beam": 3}


def test_policy_dotted_override():
    config = Config.load(**{"search.policy": str(OPENEVOLVE)})
    assert config.climber.label == "openevolve" and config.climber.selector_policy == class_ref("openevolve", "MapElites")


def test_subscription_env_strips_api_key(monkeypatch):
    from hillclimb.agents.claude_code import subscription_env

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "keep-me")
    env = subscription_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "keep-me"


def test_dotenv_beside_config_is_loaded(tmp_path: Path, monkeypatch):
    hillclimb_dir = tmp_path / "hillclimb"
    hillclimb_dir.mkdir()
    (hillclimb_dir / "hillclimb.yaml").write_text("")
    (hillclimb_dir / ".env").write_text("# a comment\n\nOPENROUTER_API_KEY=sk-or-test\nQUOTED='sk-quoted'\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
    monkeypatch.chdir(hillclimb_dir)

    Config.load()

    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-test"
    assert os.environ["QUOTED"] == "sk-quoted"


def test_dotenv_never_overrides_a_real_env_var(tmp_path: Path, monkeypatch):
    hillclimb_dir = tmp_path / "hillclimb"
    hillclimb_dir.mkdir()
    (hillclimb_dir / "hillclimb.yaml").write_text("")
    (hillclimb_dir / ".env").write_text("OPENROUTER_API_KEY=from-file\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-shell")
    monkeypatch.chdir(hillclimb_dir)

    Config.load()

    assert os.environ["OPENROUTER_API_KEY"] == "from-shell"


def test_openrouter_auth_requires_the_codex_agent():
    import pytest

    with pytest.raises(ValueError, match="needs agent: codex"):
        Config(agent="claude-code", agent_auth="openrouter")
    with pytest.raises(ValueError, match="routing.draft"):
        Config(
            agent="codex",
            agent_auth="openrouter",
            routing={"draft": {"agent": "claude-code"}},
        )
    # a route that names its own auth is fine
    Config(
        agent="codex",
        agent_auth="openrouter",
        routing={"draft": {"agent": "claude-code", "agent_auth": "subscription"}},
    )
    # pi implements OpenRouter natively as well.
    Config(agent="pi", agent_auth="openrouter")


def test_unknown_agent_auth_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="unknown agent_auth"):
        Config(agent_auth="open-router")


def test_sampling_requires_pi_and_rejects_route_typos():
    import pytest

    Config(
        agent="pi",
        routing={"draft": {"sampling": {"temperature": 0.9, "top_p": 0.95}}},
    )
    Config(
        agent="claude-code",
        routing={"default": {"agent": "pi"}, "draft": {"sampling": {"temperature": 0.9}}},
    )
    with pytest.raises(ValueError, match="sampling needs agent: pi"):
        Config(routing={"draft": {"sampling": {"temperature": 0.9}}})
    with pytest.raises(ValueError, match="samplids"):
        Config(routing={"draft": {"agent": "pi", "samplids": {"temperature": 0.9}}})


def test_sampling_dotted_override_is_revalidated():
    config = Config(agent="pi")
    config.apply_overrides({"routing.draft.sampling.temperature": 0.7})

    assert config.routing["draft"].sampling == {"temperature": 0.7}

    config = Config()
    with pytest.raises(ValueError, match="sampling needs agent: pi"):
        config.apply_overrides({"routing.draft.sampling.temperature": 0.7})


def test_pi_models_file_resolves_from_the_hillclimb_dir(tmp_path: Path, monkeypatch):
    hillclimb_dir = tmp_path / "hillclimb"
    hillclimb_dir.mkdir()
    (hillclimb_dir / "hillclimb.yaml").write_text("pi:\n  models_file: models.json\n")
    monkeypatch.chdir(hillclimb_dir)

    config = Config.load()

    assert config.pi.models_file == hillclimb_dir / "models.json"


def test_sampling_validation_checks_inherited_routes_and_action_override():
    from hillclimb.modules.policies.base import Route
    from hillclimb.harness.routing import Router

    with pytest.raises(ValueError, match="routing.improve: sampling"):
        Config(routing={
            "default": {"agent": "pi", "sampling": {"temperature": 0.8}},
            "improve": {"agent": "codex"},
        })
    config = Config(routing={
        "default": {"agent": "pi", "sampling": {"temperature": 0.8}},
        "improve": {"agent": "codex", "sampling": {}},
    })
    assert Router(config).resolve("improve").sampling == {}
    with pytest.raises(ValueError, match="sampling needs agent"):
        Router(config).resolve("draft", Route(agent="codex"))


def test_sampling_overrides_are_atomic_and_keep_integer_parameters():
    from hillclimb.config import RouteConfig

    config = Config(agent="pi", routing={"draft": RouteConfig()})
    config.apply_overrides({"routing.draft.sampling.top_k": 40})
    assert type(config.routing["draft"].sampling["top_k"]) is int
    with pytest.raises(ValueError):
        config.apply_overrides({"agent": "codex"})
    assert config.agent == "pi"
    with pytest.raises(ValueError, match="finite"):
        config.apply_overrides({"routing.draft.sampling.temperature": float("nan")})
    assert config.routing["draft"].sampling == {"top_k": 40}


def test_explicit_config_loads_dotenv_and_local_models_path(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / "config.yaml").write_text("pi: {models_file: models.json}\n")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=local-test\n")
    config = Config.load(path=tmp_path / "config.yaml")
    assert config.pi.models_file == tmp_path / "models.json"
    assert os.environ["OPENROUTER_API_KEY"] == "local-test"


# --- the config surface: one `climber:` block, harness-only everything else ---


def test_the_climber_block_is_the_climber():
    """hillclimb.yaml's `climber:` is the same block a run spec entry takes:
    it DEFINES the folder's climber. A bare name is a preset."""
    from hillclimb.config import parse_set_overrides

    assert Config.model_validate({"climber": str(OPENEVOLVE)}).climber.selector_policy == class_ref("openevolve", "MapElites")
    assert Config.model_validate({"climber": str(GEPA)}).climber.block()["loop"].endswith("gepa/loop.py:GepaLoop")
    config = Config.model_validate(
        {"climber": {"operator_policy": class_ref("greedy", "Greedy"), "params": {"num_drafts": 5}, "tuner": "optuna", "memory": "none",
                     "operators": ["draft", "improve"], "operator_params": {"draft": {"retrieval": False}}}}
    )
    # a schedule knob under `params` is the selector's (the only way to write it before 0.7)
    assert (config.climber.params, config.climber.tuner, config.climber.memory) == ({}, "optuna", "none")
    assert config.climber.selector_params == {"num_drafts": 5}
    assert config.climber.operators == ["draft", "improve"]
    # `--set` edits the block's fields; one operator's params are addressed by its name
    config.apply_overrides(parse_set_overrides(
        ["climber.params.num_drafts=1", "climber.operators.improve.ablation=false", "climber.tuner_params.seed=3"]
    ))
    # a schedule knob set as `climber.params.X` (the old spelling) lands where the selector reads it
    assert config.climber.selector_params == {"num_drafts": 1} and config.climber.tuner_params == {"seed": 3}
    assert config.climber.operator_params == {"draft": {"retrieval": False}, "improve": {"ablation": False}}
    assert config.climber.operators == ["draft", "improve"]  # which operators run is untouched
    # naming a climber replaces the block — whole, so one policy's params never reach another
    config.apply_overrides(parse_set_overrides([f"climber={GEPA}", "climber.params.max_metric_calls=9"]))
    assert config.climber.block() == {**block("gepa"), "params": {"max_metric_calls": 9}}
    # a block, inline, as a child engine receives it
    config.apply_overrides(parse_set_overrides(['climber={"operator_policy": "mine.py", "params": {"k": 2}}']))
    assert (config.climber.operator_policy, config.climber.loop, config.climber.params) == ("mine.py", None, {"k": 2})
    # a climber has a policy or a loop: naming one drops the other
    config.apply_overrides(parse_set_overrides(["climber.loop=gepa"]))
    assert (config.climber.operator_policy, config.climber.loop) == (None, "gepa")
    with pytest.raises(ValueError):
        Config.model_validate({"climber": {"operator_policy": class_ref("greedy", "Greedy"), "polcy": "x"}})  # a typo is not silently ignored
    with pytest.raises(ValueError, match="Unknown climber: nope"):
        Config().apply_overrides({"climber": "nope"})
    # the memory kind: `files` (the YAML under hillclimb/knowledge/), its pre-0.4
    # spelling `knowledge-graph` mapped on read; a memory is named like any module,
    # so an unknown one is refused by name when the climber is resolved
    assert Config.model_validate({"climber": {"operator_policy": class_ref("greedy", "Greedy"), "memory": "knowledge-graph"}}).climber.memory == "files"
    config = Config()
    pin(config)  # a knob needs a climber to land on
    config.apply_overrides(parse_set_overrides(["climber.memory=knowledge-graph"]))
    assert config.climber.memory == "files"
    from hillclimb.harness.glue import search_climber

    with pytest.raises(ValueError, match="unknown memory 'sqlite' .available: files, none"):
        search_climber(Config.model_validate({"climber": {"operator_policy": class_ref("greedy", "Greedy"), "memory": "sqlite"}})).memory()


def test_the_0_5_climber_block_still_loads():
    """0.4/0.5 wrote `climber: {ref: <name>, ...}`: a reference plus what the
    user laid over it (`operators` a mapping by name, None for "the
    climber's"). Read as the block it meant."""
    from hillclimb.config import parse_set_overrides

    config = Config.model_validate(
        {"climber": {"ref": str(OPENEVOLVE), "params": {"num_islands": 3}, "tuner": None, "memory": None, "graph": None,
                     "tuner_params": {"seed": 2}, "operators": {"draft": {"retrieval": False}}}}
    )
    # ...and 0.5's openevolve kept MAP-Elites' settings among its params: they are the selector's
    assert config.climber.block() == Config.model_validate({"climber": {
        **block("openevolve"),  # the file's whole block (its operators too)...
        "params": {"num_islands": 3},  # ...named by file, so the 0.5 openevolve folding (by name) no longer applies
        "tuner_params": {"seed": 2}, "operator_params": {"draft": {"retrieval": False}},
    }}).climber.block()
    config.apply_overrides(parse_set_overrides([f"climber.ref={GEPA}"]))  # the 0.5 way to name one
    assert config.climber.loop.endswith("gepa/loop.py:GepaLoop") and config.climber.params == {}
    with pytest.raises(ValueError, match="Unknown climber: climbers/mine .*a folder holding policy.py"):
        Config.model_validate({"climber": {"ref": "climbers/mine"}})  # a directory: show it as a block


def test_the_climber_is_a_run_default_and_the_user_config_keeps_only_agent_and_model(tmp_path, monkeypatch):
    """The folder's climber lives in runs/config.yaml; the user config may
    keep a personal agent/model under it, and nothing else of a run's."""
    from hillclimb.config import ConfigError

    user = tmp_path / "user" / "config.yaml"
    user.parent.mkdir()
    user.write_text("model: opus\nagent: codex\n")
    folder = tmp_path / "proj"
    (folder / "runs").mkdir(parents=True)
    (folder / "hillclimb.yaml").write_text("")
    (folder / "runs" / "config.yaml").write_text(f"climber: {GEPA}\nmodel: sonnet\n")
    monkeypatch.setattr("hillclimb.config.user_config_path", lambda: user)
    monkeypatch.setattr("hillclimb.config.user_env_path", lambda: tmp_path / "user" / ".env")
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    monkeypatch.chdir(folder)
    config = Config.load()
    assert config.climber.block() == block("gepa")
    assert config.model == "sonnet" and config.agent == "codex"  # the run defaults win; the rest merges
    user.write_text("climber: {operator_policy: mine.py}\n")
    with pytest.raises(ConfigError, match="`climber:` is a run default now"):
        Config.load()
    user.write_text("")
    (folder / "hillclimb.yaml").write_text("budget: {total_s: 60}\n")
    with pytest.raises(ConfigError, match="hillclimb.yaml: `budget:` is a run default now: move it to runs/config.yaml"):
        Config.load()
    (folder / "hillclimb.yaml").write_text("concurrency: {machine_max_agents: 2}\n")  # the machine's, not a run's
    assert Config.load().concurrency.machine_max_agents == 2


def test_a_config_file_written_for_0_3_still_loads():
    old = {
        "model": "opus",
        "search": {"policy": str(OPENEVOLVE), "policy_params": {"population_size": 50}, "num_drafts": 2,
                   "tuner": "optuna", "parallel_agents": 3, "n_trials": 4, "noise_k": 2.0},
        "ensemble": {"enabled": False, "top_k": 4},
        "operators": {"draft_retrieval": False, "knowledge_tool": False},
    }
    config = Config.model_validate(old)
    assert (config.climber.label, config.climber.selector_policy, config.climber.tuner) == ("openevolve", class_ref("openevolve", "MapElites"), "optuna")
    assert config.climber.params == {"population_size": 50}  # named by file: no openevolve-by-name folding
    # the schedule's knobs: the selector's
    assert config.climber.selector_params == {"num_drafts": 2, "ensemble": False, "ensemble_top_k": 4}
    assert config.climber.operator_params == {"draft": {"retrieval": False}}
    assert config.learning.tool is False
    assert (config.concurrency.parallel_agents, config.evaluation.n_replicates, config.evaluation.noise_k) == (3, 4, 2.0)
    assert not hasattr(config, "search") and not hasattr(config, "ensemble")


def test_legacy_set_overrides_keep_working_and_removed_keys_say_what_to_do():
    from hillclimb.config import current_setting, parse_set_overrides

    config = Config()
    config.apply_overrides(parse_set_overrides(
        [f"search.policy={OPENEVOLVE}", "search.policy_params.population_size=9", "ensemble.top_k=5",
         "search.parallel_agents=2", "operators.improve_ablation=false"]
    ))
    assert config.climber.label == "openevolve"
    assert config.climber.params == {"population_size": 9}
    assert config.climber.selector_params == {"ensemble_top_k": 5}
    assert config.concurrency.parallel_agents == 2
    assert config.climber.operator_params == {"improve": {"ablation": False}}
    assert current_setting("budget.total_s") == "budget.total_s"  # today's keys pass through
    with pytest.raises(KeyError, match="prompts belong to the climber"):
        config.apply_overrides({"paths.prompts_dir": "x"})
    with pytest.raises(KeyError, match="climber: .prompts: prompts/."):
        Config.model_validate({"paths": {"prompts_dir": "hillclimb/prompts"}})


def test_the_blocks_operator_params_reach_the_operators():
    from hillclimb.harness.glue import build_operators, effective_memory

    config = Config()
    pin(config)
    assert build_operators(config).get("draft").params == {}  # the operator's own defaults
    config.climber.operator_params = {"draft": {"retrieval": False}}
    assert build_operators(config).get("draft").params == {"retrieval": False}
    assert build_operators(config).get("improve").params == {}
    config.climber.operator_params = {"crossover": {"x": 1}}
    with pytest.raises(ValueError, match="this climber has no operator 'crossover'"):
        build_operators(config)
    # memory: the block's, and the master switch
    config = Config()
    pin(config)
    assert effective_memory(config) == "files"
    config.climber.memory = "none"
    assert effective_memory(config) == "none"
    config.climber.memory = "files"
    config.learning.enabled = False
    assert effective_memory(config) == "none"


def test_the_blocks_graph_module_is_used_and_reading_memory_survives_a_broken_climber(tmp_path):
    from hillclimb.harness.glue import build_graph_module, search_climber

    config = Config()
    pin(config)
    assert build_graph_module(config).name == "knowledge-graph"  # the default
    config.climber.memory_params["graph"] = "hillclimb.modules.memory.graph:KnowledgeGraphBuilder"
    assert build_graph_module(config).key == "hillclimb.modules.memory.graph:KnowledgeGraphBuilder"
    # outside a search, a climber that will not load falls back to the built-in:
    # `hillclimb knowledge …` must keep working whatever the block says
    for broken in ({"operator_policy": class_ref("greedy", "Greedy"), "graph": "nope"}, {"operator_policy": str(tmp_path / "missing.py")}):
        config = Config.model_validate({"climber": broken})
        notes = []
        assert build_graph_module(config, log=notes.append).name == "knowledge-graph"
        assert notes and "using knowledge-graph" in notes[0]
    # a search does not start on it
    with pytest.raises(ValueError, match="unknown graph module 'nope' \\(available: knowledge-graph"):
        search_climber(Config.model_validate({"climber": {"operator_policy": class_ref("greedy", "Greedy"), "graph": "nope"}})).graph_module()
