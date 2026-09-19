import os
from pathlib import Path

import pytest

from hillclimb.config import Config


def test_defaults_load():
    config = Config.load()
    assert config.backend == "claude-code"
    assert config.search.num_drafts == 3


def test_overrides():
    config = Config.load(backend="dummy", model="opus", **{"budget.total_s": 60})
    assert config.backend == "dummy"
    assert config.model == "opus"
    assert config.budget.total_s == 60


def test_none_overrides_ignored():
    config = Config.load(backend=None)
    assert config.backend == "claude-code"


def test_missing_file_uses_defaults(tmp_path: Path):
    config = Config.load(path=tmp_path / "nope.yaml")
    assert config.search.max_debug_depth == 3


def test_policy_and_routing_defaults():
    config = Config()
    assert config.search.policy == "greedy"
    assert config.search.policy_params == {}
    assert config.routing == {}


def test_routing_block_round_trip(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
routing:
  draft: {backend: claude-code, model: opus-4.8}
  improve: {model: haiku}
search:
  policy: greedy
  policy_params: {beam: 3}
"""
    )
    config = Config.load(path=cfg_file)
    assert config.routing["draft"].backend == "claude-code"
    assert config.routing["draft"].model == "opus-4.8"
    assert config.routing["improve"].backend is None  # inherits global backend
    assert config.routing["improve"].model == "haiku"
    assert config.search.policy == "greedy"
    assert config.search.policy_params == {"beam": 3}


def test_policy_dotted_override():
    config = Config.load(**{"search.policy": "greedy"})
    assert config.search.policy == "greedy"


def test_subscription_env_strips_api_key(monkeypatch):
    from hillclimb.backends.claude_code import subscription_env

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "keep-me")
    env = subscription_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "keep-me"


def test_dotenv_beside_config_is_loaded(tmp_path: Path, monkeypatch):
    hillclimb_dir = tmp_path / "hillclimb"
    hillclimb_dir.mkdir()
    (hillclimb_dir / "config.yaml").write_text("model: sonnet\n")
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
    (hillclimb_dir / "config.yaml").write_text("model: sonnet\n")
    (hillclimb_dir / ".env").write_text("OPENROUTER_API_KEY=from-file\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-shell")
    monkeypatch.chdir(hillclimb_dir)

    Config.load()

    assert os.environ["OPENROUTER_API_KEY"] == "from-shell"


def test_dotenv_is_found_one_level_above_the_hillclimb_dir(tmp_path: Path, monkeypatch):
    hillclimb_dir = tmp_path / "hillclimb"
    hillclimb_dir.mkdir()
    (hillclimb_dir / "config.yaml").write_text("model: sonnet\n")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from-repo-root\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(hillclimb_dir)

    Config.load()

    assert os.environ["OPENROUTER_API_KEY"] == "from-repo-root"


def test_openrouter_auth_requires_the_codex_backend():
    import pytest

    with pytest.raises(ValueError, match="needs backend: codex"):
        Config(backend="claude-code", backend_auth="openrouter")
    with pytest.raises(ValueError, match="routing.draft"):
        Config(
            backend="codex",
            backend_auth="openrouter",
            routing={"draft": {"backend": "claude-code"}},
        )
    # a route that names its own auth is fine
    Config(
        backend="codex",
        backend_auth="openrouter",
        routing={"draft": {"backend": "claude-code", "backend_auth": "subscription"}},
    )
    # pi implements OpenRouter natively as well.
    Config(backend="pi", backend_auth="openrouter")


def test_unknown_backend_auth_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="unknown backend_auth"):
        Config(backend_auth="open-router")


def test_sampling_requires_pi_and_rejects_route_typos():
    import pytest

    Config(
        backend="pi",
        routing={"draft": {"sampling": {"temperature": 0.9, "top_p": 0.95}}},
    )
    Config(
        backend="claude-code",
        routing={"default": {"backend": "pi"}, "draft": {"sampling": {"temperature": 0.9}}},
    )
    with pytest.raises(ValueError, match="sampling needs backend: pi"):
        Config(routing={"draft": {"sampling": {"temperature": 0.9}}})
    with pytest.raises(ValueError, match="samplids"):
        Config(routing={"draft": {"backend": "pi", "samplids": {"temperature": 0.9}}})


def test_sampling_dotted_override_is_revalidated():
    config = Config(backend="pi")
    config.apply_overrides({"routing.draft.sampling.temperature": 0.7})

    assert config.routing["draft"].sampling == {"temperature": 0.7}

    config = Config()
    with pytest.raises(ValueError, match="sampling needs backend: pi"):
        config.apply_overrides({"routing.draft.sampling.temperature": 0.7})


def test_pi_models_file_resolves_from_project_root(tmp_path: Path, monkeypatch):
    hillclimb_dir = tmp_path / "hillclimb"
    hillclimb_dir.mkdir()
    (hillclimb_dir / "config.yaml").write_text("pi:\n  models_file: models.json\n")
    monkeypatch.chdir(hillclimb_dir)

    config = Config.load()

    assert config.pi.models_file == tmp_path / "models.json"


def test_sampling_validation_checks_inherited_routes_and_action_override():
    from hillclimb.policy import Route
    from hillclimb.routing import Router

    with pytest.raises(ValueError, match="routing.improve: sampling"):
        Config(routing={
            "default": {"backend": "pi", "sampling": {"temperature": 0.8}},
            "improve": {"backend": "codex"},
        })
    config = Config(routing={
        "default": {"backend": "pi", "sampling": {"temperature": 0.8}},
        "improve": {"backend": "codex", "sampling": {}},
    })
    assert Router(config).resolve("improve").sampling == {}
    with pytest.raises(ValueError, match="sampling needs backend"):
        Router(config).resolve("draft", Route(backend="codex"))


def test_sampling_overrides_are_atomic_and_keep_integer_parameters():
    from hillclimb.config import RouteConfig

    config = Config(backend="pi", routing={"draft": RouteConfig()})
    config.apply_overrides({"routing.draft.sampling.top_k": 40})
    assert type(config.routing["draft"].sampling["top_k"]) is int
    with pytest.raises(ValueError):
        config.apply_overrides({"backend": "codex"})
    assert config.backend == "pi"
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
