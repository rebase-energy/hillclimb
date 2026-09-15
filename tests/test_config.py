import os
from pathlib import Path

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


def test_unknown_backend_auth_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="unknown backend_auth"):
        Config(backend_auth="open-router")
