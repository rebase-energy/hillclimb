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
