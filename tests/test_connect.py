from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from hillclimb import cli, connect
from hillclimb.config import Config

INIT_SHAPED = """\
# hillclimb config — this file marks the hillclimb dir.

model: sonnet
# backend: claude-code

# routing:
#   draft: {backend: codex}
"""


# --------------------------------------------------------------- config.yaml


def test_apply_config_defaults_uncomments_in_place_and_keeps_comments():
    out = connect.apply_config_defaults(
        INIT_SHAPED, {"backend": "codex", "backend_auth": "openrouter"}
    )
    lines = out.splitlines()
    assert "backend: codex" in lines
    # the commented template line became the setting, where its comment is
    assert lines.index("backend: codex") == 3
    # a key with nowhere to go lands next to the one written just before it
    assert lines[4] == "backend_auth: openrouter"
    assert lines[0].startswith("# hillclimb config")
    assert "#   draft: {backend: codex}" in out


def test_apply_config_defaults_replaces_existing_and_ignores_nested_keys():
    text = "backend: claude-code\nrouting:\n  draft:\n    backend: codex\n"
    out = connect.apply_config_defaults(text, {"backend": "pi"})
    assert out == "backend: pi\nrouting:\n  draft:\n    backend: codex\n"


def test_apply_config_defaults_on_an_empty_file():
    assert connect.apply_config_defaults("", {"backend": "pi"}) == "backend: pi\n"


def test_pins_backend_only_for_an_active_top_level_key():
    assert connect.pins_backend("backend: codex\n")
    assert not connect.pins_backend(INIT_SHAPED)
    assert not connect.pins_backend("routing:\n  draft:\n    backend: codex\n")


# ---------------------------------------------------------------------- .env


def test_upsert_env_adds_replaces_and_preserves():
    text = "# keys\nHF_TOKEN=abc\nOPENROUTER_API_KEY=old\n"
    out = connect.upsert_env(text, "OPENROUTER_API_KEY", "new")
    assert out == "# keys\nHF_TOKEN=abc\nOPENROUTER_API_KEY=new\n"
    assert connect.upsert_env("HF_TOKEN=abc\n", "OPENROUTER_API_KEY", "k").endswith(
        "OPENROUTER_API_KEY=k\n"
    )
    assert connect.upsert_env("export OPENROUTER_API_KEY=old\n", "OPENROUTER_API_KEY", "new") == (
        "OPENROUTER_API_KEY=new\n"
    )


def test_write_env_key_is_owner_only(tmp_path: Path):
    path = tmp_path / ".env"
    connect.write_env_key(path, "OPENROUTER_API_KEY", "k")
    assert path.read_text() == "OPENROUTER_API_KEY=k\n"
    assert path.stat().st_mode & 0o077 == 0


def test_env_file_prefers_the_hillclimb_dir_then_the_repo_root(tmp_path: Path):
    config = Config()
    config.hillclimb_dir = tmp_path / "hillclimb"
    config.hillclimb_dir.mkdir()
    assert connect.env_file(config) == config.hillclimb_dir / ".env"
    # Config.load reads the parent's .env when the folder has none; writing
    # anywhere else would store a key nothing loads
    (tmp_path / ".env").write_text("HF_TOKEN=abc\n")
    assert connect.env_file(config) == tmp_path / ".env"
    assert connect.env_file(Config()) is None


# ------------------------------------------------------- reading the CLIs


def test_parse_claude_status_reads_the_json():
    ok, detail = connect.parse_claude_status(
        0, json.dumps({"loggedIn": True, "authMethod": "claude.ai", "email": "a@b.c"})
    )
    assert ok and "claude.ai" in detail and "a@b.c" in detail
    assert connect.parse_claude_status(0, json.dumps({"loggedIn": False}))[0] is False


def test_parse_claude_status_flags_an_api_key_that_would_be_billed():
    ok, detail = connect.parse_claude_status(
        0, json.dumps({"loggedIn": True, "authMethod": "claude.ai", "apiKeySource": "ANTHROPIC_API_KEY"})
    )
    assert ok
    assert "ANTHROPIC_API_KEY is set and would be billed" in detail


def test_parse_claude_status_falls_back_when_there_is_no_json():
    assert connect.parse_claude_status(1, "Unknown command: auth")[0] is False
    assert connect.parse_claude_status(0, "all good")[0] is True


def test_parse_codex_status_reads_the_line_codex_puts_on_stderr():
    ok, detail = connect.parse_codex_status(0, "\nLogged in using ChatGPT\n")
    assert ok and detail == "Logged in using ChatGPT"
    assert connect.parse_codex_status(1, "Not logged in")[0] is False


def test_parse_pi_credentials_treats_an_empty_login_file_as_logged_out():
    assert connect.parse_pi_credentials("{}") == []
    assert connect.parse_pi_credentials('{"anthropic": {}, "openai": {}}') == ["anthropic", "openai"]
    assert connect.parse_pi_credentials("not json") == []


def test_describe_key():
    assert connect.describe_key({"label": "hill", "usage": 1.5, "limit": 10.0}) == (
        "hill, $1.50 used of $10.00"
    )
    assert connect.describe_key({"usage": 0, "limit": None}) == "key, $0.00 used"


# ------------------------------------------------------------------ routing


def test_configured_auth_follows_the_scalar_then_the_routes():
    config = Config(backend="codex", backend_auth="openrouter")
    assert connect.configured_auth(config, "codex") == "openrouter"
    assert connect.configured_auth(config, "claude") is None
    assert connect.configured_auth(config, "openrouter") == "openrouter"

    routed = Config.model_validate(
        {
            "backend": "claude-code",
            "routing": {"improve": {"backend": "codex", "backend_auth": "openrouter"}},
        }
    )
    assert routed.model_dump()  # the route validates
    assert connect.configured_auth(routed, "codex") == "openrouter"
    assert connect.configured_auth(routed, "claude") == "subscription"


def test_check_rejects_a_route_a_backend_does_not_implement():
    status = connect.check("claude", "openrouter")
    assert status.state == "unsupported" and not status.ok


def test_status_rows_marks_the_configured_backend(monkeypatch):
    monkeypatch.setattr(
        connect, "check", lambda target, auth: connect.Status(target, auth, "ready", "fine")
    )
    config = Config(backend="codex")
    rows = connect.status_rows(config)
    assert [status.target for status, _ in rows] == list(connect.TARGETS)
    assert [target for (status, is_default) in rows if is_default for target in [status.target]] == [
        "codex"
    ]


# ---------------------------------------------------------------------- CLI


def _hillclimb_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "hillclimb"
    folder.mkdir()
    (folder / "config.yaml").write_text(INIT_SHAPED)
    return folder


def test_connect_lists_every_target_as_json(monkeypatch, tmp_path):
    monkeypatch.setenv("HILLCLIMB_DIR", str(_hillclimb_dir(tmp_path)))
    monkeypatch.setattr(
        connect, "check", lambda target, auth: connect.Status(target, auth, "ready", "fine")
    )
    result = CliRunner().invoke(cli.app, ["connect", "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert [row["target"] for row in rows] == list(connect.TARGETS)
    assert [row["default"] for row in rows] == [True, False, False, False]


def test_connect_a_backend_checks_stages_and_pins(monkeypatch, tmp_path):
    folder = _hillclimb_dir(tmp_path)
    monkeypatch.setenv("HILLCLIMB_DIR", str(folder))
    monkeypatch.setattr(
        connect, "check", lambda target, auth: connect.Status(target, auth, "ready", "fine")
    )
    staged = []
    monkeypatch.setattr(
        connect,
        "import_credentials",
        lambda target, auth, models_file=None: staged.append((target, auth)) or tmp_path / "home",
    )
    result = CliRunner().invoke(cli.app, ["connect", "codex", "--no-probe"])
    assert result.exit_code == 0, result.output
    assert staged == [("codex", "subscription")]
    text = (folder / "config.yaml").read_text()
    assert "backend: codex" in text and "backend_auth: subscription" in text


def test_connect_leaves_a_config_that_already_pins_a_backend(monkeypatch, tmp_path):
    folder = _hillclimb_dir(tmp_path)
    (folder / "config.yaml").write_text("backend: claude-code\n")
    monkeypatch.setenv("HILLCLIMB_DIR", str(folder))
    monkeypatch.setattr(
        connect, "check", lambda target, auth: connect.Status(target, auth, "ready", "fine")
    )
    monkeypatch.setattr(connect, "import_credentials", lambda *a, **k: None)

    result = CliRunner().invoke(cli.app, ["connect", "codex", "--no-probe"])
    assert result.exit_code == 0, result.output
    assert (folder / "config.yaml").read_text() == "backend: claude-code\n"
    assert "already pins a backend" in result.output

    result = CliRunner().invoke(cli.app, ["connect", "codex", "--no-probe", "--default"])
    assert result.exit_code == 0, result.output
    assert (folder / "config.yaml").read_text() == "backend: codex\nbackend_auth: subscription\n"


def test_connect_stops_on_a_missing_cli(monkeypatch, tmp_path):
    monkeypatch.setenv("HILLCLIMB_DIR", str(_hillclimb_dir(tmp_path)))
    monkeypatch.setattr(
        connect,
        "check",
        lambda target, auth: connect.Status(target, auth, "missing-cli", "pi is not on PATH", "install pi"),
    )
    result = CliRunner().invoke(cli.app, ["connect", "pi", "--no-probe"])
    assert result.exit_code == 1
    assert "install pi" in result.output


def test_connect_runs_the_agents_own_login_when_logged_out(monkeypatch, tmp_path):
    monkeypatch.setenv("HILLCLIMB_DIR", str(_hillclimb_dir(tmp_path)))
    states = iter(
        [
            connect.Status("claude", "subscription", "logged-out", "not logged in", "claude auth login"),
            connect.Status("claude", "subscription", "ready", "logged in"),
        ]
    )
    monkeypatch.setattr(connect, "check", lambda target, auth: next(states))
    logins: list[str] = []
    monkeypatch.setattr(connect, "run_login", lambda target: logins.append(target) or 0)

    result = CliRunner().invoke(cli.app, ["connect", "claude", "--no-probe"])
    assert result.exit_code == 0, result.output
    assert logins == ["claude"]


def test_connect_fails_when_the_ping_fails(monkeypatch, tmp_path):
    from hillclimb.backends.base import OperatorResult

    folder = _hillclimb_dir(tmp_path)
    monkeypatch.setenv("HILLCLIMB_DIR", str(folder))
    monkeypatch.setattr(
        connect, "check", lambda target, auth: connect.Status(target, auth, "ready", "fine")
    )
    monkeypatch.setattr(connect, "import_credentials", lambda *a, **k: None)
    monkeypatch.setattr(
        connect,
        "ping",
        lambda *a, **k: OperatorResult(ok=False, error_kind="error", error_message="model not supported"),
    )
    result = CliRunner().invoke(cli.app, ["connect", "codex", "--model", "gpt-5"])
    assert result.exit_code == 1
    assert "model not supported" in result.output
    # a route that does not work must not become the default
    assert not connect.pins_backend((folder / "config.yaml").read_text())


def test_connect_openrouter_validates_before_storing(monkeypatch, tmp_path):
    folder = _hillclimb_dir(tmp_path)
    monkeypatch.setenv("HILLCLIMB_DIR", str(folder))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "hillclimb.openrouter.key_info",
        lambda key=None, timeout=10: {"label": "hill", "usage": 0.0, "limit": None},
    )
    result = CliRunner().invoke(cli.app, ["connect", "openrouter", "--key", "sk-or-test"])
    assert result.exit_code == 0, result.output
    assert (folder / ".env").read_text() == "OPENROUTER_API_KEY=sk-or-test\n"


def test_connect_openrouter_stores_nothing_when_the_key_is_refused(monkeypatch, tmp_path):
    from hillclimb.openrouter import OpenRouterError

    folder = _hillclimb_dir(tmp_path)
    monkeypatch.setenv("HILLCLIMB_DIR", str(folder))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def refuse(key=None, timeout=10):
        raise OpenRouterError("OpenRouter key HTTP 401: User not found.")

    monkeypatch.setattr("hillclimb.openrouter.key_info", refuse)
    result = CliRunner().invoke(cli.app, ["connect", "openrouter", "--key", "nope"])
    assert result.exit_code == 1
    assert "401" in result.output
    assert not (folder / ".env").exists()


def test_connect_writes_user_defaults_with_user_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("HILLCLIMB_DIR", str(_hillclimb_dir(tmp_path)))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        connect, "check", lambda target, auth: connect.Status(target, auth, "ready", "fine")
    )
    monkeypatch.setattr(connect, "import_credentials", lambda *a, **k: None)
    result = CliRunner().invoke(cli.app, ["connect", "claude", "--no-probe", "--user"])
    assert result.exit_code == 0, result.output
    user_config = tmp_path / "xdg" / "hillclimb" / "config.yaml"
    assert user_config.read_text() == "backend: claude-code\nbackend_auth: subscription\n"


def test_connect_works_without_a_hillclimb_dir(monkeypatch, tmp_path):
    """Checking a credential needs no hillclimb dir — only writing does."""
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    monkeypatch.delenv("HILLCLIMB_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        connect, "check", lambda target, auth: connect.Status(target, auth, "ready", "fine")
    )
    monkeypatch.setattr(connect, "import_credentials", lambda *a, **k: None)

    listing = CliRunner().invoke(cli.app, ["connect"])
    assert listing.exit_code == 0, listing.output
    assert "no hillclimb dir here" in listing.output

    # asking for the defaults anyway is an error, not a silent no-op
    refused = CliRunner().invoke(cli.app, ["connect", "claude", "--no-probe", "--default"])
    assert refused.exit_code == 1
    assert "hillclimb init" in refused.output

    # without --default the check still runs and reports
    quiet = CliRunner().invoke(cli.app, ["connect", "claude", "--no-probe"])
    assert quiet.exit_code == 0, quiet.output
