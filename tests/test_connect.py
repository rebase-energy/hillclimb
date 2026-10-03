from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from hillclimb import cli, connect
from hillclimb.config import Config

INIT_SHAPED = """\
# hillclimb config — this file marks the hillclimb dir.

model: sonnet
# agent: claude-code

# routing:
#   draft: {agent: codex}
"""


import pytest


@pytest.fixture(autouse=True)
def _no_pinned_dir(tmp_path: Path, monkeypatch):
    """Each test decides its own hillclimb dir (conftest already points the
    user level at tmp_path/xdg); the machine cache — where `connect` leaves
    its record — is the test's too, so no test marks the developer's
    machine as connected."""
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    monkeypatch.setenv("HILLCLIMB_CACHE_DIR", str(tmp_path / "cache"))


def _user_config(tmp_path: Path) -> Path:
    return tmp_path / "xdg" / "hillclimb" / "config.yaml"


# --------------------------------------------------------------- config.yaml


def test_apply_config_defaults_uncomments_in_place_and_keeps_comments():
    out = connect.apply_config_defaults(
        INIT_SHAPED, {"agent": "codex", "agent_auth": "openrouter"}
    )
    lines = out.splitlines()
    assert "agent: codex" in lines
    # the commented template line became the setting, where its comment is
    assert lines.index("agent: codex") == 3
    # a key with nowhere to go lands next to the one written just before it
    assert lines[4] == "agent_auth: openrouter"
    assert lines[0].startswith("# hillclimb config")
    assert "#   draft: {agent: codex}" in out


def test_apply_config_defaults_replaces_existing_and_ignores_nested_keys():
    text = "agent: claude-code\nrouting:\n  draft:\n    agent: codex\n"
    out = connect.apply_config_defaults(text, {"agent": "pi"})
    assert out == "agent: pi\nrouting:\n  draft:\n    agent: codex\n"


def test_apply_config_defaults_on_an_empty_file():
    assert connect.apply_config_defaults("", {"agent": "pi"}) == "agent: pi\n"


def test_pins_agent_only_for_an_active_top_level_key():
    assert connect.pins_agent("agent: codex\n")
    assert not connect.pins_agent(INIT_SHAPED)
    assert not connect.pins_agent("routing:\n  draft:\n    agent: codex\n")


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


def test_env_file_is_the_user_level_one_unless_local(tmp_path: Path):
    """Keys go beside the user config by default — every folder reads that
    file under its own — and `local` picks this folder's: the `.env` beside
    hillclimb.yaml."""
    user_env = tmp_path / "xdg" / "hillclimb" / ".env"
    config = Config()
    config.hillclimb_dir = tmp_path / "hillclimb"
    config.hillclimb_dir.mkdir()
    assert connect.env_file(config) == user_env
    assert connect.env_file(Config()) == user_env
    assert connect.env_file(config, local=True) == config.hillclimb_dir / ".env"
    assert connect.env_file(Config(), local=True) is None


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
    config = Config(agent="codex", agent_auth="openrouter")
    assert connect.configured_auth(config, "codex") == "openrouter"
    assert connect.configured_auth(config, "claude") is None
    assert connect.configured_auth(config, "openrouter") == "openrouter"

    routed = Config.model_validate(
        {
            "agent": "claude-code",
            "routing": {"improve": {"agent": "codex", "agent_auth": "openrouter"}},
        }
    )
    assert routed.model_dump()  # the route validates
    assert connect.configured_auth(routed, "codex") == "openrouter"
    assert connect.configured_auth(routed, "claude") == "subscription"


def test_check_rejects_a_route_a_agent_does_not_implement():
    status = connect.check("claude", "openrouter")
    assert status.state == "unsupported" and not status.ok


def test_a_working_login_is_logged_in_until_connect_marks_it_ready(monkeypatch, tmp_path):
    """The three states: `logged-out` (the agent's own login is missing),
    `logged-in` (it works, hillclimb has not connected it), `ready` (both).
    The mark is `connect`'s record under the machine cache, and for codex
    the staged home must be there too."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        connect, "_check_claude", lambda auth: connect.Status("claude", auth, "ready", "logged in")
    )
    status = connect.check("claude", "subscription")
    assert status.state == "logged-in" and status.ok and not status.connected
    assert status.fix == "hillclimb connect claude" and status.detail == "logged in"
    connect.mark_connected("claude", "subscription", "sonnet")
    status = connect.check("claude", "subscription")
    assert status.state == "ready" and status.connected and status.fix == ""

    monkeypatch.setattr(
        connect, "_check_codex", lambda auth: connect.Status("codex", auth, "ready", "Logged in using ChatGPT")
    )
    connect.mark_connected("codex", "subscription", None)
    assert connect.check("codex", "subscription").state == "logged-in"  # no staged home yet
    (tmp_path / ".cache" / "hillclimb" / "codex-home" / "subscription").mkdir(parents=True)
    assert connect.check("codex", "subscription").state == "ready"
    assert connect.remove_staged("codex")  # what `disconnect` does
    assert connect.check("codex", "subscription").state == "logged-in"

    # a missing login is logged-out whatever the record says
    monkeypatch.setattr(
        connect, "_check_claude",
        lambda auth: connect.Status("claude", auth, "logged-out", "not logged in", "claude auth login"),
    )
    assert connect.check("claude", "subscription").state == "logged-out"

    # the OpenRouter route: no-key / key-set / ready
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    monkeypatch.setattr(
        "hillclimb.agents.openrouter.key_info", lambda key=None, timeout=10: {"label": "hill", "usage": 0.0, "limit": None}
    )
    assert connect.check("openrouter", "openrouter").state == "key-set"
    connect.mark_connected("openrouter", "openrouter", None)
    assert connect.check("openrouter", "openrouter").state == "ready"


def test_status_rows_marks_the_configured_agent(monkeypatch):
    monkeypatch.setattr(
        connect, "check", lambda target, auth: connect.Status(target, auth, "ready", "fine")
    )
    config = Config(agent="codex")
    rows = connect.status_rows(config)
    assert [status.target for status, _ in rows] == list(connect.TARGETS)
    assert [target for (status, is_default) in rows if is_default for target in [status.target]] == [
        "codex"
    ]


# ---------------------------------------------------------------------- CLI


def _hillclimb_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "hillclimb"
    folder.mkdir()
    (folder / "hillclimb.yaml").write_text(INIT_SHAPED)
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


def test_connect_a_agent_checks_stages_and_pins(monkeypatch, tmp_path):
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
    # connect leaves its mark: the record that turns logged-in into ready
    assert (connect.record_dir("codex", "subscription") / "connected.json").exists()
    # the defaults land at the user level — every folder on the machine —
    # and the folder's own config.yaml is left as `init` wrote it
    assert _user_config(tmp_path).read_text() == "agent: codex\nagent_auth: subscription\n"
    assert (folder / "hillclimb.yaml").read_text() == INIT_SHAPED

    # `--local` pins this folder instead: the override for it alone
    result = CliRunner().invoke(cli.app, ["connect", "claude", "--no-probe", "--local"])
    assert result.exit_code == 0, result.output
    text = (folder / "hillclimb.yaml").read_text()
    assert "agent: claude-code" in text and "agent_auth: subscription" in text
    assert _user_config(tmp_path).read_text() == "agent: codex\nagent_auth: subscription\n"


def test_connect_leaves_a_config_that_already_pins_a_agent(monkeypatch, tmp_path):
    folder = _hillclimb_dir(tmp_path)
    user_config = _user_config(tmp_path)
    user_config.parent.mkdir(parents=True)
    user_config.write_text("agent: claude-code\n")
    monkeypatch.setenv("HILLCLIMB_DIR", str(folder))
    monkeypatch.setattr(
        connect, "check", lambda target, auth: connect.Status(target, auth, "ready", "fine")
    )
    monkeypatch.setattr(connect, "import_credentials", lambda *a, **k: None)

    result = CliRunner().invoke(cli.app, ["connect", "codex", "--no-probe"])
    assert result.exit_code == 0, result.output
    assert user_config.read_text() == "agent: claude-code\n"
    assert "already pins a coding agent" in result.output

    result = CliRunner().invoke(cli.app, ["connect", "codex", "--no-probe", "--default"])
    assert result.exit_code == 0, result.output
    assert user_config.read_text() == "agent: codex\nagent_auth: subscription\n"

    # a folder that pins its own agent keeps overriding the user default,
    # and connect says so rather than leaving the reader to wonder
    (folder / "hillclimb.yaml").write_text("agent: pi\n")
    result = CliRunner().invoke(cli.app, ["connect", "codex", "--no-probe", "--default"])
    assert result.exit_code == 0, result.output
    assert "keeps overriding the user default" in result.output
    assert (folder / "hillclimb.yaml").read_text() == "agent: pi\n"


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
            connect.Status("claude", "subscription", "logged-in", "logged in", "hillclimb connect claude"),
            connect.Status("claude", "subscription", "ready", "logged in"),  # after the mark
        ]
    )
    monkeypatch.setattr(connect, "check", lambda target, auth: next(states))
    logins: list[str] = []
    monkeypatch.setattr(connect, "run_login", lambda target: logins.append(target) or 0)

    result = CliRunner().invoke(cli.app, ["connect", "claude", "--no-probe"])
    assert result.exit_code == 0, result.output
    assert logins == ["claude"]
    assert "ready" in result.output


def test_connect_fails_when_the_ping_fails(monkeypatch, tmp_path):
    from hillclimb.agents.base import AgentResult

    folder = _hillclimb_dir(tmp_path)
    monkeypatch.setenv("HILLCLIMB_DIR", str(folder))
    monkeypatch.setattr(
        connect, "check", lambda target, auth: connect.Status(target, auth, "ready", "fine")
    )
    monkeypatch.setattr(connect, "import_credentials", lambda *a, **k: None)
    monkeypatch.setattr(
        connect,
        "ping",
        lambda *a, **k: AgentResult(ok=False, error_kind="error", error_message="model not supported"),
    )
    result = CliRunner().invoke(cli.app, ["connect", "codex", "--model", "gpt-5"])
    assert result.exit_code == 1
    assert "model not supported" in result.output
    # a route that does not work must not become the default
    assert not connect.pins_agent((folder / "hillclimb.yaml").read_text())


def test_connect_openrouter_validates_before_storing(monkeypatch, tmp_path):
    folder = _hillclimb_dir(tmp_path)
    monkeypatch.setenv("HILLCLIMB_DIR", str(folder))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "hillclimb.agents.openrouter.key_info",
        lambda key=None, timeout=10: {"label": "hill", "usage": 0.0, "limit": None},
    )
    result = CliRunner().invoke(cli.app, ["connect", "openrouter", "--key", "sk-or-test"])
    assert result.exit_code == 0, result.output
    # stored at the user level, for every folder; the folder gets none
    assert (tmp_path / "xdg" / "hillclimb" / ".env").read_text() == "OPENROUTER_API_KEY=sk-or-test\n"
    assert not (folder / ".env").exists()
    result = CliRunner().invoke(cli.app, ["connect", "openrouter", "--key", "sk-or-local", "--local"])
    assert result.exit_code == 0, result.output
    assert (folder / ".env").read_text() == "OPENROUTER_API_KEY=sk-or-local\n"


def test_connect_openrouter_stores_nothing_when_the_key_is_refused(monkeypatch, tmp_path):
    from hillclimb.agents.openrouter import OpenRouterError

    folder = _hillclimb_dir(tmp_path)
    monkeypatch.setenv("HILLCLIMB_DIR", str(folder))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def refuse(key=None, timeout=10):
        raise OpenRouterError("OpenRouter key HTTP 401: User not found.")

    monkeypatch.setattr("hillclimb.agents.openrouter.key_info", refuse)
    result = CliRunner().invoke(cli.app, ["connect", "openrouter", "--key", "nope"])
    assert result.exit_code == 1
    assert "401" in result.output
    assert not (folder / ".env").exists()


def test_connect_accepts_user_as_the_spelled_out_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HILLCLIMB_DIR", str(_hillclimb_dir(tmp_path)))
    monkeypatch.setattr(
        connect, "check", lambda target, auth: connect.Status(target, auth, "ready", "fine")
    )
    monkeypatch.setattr(connect, "import_credentials", lambda *a, **k: None)
    result = CliRunner().invoke(cli.app, ["connect", "claude", "--no-probe", "--user"])
    assert result.exit_code == 0, result.output
    assert _user_config(tmp_path).read_text() == "agent: claude-code\nagent_auth: subscription\n"


def test_connect_works_before_init(monkeypatch, tmp_path):
    """`hillclimb connect` then `hillclimb init` is the README's order: with
    no hillclimb dir the defaults still land at the user level, and the
    listing says where; only `--local` needs a folder."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        connect, "check", lambda target, auth: connect.Status(target, auth, "ready", "fine")
    )
    monkeypatch.setattr(connect, "import_credentials", lambda *a, **k: None)

    listing = CliRunner().invoke(cli.app, ["connect"])
    assert listing.exit_code == 0, listing.output
    assert "no hillclimb dir here" in listing.output and "config.yaml" in listing.output

    pinned = CliRunner().invoke(cli.app, ["connect", "claude", "--no-probe"])
    assert pinned.exit_code == 0, pinned.output
    assert _user_config(tmp_path).read_text() == "agent: claude-code\nagent_auth: subscription\n"
    # the user default is what a later folder runs with, and its listing marks it
    listing = CliRunner().invoke(cli.app, ["connect"])
    assert "●" in listing.output

    refused = CliRunner().invoke(cli.app, ["connect", "codex", "--no-probe", "--local"])
    assert refused.exit_code == 1
    assert "hillclimb init" in refused.output
    assert _user_config(tmp_path).read_text() == "agent: claude-code\nagent_auth: subscription\n"

    # the key has a user-level home too
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "hillclimb.agents.openrouter.key_info",
        lambda key=None, timeout=10: {"label": "hill", "usage": 0.0, "limit": None},
    )
    stored = CliRunner().invoke(cli.app, ["connect", "openrouter", "--key", "sk-or-test"])
    assert stored.exit_code == 0, stored.output
    assert (tmp_path / "xdg" / "hillclimb" / ".env").read_text() == "OPENROUTER_API_KEY=sk-or-test\n"


# ------------------------------------------------------------- disconnect


def test_unpin_config_defaults_comments_out_only_that_targets_lines():
    pinned = "model: sonnet\nagent: codex\nagent_auth: subscription\n# routing:\n#   draft: {agent: codex}\n"
    assert connect.unpin_config_defaults(pinned, "codex") == (
        "model: sonnet\n# agent: codex\n# agent_auth: subscription\n# routing:\n#   draft: {agent: codex}\n"
    )
    # another target's pin is not this disconnect's to remove
    assert connect.unpin_config_defaults(pinned, "claude") == pinned
    # the openrouter route unpins only the auth, the agent stays
    routed = "agent: codex\nagent_auth: openrouter\n"
    assert connect.unpin_config_defaults(routed, "openrouter") == "agent: codex\n# agent_auth: openrouter\n"
    assert connect.unpin_config_defaults("", "codex") == ""


def test_remove_env_key_drops_only_that_line():
    text = "# keys\nOPENROUTER_API_KEY=sk-or\nHF_TOKEN=abc\n"
    assert connect.remove_env_key(text, "OPENROUTER_API_KEY") == "# keys\nHF_TOKEN=abc\n"
    assert connect.remove_env_key("OPENROUTER_API_KEY=sk\n", "OPENROUTER_API_KEY") == ""


def test_staged_homes_lists_the_cache_dirs_connect_made(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HILLCLIMB_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    cache = tmp_path / ".cache" / "hillclimb"
    for d in ("codex-home/subscription", "codex-home/openrouter", "pi-home/subscription",
              "connect/codex-subscription", "connect/claude-code-subscription", "connect/pi-openrouter"):
        (cache / d).mkdir(parents=True)
    names = lambda t: [str(p.relative_to(cache)) for p in connect.staged_homes(t)]
    assert names("codex") == ["codex-home/openrouter", "codex-home/subscription", "connect/codex-subscription"]
    assert names("claude") == ["connect/claude-code-subscription"]
    assert names("openrouter") == ["codex-home/openrouter", "connect/pi-openrouter"]
    assert connect.remove_staged("pi") == [cache / "pi-home" / "subscription", cache / "connect" / "pi-openrouter"]
    assert not (cache / "pi-home" / "subscription").exists()
    # the agent's own files are never touched
    assert (cache / "codex-home" / "subscription").exists()


def test_disconnect_unpins_and_forgets_but_never_logs_out(monkeypatch, tmp_path):
    """`disconnect codex` undoes `connect codex` on hillclimb's side only:
    the user-level pin is commented out in place and the staged homes go.
    The agent's own login is the person's, not hillclimb's — no logout is
    run, no agent CLI is even invoked, and the agent's own files stay."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HILLCLIMB_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    user_config = _user_config(tmp_path)
    user_config.parent.mkdir(parents=True)
    user_config.write_text("model: sonnet\nagent: codex\nagent_auth: subscription\n")
    home = tmp_path / ".cache" / "hillclimb" / "codex-home" / "subscription"
    home.mkdir(parents=True)
    (home / "auth.json").write_text("{}")
    own_login = tmp_path / ".codex" / "auth.json"
    own_login.parent.mkdir()
    own_login.write_text("{}")
    invoked = []
    monkeypatch.setattr(connect.subprocess, "call", lambda *a, **k: invoked.append(a) or 0)
    monkeypatch.setattr(connect.subprocess, "run", lambda *a, **k: invoked.append(a) or None)

    connect.mark_connected("codex", "subscription", None)
    assert connect.is_connected("codex", "subscription")

    result = CliRunner().invoke(cli.app, ["disconnect", "codex"])
    assert result.exit_code == 0, result.output
    assert user_config.read_text() == "model: sonnet\n# agent: codex\n# agent_auth: subscription\n"
    assert not home.exists()
    assert not connect.is_connected("codex", "subscription")  # ready -> logged-in
    assert own_login.read_text() == "{}"
    assert invoked == []
    assert "login on this machine is untouched" in result.output
    assert "hillclimb connect codex" in result.output
    assert "fix:" not in result.output

    # a target that is not pinned says so instead of editing anything
    result = CliRunner().invoke(cli.app, ["disconnect", "claude"])
    assert result.exit_code == 0, result.output
    assert "nothing to unpin" in result.output

    refused = CliRunner().invoke(cli.app, ["disconnect", "gemini"])
    assert refused.exit_code != 0


def test_disconnect_openrouter_drops_the_key(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    user_dir = tmp_path / "xdg" / "hillclimb"
    user_dir.mkdir(parents=True)
    (user_dir / "config.yaml").write_text("agent: codex\nagent_auth: openrouter\n")
    (user_dir / ".env").write_text("OPENROUTER_API_KEY=sk-or\nHF_TOKEN=abc\n")
    result = CliRunner().invoke(cli.app, ["disconnect", "openrouter"])
    assert result.exit_code == 0, result.output
    assert (user_dir / "config.yaml").read_text() == "agent: codex\n# agent_auth: openrouter\n"
    assert (user_dir / ".env").read_text() == "HF_TOKEN=abc\n"


def test_connect_codex_pings_the_clis_default_model_for_a_claude_alias(monkeypatch, tmp_path):
    """`connect codex` on a config whose `model` is `sonnet` (the shipped
    default) must not ping codex with a Claude alias: it pings the Codex
    CLI's default model, says so, and records what answered."""
    import json

    from hillclimb.agents.base import AgentResult

    monkeypatch.setenv("HILLCLIMB_DIR", str(_hillclimb_dir(tmp_path)))
    monkeypatch.setattr(
        connect, "check", lambda target, auth: connect.Status(target, auth, "ready", "fine")
    )
    monkeypatch.setattr(connect, "import_credentials", lambda *a, **k: None)
    pinged = []

    def fake_ping(agent, auth, model, *, models_file=None):
        pinged.append((agent, auth, model))
        return AgentResult(ok=True, model_id="gpt-5-codex", total_tokens=12, duration_s=1.0)

    monkeypatch.setattr(connect, "ping", fake_ping)
    result = CliRunner().invoke(cli.app, ["connect", "codex"])
    assert result.exit_code == 0, result.output
    assert pinged == [("codex", "subscription", "sonnet")]  # the agent omits it, as in a search
    assert "Codex CLI's default model" in result.output and "ping ok" in result.output
    record = json.loads((connect.record_dir("codex", "subscription") / "connected.json").read_text())
    assert record["model"] == "gpt-5-codex"

    # an explicit codex model is pinged as given
    result = CliRunner().invoke(cli.app, ["connect", "codex", "--model", "gpt-5"])
    assert result.exit_code == 0, result.output
    assert pinged[-1] == ("codex", "subscription", "gpt-5")
