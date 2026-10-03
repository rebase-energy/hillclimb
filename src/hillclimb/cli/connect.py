"""`hillclimb connect [claude|codex|pi|openrouter]`: who runs the operators, and who pays."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from hillclimb.cli import common
from hillclimb.cli._app import HillclimbGroup, app
from hillclimb.config import Config

# ---------------------------------------------------------------- connect --

connect_app = typer.Typer(
    cls=HillclimbGroup,
    invoke_without_command=True,
    help="Connect the coding agents (claude, codex, pi) and the OpenRouter route that pays for them.",
)


app.add_typer(connect_app, name="connect")


_CONNECT_AUTH = typer.Option(
    None, "--auth", help="Who pays: subscription (default) | api-key | openrouter"
)


# claude-code has no OpenRouter route — offering it in the help would be a lie
_CONNECT_AUTH_CLAUDE = typer.Option(
    None, "--auth", help="Who pays: subscription (default) | api-key"
)


_CONNECT_MODEL = typer.Option(None, "--model", help="Model to ping with (default: the configured one)")


_CONNECT_PROBE = typer.Option(
    True, "--probe/--no-probe", help="Make one tool-free coding agent call to prove the route works"
)


_CONNECT_DEFAULT = typer.Option(
    None,
    "--default/--no-default",
    help="Pin this coding agent as the default. Default: only when none is pinned there yet",
)


_CONNECT_LOCAL = typer.Option(
    False,
    "--local/--user",
    help=(
        "Where the defaults (and an OpenRouter key) go: --user (the default) is "
        "~/.config/hillclimb/, every folder on this machine; --local is this "
        "folder's hillclimb dir, overriding the user level for it alone"
    ),
)


def _connect_config() -> Config:
    """`connect` runs before `hillclimb init` too: checking a credential
    needs no hillclimb dir, and the defaults it pins live at the user level
    unless `--local` asks for the folder's."""
    return Config.load(require_dir=False)


def _connect_target_config(config: Config, *, local: bool) -> Path | None:
    """The config.yaml `connect` writes defaults into: the user's, or with
    `local` this folder's (None when there is no hillclimb dir)."""
    from hillclimb.project import MARKER_FILE, user_config_path

    if not local:
        return user_config_path()
    if config.hillclimb_dir is None:
        return None
    return config.hillclimb_dir / MARKER_FILE


def _write_defaults(
    path: Path | None, updates: dict[str, str], *, wanted: bool | None, config: Config
) -> None:
    """Persist `agent`/`agent_auth`, unless the config already pins a
    coding agent on purpose — connecting a second coding agent to try it out must not
    silently repoint an existing setup. A user-level write also says so
    when this folder's hillclimb.yaml pins something else and keeps winning."""
    from hillclimb import connect as connect_mod
    from hillclimb.project import MARKER_FILE, user_config_path

    if wanted is False:
        return
    if path is None:
        # only `--local` gets here: the user level always has a path
        common.fail(
            "error: no hillclimb dir to write to — run [cmd]hillclimb init[/] first, "
            "or drop [cmd]--local[/] to pin the defaults for every folder"
        )
        raise typer.Exit(1)
    text = path.read_text() if path.exists() else ""
    if wanted is None and connect_mod.pins_agent(text):
        current = f"{updates['agent']}/{updates['agent_auth']}"
        common.say(
            f"[path]{common._m(path)}[/] already pins a coding agent — left as is "
            f"[note](`hillclimb connect … --default` switches it to {common._m(current)})[/]"
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(connect_mod.apply_config_defaults(text, updates))
    settings = ", ".join(f"{key}: {value}" for key, value in updates.items())
    common.say(f"[head]wrote[/] {common._m(settings)} to [path]{common._m(path)}[/]")
    folder = config.hillclimb_dir / MARKER_FILE if config.hillclimb_dir else None
    if path == user_config_path() and folder and folder.exists() and connect_mod.pins_agent(folder.read_text()):
        common.say(
            f"[note]this folder's[/] [path]{common._m(folder)}[/] [note]pins its own coding agent "
            "and keeps overriding the user default here (`--local` changes that one)[/]"
        )


def _sandbox_line(config) -> str:
    """Whether coding agents and verifiers will run confined on this machine."""
    from hillclimb.harness import sandbox

    try:
        if not sandbox.enabled(config):
            return "[head]sandbox[/]: [warn]off[/] [note]— coding agents and solutions run with your full user rights[/]"
        kind = sandbox.backend()
    except sandbox.SandboxUnavailable as exc:
        return f"[head]sandbox[/]: [bad]cannot start[/] [note]— {common._m(exc)}[/]"
    if kind is None:
        return (
            "[head]sandbox[/]: [warn]none exists for this operating system[/] "
            "[note]— coding agents and solutions run with your full user rights[/]"
        )
    tool = {"seatbelt": "sandbox-exec", "bwrap": "bubblewrap"}[kind]
    return (
        f"[head]sandbox[/]: [ok]on[/] [note]({tool}) — coding agents and solutions write only to "
        "their candidate's folder and cannot read your keys[/]"
    )


def _verdict_style(status) -> str:
    """The theme style for a connection's state: ready is ok, an error is
    bad, everything between (logged in but not connected, logged out) warns."""
    return "ok" if status.connected else ("bad" if status.state == "error" else "warn")


def _print_status(status) -> None:
    """One connection, coloured by verdict. Provider text is escaped: an
    error message full of brackets is not rich markup."""
    detail = f" — {common._m(status.detail)}" if status.detail else ""
    common.say(
        f"[head]{common._m(status.target)}[/] ({common._m(status.auth)}): "
        f"[{_verdict_style(status)}]{common._m(status.state)}[/]{detail}"
    )
    if not status.ok and status.fix:
        common.say(f"  fix: [note]{common._m(status.fix)}[/]")


class _LoginExpired(Exception):
    """The ping failed because the login on disk is dead, not the model."""


def _run_probe(
    agent: str, auth: str, model: str, config: Config, *, raise_expired: bool = False
) -> str | None:
    """One real call through the connected route. Returns the model that
    answered, or None when the ping failed — a failure here is the whole
    reason the command exists, so it is loud. With `raise_expired`, a ping
    that failed on a dead login raises `_LoginExpired` instead, so the caller
    can log in again and retry."""
    from hillclimb import connect as connect_mod

    label = f"model [path]{common._m(model)}[/]"
    answered = model  # what to report when the coding agent does not name the model
    if agent == "codex":
        from hillclimb.agents.codex_cli import CODEX_DEFAULT_LABEL, native_model

        if native_model(model, auth) is None:
            # hillclimb's `model` is a Claude alias: the coding agent omits it and
            # the Codex CLI's own default model answers instead of refusing
            label = "the Codex CLI's default model"
            answered = CODEX_DEFAULT_LABEL
    common.say(f"pinging [path]{common._m(agent)}[/] with {label} …")
    # the configured model goes through as is: the coding agent decides what to
    # do with it, exactly as it will inside a search
    result = connect_mod.ping(agent, auth, model, models_file=config.pi.models_file)
    if not result.ok:
        detail = result.error_message or result.error_kind or "unknown error"
        if raise_expired and connect_mod.login_expired(detail):
            # not an error yet: the caller logs in again and retries
            raise _LoginExpired(detail.strip())
        common.fail(
            f"error: the ping failed ({common._m(result.error_kind or 'error')}): "
            f"{common._m(detail.strip()[:400])}"
        )
        if connect_mod.login_expired(detail):
            return None  # the login is the culprit, not the model: no --model hint
        # the model is the usual culprit: aliases like `sonnet` mean nothing
        # to codex, and pi resolves a bare one against whichever provider
        # matches first
        common.say(f"  pinged {label} — [cmd]--model <id>[/] tries another", err=True)
        return None
    tokens = result.total_tokens or 0
    model_id = result.model_id or answered
    common.say(
        f"[ok]ping ok:[/] [path]{common._m(model_id)}[/], {tokens} tokens, {result.duration_s:.1f}s"
    )
    return model_id


def _connect_agent(
    target: str,
    *,
    auth: str | None,
    model: str | None,
    probe: bool,
    login: bool,
    default: bool | None,
    local: bool,
) -> None:
    """Shared flow for claude / codex / pi: check, log in, import, ping, pin.
    A ping that fails on a dead login logs in again and pings once more."""
    from hillclimb import connect as connect_mod

    config = _connect_config()
    auth = auth or connect_mod.configured_auth(config, target) or "subscription"
    if auth not in connect_mod.AUTHS_FOR[target]:
        raise typer.BadParameter(
            f"{target} has no {auth} route (one of {', '.join(connect_mod.AUTHS_FOR[target])})",
            param_hint="--auth",
        )
    agent = connect_mod.AGENT_FOR[target]

    status = connect_mod.check(target, auth)
    if status.state == "missing-cli":
        _print_status(status)
        raise typer.Exit(1)
    if not status.ok and login and connect_mod.login_command(target):
        common.say(connect_mod.login_header(target))
        common.say(
            f"  [note]{common._m(status.detail)}[/] — starting "
            f"[cmd]{common._m(' '.join(connect_mod.login_command(target)))}[/]"
        )
        connect_mod.run_login(target)
        status = connect_mod.check(target, auth)
    if not status.ok:
        _print_status(status)
        raise typer.Exit(1)

    home = connect_mod.import_credentials(target, auth, config.pi.models_file)
    if home is not None:
        common.say(f"operator home: [path]{common._m(home)}[/]")

    model = model or config.model
    pinged: str | None = None
    if probe:
        if auth == "openrouter" and "/" not in model:
            common.warn(
                f"no OpenRouter model id to ping with ({common._m(repr(model))}) — "
                "re-run with [cmd]--model <provider>/<model>[/] to check the route"
            )
        else:
            # `login status` only sees a credential file: a login that is
            # dead (codex spends its refresh token once) shows up here, in the
            # first real call. Replace it and try once more.
            relogin = login and auth == "subscription" and bool(connect_mod.login_command(target))
            try:
                pinged = _run_probe(agent, auth, model, config, raise_expired=relogin)
            except _LoginExpired as expired:
                steps = [connect_mod.logout_command(target), connect_mod.login_command(target)]
                shown = ", then ".join(" ".join(step) for step in steps if step)
                common.warn(
                    f"the {common._m(target)} login no longer works "
                    f"({common._m(str(expired)[:400])})"
                )
                common.say(connect_mod.login_header(target))
                common.say(f"  logging in again: [cmd]{common._m(shown)}[/]")
                connect_mod.run_relogin(target)
                status = connect_mod.check(target, auth)
                if not status.ok:
                    _print_status(status)
                    raise typer.Exit(1) from None
                connect_mod.import_credentials(target, auth, config.pi.models_file)
                pinged = _run_probe(agent, auth, model, config)
            if pinged is None:
                raise typer.Exit(1)

    # the mark that turns `logged-in` into `ready`
    connect_mod.mark_connected(target, auth, pinged)
    _print_status(connect_mod.check(target, auth))
    _write_defaults(
        _connect_target_config(config, local=local),
        {"agent": agent, "agent_auth": auth},
        wanted=default,
        config=config,
    )


@connect_app.callback()
def connect(
    ctx: typer.Context,
    as_json: bool = typer.Option(False, "--json", help="The same rows as data"),
):
    """Which coding agents this machine can run operators with, and who pays.

    A bare `hillclimb connect` checks every agent — the credential is read
    through the same environment an operator gets, so an inherited
    `ANTHROPIC_API_KEY` shadowing your subscription shows up here instead of
    on a bill. `●` marks the coding agent this config runs by default.

    `hillclimb connect <claude|codex|pi|openrouter>` sets one up: it runs the
    coding agent's login inside the operator home searches use, pings the
    route with one tool-free call, and pins the defaults in
    `~/.config/hillclimb/config.yaml` — every folder on this machine, so it
    works before `hillclimb init`; a folder's own hillclimb.yaml overrides them
    (`--local` writes there instead). `hillclimb smoke` is the next step up
    — a whole DRAFT on a real problem.
    """
    if ctx.invoked_subcommand is not None:
        return
    from hillclimb import connect as connect_mod
    from hillclimb.project import user_config_path

    config = _connect_config()
    rows = connect_mod.status_rows(config)
    if as_json:
        typer.echo(
            json.dumps(
                [{**status.as_dict(), "default": is_default} for status, is_default in rows],
                indent=2,
            )
        )
        return
    # the table speaks through the shared console, so its styles are the
    # theme's own (`head`, `cmd`, `path`, the verdict colours) and not a
    # palette of this command's making
    common.say()
    common.table(
        [("", None), ("agent", "cmd"), ("billing", None), ("state", None), ("", None)],
        [
            (
                "[path]●[/]" if is_default else " ",
                common._m(status.target),
                common._m(status.auth),
                f"[{_verdict_style(status)}]{common._m(status.state)}[/]",
                common._m(status.detail),
            )
            for status, is_default in rows
        ],
    )
    common.say()
    for status, _ in rows:
        if not status.connected and status.fix:
            common.say(f"  [head]{common._m(status.target)}[/]: [note]{common._m(status.fix)}[/]")
    common.say(f"  {_sandbox_line(config)}")
    if config.hillclimb_dir is None:
        common.say(
            "  [note]no hillclimb dir here — connecting pins the defaults in[/] "
            f"[path]{common._m(user_config_path())}[/] [note]for every folder;[/] "
            "[cmd]hillclimb init[/] [note]makes a folder whose hillclimb.yaml can override them[/]"
        )
    common.say()


@connect_app.command("claude")
def connect_claude(
    auth: str = _CONNECT_AUTH_CLAUDE,
    model: str = _CONNECT_MODEL,
    probe: bool = _CONNECT_PROBE,
    login: bool = typer.Option(True, "--login/--no-login", help="Run `claude auth login` when logged out"),
    default: bool = _CONNECT_DEFAULT,
    local: bool = _CONNECT_LOCAL,
):
    """Claude Code as the coding agent, billed to your Claude subscription.

    Operators run in a Claude Code home of hillclimb's own
    (`~/.cache/hillclimb/claude-home`), never your `~/.claude`: your
    settings, plugins, skills, MCP servers and history neither change a
    search nor collect what it leaves behind. That home logs in once, here
    (`claude auth login`, run inside it); your own login is left as it is.
    The check strips `ANTHROPIC_API_KEY`, so a key left in the environment
    cannot masquerade as the subscription. `--auth api-key` keeps the key
    instead, for headless machines.
    """
    _connect_agent(
        "claude", auth=auth, model=model, probe=probe, login=login, default=default, local=local
    )


@connect_app.command("codex")
def connect_codex(
    auth: str = _CONNECT_AUTH,
    model: str = _CONNECT_MODEL,
    probe: bool = _CONNECT_PROBE,
    login: bool = typer.Option(
        True, "--login/--no-login", help="Run `codex login` when logged out or when the login no longer works"
    ),
    default: bool = _CONNECT_DEFAULT,
    local: bool = _CONNECT_LOCAL,
):
    """The Codex CLI as the coding agent.

    Operators run in a codex home of hillclimb's own
    (`~/.cache/hillclimb/codex-home`, with a HOME of its own too), never your
    `~/.codex` or `~/.agents`, so your settings, plugins and skills change
    neither a search's results nor its token bill. That home logs in once,
    here (`codex login`, run inside it) — a login of its own, not a copy of
    yours, which codex would invalidate the first time either refreshed.
    `--auth openrouter` bills OpenRouter credits instead (`hillclimb connect
    openrouter` first).
    """
    _connect_agent(
        "codex", auth=auth, model=model, probe=probe, login=login, default=default, local=local
    )


@connect_app.command("pi")
def connect_pi(
    auth: str = _CONNECT_AUTH,
    model: str = _CONNECT_MODEL,
    probe: bool = _CONNECT_PROBE,
    default: bool = _CONNECT_DEFAULT,
    local: bool = _CONNECT_LOCAL,
):
    """pi as the coding agent — the one that can sample.

    pi logs in inside its own TUI, so this imports what that login wrote
    (`~/.pi/agent/auth.json`) into pi's isolated hillclimb home, together
    with `pi.models_file` if the config names one.
    """
    _connect_agent(
        "pi", auth=auth, model=model, probe=probe, login=False, default=default, local=local
    )


@connect_app.command("openrouter")
def connect_openrouter(
    key: str = typer.Option(None, "--key", help="The API key; omitted, connect asks for it (input hidden)"),
    agent: str = typer.Option(None, "--agent", help="Also route this coding agent through OpenRouter: codex | pi"),
    model: str = typer.Option(None, "--model", help="OpenRouter model id, e.g. qwen/qwen3-coder"),
    probe: bool = _CONNECT_PROBE,
    default: bool = _CONNECT_DEFAULT,
    local: bool = _CONNECT_LOCAL,
):
    """OpenRouter credits as the bill for codex or pi operators.

    The only credential hillclimb stores itself: the key is validated against
    OpenRouter (one unbilled call), then written to a `.env` — the user-level
    one in `~/.config/hillclimb/`, or with `--local` the one beside this
    folder's hillclimb.yaml that `hillclimb init` gitignores — never into
    hillclimb.yaml, where it could be journaled. `--agent codex` also pins
    the route.
    """
    from hillclimb import connect as connect_mod
    from hillclimb.agents.openrouter import OpenRouterError, key_info

    config = _connect_config()
    # an ambient key is already usable — only a key typed here gets stored
    ambient = os.environ.get("OPENROUTER_API_KEY")
    provided = key
    if provided is None and not ambient:
        provided = typer.prompt("OpenRouter API key", hide_input=True).strip()
    candidate = provided or ambient
    try:
        info = key_info(candidate)
    except OpenRouterError as exc:
        common.fail(f"error: {common._m(exc)}")
        raise typer.Exit(1)
    common.say(f"[ok]key accepted:[/] {common._m(connect_mod.describe_key(info))}")

    if provided:
        env_path = connect_mod.env_file(config, local=local)
        if env_path is None:
            common.fail(
                "error: no hillclimb dir to store the key in — run [cmd]hillclimb init[/] first, "
                "drop [cmd]--local[/] to store it for every folder, "
                "or export [path]OPENROUTER_API_KEY[/] yourself"
            )
            raise typer.Exit(1)
        connect_mod.write_env_key(env_path, "OPENROUTER_API_KEY", candidate)
        os.environ["OPENROUTER_API_KEY"] = candidate  # usable by the ping below
        common.say(f"[head]stored[/] [path]OPENROUTER_API_KEY[/] in [path]{common._m(env_path)}[/]")
    else:
        common.say("key came from the environment — [note]nothing stored[/]")
    # a validated key is a connected route: `key-set` becomes `ready`
    connect_mod.mark_connected("openrouter", "openrouter", None)
    _print_status(connect_mod.check("openrouter", "openrouter"))

    if agent is None:
        common.say(
            "pin it to a coding agent with: [cmd]hillclimb connect openrouter --agent codex --model <id>[/]"
        )
        return
    if agent not in ("codex", "pi"):
        raise typer.BadParameter("OpenRouter runs through codex or pi", param_hint="--agent")
    _connect_agent(
        agent, auth="openrouter", model=model, probe=probe, login=False, default=default, local=local
    )


@app.command("disconnect", short_help="Undo `hillclimb connect <agent>`: unpin it and forget its credentials.")
def disconnect(
    target: str = typer.Argument(..., metavar="AGENT", help="claude | codex | pi | openrouter"),
    local: bool = _CONNECT_LOCAL,
):
    """Undo `hillclimb connect <agent>` on hillclimb's side: unpin it as the
    default and forget the staged credentials, so the next `connect` sets it
    up from scratch again.

    Unpinning edits the same config.yaml `connect` wrote (`~/.config/hillclimb/`,
    or this folder's with `--local`), commenting the lines out in place; the
    isolated homes under `~/.cache/hillclimb/` go; `openrouter` drops the key
    from the `.env`. The coding agent's own login is left exactly as it is — hillclimb
    never logs you out of claude, codex or pi.
    """
    from hillclimb import connect as connect_mod
    from hillclimb.project import MARKER_FILE, user_config_path

    if target not in connect_mod.TARGETS:
        raise typer.BadParameter(f"{target} is not one of {', '.join(connect_mod.TARGETS)}", param_hint="agent")
    config = _connect_config()

    # 1. the default it pinned
    path = user_config_path() if not local else (config.hillclimb_dir / MARKER_FILE if config.hillclimb_dir else None)
    if path is None:
        common.fail("error: no hillclimb dir here — drop [cmd]--local[/] to unpin the user default")
        raise typer.Exit(1)
    if path.exists():
        before = path.read_text()
        after = connect_mod.unpin_config_defaults(before, target)
        if after != before:
            path.write_text(after)
            common.say(f"[head]unpinned[/] {common._m(target)} in [path]{common._m(path)}[/]")
        else:
            common.say(f"[note]{common._m(path)} does not pin {common._m(target)} — nothing to unpin[/]")

    # 2. the key, for the OpenRouter route
    if target == "openrouter":
        env_path = connect_mod.env_file(config, local=local)
        if env_path is not None and env_path.exists():
            before = env_path.read_text()
            after = connect_mod.remove_env_key(before, "OPENROUTER_API_KEY")
            if after != before:
                env_path.write_text(after)
                common.say(f"[head]removed[/] [path]OPENROUTER_API_KEY[/] from [path]{common._m(env_path)}[/]")

    # 3. what searches read
    for removed in connect_mod.remove_staged(target):
        common.say(f"[head]removed[/] [path]{common._m(removed)}[/]")

    common.say(
        f"[head]{common._m(target)} is disconnected from hillclimb.[/] "
        + ("" if target == "openrouter" else "[note]Your own login on this machine is untouched.[/]")
    )
    common.next_steps([(f"hillclimb connect {target}", "set it up again")])
