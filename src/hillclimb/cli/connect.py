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
    help="Connect the agents that run operators (claude, codex, pi) and the OpenRouter route that pays for them.",
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
    True, "--probe/--no-probe", help="Make one tool-free agent call to prove the route works"
)


_CONNECT_DEFAULT = typer.Option(
    None,
    "--default/--no-default",
    help="Write this backend into config.yaml. Default: only when no backend is pinned there yet",
)


_CONNECT_USER = typer.Option(
    False, "--user", help="Write the defaults to ~/.config/hillclimb/config.yaml instead of the hillclimb dir"
)


def _connect_config() -> Config:
    """`connect` runs before `hillclimb init` too: checking a credential
    needs no hillclimb dir, only writing defaults and keys does."""
    from hillclimb.project import HillclimbDirNotFound

    try:
        return common.load_config(raise_not_found=True)
    except HillclimbDirNotFound:
        return Config()


def _connect_target_config(config: Config, *, user: bool) -> Path | None:
    """The config.yaml `connect` would write defaults into."""
    from hillclimb.project import MARKER_FILE, user_config_path

    if user:
        return user_config_path()
    if config.hillclimb_dir is None:
        return None
    return config.hillclimb_dir / MARKER_FILE


def _write_defaults(path: Path | None, updates: dict[str, str], *, wanted: bool | None) -> None:
    """Persist `backend`/`backend_auth`, unless the config already pins a
    backend on purpose — connecting a second agent to try it out must not
    silently repoint an existing setup."""
    from hillclimb import connect as connect_mod

    if wanted is False:
        return
    if path is None:
        if wanted:
            typer.echo(
                "error: no hillclimb dir to write to — run `hillclimb init`, or pass --user",
                err=True,
            )
            raise typer.Exit(1)
        return
    text = path.read_text() if path.exists() else ""
    if wanted is None and connect_mod.pins_backend(text):
        current = f"{updates['backend']}/{updates['backend_auth']}"
        typer.echo(
            f"{path} already pins a backend — left as is "
            f"(`hillclimb connect … --default` switches it to {current})"
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(connect_mod.apply_config_defaults(text, updates))
    settings = ", ".join(f"{key}: {value}" for key, value in updates.items())
    typer.echo(f"wrote {settings} to {path}")


def _print_status(status) -> None:
    """One connection, coloured by verdict. Provider text is escaped: an
    error message full of brackets is not rich markup."""
    from rich.console import Console
    from rich.markup import escape

    colour = "green" if status.ok else ("yellow" if status.state != "error" else "red")
    detail = f" — {escape(status.detail)}" if status.detail else ""
    Console(highlight=False).print(
        f"[bold]{status.target}[/] ({status.auth}): [{colour}]{status.state}[/]{detail}"
    )
    if not status.ok and status.fix:
        typer.echo(f"  fix: {status.fix}")


def _run_probe(backend: str, auth: str, model: str, config: Config) -> bool:
    """One real call through the connected route. Returns whether it worked;
    a failure here is the whole reason the command exists, so it is loud."""
    from hillclimb import connect as connect_mod

    typer.echo(f"pinging {backend} with model {model} …")
    result = connect_mod.ping(backend, auth, model, models_file=config.pi.models_file)
    if not result.ok:
        detail = result.error_message or result.error_kind or "unknown error"
        typer.echo(f"error: the ping failed ({result.error_kind or 'error'}): {detail.strip()[:400]}", err=True)
        # the model is the usual culprit: aliases like `sonnet` mean nothing
        # to codex, and pi resolves a bare one against whichever provider
        # matches first
        typer.echo(f"  pinged model {model!r} — `--model <id>` tries another", err=True)
        return False
    tokens = result.total_tokens or 0
    model_id = result.model_id or model
    typer.echo(f"ping ok: {model_id}, {tokens} tokens, {result.duration_s:.1f}s")
    return True


def _connect_backend(
    target: str,
    *,
    auth: str | None,
    model: str | None,
    probe: bool,
    login: bool,
    default: bool | None,
    user: bool,
) -> None:
    """Shared flow for claude / codex / pi: check, log in, import, ping, pin."""
    from hillclimb import connect as connect_mod

    config = _connect_config()
    auth = auth or connect_mod.configured_auth(config, target) or "subscription"
    if auth not in connect_mod.AUTHS_FOR[target]:
        raise typer.BadParameter(
            f"{target} has no {auth} route (one of {', '.join(connect_mod.AUTHS_FOR[target])})",
            param_hint="--auth",
        )
    backend = connect_mod.BACKEND_FOR[target]

    status = connect_mod.check(target, auth)
    if status.state == "missing-cli":
        _print_status(status)
        raise typer.Exit(1)
    if not status.ok and login and connect_mod.login_command(target):
        typer.echo(f"{status.detail} — starting `{' '.join(connect_mod.login_command(target))}`")
        connect_mod.run_login(target)
        status = connect_mod.check(target, auth)
    if not status.ok:
        _print_status(status)
        raise typer.Exit(1)

    home = connect_mod.import_credentials(target, auth, config.pi.models_file)
    if home is not None:
        typer.echo(f"credentials staged for searches in {home}")
    _print_status(status)

    model = model or config.model
    if probe:
        if auth == "openrouter" and "/" not in model:
            typer.echo(
                f"no OpenRouter model id to ping with ({model!r}) — "
                "re-run with --model <provider>/<model> to check the route"
            )
        elif not _run_probe(backend, auth, model, config):
            raise typer.Exit(1)

    _write_defaults(
        _connect_target_config(config, user=user),
        {"backend": backend, "backend_auth": auth},
        wanted=default,
    )


@connect_app.callback()
def connect(
    ctx: typer.Context,
    as_json: bool = typer.Option(False, "--json", help="The same rows as data"),
):
    """Which agents this machine can run operators with, and who pays.

    A bare `hillclimb connect` checks every target — the credential is read
    through the same environment an operator gets, so an inherited
    `ANTHROPIC_API_KEY` shadowing your subscription shows up here instead of
    on a bill. `●` marks the backend this config runs by default.

    `hillclimb connect <claude|codex|pi|openrouter>` sets one up: it runs the
    agent's own login, stages the credentials searches will read, pings the
    route with one tool-free call, and pins the defaults in config.yaml.
    `hillclimb smoke` is the next step up — a whole DRAFT on a real problem.
    """
    if ctx.invoked_subcommand is not None:
        return
    from rich.console import Console
    from rich.table import Table

    from hillclimb import connect as connect_mod

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
    from rich.markup import escape

    console = Console(highlight=False)
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("")
    table.add_column("target", style="bold cyan")
    table.add_column("billing")
    table.add_column("state")
    table.add_column("")
    for status, is_default in rows:
        colour = "green" if status.ok else ("red" if status.state == "error" else "yellow")
        table.add_row(
            "[cyan]●[/]" if is_default else " ",
            status.target,
            status.auth,
            f"[{colour}]{status.state}[/]",
            escape(status.detail),
        )
    console.print()
    console.print(table)
    console.print()
    for status, _ in rows:
        if not status.ok and status.fix:
            console.print(f"  [bold]{status.target}[/]: {status.fix}")
    if config.hillclimb_dir is None:
        console.print("  no hillclimb dir here — `hillclimb init` before connecting anything")
    console.print()


@connect_app.command("claude")
def connect_claude(
    auth: str = _CONNECT_AUTH_CLAUDE,
    model: str = _CONNECT_MODEL,
    probe: bool = _CONNECT_PROBE,
    login: bool = typer.Option(True, "--login/--no-login", help="Run `claude auth login` when logged out"),
    default: bool = _CONNECT_DEFAULT,
    user: bool = _CONNECT_USER,
):
    """Claude Code as the operator backend, billed to your Claude subscription.

    The login is Claude Code's own (`claude auth login`); hillclimb only
    checks it the way an operator will — with `ANTHROPIC_API_KEY` stripped,
    so a key left in the environment cannot masquerade as the subscription.
    `--auth api-key` keeps the key instead, for headless machines.
    """
    _connect_backend(
        "claude", auth=auth, model=model, probe=probe, login=login, default=default, user=user
    )


@connect_app.command("codex")
def connect_codex(
    auth: str = _CONNECT_AUTH,
    model: str = _CONNECT_MODEL,
    probe: bool = _CONNECT_PROBE,
    login: bool = typer.Option(True, "--login/--no-login", help="Run `codex login` when logged out"),
    default: bool = _CONNECT_DEFAULT,
    user: bool = _CONNECT_USER,
):
    """The Codex CLI as the operator backend.

    Runs `codex login`, then copies the credential into the isolated
    `CODEX_HOME` searches use, so your personal `~/.codex` settings change
    neither a search's results nor its token bill. `--auth openrouter` bills
    OpenRouter credits instead (`hillclimb connect openrouter` first).
    """
    _connect_backend(
        "codex", auth=auth, model=model, probe=probe, login=login, default=default, user=user
    )


@connect_app.command("pi")
def connect_pi(
    auth: str = _CONNECT_AUTH,
    model: str = _CONNECT_MODEL,
    probe: bool = _CONNECT_PROBE,
    default: bool = _CONNECT_DEFAULT,
    user: bool = _CONNECT_USER,
):
    """The pi coding agent as the operator backend — the one that can sample.

    pi logs in inside its own TUI, so this imports what that login wrote
    (`~/.pi/agent/auth.json`) into pi's isolated hillclimb home, together
    with `pi.models_file` if the config names one.
    """
    _connect_backend(
        "pi", auth=auth, model=model, probe=probe, login=False, default=default, user=user
    )


@connect_app.command("openrouter")
def connect_openrouter(
    key: str = typer.Option(None, "--key", help="The API key; omitted, connect asks for it (input hidden)"),
    backend: str = typer.Option(None, "--backend", help="Also route this backend through OpenRouter: codex | pi"),
    model: str = typer.Option(None, "--model", help="OpenRouter model id, e.g. qwen/qwen3-coder"),
    probe: bool = _CONNECT_PROBE,
    default: bool = _CONNECT_DEFAULT,
    user: bool = _CONNECT_USER,
):
    """OpenRouter credits as the bill for codex or pi operators.

    The only credential hillclimb stores itself: the key is validated against
    OpenRouter (one unbilled call), then written to the `.env` beside
    config.yaml that `hillclimb init` gitignores — never into config.yaml,
    where it could be journaled. `--backend codex` also pins the route.
    """
    from hillclimb import connect as connect_mod
    from hillclimb.backends.openrouter import OpenRouterError, key_info

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
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"key accepted: {connect_mod.describe_key(info)}")

    if provided:
        env_path = connect_mod.env_file(config)
        if env_path is None:
            typer.echo(
                "error: no hillclimb dir to store the key in — run `hillclimb init`, "
                "or export OPENROUTER_API_KEY yourself",
                err=True,
            )
            raise typer.Exit(1)
        connect_mod.write_env_key(env_path, "OPENROUTER_API_KEY", candidate)
        os.environ["OPENROUTER_API_KEY"] = candidate  # usable by the ping below
        typer.echo(f"stored OPENROUTER_API_KEY in {env_path}")
    else:
        typer.echo("key came from the environment — nothing stored")

    if backend is None:
        typer.echo("pin it to a backend with: hillclimb connect openrouter --backend codex --model <id>")
        return
    if backend not in ("codex", "pi"):
        raise typer.BadParameter("OpenRouter runs through codex or pi", param_hint="--backend")
    _connect_backend(
        backend, auth="openrouter", model=model, probe=probe, login=False, default=default, user=user
    )
