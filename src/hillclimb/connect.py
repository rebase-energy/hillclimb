"""Connecting hillclimb to the agents that run its operators.

`backend` + `backend_auth` say WHO runs an operator call and WHO pays for it.
Both were only settings: the credential behind them was discovered at the
first spawn inside a search, so a missing login surfaced as a dead operator
minutes into a climb, and a stale one as a bill on the wrong account. This
module makes it a step you take on purpose.

Two things make a check trustworthy here:

* It runs **through the same env builders the backends use** —
  `subscription_env`, `codex_env`, `pi_env` — so what it reports is what an
  operator will see, not what the personal CLI config happens to hold. The
  difference is real: an inherited `ANTHROPIC_API_KEY` silently rebills a
  "subscription" search to the API, and `subscription_env` drops it. A check
  against the ambient environment would miss exactly the failure worth
  catching.
* A **target is not a backend**. `claude`, `codex` and `pi` are backends and
  own their own login flow — hillclimb shells out to it and then materializes
  the isolated per-auth home the searches read (`codex_home`, `pi_home`).
  `openrouter` is a billing route for codex and pi, and is the one credential
  hillclimb stores itself: in the `.env` beside `config.yaml` that
  `hillclimb init` already gitignores. Keys never enter `Config`, so they
  cannot be journaled.

Connecting writes at most two things: that `.env` line, and the `backend` /
`backend_auth` defaults in `config.yaml`. Everything else lives where the
agent's own CLI put it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# targets in the order the status table lists them: the backends that run
# operators, then the route that pays for them
TARGETS = ("claude", "codex", "pi", "openrouter")
BACKEND_FOR = {"claude": "claude-code", "codex": "codex", "pi": "pi"}
# which billing modes each backend implements (config.BACKEND_AUTHS is the
# full vocabulary; claude-code has no OpenRouter route)
AUTHS_FOR = {
    "claude": ("subscription", "api-key"),
    "codex": ("subscription", "api-key", "openrouter"),
    "pi": ("subscription", "api-key", "openrouter"),
    "openrouter": ("openrouter",),
}
INSTALL_HINT = {
    "claude": "npm install -g @anthropic-ai/claude-code",
    "codex": "npm install -g @openai/codex",
    "pi": "install the pi coding agent, then run `pi` once",
}
# pi has no login subcommand — its login lives inside the TUI, so hillclimb
# can only tell you to go there and then import what it wrote
PI_LOGIN_HINT = "run `pi`, log in with /login, then `hillclimb connect pi` again"

PING_PROMPT = "Reply with exactly: pong. Do not use any tools."
PING_TIMEOUT_S = 180


@dataclass(frozen=True)
class Status:
    """One (target, auth) pair, as an operator would find it.

    `state` is the machine-readable verdict; `detail` is what was found and
    `fix` the command that would change it. Only `ready` means a search
    would get through its first operator call.
    """

    target: str
    auth: str
    state: str  # ready | missing-cli | logged-out | no-key | unsupported | error
    detail: str = ""
    fix: str = ""

    @property
    def ok(self) -> bool:
        return self.state == "ready"

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            "auth": self.auth,
            "state": self.state,
            "ok": self.ok,
            "detail": self.detail,
            "fix": self.fix,
        }


# --------------------------------------------------------------------------
# reading what the agent CLIs report (pure: the subprocess result in, a
# verdict out — so the interpretation is testable without a login)
# --------------------------------------------------------------------------


def parse_claude_status(returncode: int, stdout: str) -> tuple[bool, str]:
    """`claude auth status` emits JSON. Logged in through the subscription
    means `loggedIn` with a non-apiKey method; an `apiKeySource` that
    survived `subscription_env` would mean the calls bill the API instead,
    which is worth saying out loud."""
    try:
        payload = json.loads(stdout)
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        # older CLIs have no `auth status`; the text is the best evidence left
        text = (stdout or "").strip().splitlines()
        first = text[0][:120] if text else ""
        if returncode == 0 and "not logged in" not in (stdout or "").lower():
            return True, first or "logged in"
        return False, first or "not logged in"
    if not payload.get("loggedIn"):
        return False, "not logged in"
    method = payload.get("authMethod") or "unknown"
    who = payload.get("email") or payload.get("orgName") or ""
    detail = f"logged in ({method}{', ' + who if who else ''})"
    source = payload.get("apiKeySource")
    if source:
        detail += f" — but {source} is set and would be billed"
    return True, detail


def parse_codex_status(returncode: int, output: str) -> tuple[bool, str]:
    """`codex login status` prints one line ("Logged in using ChatGPT") — on
    stderr, so callers pass both streams joined."""
    text = (output or "").strip().splitlines()
    first = text[0][:120] if text else ""
    if returncode == 0 and "not logged in" not in (output or "").lower():
        return True, first or "logged in"
    return False, first or "not logged in"


def parse_pi_credentials(text: str) -> list[str]:
    """The providers pi has logged in, from its `auth.json`.

    pi writes the file with an empty object before anyone logs in, so its
    mere existence proves nothing — the check that matters is whether there
    is an entry in it.
    """
    try:
        data = json.loads(text)
    except ValueError:
        return []
    if isinstance(data, dict):
        return sorted(str(key) for key in data)
    if isinstance(data, list):
        return sorted(str(item) for item in data if isinstance(item, str))
    return []


def describe_key(info: dict) -> str:
    """OpenRouter's key record as one line: label, spend, and the credit
    ceiling if the key has one (`limit: null` is pay-as-you-go)."""
    label = str(info.get("label") or "key")
    usage = info.get("usage")
    limit = info.get("limit")
    parts = [label]
    if isinstance(usage, (int, float)):
        spent = f"${usage:.2f} used"
        if isinstance(limit, (int, float)):
            spent += f" of ${limit:.2f}"
        parts.append(spent)
    if info.get("is_free_tier"):
        parts.append("free tier")
    return ", ".join(parts)


# --------------------------------------------------------------------------
# checking
# --------------------------------------------------------------------------


def _run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)


def _check_claude(auth: str) -> Status:
    from hillclimb.backends.claude_code import subscription_env

    binary = shutil.which("claude")
    if binary is None:
        return Status("claude", auth, "missing-cli", "claude is not on PATH", INSTALL_HINT["claude"])
    env = subscription_env(auth)
    if auth == "api-key":
        if not env.get("ANTHROPIC_API_KEY"):
            return Status(
                "claude", auth, "no-key", "ANTHROPIC_API_KEY is not set",
                "export ANTHROPIC_API_KEY=… (or connect --auth subscription)",
            )
        return Status("claude", auth, "ready", "ANTHROPIC_API_KEY is set")
    try:
        proc = _run([binary, "auth", "status"], env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Status("claude", auth, "error", str(exc)[:120])
    ok, detail = parse_claude_status(proc.returncode, proc.stdout or proc.stderr)
    if not ok:
        return Status("claude", auth, "logged-out", detail, "claude auth login")
    return Status("claude", auth, "ready", _with_quota(detail))


def _with_quota(detail: str) -> str:
    """Append subscription window utilization when it is readable — the
    number that decides whether a long search will finish on this plan."""
    from hillclimb.harness import quota

    snapshot = quota.snapshot() or {}
    window = snapshot.get("five_hour") or {}
    utilization = window.get("utilization")
    if not isinstance(utilization, (int, float)):
        return detail
    return f"{detail} — {utilization:.0f}% of the 5-hour window used"


def _check_codex(auth: str) -> Status:
    from hillclimb.backends.codex_cli import codex_env

    binary = shutil.which("codex")
    if binary is None:
        return Status("codex", auth, "missing-cli", "codex is not on PATH", INSTALL_HINT["codex"])
    if auth == "openrouter":
        # the credential is OpenRouter's, not codex's — codex is only the
        # process that carries it
        key = _check_openrouter("openrouter")
        detail = f"codex on OpenRouter ({key.detail})" if key.ok else key.detail
        return Status("codex", auth, key.state, detail, key.fix)
    try:
        # materializes the isolated CODEX_HOME, exactly as the first operator
        # call would: the check and the search read the same credential
        env = codex_env(auth)
    except (RuntimeError, OSError) as exc:
        return Status("codex", auth, "error", str(exc)[:160])
    if auth == "api-key":
        if not env.get("OPENAI_API_KEY"):
            return Status(
                "codex", auth, "no-key", "OPENAI_API_KEY is not set",
                "export OPENAI_API_KEY=… (or connect --auth subscription)",
            )
        return Status("codex", auth, "ready", "OPENAI_API_KEY is set")
    try:
        proc = _run([binary, "login", "status"], env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Status("codex", auth, "error", str(exc)[:120])
    ok, detail = parse_codex_status(proc.returncode, f"{proc.stdout}\n{proc.stderr}")
    if not ok:
        return Status("codex", auth, "logged-out", detail, "codex login")
    return Status("codex", auth, "ready", detail)


def pi_credentials_path() -> Path:
    """Where pi's own login stores the credential `pi_home` imports."""
    return Path.home() / ".pi" / "agent" / "auth.json"


def _check_pi(auth: str) -> Status:
    binary = shutil.which("pi")
    if binary is None:
        return Status("pi", auth, "missing-cli", "pi is not on PATH", INSTALL_HINT["pi"])
    if auth == "openrouter":
        key = _check_openrouter("openrouter")
        detail = f"pi on OpenRouter ({key.detail})" if key.ok else key.detail
        return Status("pi", auth, key.state, detail, key.fix)
    if auth == "api-key":
        present = [name for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if os.environ.get(name)]
        if not present:
            return Status(
                "pi", auth, "no-key", "no provider key in the environment",
                "export the provider key pi's model needs, e.g. ANTHROPIC_API_KEY=…",
            )
        return Status("pi", auth, "ready", f"{', '.join(present)} set")
    source = pi_credentials_path()
    try:
        providers = parse_pi_credentials(source.read_text())
    except OSError:
        providers = []
    if not providers:
        return Status(
            "pi", auth, "logged-out", f"no provider logged in at {_tilde(source)}", PI_LOGIN_HINT
        )
    return Status("pi", auth, "ready", f"logged in: {', '.join(providers)}")


def _check_openrouter(auth: str = "openrouter") -> Status:
    from hillclimb.backends.openrouter import OpenRouterError, key_info

    if not os.environ.get("OPENROUTER_API_KEY"):
        return Status(
            "openrouter", auth, "no-key", "OPENROUTER_API_KEY is not set",
            "hillclimb connect openrouter",
        )
    try:
        info = key_info()
    except OpenRouterError as exc:
        return Status("openrouter", auth, "error", str(exc)[:160], "hillclimb connect openrouter")
    return Status("openrouter", auth, "ready", describe_key(info))


def check(target: str, auth: str) -> Status:
    """The state of one (target, auth) pair, read the way an operator reads it.

    Never raises for an unusable credential — an unusable credential is a
    `Status`, which is the whole point of having the command.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r} (one of {', '.join(TARGETS)})")
    if auth not in AUTHS_FOR[target]:
        return Status(
            target, auth, "unsupported",
            f"{target} has no {auth} route (one of {', '.join(AUTHS_FOR[target])})",
        )
    if target == "claude":
        return _check_claude(auth)
    if target == "codex":
        return _check_codex(auth)
    if target == "pi":
        return _check_pi(auth)
    return _check_openrouter(auth)


def configured_auth(config, target: str) -> str | None:
    """The billing mode this config actually selects for a backend, or None
    when nothing routes to it. Routing overrides the scalar per operator, so
    a backend can appear under several modes; the global one wins the row,
    else the first routed one in sorted order."""
    backend = BACKEND_FOR.get(target)
    if backend is None:  # openrouter is a route, not a backend
        return "openrouter"
    if config.backend == backend:
        return config.backend_auth
    default = config.routing.get("default")
    routed: list[str] = []
    for name in sorted(config.routing):
        route = config.routing[name]
        route_backend = route.backend or (default.backend if default else None) or config.backend
        if route_backend != backend:
            continue
        routed.append(
            route.backend_auth
            or (default.backend_auth if default else None)
            or config.backend_auth
        )
    return routed[0] if routed else None


def status_rows(config) -> list[tuple[Status, bool]]:
    """Every target with the auth this config would use, `(status, is_default)`.

    A target the config does not route to is still checked, on its natural
    mode, so the table answers "what else could I run on this machine".
    """
    rows: list[tuple[Status, bool]] = []
    for target in TARGETS:
        auth = configured_auth(config, target) or "subscription"
        if auth not in AUTHS_FOR[target]:
            auth = AUTHS_FOR[target][0]
        is_default = BACKEND_FOR.get(target) == config.backend
        rows.append((check(target, auth), is_default))
    return rows


# --------------------------------------------------------------------------
# connecting
# --------------------------------------------------------------------------


def login_command(target: str) -> list[str] | None:
    """The agent's own login flow, or None when it has no CLI entry point."""
    return {"claude": ["claude", "auth", "login"], "codex": ["codex", "login"]}.get(target)


def run_login(target: str) -> int:
    """Hand the terminal to the agent's login (browser flow, prompts, all of
    it) and return its exit code. hillclimb never sees the credential."""
    cmd = login_command(target)
    if cmd is None:
        raise ValueError(f"{target} has no login command")
    binary = shutil.which(cmd[0])
    if binary is None:
        raise RuntimeError(f"{cmd[0]} is not on PATH")
    return subprocess.call([binary, *cmd[1:]])


def import_credentials(target: str, auth: str, models_file: Path | None = None) -> Path | None:
    """Materialize the isolated per-auth home a search will read, now rather
    than inside the first operator call. Returns the directory, or None for a
    target that has none (claude-code uses the ambient login)."""
    if target == "codex":
        from hillclimb.backends.codex_cli import codex_home

        return codex_home(auth)
    if target == "pi":
        from hillclimb.backends.pi_cli import pi_home

        return pi_home(auth, models_file)
    return None


def ping(backend_name: str, auth: str, model: str, *, models_file: Path | None = None):
    """One real, tool-free operator call — the proof that the credential, the
    model id and the provider route all work together. Costs a handful of
    tokens, which is the cheapest honest answer available; a key that is
    valid for the API but not for the requested model fails only here.

    Runs in the machine cache, never in a run: it is not a candidate.
    """
    from hillclimb.backends import get_backend
    from hillclimb.backends.base import OperatorRequest
    from hillclimb.project import machine_cache_dir

    work_dir = machine_cache_dir() / "connect" / f"{backend_name}-{auth}"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    backend = get_backend(backend_name, auth=auth, pi_models_file=models_file)
    request = OperatorRequest(
        operator="draft",  # the routed operators all look alike to a provider
        prompt=PING_PROMPT,
        candidate_dir=work_dir,
        timeout_s=PING_TIMEOUT_S,
        model=model,
    )
    # pi's preflight is the same call without tools; the others have no
    # cheaper door than invoke()
    preflight = getattr(backend, "preflight", None)
    return preflight(request) if preflight else backend.invoke(request)


# --------------------------------------------------------------------------
# the two files connecting may write
# --------------------------------------------------------------------------


def apply_config_defaults(text: str, updates: dict[str, str]) -> str:
    """Set top-level scalars in a config.yaml, in place, comments intact.

    A line-level edit rather than a YAML round-trip: `config.yaml` is
    hand-written and heavily commented (`hillclimb init` ships it that way),
    and re-emitting it from a parsed dict would throw all of that away. Only
    column-zero keys match, so `routing:`'s indented `backend:` is never
    touched, and the commented-out `# backend:` line `init` leaves behind is
    uncommented in place — the setting appears where its comment explains it.
    A key with nowhere to go lands next to the one written just before it, so
    `backend` and `backend_auth` end up adjacent instead of scattered.
    """
    lines = text.splitlines()
    anchor: int | None = None
    for key, value in updates.items():
        pattern = re.compile(rf"^#?\s*{re.escape(key)}:")
        for index, line in enumerate(lines):
            if pattern.match(line):
                lines[index] = f"{key}: {value}"
                anchor = index
                break
        else:
            insert_at = (
                anchor + 1
                if anchor is not None
                else next(
                    (
                        index
                        for index, line in enumerate(lines)
                        if line.strip() and not line.lstrip().startswith("#")
                    ),
                    len(lines),
                )
            )
            lines.insert(insert_at, f"{key}: {value}")
            anchor = insert_at
    return "\n".join(lines) + "\n"


def pins_backend(text: str) -> bool:
    """Does this config.yaml already choose a backend on purpose?

    Connecting should not silently repoint an existing setup at whatever was
    connected last; it should complete a fresh one. An active column-zero
    `backend:` is the signal that someone already decided.
    """
    return any(re.match(r"^backend:", line) for line in text.splitlines())


def env_file(config) -> Path | None:
    """Where provider keys belong for this config: the `.env` beside
    `config.yaml`, or the repo-root one already in use (`Config.load` reads
    the hillclimb dir's first, then its parent's — writing anywhere else
    would store a key nothing loads)."""
    if config.hillclimb_dir is None:
        return None
    beside = config.hillclimb_dir / ".env"
    parent = config.hillclimb_dir.parent / ".env"
    if not beside.exists() and parent.exists():
        return parent
    return beside


def upsert_env(text: str, key: str, value: str) -> str:
    """Set `KEY=value` in a .env, replacing an existing assignment in place
    and leaving every other line (comments, other keys) alone."""
    lines = text.splitlines()
    pattern = re.compile(rf"^\s*(export\s+)?{re.escape(key)}\s*=")
    for index, line in enumerate(lines):
        if pattern.match(line):
            lines[index] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def write_env_key(path: Path, key: str, value: str) -> None:
    """Store a provider key, readable only by its owner. The file may already
    hold other keys, so it is edited, never replaced."""
    text = path.read_text() if path.exists() else ""
    path.write_text(upsert_env(text, key, value))
    path.chmod(0o600)


def _tilde(path: Path) -> str:
    """Paths under $HOME print as ~/… — shorter, and safe to paste."""
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)
