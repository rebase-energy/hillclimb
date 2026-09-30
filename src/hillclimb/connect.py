"""Connecting hillclimb to the agents that run its operators.

`agent` + `agent_auth` say WHO runs an operator call and WHO pays for it.
Both were only settings: the credential behind them was discovered at the
first spawn inside a search, so a missing login surfaced as a dead operator
minutes into a climb, and a stale one as a bill on the wrong account. This
module makes it a step you take on purpose.

Two things make a check trustworthy here:

* It runs **through the same env builders the agents use** —
  `subscription_env`, `codex_env`, `pi_env` — so what it reports is what an
  operator will see, not what the personal CLI config happens to hold. The
  difference is real: an inherited `ANTHROPIC_API_KEY` silently rebills a
  "subscription" search to the API, and `subscription_env` drops it. A check
  against the ambient environment would miss exactly the failure worth
  catching.
* A **target is not a agent**. `claude`, `codex` and `pi` are agents and
  own their own login flow — hillclimb shells out to it and then materializes
  the isolated per-auth home the searches read (`codex_home`, `pi_home`).
  `openrouter` is a billing route for codex and pi, and is the one credential
  hillclimb stores itself: in a `.env` — the user-level one beside
  `~/.config/hillclimb/config.yaml`, or with `--local` the one beside the
  folder's `hillclimb.yaml` that `hillclimb init` already gitignores. Keys never
  enter `Config`, so they cannot be journaled.

Connecting writes at most two things: that `.env` line, and the `agent` /
`agent_auth` defaults in a config file. Both land at the USER level by
default — `~/.config/hillclimb/` — so one login serves every folder on the
machine and `hillclimb connect` works before `hillclimb init`; a folder's
own `hillclimb.yaml`/`.env` overrides them (`--local` writes there instead).
Everything else lives where the agent's own CLI put it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# targets in the order the status table lists them: the agents that run
# operators, then the route that pays for them
TARGETS = ("claude", "codex", "pi", "openrouter")
AGENT_FOR = {"claude": "claude-code", "codex": "codex", "pi": "pi"}
# which billing modes each agent implements (config.AGENT_AUTHS is the
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
    `fix` the command that would change it. Three states matter for an
    agent: `logged-out` (its own login is missing), `logged-in` (the login
    works but hillclimb has not connected it — `hillclimb connect <target>`
    has not completed on this machine, or its cache was wiped) and `ready`
    (logged in AND connected: the ping passed and what a search reads is
    staged). The OpenRouter route says `no-key` / `key-set` / `ready` the
    same way. Only `ready` means a search would get through its first
    operator call without `connect` first.
    """

    target: str
    auth: str
    state: str  # ready | logged-in | key-set | logged-out | no-key | missing-cli | unsupported | error
    detail: str = ""
    fix: str = ""

    @property
    def ok(self) -> bool:
        """The credential itself works — what `connect` needs to proceed."""
        return self.state in ("ready", "logged-in", "key-set")

    @property
    def connected(self) -> bool:
        return self.state == "ready"

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            "auth": self.auth,
            "state": self.state,
            "ok": self.ok,
            "connected": self.connected,
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
    from hillclimb.harness.oscompat import runnable

    return subprocess.run(runnable(cmd), capture_output=True, text=True, env=env, timeout=60)


def _check_claude(auth: str) -> Status:
    from hillclimb.agents.claude_code import subscription_env

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
    # The status says who is logged in and nothing about the plan's window:
    # quota is the search's concern (journaled per candidate by
    # `harness.quota`), not the connection's.
    return Status("claude", auth, "ready", detail)


def _check_codex(auth: str) -> Status:
    from hillclimb.agents.codex_cli import codex_env

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
    from hillclimb.agents.openrouter import OpenRouterError, key_info

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
        status = _check_claude(auth)
    elif target == "codex":
        status = _check_codex(auth)
    elif target == "pi":
        status = _check_pi(auth)
    else:
        status = _check_openrouter(auth)
    return _with_connection(status)


def _with_connection(status: Status) -> Status:
    """A working credential is `ready` only once `connect` has completed for
    it on this machine; until then it is `logged-in` (`key-set` for the
    OpenRouter route) with `connect` as the fix."""
    if status.state != "ready" or is_connected(status.target, status.auth):
        return status
    state = "key-set" if status.target == "openrouter" else "logged-in"
    return Status(
        status.target, status.auth, state, status.detail, f"hillclimb connect {status.target}"
    )


def record_dir(target: str, auth: str) -> Path:
    """Where `connect <target>` leaves its mark for this auth mode: the
    ping's scratch dir under the machine cache, named like the staged
    homes (`claude-code-subscription`, `openrouter-openrouter`), so
    `staged_homes` — and with it `disconnect` — already covers it."""
    from hillclimb.project import machine_cache_dir

    return machine_cache_dir() / "connect" / f"{AGENT_FOR.get(target, target)}-{auth}"


def mark_connected(target: str, auth: str, model: str | None) -> Path:
    """Record that `connect` completed: when, and the model it pinged."""
    import json
    from datetime import datetime, timezone

    path = record_dir(target, auth) / "connected.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "target": target, "auth": auth, "model": model,
        "connected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    return path


def is_connected(target: str, auth: str) -> bool:
    """`connect` completed for this pair and what a search reads is still
    there: the record, and for codex/pi the staged home too."""
    if not (record_dir(target, auth) / "connected.json").exists():
        return False
    homes = Path.home() / ".cache" / "hillclimb"  # where codex_home/pi_home write
    if target == "codex":
        return (homes / "codex-home" / auth).is_dir()
    if target == "pi":
        return (homes / "pi-home" / auth).is_dir()
    return True


def configured_auth(config, target: str) -> str | None:
    """The billing mode this config actually selects for a agent, or None
    when nothing routes to it. Routing overrides the scalar per operator, so
    a agent can appear under several modes; the global one wins the row,
    else the first routed one in sorted order."""
    agent = AGENT_FOR.get(target)
    if agent is None:  # openrouter is a route, not a agent
        return "openrouter"
    if config.agent == agent:
        return config.agent_auth
    default = config.routing.get("default")
    routed: list[str] = []
    for name in sorted(config.routing):
        route = config.routing[name]
        route_agent = route.agent or (default.agent if default else None) or config.agent
        if route_agent != agent:
            continue
        routed.append(
            route.agent_auth
            or (default.agent_auth if default else None)
            or config.agent_auth
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
        is_default = AGENT_FOR.get(target) == config.agent
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


def staged_homes(target: str) -> list[Path]:
    """Everything `connect <target>` materialized under the machine cache
    for searches to read — the isolated per-auth homes (every auth mode of
    the target) and the ping's scratch dirs. What `disconnect` removes. The
    agent's own login (`~/.claude`, `~/.codex`, `~/.pi`) is never among
    them: hillclimb may start a login it needs, never end one — the account
    belongs to the person, not to hillclimb."""
    from hillclimb.project import machine_cache_dir

    cache = Path.home() / ".cache" / "hillclimb"  # where codex_home/pi_home write
    pings = machine_cache_dir() / "connect"
    found: list[Path] = []
    if target == "codex":
        found += sorted((cache / "codex-home").glob("*"))
        found += sorted(pings.glob("codex-*"))
    elif target == "pi":
        found += sorted((cache / "pi-home").glob("*"))
        found += sorted(pings.glob("pi-*"))
    elif target == "claude":
        found += sorted(pings.glob("claude-code-*"))
    elif target == "openrouter":
        found += [p for p in (cache / "codex-home" / "openrouter", cache / "pi-home" / "openrouter") if p.exists()]
        found += sorted(pings.glob("*-openrouter"))
    return [p for p in found if p.is_dir()]


def remove_staged(target: str) -> list[Path]:
    """Delete `staged_homes(target)`; returns what was removed."""
    removed = []
    for path in staged_homes(target):
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
    return removed


def import_credentials(target: str, auth: str, models_file: Path | None = None) -> Path | None:
    """Materialize the isolated per-auth home a search will read, now rather
    than inside the first operator call. Returns the directory, or None for a
    target that has none (claude-code uses the ambient login)."""
    if target == "codex":
        from hillclimb.agents.codex_cli import codex_home

        return codex_home(auth)
    if target == "pi":
        from hillclimb.agents.pi_cli import pi_home

        return pi_home(auth, models_file)
    return None


def ping(agent_name: str, auth: str, model: str, *, models_file: Path | None = None):  # noqa: D417
    """One real, tool-free operator call — the proof that the credential, the
    model id and the provider route all work together. Costs a handful of
    tokens, which is the cheapest honest answer available; a key that is
    valid for the API but not for the requested model fails only here.

    Runs in the machine cache, never in a run: it is not a candidate.
    """
    from hillclimb.agents import get_agent
    from hillclimb.agents.base import AgentRequest
    from hillclimb.project import machine_cache_dir

    # the same dir `mark_connected` records into, so a failed re-ping also
    # clears the old mark: a connection is only as current as its last ping
    work_dir = machine_cache_dir() / "connect" / f"{agent_name}-{auth}"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    agent = get_agent(agent_name, auth=auth, pi_models_file=models_file)
    request = AgentRequest(
        operator="draft",  # the routed operators all look alike to a provider
        prompt=PING_PROMPT,
        candidate_dir=work_dir,
        timeout_s=PING_TIMEOUT_S,
        model=model,
    )
    # pi's preflight is the same call without tools; the others have no
    # cheaper door than invoke()
    preflight = getattr(agent, "preflight", None)
    return preflight(request) if preflight else agent.invoke(request)


# --------------------------------------------------------------------------
# the two files connecting may write
# --------------------------------------------------------------------------


def apply_config_defaults(text: str, updates: dict[str, str]) -> str:
    """Set top-level scalars in a config.yaml, in place, comments intact.

    A line-level edit rather than a YAML round-trip: `config.yaml` is
    hand-written and heavily commented (`hillclimb init` ships it that way),
    and re-emitting it from a parsed dict would throw all of that away. Only
    column-zero keys match, so `routing:`'s indented `agent:` is never
    touched, and the commented-out `# agent:` line `init` leaves behind is
    uncommented in place — the setting appears where its comment explains it.
    A key with nowhere to go lands next to the one written just before it, so
    `agent` and `agent_auth` end up adjacent instead of scattered.
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


def unpin_config_defaults(text: str, target: str) -> str:
    """The inverse of `apply_config_defaults` for one target: comment out
    the column-zero `agent:` (and `agent_auth:`) lines when they name
    that target's agent — a agent `disconnect` is not what pins another
    one — or, for the `openrouter` route, just the `agent_auth:` line that
    names it. Anything else is left byte-for-byte."""
    lines = text.splitlines()
    agent = next((line for line in lines if re.match(r"^agent:", line)), None)
    pinned = agent.split(":", 1)[1].strip() if agent else None
    if target == "openrouter":
        keys = ["agent_auth"] if any(re.match(r"^agent_auth:\s*openrouter\s*$", l) for l in lines) else []
    else:
        keys = ["agent", "agent_auth"] if pinned == AGENT_FOR[target] else []
    for key in keys:
        pattern = re.compile(rf"^{re.escape(key)}:")
        lines = [f"# {line}" if pattern.match(line) else line for line in lines]
    return "\n".join(lines) + ("\n" if lines else "")


def pins_agent(text: str) -> bool:
    """Does this config.yaml already choose a agent on purpose?

    Connecting should not silently repoint an existing setup at whatever was
    connected last; it should complete a fresh one. An active column-zero
    `agent:` is the signal that someone already decided.
    """
    return any(re.match(r"^agent:", line) for line in text.splitlines())


def env_file(config, *, local: bool = False) -> Path | None:
    """Where a provider key belongs: the user-level `.env` beside
    `~/.config/hillclimb/config.yaml` (every folder reads it, under its
    own), or with `local` this folder's — the `.env` beside
    `hillclimb.yaml`, the one `Config.load` reads. `local` without a
    hillclimb dir is None."""
    from hillclimb.project import user_env_path

    if not local:
        return user_env_path()
    if config.hillclimb_dir is None:
        return None
    return config.hillclimb_dir / ".env"


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


def remove_env_key(text: str, key: str) -> str:
    """Drop `KEY=…` from a .env, every other line intact."""
    pattern = re.compile(rf"^\s*(export\s+)?{re.escape(key)}\s*=")
    lines = [line for line in text.splitlines() if not pattern.match(line)]
    return "\n".join(lines) + ("\n" if lines else "")


def write_env_key(path: Path, key: str, value: str) -> None:
    """Store a provider key, readable only by its owner. The file may already
    hold other keys, so it is edited, never replaced. The user-level file
    is the first thing written under `~/.config/hillclimb/` on a fresh
    machine, so the folder is made on the way."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text() if path.exists() else ""
    path.write_text(upsert_env(text, key, value))
    path.chmod(0o600)


def _tilde(path: Path) -> str:
    """Paths under $HOME print as ~/… — shorter, and safe to paste."""
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)
