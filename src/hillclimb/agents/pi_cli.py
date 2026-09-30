"""pi coding-agent operator agent.

One Hillclimb operator call is one headless ``pi`` process.  pi's JSON event
stream is kept verbatim for diagnostics while this module extracts the small
amount of accounting and status information the search engine needs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from hillclimb.agents.base import OperatorRequest, OperatorResult
from hillclimb.agents.claude_code import (
    PID_FILE,
    RATE_LIMIT_MARKERS,
    STREAM_FILE,
    usage_total_tokens,
)
from hillclimb.harness import sandbox
from hillclimb.harness.oscompat import new_group_kwargs, runnable
from hillclimb.harness.pricing import cost_usd
from hillclimb.harness.procs import Reaper


SAMPLING_EXTENSION = Path(__file__).with_name("pi_ext") / "hillclimb-sampling.ts"
# what pi has to reach without internet: the providers its logins and keys
# speak to (`sandbox.allow_hosts` adds more; a models_file adds its own)
MODEL_HOSTS = (
    "api.anthropic.com", "console.anthropic.com", "claude.ai",
    "api.openai.com", "auth.openai.com", "chatgpt.com",
    "generativelanguage.googleapis.com", "oauth2.googleapis.com",
)
OPENROUTER_HOSTS = ("openrouter.ai",)
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
TOOLS = "read,write,edit,bash,grep,find,ls"
CREDIT_MARKERS = (
    "payment required",
    "requires more credits",
    "insufficient credits",
    "insufficient_quota",
)


def _has_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _atomic_copy(source: Path, target: Path) -> None:
    """Copy a shared credential/config file without exposing a partial file."""
    fd, staging = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=target.suffix, dir=target.parent)
    os.close(fd)
    try:
        shutil.copy2(source, staging)
        os.replace(staging, target)
    finally:
        Path(staging).unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict) -> None:
    fd, staging = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(staging, path)
    finally:
        Path(staging).unlink(missing_ok=True)


def pi_home(auth: str, models_file: Path | None = None) -> Path:
    """Return pi's isolated config dir and populate only Hillclimb-owned data."""
    home = Path.home() / ".cache" / "hillclimb" / "pi-home" / auth
    models_data = None
    if models_file is not None:
        source = Path(models_file).expanduser()
        if not source.is_file():
            raise RuntimeError(f"pi.models_file does not exist: {source}")
        models_data = source.read_bytes()
        try:
            models = json.loads(models_data)
            if not isinstance(models, dict) or not isinstance(models.get("providers"), dict):
                raise ValueError("expected an object with a providers mapping")
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError(f"invalid pi.models_file {source}: {exc}") from exc
        # Concurrent searches can use different local servers. Never replace
        # the configuration another live process is about to load.
        home /= hashlib.sha256(models_data).hexdigest()[:16]
    home.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        home / "settings.json",
        {
            "enableInstallTelemetry": False,
            "quietStartup": True,
        },
    )

    auth_target = home / "auth.json"
    if auth == "subscription":
        source = Path.home() / ".pi" / "agent" / "auth.json"
        if source.exists() and (
            not auth_target.exists() or source.stat().st_mtime > auth_target.stat().st_mtime
        ):
            _atomic_copy(source, auth_target)
    else:
        # A stale interactive login must not become a fallback for a billed
        # API-key/OpenRouter route.
        auth_target.unlink(missing_ok=True)

    models_target = home / "models.json"
    if models_data is not None and not models_target.exists():
        _atomic_write_json(models_target, models)
    return home


def pi_env(auth: str = "subscription", models_file: Path | None = None) -> dict[str, str]:
    """Build pi's isolated child environment for the selected billing mode."""
    from hillclimb.harness.executor import single_threaded

    env = os.environ.copy()
    if auth not in {"subscription", "api-key", "openrouter"}:
        raise RuntimeError(f"unknown pi agent_auth: {auth!r}")
    if auth == "openrouter":
        if not env.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "agent_auth: openrouter needs OPENROUTER_API_KEY — export it "
                "or put it in a .env beside hillclimb.yaml"
            )
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("OPENAI_API_KEY", None)
    elif auth == "subscription":
        # Provider keys take precedence over pi's stored OAuth credentials.
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("OPENAI_API_KEY", None)
        env.pop("OPENROUTER_API_KEY", None)
    env["PI_CODING_AGENT_DIR"] = str(pi_home(auth, models_file))
    # Keep catalogue/startup behavior fixed during a search. Updating pi (and
    # its bundled model catalogue) is an explicit setup action.
    env["PI_OFFLINE"] = "1"
    env["PI_TELEMETRY"] = "0"
    return single_threaded(env)


def _normalized_usage(usage: dict) -> dict[str, int]:
    return {
        "input_tokens": int(usage.get("input") or 0),
        "output_tokens": int(usage.get("output") or 0),
        "cache_creation_input_tokens": int(usage.get("cacheWrite") or 0),
        "cache_read_input_tokens": int(usage.get("cacheRead") or 0),
    }


def _served_model(requested: str, message: dict) -> str | None:
    model = message.get("responseModel") or message.get("model")
    if not model:
        return None
    provider = message.get("provider")
    qualified = f"{provider}/{model}" if provider and not str(model).startswith(f"{provider}/") else str(model)
    if requested in {model, qualified}:
        return requested
    return qualified


class _PiStreamReader(threading.Thread):
    """Drain pi JSONL, preserving it verbatim and accumulating final events."""

    def __init__(self, stdout, stream_path: Path, requested_model: str):
        super().__init__(daemon=True, name="pi-stream-reader")
        self.stdout = stdout
        self.stream_path = stream_path
        self.requested_model = requested_model
        self.session_id: str | None = None
        self.model_id: str | None = None
        self.providers: set[str] = set()
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        self.cost_usd = 0.0
        self.num_turns = 0
        self.last_stop_reason: str | None = None
        self.error_message = ""
        self.rate_limited = False
        self.out_of_credits = False
        self.reader_error = ""

    def _read_message(self, event: dict) -> None:
        if event.get("type") == "session" and event.get("id"):
            self.session_id = str(event["id"])
            return
        if event.get("type") == "turn_end":
            self.num_turns += 1
            return
        if event.get("type") == "agent_end":
            # Agent-core can report runtime failures only here, without a
            # message_end. Do not re-count the messages repeated in this event.
            assistants = [m for m in event.get("messages", []) if m.get("role") == "assistant"]
            if assistants and assistants[-1].get("stopReason") in {"error", "aborted"}:
                self._read_status(assistants[-1])
            return
        if event.get("type") != "message_end":
            return
        message = event.get("message") or {}
        if message.get("role") != "assistant":
            return
        if self.model_id is None:
            self.model_id = _served_model(self.requested_model, message)
        if message.get("provider"):
            self.providers.add(str(message["provider"]))
        for key, count in _normalized_usage(message.get("usage") or {}).items():
            self.usage[key] += count
        cost = (message.get("usage") or {}).get("cost") or {}
        self.cost_usd += float(cost.get("total") or 0.0)
        self._read_status(message)

    def _read_status(self, message: dict) -> None:
        self.last_stop_reason = message.get("stopReason")
        self.error_message = str(message.get("errorMessage") or "")
        # A successful retry supersedes a previous provider error.
        self.out_of_credits = self.last_stop_reason == "error" and (
            _has_marker(self.error_message, CREDIT_MARKERS)
            or bool(re.search(r"\b402\b", self.error_message))
        )
        self.rate_limited = self.last_stop_reason == "error" and not self.out_of_credits and (
            _has_marker(self.error_message, RATE_LIMIT_MARKERS)
            or bool(re.search(r"\b429\b", self.error_message))
        )

    def run(self) -> None:
        with self.stream_path.open("w") as sink:
            for line in self.stdout:
                sink.write(line)
                sink.flush()
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    if _has_marker(line, RATE_LIMIT_MARKERS):
                        self.rate_limited = True
                    if _has_marker(line, CREDIT_MARKERS):
                        self.out_of_credits = True
                    continue
                if isinstance(event, dict):
                    try:
                        self._read_message(event)
                    except (AttributeError, TypeError, ValueError) as exc:
                        self.reader_error = f"invalid pi stream event: {exc}"


class PiCliAgent:
    """Run Hillclimb operators through an isolated local pi CLI."""

    name = "pi"

    def __init__(
        self,
        pi_bin: str = "pi",
        auth: str = "subscription",
        abort: threading.Event | None = None,
        models_file: Path | None = None,
    ):
        self.pi_bin = pi_bin
        self.auth = auth
        self.abort = abort
        self.models_file = Path(models_file) if models_file is not None else None

    @staticmethod
    def _session_dir(candidate_dir: Path) -> Path:
        search_dir = (
            candidate_dir.parents[1]
            if candidate_dir.parent.name == "candidates" and len(candidate_dir.parents) > 1
            else candidate_dir
        )
        session_dir = search_dir / "pi-sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _command(self, request: OperatorRequest, *, no_tools: bool = False) -> list[str]:
        cmd = [
            self.pi_bin,
            "-p",
            "--mode",
            "json",
            "--no-extensions",
            "-e",
            str(SAMPLING_EXTENSION),
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
        ]
        cmd += ["--no-tools"] if no_tools else ["--tools", TOOLS]
        cmd += [
            "--session-dir",
            str(self._session_dir(Path(request.candidate_dir).resolve())),
        ]
        if request.resume_session_id:
            # --session restores the parent's cwd; --fork keeps its history
            # but binds tools to the new candidate and avoids sibling races.
            cmd += ["--fork", request.resume_session_id]
        if self.auth == "openrouter":
            cmd += ["--provider", "openrouter", "--model", request.model.removeprefix("openrouter/")]
        else:
            cmd += ["--model", request.model]
        if no_tools:
            cmd += ["--system-prompt", "Reply briefly to the user's request."]
        return cmd

    def _provider_endpoints(self) -> tuple[list[str], list[int]]:
        """(hosts, localhost ports) of the providers in `pi.models_file`."""
        hosts: list[str] = []
        ports: list[int] = []
        if self.models_file is None:
            return hosts, ports
        from urllib.parse import urlsplit

        try:
            providers = json.loads(Path(self.models_file).expanduser().read_text())["providers"]
        except (OSError, ValueError, KeyError, TypeError):
            return hosts, ports
        for provider in providers.values() if isinstance(providers, dict) else ():
            url = urlsplit(str(provider.get("baseUrl", ""))) if isinstance(provider, dict) else None
            if url is None or not url.hostname:
                continue
            if url.hostname in LOCAL_HOSTS:
                ports.append(url.port or (443 if url.scheme == "https" else 80))
            else:
                hosts.append(url.hostname)
        return hosts, ports

    def _sandboxed(
        self, cmd: list[str], request: OperatorRequest, candidate_dir: Path, env: dict[str, str]
    ) -> sandbox.Launch:
        """The command as it starts: inside the sandbox, writing only to the
        candidate dir, its sessions and pi's isolated home, and without
        internet reaching only its model providers."""
        policy = request.sandbox
        if policy is None:
            if not request.allow_internet:
                raise sandbox.SandboxUnavailable(sandbox.NEEDS_SANDBOX)
            return sandbox.Launch(list(cmd), {})
        hosts, ports = self._provider_endpoints()
        hosts += OPENROUTER_HOSTS if self.auth == "openrouter" else MODEL_HOSTS
        policy = policy.writable(
            candidate_dir, self._session_dir(candidate_dir), env["PI_CODING_AGENT_DIR"]
        ).for_agent(request.allow_internet, hosts=hosts, local_ports=ports)
        return sandbox.launch(list(cmd), policy)

    def preflight(self, request: OperatorRequest) -> OperatorResult:
        """Make the cheap, tool-free provider call used before search work."""
        return self._invoke(request, no_tools=True)

    def invoke(self, request: OperatorRequest) -> OperatorResult:
        return self._invoke(request, no_tools=False)

    def _invoke(self, request: OperatorRequest, *, no_tools: bool) -> OperatorResult:
        candidate_dir = Path(request.candidate_dir).resolve()
        candidate_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._command(request, no_tools=no_tools)
        stream_path = candidate_dir / STREAM_FILE
        pid_path = candidate_dir / PID_FILE
        stderr_path = candidate_dir / "agent_stderr.log"
        raw_path = candidate_dir / "agent_raw.json"
        start = time.monotonic()
        timed_out = False
        aborted = False
        spawn_error = ""
        proc: subprocess.Popen | None = None
        reaper: Reaper | None = None
        reader: _PiStreamReader | None = None

        try:
            child_env = pi_env(self.auth, self.models_file)
            child_env.pop("HILLCLIMB_SAMPLING", None)
            started = self._sandboxed(cmd, request, candidate_dir, child_env)
            cmd = started.argv
            child_env.update(started.env)
            if request.sampling:
                child_env["HILLCLIMB_SAMPLING"] = json.dumps(
                    request.sampling, sort_keys=True, separators=(",", ":")
                )
        except (RuntimeError, OSError, ValueError) as exc:
            return OperatorResult(ok=False, error_kind="error", error_message=str(exc))

        try:
            with stderr_path.open("w") as stderr_sink:
                try:
                    proc = subprocess.Popen(
                        runnable(cmd),
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=stderr_sink,
                        text=True,
                        cwd=candidate_dir,
                        env=child_env,
                        **new_group_kwargs(),
                    )
                except OSError as exc:
                    spawn_error = str(exc)
                if proc is not None:
                    reaper = Reaper(proc)  # reaps through wait4: the call's CPU rides along
                    pid_path.write_text(str(proc.pid))
                    reader = _PiStreamReader(proc.stdout, stream_path, request.model)
                    reader.start()
                    try:
                        proc.stdin.write(request.prompt)
                        proc.stdin.close()
                    except BrokenPipeError:
                        pass
                    deadline = time.monotonic() + request.timeout_s
                    while reaper.wait(timeout=1.0) is None:
                        if self.abort is not None and self.abort.is_set():
                            aborted = True
                            reaper.kill_group()
                            break
                        if time.monotonic() >= deadline:
                            timed_out = True
                            reaper.kill_group()
                            break
        finally:
            pid_path.unlink(missing_ok=True)
            if reaper is not None and reaper.poll() is None:
                reaper.kill_group()
            if reader is not None:
                reader.join(timeout=5)

        duration = time.monotonic() - start
        stderr_text = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""
        raw_path.write_text(
            json.dumps(
                {
                    "cmd": cmd,
                    "returncode": proc.returncode if proc is not None else None,
                    "stderr": stderr_text[-4000:],
                    "session_id": reader.session_id if reader is not None else None,
                    "last_stop_reason": reader.last_stop_reason if reader is not None else None,
                }
            )
        )
        common: dict = {
            "duration_s": duration,
            "raw_output_path": str(raw_path),
            "cpu_s": reaper.cpu_s if reaper is not None else None,
        }
        if reader is not None:
            token_usage = {key: value for key, value in reader.usage.items() if value}
            reported_cost = reader.cost_usd
            if reported_cost == 0 and (
                self.auth == "openrouter"
                or "openrouter" in reader.providers
                or request.model.startswith("openrouter/")
            ):
                reported_cost = cost_usd(
                    (reader.model_id or request.model).removeprefix("openrouter/"), token_usage
                )
            common.update(
                session_id=reader.session_id,
                model_id=reader.model_id,
                cost_usd=reported_cost,
                num_turns=reader.num_turns,
                total_tokens=usage_total_tokens(reader.usage) or None,
                token_usage=token_usage,
            )

        if spawn_error:
            return OperatorResult(
                ok=False,
                error_kind="error",
                error_message=f"could not start pi CLI: {spawn_error}",
                **common,
            )
        if aborted:
            return OperatorResult(
                ok=False,
                error_kind="aborted",
                error_message="agent call aborted (stop requested)",
                **common,
            )
        if timed_out:
            return OperatorResult(
                ok=False,
                error_kind="timeout",
                error_message=f"agent call exceeded {request.timeout_s}s",
                **common,
            )
        assert proc is not None and reader is not None
        if reader.reader_error:
            return OperatorResult(
                ok=False, error_kind="error", error_message=reader.reader_error, **common
            )
        error_text = reader.error_message or stderr_text
        failed = proc.returncode != 0 or reader.last_stop_reason not in {"stop", "toolUse"}
        if failed and (reader.out_of_credits or _has_marker(error_text, CREDIT_MARKERS)):
            return OperatorResult(
                ok=False,
                error_kind="out_of_credits",
                error_message=error_text[:500],
                **common,
            )
        if failed and (reader.rate_limited or _has_marker(error_text, RATE_LIMIT_MARKERS)):
            return OperatorResult(
                ok=False,
                error_kind="rate_limited",
                error_message=error_text[:500],
                **common,
            )
        if reader.last_stop_reason in {"error", "aborted", "length"}:
            return OperatorResult(
                ok=False,
                error_kind="aborted" if reader.last_stop_reason == "aborted" else "error",
                error_message=(error_text or f"pi stopped with {reader.last_stop_reason}")[:500],
                **common,
            )
        if reader.last_stop_reason not in {"stop", "toolUse"}:
            message = error_text or (
                f"pi exited {proc.returncode} without a completed assistant message"
            )
            return OperatorResult(
                ok=False, error_kind="error", error_message=message[:500], **common
            )
        if proc.returncode != 0:
            return OperatorResult(
                ok=False,
                error_kind="error",
                error_message=(stderr_text or f"pi exited {proc.returncode}")[:500],
                **common,
            )
        return OperatorResult(ok=True, **common)
