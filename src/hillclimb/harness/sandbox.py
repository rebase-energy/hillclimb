"""The OS sandbox every coding agent and every verifier run is started in.

Coding agents run with their permission prompts off and a verifier runs code a
coding agent wrote, both as the user. The sandbox is what keeps that from costing
the user anything: a `SandboxPolicy` says what the process tree may write,
what it may not read and what network it has, and `launch()` wraps an argv in
the operating system's own mechanism to enforce it:

  macOS   sandbox-exec (Seatbelt), part of the OS
  Linux   bwrap (bubblewrap), a system package
  other   none — `backend()` is None and the caller runs unsandboxed

What a policy means, on both:

  write       only the listed paths (plus the temp dirs and /dev)
  read_only   never writable, even inside a writable path or a temp dir
              (the problem: what scores a solution is not the solution's)
  deny_read   unreadable, and unix sockets under them unreachable
  network     `open`, `none`, or `proxy`: nothing but the engine's allowlist
              proxy, which tunnels HTTPS to `allow_hosts` and refuses the
              rest. That is how a coding agent without internet still reaches its
              model. With `none`/`proxy` localhost is closed too, except
              `local_ports` (on Linux the sandbox has a loopback of its own).

macOS refuses a sandbox inside a sandbox, so a policy is applied ONCE, round
the whole process: codex, which applies its own, is never wrapped, and a
meta-problem's verifier (it starts searches that sandbox themselves) is not
either.

Stdlib only at top level: inside a Linux sandbox without network this file
runs as a script (`bridge`) under the engine's interpreter.
"""

from __future__ import annotations

import asyncio
import glob
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable

ENV_SWITCH = "HILLCLIMB_SANDBOX"  # `off` runs everything unsandboxed (CI, containers)
BRIDGE_PORT = 3128  # where the proxy appears inside a Linux sandbox's own loopback

# never readable, relative to the home folder: credentials, and what reaches them
SECRET_PATHS = (
    ".ssh", ".aws", ".azure", ".gnupg", ".kube", ".docker", ".netrc", ".npmrc",
    ".pypirc", ".git-credentials", ".password-store", ".config/gcloud",
    ".config/gh", ".config/op", ".config/hillclimb/.env",
    ".bash_history", ".zsh_history", ".python_history",
    ".mozilla", ".config/google-chrome", ".config/chromium",
    "Library/Cookies", "Library/Safari",
    "Library/Application Support/Google/Chrome",
    "Library/Application Support/Firefox",
    "Library/Application Support/com.apple.TCC",
)
# a process that reaches a container runtime's socket has left the sandbox
RUNTIME_SOCKETS = (
    "/var/run/docker.sock", "/run/docker.sock", "/run/podman/podman.sock",
    "/run/containerd/containerd.sock",
)
# writable for everyone: what numeric and ML libraries cache under the home folder
CACHE_PATHS = (".cache/huggingface", ".cache/torch", ".cache/matplotlib", ".matplotlib")
# the user's own coding-agent setups (settings, plugins, skills, MCP servers,
# logins, history): never read by anything a search starts. Operators get an
# isolated home of hillclimb's instead (`claude-home`, `codex-home` below)
PERSONAL_AGENT_PATHS = (".claude", ".claude.json", ".codex", ".agents")
# each coding agent's operator home under hillclimb's machine cache (and, for
# pi, still its own folder); a coding agent reads only its own
AGENT_HOMES = {
    "claude-code": ((), ("claude-home",)),
    "codex": ((), ("codex-home",)),
    "pi": ((".pi",), ("pi-home",)),
}
# the search's own records: the journal carries holdout scores
SEARCH_RECORDS = ("journal.jsonl", "holdout-eval")
STORE_FILES = ("store.sqlite", "store.sqlite-wal", "store.sqlite-shm", ".env")

log: Callable[[str], None] | None = None  # the engine's log, for what the proxy refuses


class SandboxUnavailable(RuntimeError):
    """The sandbox is on and this machine cannot start one."""


def real(path: str | os.PathLike) -> str:
    """The path the OS enforces on: absolute, symlinks resolved (/tmp is
    /private/tmp on macOS)."""
    return os.path.realpath(os.path.expanduser(str(path)))


def _home(*relative: str) -> tuple[str, ...]:
    return tuple(real(Path.home() / name) for name in relative)


@dataclass(frozen=True)
class SandboxPolicy:
    write: tuple[str, ...] = ()
    # `<path>*`: a file and the siblings its owner writes beside it (locks, backups)
    write_prefix: tuple[str, ...] = ()
    read_only: tuple[str, ...] = ()
    deny_read: tuple[str, ...] = ()
    network: str = "open"  # open | none | proxy
    allow_hosts: tuple[str, ...] = ()  # proxy: these hosts and their subdomains
    local_ports: tuple[int, ...] = ()  # none/proxy: localhost ports left open

    def writable(self, *paths: str | os.PathLike, prefix: tuple[str, ...] = ()) -> "SandboxPolicy":
        return replace(
            self,
            write=_unique(self.write + tuple(real(p) for p in paths)),
            write_prefix=_unique(self.write_prefix + tuple(real(p) for p in prefix)),
        )

    def protected(self, *paths: str | os.PathLike) -> "SandboxPolicy":
        """Never writable, whatever else is (the problem and its data)."""
        return replace(self, read_only=_unique(self.read_only + tuple(real(p) for p in paths)))

    def unreadable(self, *paths: str | os.PathLike) -> "SandboxPolicy":
        return replace(self, deny_read=_unique(self.deny_read + tuple(real(p) for p in paths)))

    def offline(self) -> "SandboxPolicy":
        return replace(self, network="none", allow_hosts=())

    def for_agent(
        self, allow_internet: bool, *, hosts=(), local_ports=()
    ) -> "SandboxPolicy":
        """An coding agent's network: all of it, or only its model provider's hosts."""
        return self if allow_internet else self.through_proxy(hosts, local_ports)

    def through_proxy(self, hosts, local_ports=()) -> "SandboxPolicy":
        return replace(
            self,
            network="proxy",
            allow_hosts=_unique(tuple(h.lower().strip(".") for h in (*self.allow_hosts, *hosts) if h)),
            local_ports=_unique(self.local_ports + tuple(int(p) for p in local_ports)),
        )


def _unique(items: tuple) -> tuple:
    return tuple(dict.fromkeys(items))


# --- which sandbox this machine has ---


def switched_off() -> bool:
    return os.environ.get(ENV_SWITCH, "").strip().lower() in {"off", "0", "false", "no"}


def enabled(config) -> bool:
    """Is the sandbox asked for? (`sandbox: off` or $HILLCLIMB_SANDBOX=off say no.)"""
    return bool(config.sandbox.enabled) and not switched_off()


OPT_OUT = (
    "To run without a sandbox, set `sandbox: off` in hillclimb.yaml — coding agents and "
    "solutions then run with your full user rights."
)
_BWRAP_PROBE = [
    "bwrap", "--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc",
    "--unshare-pid", "--unshare-net", "--die-with-parent", "true",
]


@lru_cache(maxsize=1)
def backend() -> str | None:
    """`seatbelt`, `bwrap`, or None where no sandbox exists for the OS
    (Windows). Started once for real; raises SandboxUnavailable with the fix
    when the OS has one and it does not start here."""
    if sys.platform == "darwin":
        name, probe = "seatbelt", ["sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"]
    elif sys.platform.startswith("linux"):
        name, probe = "bwrap", _BWRAP_PROBE
    else:
        return None
    if shutil.which(probe[0]) is None:
        if name == "bwrap":
            raise SandboxUnavailable(
                "the sandbox needs bubblewrap, which is not installed. Install it with "
                "`sudo apt install bubblewrap` (Debian, Ubuntu), `sudo dnf install bubblewrap` "
                f"(Fedora) or `sudo pacman -S bubblewrap` (Arch). {OPT_OUT}"
            )
        raise SandboxUnavailable(f"the sandbox needs {probe[0]}, which is not on PATH. {OPT_OUT}")
    try:
        done = subprocess.run(probe, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SandboxUnavailable(f"the sandbox ({probe[0]}) does not start: {exc}. {OPT_OUT}") from exc
    if done.returncode == 0:
        return name
    detail = (done.stderr or "").strip()[:300] or f"exit {done.returncode}"
    if name == "seatbelt":
        hint = (
            "hillclimb is probably running inside another sandbox already (a coding "
            "agent's, for example), and macOS allows no sandbox inside a sandbox. Run it "
            "from a plain terminal."
        )
    else:
        hint = (
            "Unprivileged user namespaces are probably restricted. On Ubuntu 24.04 and later: "
            "`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`. Inside a "
            "container, the container itself has to allow them."
        )
    raise SandboxUnavailable(f"the sandbox ({probe[0]}) does not start: {detail}. {hint} {OPT_OUT}")


def active(config) -> bool:
    """Will processes of this config be sandboxed? Raises SandboxUnavailable
    when they should be and cannot."""
    return enabled(config) and backend() is not None


# --- the policies hillclimb uses ---


def _base(config, search_dir: Path | None, agent: str | None) -> SandboxPolicy:
    deny = list(_home(*SECRET_PATHS)) + list(_home(*PERSONAL_AGENT_PATHS)) + [real(p) for p in RUNTIME_SOCKETS]
    from hillclimb.project import machine_cache_dir, user_env_path

    deny.append(real(user_env_path()))
    for name, (in_home, in_cache) in AGENT_HOMES.items():
        if name != agent:
            deny += _home(*in_home)
            deny += [real(machine_cache_dir() / part) for part in in_cache]
    hillclimb_dir = getattr(config, "hillclimb_dir", None)
    if hillclimb_dir is not None:
        deny += [real(Path(hillclimb_dir) / name) for name in STORE_FILES]
    if search_dir is not None:
        deny += [real(Path(search_dir) / name) for name in SEARCH_RECORDS]
    settings = config.sandbox

    def anchored(paths) -> tuple[str, ...]:
        # like every path in the config: relative to the hillclimb dir
        return tuple(real(Path(hillclimb_dir or ".") / Path(p).expanduser()) for p in paths)

    return SandboxPolicy(
        write=_unique(_home(*CACHE_PATHS) + anchored(settings.write)),
        deny_read=_unique(tuple(deny) + anchored(settings.deny_read)),
        allow_hosts=tuple(settings.allow_hosts),
        local_ports=tuple(settings.local_ports),
    )


def _for_problem(policy: SandboxPolicy, problem) -> SandboxPolicy:
    """The problem as everything run on its behalf sees it: its folder and
    data read-only (a solution or an agent that rewrote the verifier would
    choose its own score), and its private paths, when it names any, kept
    from the reader too."""
    if problem is None:
        return policy
    dirs = [d for d in (getattr(problem, "problem_dir", None), getattr(problem, "data_dir", None)) if d]
    return policy.protected(*dirs)


def private_paths(problem) -> tuple[str, ...]:
    """What only the problem's scorer may read (`private:` in problem.yaml)."""
    return tuple(real(p) for p in getattr(problem, "private_paths", None) or ())


def holdout_input_paths(problem) -> tuple[str, ...]:
    """What only holdout runs may read (`holdout_inputs:` in problem.yaml)."""
    return tuple(real(p) for p in getattr(problem, "holdout_inputs", None) or ())


def hidden_from_agents(problem) -> tuple[str, ...]:
    """Everything a coding agent, a unit test or a validation run of the
    solution must not read: the scorer's private data and the holdout inputs."""
    return private_paths(problem) + holdout_input_paths(problem)


def agent_policy(config, search_dir: Path | None, agent: str, problem=None) -> SandboxPolicy | None:
    """What a coding agent runs under; None = unsandboxed. The coding agent adds
    its candidate dir, its own state and — without internet — the proxy. With
    the `problem` it works on, the problem is read-only and its private paths
    unreadable: an agent never sees what only the scorer may."""
    if not active(config):
        return None
    policy = _for_problem(_base(config, search_dir, agent), problem)
    return policy.unreadable(*hidden_from_agents(problem)) if problem is not None else policy


def verifier_policy(
    config, problem, search_dir: Path | None, *, holdout: bool = False
) -> SandboxPolicy | None:
    """What a verifier run (validation, unit tests, holdout) runs under; the
    executor adds the dir it runs in. The network is the problem's
    `allow_internet_during_solution`; a holdout that fetches gated data has it."""
    if not active(config):
        return None
    if getattr(problem, "solution_kind", "program") == "climber":
        return None  # its verifier starts searches, and those sandbox themselves
    policy = _for_problem(_base(config, search_dir, None), problem)
    if holdout and search_dir is not None:
        here = real(Path(search_dir) / "holdout-eval")
        policy = replace(policy, deny_read=tuple(p for p in policy.deny_read if p != here))
    online = problem.allow_internet_during_solution or (
        holdout and getattr(problem, "holdout_needs_credentials", False)
    )
    return policy if online else policy.offline()


def candidate_root(path: Path) -> Path:
    """The candidate dir a trial or replicate dir belongs to (else `path`)."""
    path = Path(path)
    for parent in (path, *path.parents):
        if parent.parent.name == "candidates":
            return parent
    return path


# --- starting a process inside ---


@dataclass(frozen=True)
class Launch:
    argv: list[str]
    env: dict[str, str]  # to lay over the child's environment


def launch(argv: list[str], policy: SandboxPolicy | None) -> Launch:
    """`argv` as it must be started to run under `policy` (None = as it is)."""
    if policy is None:
        return Launch(list(argv), {})
    kind = backend()
    if kind is None:
        return Launch(list(argv), {})
    proxy = proxy_for(frozenset(policy.allow_hosts)) if policy.network == "proxy" else None
    if kind == "seatbelt":
        port = proxy.port if proxy else None
        return Launch(
            ["sandbox-exec", "-p", seatbelt_profile(policy, port), *argv],
            proxy_env(port) if port else {},
        )
    inner = list(argv)
    if proxy is not None:
        inner = [sys.executable, real(__file__), "bridge", str(BRIDGE_PORT), proxy.socket_path, "--", *argv]
    return Launch(
        ["bwrap", *bwrap_args(policy, proxy.socket_dir if proxy else None), "--", *inner],
        proxy_env(BRIDGE_PORT) if proxy else {},
    )


def proxy_env(port: int) -> dict[str, str]:
    url = f"http://127.0.0.1:{port}"
    env = {name: url for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")}
    env.update(NO_PROXY="localhost,127.0.0.1", no_proxy="localhost,127.0.0.1")
    env["NODE_USE_ENV_PROXY"] = "1"  # node reads the proxy variables only when told to
    return env


def _quoted(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _regex_quoted(text: str) -> str:
    escaped = "".join(ch if ch.isalnum() or ch in "/_-" else "\\" + ch for ch in text)
    return '#"^' + escaped.replace('"', '\\"') + '"'


def temp_dirs() -> tuple[str, ...]:
    dirs = ["/tmp", "/var/tmp", tempfile.gettempdir()]
    if sys.platform == "darwin":
        dirs.append("/private/var/folders")  # every per-user temp and cache dir
    return _unique(tuple(real(d) for d in dirs))


def seatbelt_profile(policy: SandboxPolicy, proxy_port: int | None = None) -> str:
    """The Seatbelt profile for `policy`. Later rules win."""
    writable = [f"(subpath {_quoted(p)})" for p in (*temp_dirs(), *policy.write)]
    writable += [f"(regex {_regex_quoted(p)})" for p in policy.write_prefix]
    rules = [
        "(version 1)",
        "(allow default)",
        "(deny file-write*)",
        f'(allow file-write* (regex #"^/dev/") {" ".join(writable)})',
        *(
            [f'(deny file-write* {" ".join(f"(subpath {_quoted(p)})" for p in policy.read_only)})']
            if policy.read_only else []
        ),
        # its own process tree is its to stop; nothing outside is
        "(deny signal)",
        "(allow signal (target same-sandbox))",
    ]
    if policy.deny_read:
        denied = " ".join(f"(subpath {_quoted(p)})" for p in policy.deny_read)
        rules.append(f"(deny file-read* file-write* {denied})")
        rules.append(f"(deny network-outbound {denied})")  # the unix sockets under them
    if policy.network != "open":
        rules.append("(deny network-outbound (remote ip))")
        # name lookups leave through this socket; a proxy resolves for its clients
        rules.append('(deny network-outbound (literal "/private/var/run/mDNSResponder"))')
        ports = [*policy.local_ports, *([proxy_port] if proxy_port else [])]
        if ports:
            open_ports = " ".join(f'(remote ip "localhost:{port}")' for port in ports)
            rules.append(f"(allow network-outbound {open_ports})")
    return "".join(rules)


def bwrap_args(policy: SandboxPolicy, proxy_dir: str | None = None) -> list[str]:
    """bubblewrap's arguments for `policy`: the whole disk read-only, then
    what is writable, then what is hidden (a later mount covers an earlier)."""
    args = ["--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc"]
    for path in (*temp_dirs(), *policy.write):
        args += ["--bind-try", path, path]
    for prefix in policy.write_prefix:
        for path in sorted(glob.glob(glob.escape(prefix) + "*")):
            args += ["--bind-try", path, path]
    for path in policy.read_only:
        args += ["--ro-bind-try", path, path]
    for path in policy.deny_read:
        if os.path.isdir(path):
            args += ["--tmpfs", path]
        elif os.path.exists(path):
            args += ["--ro-bind", "/dev/null", path]
    if proxy_dir is not None:
        args += ["--ro-bind", proxy_dir, proxy_dir]
    args += ["--unshare-pid", "--unshare-ipc", "--die-with-parent"]
    if policy.network != "open":
        args.append("--unshare-net")
    return args


# --- the allowlist proxy ---


def host_allowed(host: str, allowed) -> bool:
    host = host.lower().strip(".")
    return any(host == name or host.endswith("." + name) for name in allowed)


class AllowlistProxy:
    """An HTTPS (CONNECT) proxy on a thread of the engine: tunnels to the
    allowed hosts and answers 403 to everything else, plain HTTP included.
    Listens on localhost (macOS) or on a unix socket a Linux sandbox gets bound
    in — its network namespace has no way to the host's localhost."""

    def __init__(self, hosts, *, unix: bool):
        self.hosts = frozenset(hosts)
        self.unix = unix
        self.port: int | None = None
        self.socket_dir: str | None = None
        self.socket_path: str | None = None
        self.refused: set[str] = set()
        self._ready = threading.Event()
        self._error: BaseException | None = None

    def start(self) -> "AllowlistProxy":
        threading.Thread(target=self._run, name="hillclimb-sandbox-proxy", daemon=True).start()
        if not self._ready.wait(timeout=15) or self._error is not None:
            raise SandboxUnavailable(f"the sandbox's proxy did not start: {self._error or 'timeout'}")
        return self

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:  # noqa: BLE001 — reported by start()
            self._error = exc
            self._ready.set()

    async def _serve(self) -> None:
        if self.unix:
            self.socket_dir = tempfile.mkdtemp(prefix="hillclimb-proxy-")
            self.socket_path = os.path.join(self.socket_dir, "proxy.sock")
            server = await asyncio.start_unix_server(self._handle, self.socket_path)
        else:
            server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
            self.port = server.sockets[0].getsockname()[1]
        self._ready.set()
        async with server:
            await server.serve_forever()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        upstream = None
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=30)
            method, target, _ = head.split(b"\r\n", 1)[0].decode("latin-1").split(" ", 2)
            host, _, port = target.rpartition(":")
            host = host.strip("[]")
            if method.upper() != "CONNECT" or not port.isdigit() or not host_allowed(host, self.hosts):
                self._refuse(host if method.upper() == "CONNECT" else target)
                writer.write(b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n")
                await writer.drain()
                return
            try:
                upstream_reader, upstream = await asyncio.wait_for(
                    asyncio.open_connection(host, int(port)), timeout=30
                )
            except (OSError, asyncio.TimeoutError):
                writer.write(b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n")
                await writer.drain()
                return
            writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            await writer.drain()
            await asyncio.gather(_pump(reader, upstream), _pump(upstream_reader, writer))
        except (OSError, ValueError, asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            pass
        finally:
            for stream in (writer, upstream):
                if stream is not None:
                    stream.close()

    def _refuse(self, host: str) -> None:
        if host in self.refused:
            return
        self.refused.add(host)
        if log is not None:
            log(f"sandbox: no internet for coding agents, refused {host}")


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except OSError:
        pass
    finally:
        writer.close()


@lru_cache(maxsize=None)
def proxy_for(hosts: frozenset) -> AllowlistProxy:
    """The engine's proxy for one set of hosts, started on first use."""
    return AllowlistProxy(hosts, unix=sys.platform.startswith("linux")).start()


# --- inside a Linux sandbox: localhost:port -> the proxy's unix socket ---


def _relay(source: socket.socket, sink: socket.socket) -> None:
    try:
        while data := source.recv(65536):
            sink.sendall(data)
    except OSError:
        pass
    finally:
        for sock in (source, sink):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _accept(listener: socket.socket, socket_path: str) -> None:
    while True:
        client, _ = listener.accept()
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            upstream.connect(socket_path)
        except OSError:
            client.close()
            continue
        for pair in ((client, upstream), (upstream, client)):
            threading.Thread(target=_relay, args=pair, daemon=True).start()


def bridge(port: int, socket_path: str, argv: list[str]) -> int:
    """Listen on the sandbox's own localhost:port, hand every connection to
    the proxy's socket, and run `argv` beside it; its exit code is ours."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(64)
    threading.Thread(target=_accept, args=(listener, socket_path), daemon=True).start()
    child = subprocess.Popen(argv)
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, lambda received, _frame: child.send_signal(received))
    code = child.wait()
    return code if code >= 0 else 128 - code


def _main(argv: list[str]) -> int:
    if len(argv) < 5 or argv[0] != "bridge" or argv[3] != "--":
        print("usage: sandbox.py bridge PORT SOCKET -- COMMAND...", file=sys.stderr)
        return 2
    return bridge(int(argv[1]), argv[2], argv[4:])


# where hillclimb runs sandboxed on a Windows machine: WSL2 has a real Linux
# kernel, so bubblewrap works there exactly as on Linux
WINDOWS_HINT = (
    "on Windows, run hillclimb inside WSL2 (Ubuntu), where the Linux sandbox works: "
    "https://docs.hillclimb.sh/security-and-sandboxes#windows-via-wsl2"
)


def no_sandbox_reason() -> str:
    """Why there is no sandbox on this machine, and on Windows where to get one."""
    reason = "none exists for this operating system"
    return f"{reason} ({WINDOWS_HINT})" if sys.platform == "win32" else reason


NEEDS_SANDBOX = (
    "allow_internet_for_agents: false needs the sandbox, and it is off or does not "
    "exist on this operating system"
)


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
