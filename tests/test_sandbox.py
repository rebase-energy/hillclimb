"""The OS sandbox (harness/sandbox.py): what a policy turns into, how the
agents and the verifier are started under it, and — where this machine can
start one — that it holds."""

from __future__ import annotations

import json
import os
import socket
import stat
import sys
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hillclimb import api
from hillclimb.agents.claude_code import ClaudeCodeAgent
from hillclimb.agents.codex_cli import CodexCliAgent
from hillclimb.agents.pi_cli import PiCliAgent
from hillclimb.config import Config
from hillclimb.harness import sandbox
from hillclimb.harness.executor import run_logged
from hillclimb.harness.sandbox import SandboxPolicy
from tests import test_claude_agent, test_codex_agent, test_pi_agent


# Windows has no sandbox (the harness runs unsandboxed there, and says so):
# what wraps an agent or writes a seatbelt profile is POSIX-only, and the
# agent stubs below are shebang scripts Windows cannot start.
posix_only = pytest.mark.skipif(os.name == "nt", reason="no sandbox on Windows; the agent stubs are shebang scripts")


@pytest.fixture
def sandbox_on(monkeypatch):
    """The sandbox asked for, on a machine that has one (nothing is started)."""
    monkeypatch.delenv(sandbox.ENV_SWITCH, raising=False)
    monkeypatch.setattr(sandbox, "backend", lambda: "seatbelt")


@pytest.fixture
def launches(monkeypatch):
    """Record the policy each agent asks for instead of starting a sandbox."""
    seen: list[SandboxPolicy] = []

    def fake_launch(argv, policy):
        seen.append(policy)
        return sandbox.Launch(list(argv), {"SANDBOX_MARK": "1"})

    monkeypatch.setattr(sandbox, "launch", fake_launch)
    return seen


def problem(**fields):
    base = dict(allow_internet_during_solution=False, solution_kind="program", holdout_needs_credentials=False)
    return SimpleNamespace(**{**base, **fields})


def recording_stub(tmp_path: Path, body: str) -> str:
    """An agent stub that first records its argv and the sandbox's mark."""
    first, rest = body.split("\n", 1)
    record = (
        "import json as _j, os as _o, sys as _s\n"
        f"open({str(tmp_path / 'argv.json')!r}, 'w').write(_j.dumps("
        "{'argv': _s.argv[1:], 'mark': _o.environ.get('SANDBOX_MARK')}))\n"
    )
    stub = tmp_path / "agent-stub"
    stub.write_text(f"{first}\n{record}{rest}")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return str(stub)


def recorded(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "argv.json").read_text())


def request_with(request, *, allow_internet=True, policy=SandboxPolicy()):
    return request.model_copy(update={"allow_internet": allow_internet, "sandbox": policy})


# --- config ---


def test_sandbox_is_on_by_default():
    assert Config().sandbox.enabled is True


@pytest.mark.parametrize("text, expected", [("sandbox: off", False), ("sandbox: on", True),
                                            ("sandbox: {enabled: false}", False), ("sandbox: 'off'", False)])
def test_sandbox_switch_spellings(text, expected):
    assert Config(**yaml.safe_load(text)).sandbox.enabled is expected


def test_sandbox_settings_are_checked():
    with pytest.raises(ValueError):
        Config(sandbox="maybe")
    with pytest.raises(ValueError):
        Config(sandbox={"enabeld": False})


def test_env_switch_turns_it_off(monkeypatch):
    monkeypatch.setenv(sandbox.ENV_SWITCH, "off")
    assert not sandbox.enabled(Config())
    assert sandbox.agent_policy(Config(), None, "claude-code") is None
    monkeypatch.delenv(sandbox.ENV_SWITCH)
    assert sandbox.enabled(Config())
    assert not sandbox.enabled(Config(sandbox=False))


# --- which sandbox ---


def test_no_sandbox_exists_on_windows(monkeypatch):
    sandbox.backend.cache_clear()
    monkeypatch.setattr(sandbox.sys, "platform", "win32")
    try:
        assert sandbox.backend() is None
        monkeypatch.delenv(sandbox.ENV_SWITCH, raising=False)
        assert sandbox.agent_policy(Config(), None, "claude-code") is None
    finally:
        sandbox.backend.cache_clear()


def test_missing_bubblewrap_names_the_install(monkeypatch):
    sandbox.backend.cache_clear()
    monkeypatch.setattr(sandbox.sys, "platform", "linux")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
    try:
        with pytest.raises(sandbox.SandboxUnavailable) as caught:
            sandbox.backend()
    finally:
        sandbox.backend.cache_clear()
    assert "apt install bubblewrap" in str(caught.value)
    assert "sandbox: off" in str(caught.value)


def test_a_sandbox_that_does_not_start_is_explained(monkeypatch):
    sandbox.backend.cache_clear()
    monkeypatch.setattr(sandbox.sys, "platform", "darwin")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        sandbox.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=71, stderr="sandbox_apply: Operation not permitted"),
    )
    try:
        with pytest.raises(sandbox.SandboxUnavailable, match="inside another sandbox"):
            sandbox.backend()
    finally:
        sandbox.backend.cache_clear()


# --- policies ---


def test_verifier_has_no_network_unless_the_problem_allows_it(sandbox_on, tmp_path):
    config = Config()
    assert sandbox.verifier_policy(config, problem(), tmp_path).network == "none"
    online = problem(allow_internet_during_solution=True)
    assert sandbox.verifier_policy(config, online, tmp_path).network == "open"
    gated = problem(holdout_needs_credentials=True)
    assert sandbox.verifier_policy(config, gated, tmp_path).network == "none"
    assert sandbox.verifier_policy(config, gated, tmp_path, holdout=True).network == "open"


def test_a_meta_problems_verifier_is_not_wrapped(sandbox_on, tmp_path):
    """It starts searches that sandbox themselves; macOS allows no nesting."""
    assert sandbox.verifier_policy(Config(), problem(solution_kind="climber"), tmp_path) is None


def test_the_search_records_and_secrets_are_unreadable(sandbox_on, tmp_path):
    config = Config(hillclimb_dir=tmp_path)
    search_dir = tmp_path / "runs" / "r" / "searches" / "s"
    policy = sandbox.verifier_policy(config, problem(), search_dir)
    for path in (search_dir / "journal.jsonl", search_dir / "holdout-eval", tmp_path / ".env",
                 Path.home() / ".ssh", Path.home() / ".claude"):
        assert sandbox.real(path) in policy.deny_read
    holdout = sandbox.verifier_policy(config, problem(), search_dir, holdout=True)
    assert sandbox.real(search_dir / "holdout-eval") not in holdout.deny_read  # it runs there


def test_an_agent_reads_its_own_home_and_no_other(sandbox_on, tmp_path):
    """Operators read their operator home and nothing of the user's own
    agent setups: not ~/.claude (claude-code included), ~/.codex or ~/.agents."""
    from hillclimb.project import machine_cache_dir

    policy = sandbox.agent_policy(Config(), tmp_path, "claude-code")
    for personal in (".claude", ".claude.json", ".codex", ".agents"):
        assert sandbox.real(Path.home() / personal) in policy.deny_read
    assert sandbox.real(machine_cache_dir() / "claude-home") not in policy.deny_read
    assert sandbox.real(machine_cache_dir() / "codex-home") in policy.deny_read
    verifier = sandbox.verifier_policy(Config(), problem(), tmp_path)
    assert sandbox.real(Path.home() / ".claude") in verifier.deny_read
    assert policy.network == "open"


def test_the_problem_is_read_only_and_its_private_paths_are_the_scorers(sandbox_on, tmp_path):
    """Nothing run on a problem's behalf may rewrite it (a solution that
    edited the scorer would choose its own score); agents cannot read its
    private paths, and the verifier policy leaves them to the executor, which
    hides them from the solution step only."""
    hidden = tmp_path / "p" / "hidden"
    hidden.mkdir(parents=True)
    spec = problem(problem_dir=tmp_path / "p", data_dir=tmp_path / "data", private_paths=[hidden])
    agent = sandbox.agent_policy(Config(), None, "claude-code", spec)
    verifier = sandbox.verifier_policy(Config(), spec, None)
    for policy in (agent, verifier):
        assert {sandbox.real(tmp_path / "p"), sandbox.real(tmp_path / "data")} <= set(policy.read_only)
    assert sandbox.real(hidden) in agent.deny_read
    assert sandbox.real(hidden) not in verifier.deny_read
    assert sandbox.private_paths(spec) == (sandbox.real(hidden),)


def test_config_adds_paths_relative_to_the_hillclimb_dir(sandbox_on, tmp_path):
    config = Config(
        hillclimb_dir=tmp_path,
        sandbox={"write": ["scratch"], "deny_read": ["private"], "allow_hosts": ["example.org"], "local_ports": [8000]},
    )
    policy = sandbox.agent_policy(config, None, "pi")
    assert sandbox.real(tmp_path / "scratch") in policy.write
    assert sandbox.real(tmp_path / "private") in policy.deny_read
    proxied = policy.for_agent(False, hosts=["API.Model.test"])
    assert proxied.network == "proxy"
    assert proxied.allow_hosts == ("example.org", "api.model.test")
    assert proxied.local_ports == (8000,)


def test_candidate_root():
    root = Path("/x/searches/s/candidates/c001")
    assert sandbox.candidate_root(root / "trials" / "t0" / "replicates" / "r1") == root
    assert sandbox.candidate_root(root) == root
    assert sandbox.candidate_root(Path("/x/elsewhere")) == Path("/x/elsewhere")


@posix_only
def test_seatbelt_profile(tmp_path):
    policy = SandboxPolicy().writable(tmp_path / 'a "b"', prefix=(str(tmp_path / ".state.json"),))
    profile = sandbox.seatbelt_profile(policy.unreadable(tmp_path / "secret").through_proxy(["h"], [8000]), 4321)
    assert profile.startswith("(version 1)(allow default)(deny file-write*)(allow file-write* ")
    assert '(subpath "' + sandbox.real(tmp_path / 'a "b"').replace('"', '\\"') + '")' in profile
    assert "\\.state\\.json" in profile
    assert "(deny signal)(allow signal (target same-sandbox))" in profile
    secret = sandbox.real(tmp_path / "secret")
    assert f'(deny file-read* file-write* (subpath "{secret}"))' in profile
    assert f'(deny network-outbound (subpath "{secret}"))' in profile
    assert profile.endswith(
        '(deny network-outbound (remote ip))'
        '(deny network-outbound (literal "/private/var/run/mDNSResponder"))'
        '(allow network-outbound (remote ip "localhost:8000") (remote ip "localhost:4321"))'
    )
    assert "network" not in sandbox.seatbelt_profile(SandboxPolicy())
    assert "(allow network-outbound" not in sandbox.seatbelt_profile(SandboxPolicy().offline())


@posix_only
def test_read_only_paths_are_denied_after_the_writable_ones(tmp_path):
    policy = SandboxPolicy().writable(tmp_path / "work").protected(tmp_path / "work" / "problem")
    profile = sandbox.seatbelt_profile(policy)
    problem_dir = sandbox.real(tmp_path / "work" / "problem")
    assert profile.index("(allow file-write*") < profile.index(f'(deny file-write* (subpath "{problem_dir}"))')
    args = " ".join(sandbox.bwrap_args(policy))
    work = sandbox.real(tmp_path / "work")
    assert args.index(f"--bind-try {work} {work}") < args.index(f"--ro-bind-try {problem_dir} {problem_dir}")


def test_bwrap_args(tmp_path):
    (tmp_path / "secret").mkdir()
    (tmp_path / "token").write_text("x")
    policy = SandboxPolicy().writable(tmp_path / "work").unreadable(
        tmp_path / "secret", tmp_path / "token", tmp_path / "absent"
    )
    args = sandbox.bwrap_args(policy.offline())
    assert args[:8] == ["--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc"]
    work, secret, token = (sandbox.real(tmp_path / n) for n in ("work", "secret", "token"))
    joined = " ".join(args)
    assert f"--bind-try {work} {work}" in joined
    assert f"--tmpfs {secret}" in joined
    assert f"--ro-bind /dev/null {token}" in joined
    assert "absent" not in joined
    assert joined.index(work) < joined.index(secret)  # what hides comes last
    assert args[-1] == "--unshare-net"
    assert "--unshare-net" not in sandbox.bwrap_args(policy)


def test_host_allowed():
    allowed = ("api.anthropic.com", "openrouter.ai")
    assert sandbox.host_allowed("api.anthropic.com", allowed)
    assert sandbox.host_allowed("EU.OpenRouter.ai.", allowed)
    assert not sandbox.host_allowed("anthropic.com", allowed)
    assert not sandbox.host_allowed("evil-openrouter.ai", allowed)
    assert not sandbox.host_allowed("api.anthropic.com.evil.test", allowed)


# --- how the agents start ---


@posix_only
def test_claude_runs_inside_the_sandbox(tmp_path, launches):
    agent = ClaudeCodeAgent(claude_bin=recording_stub(tmp_path, test_claude_agent.STUB_OK))
    request = request_with(test_claude_agent.make_request(tmp_path))
    assert agent.invoke(request).ok
    [policy] = launches
    assert sandbox.real(request.candidate_dir) in policy.write
    # its own operator home, never the user's ~/.claude
    from hillclimb.agents.claude_code import claude_home

    assert sandbox.real(claude_home()) in policy.write
    assert sandbox.real(Path.home() / ".claude") not in policy.write
    assert policy.write_prefix == ()
    assert policy.network == "open"
    seen = recorded(tmp_path)
    assert seen["mark"] == "1"  # the sandbox's environment reaches the agent
    assert "--disallowedTools" not in seen["argv"]


@posix_only
def test_claude_without_internet_reaches_only_anthropic(tmp_path, launches):
    agent = ClaudeCodeAgent(claude_bin=recording_stub(tmp_path, test_claude_agent.STUB_OK))
    request = request_with(test_claude_agent.make_request(tmp_path), allow_internet=False)
    assert agent.invoke(request).ok
    [policy] = launches
    assert policy.network == "proxy"
    assert "api.anthropic.com" in policy.allow_hosts
    argv = recorded(tmp_path)["argv"]
    assert argv[argv.index("--disallowedTools") + 1] == "WebSearch,WebFetch"
    assert "--strict-mcp-config" in argv


@posix_only
def test_claude_unsandboxed_is_started_as_before(tmp_path, launches):
    agent = ClaudeCodeAgent(claude_bin=recording_stub(tmp_path, test_claude_agent.STUB_OK))
    assert agent.invoke(test_claude_agent.make_request(tmp_path)).ok
    assert launches == []
    assert recorded(tmp_path) == {
        # no MCP servers or account connectors, sandbox or not
        "argv": ["-p", "--output-format", "stream-json", "--verbose",
                 "--permission-mode", "bypassPermissions", "--model", "sonnet", "--strict-mcp-config"],
        "mark": None,
    }


@pytest.mark.parametrize("make_agent, requests", [
    (lambda stub: ClaudeCodeAgent(claude_bin=stub), test_claude_agent),
    (lambda stub: PiCliAgent(pi_bin=stub), test_pi_agent),
])
def test_no_internet_without_a_sandbox_is_refused(tmp_path, make_agent, requests):
    agent = make_agent(recording_stub(tmp_path, requests.STUB_OK))
    request = request_with(requests.make_request(tmp_path), allow_internet=False, policy=None)
    result = agent.invoke(request)
    assert not result.ok and "needs the sandbox" in result.error_message
    assert not (tmp_path / "argv.json").exists()  # nothing was started


@posix_only
def test_codex_keeps_its_own_sandbox(tmp_path, launches):
    agent = CodexCliAgent(codex_bin=recording_stub(tmp_path, test_codex_agent.STUB_OK))
    request = request_with(test_codex_agent.make_request(tmp_path), allow_internet=False)
    assert agent.invoke(request).ok
    assert launches == []  # never wrapped: macOS allows no sandbox inside a sandbox
    argv = recorded(tmp_path)["argv"]
    assert argv[argv.index('web_search="disabled"') - 1] == "-c"
    assert "sandbox_workspace_write.network_access=false" in argv
    assert argv[argv.index("--sandbox") + 1] == "workspace-write"


@posix_only
def test_pi_runs_inside_the_sandbox(tmp_path, monkeypatch, launches):
    monkeypatch.setenv("STUB_EXPECT_SAMPLING", "absent")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    models = tmp_path / "models.json"
    models.write_text(json.dumps({"providers": {
        "vllm": {"baseUrl": "http://localhost:8000/v1"},
        "box": {"baseUrl": "https://gpu.example.test/v1"},
    }}))
    agent = PiCliAgent(
        pi_bin=recording_stub(tmp_path, test_pi_agent.STUB_OK), auth="openrouter", models_file=models
    )
    request = request_with(test_pi_agent.make_request(tmp_path), allow_internet=False)
    assert agent.invoke(request).ok
    [policy] = launches
    candidate_dir = request.candidate_dir
    assert sandbox.real(candidate_dir) in policy.write
    assert sandbox.real(candidate_dir.parents[1] / "pi-sessions") in policy.write
    assert any("pi-home" in path for path in policy.write)
    assert policy.network == "proxy"
    assert policy.allow_hosts == ("gpu.example.test", "openrouter.ai")
    assert policy.local_ports == (8000,)
    assert recorded(tmp_path)["mark"] == "1"


# --- the search's start ---


def test_preflight_says_what_runs_and_refuses_what_cannot(monkeypatch, sandbox_on):
    lines: list[str] = []
    api._preflight_sandbox(Config(), lines.append)
    assert lines == ["sandbox: on (seatbelt)"]

    lines.clear()
    api._preflight_sandbox(Config(sandbox=False), lines.append)
    assert lines == ["sandbox: off — coding agents and solutions run with your full user rights"]

    lines.clear()
    api._preflight_sandbox(Config(allow_internet_for_agents=False), lines.append)
    assert lines == ["sandbox: on (seatbelt)", "agents: no internet"]

    with pytest.raises(sandbox.SandboxUnavailable, match="needs the sandbox"):
        api._preflight_sandbox(Config(sandbox=False, allow_internet_for_agents=False), lines.append)
    # codex brings its own
    api._preflight_sandbox(Config(agent="codex", sandbox=False, allow_internet_for_agents=False), lines.append)

    monkeypatch.setattr(sandbox, "backend", lambda: None)
    lines.clear()
    api._preflight_sandbox(Config(), lines.append)
    assert "none exists for this operating system" in lines[0]


def test_a_machine_that_cannot_sandbox_does_not_start_a_search(monkeypatch):
    def unavailable():
        raise sandbox.SandboxUnavailable("bubblewrap is not installed")

    monkeypatch.delenv(sandbox.ENV_SWITCH, raising=False)
    monkeypatch.setattr(sandbox, "backend", unavailable)
    with pytest.raises(sandbox.SandboxUnavailable, match="not installed"):
        api._preflight_sandbox(Config(), lambda line: None)


# --- the proxy ---


@pytest.fixture
def echo_server():
    """A TCP server that answers every message with `pong:<message>`."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()

    def serve():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            with conn:
                data = conn.recv(1024)
                if data:
                    conn.sendall(b"pong:" + data)

    threading.Thread(target=serve, daemon=True).start()
    yield server.getsockname()[1]
    server.close()


CONNECT_CLIENT = """
import os, socket, sys
host, port = os.environ["HTTPS_PROXY"].removeprefix("http://").split(":")
def connect(target):
    sock = socket.create_connection((host, int(port)), timeout=10)
    sock.sendall(f"CONNECT {target} HTTP/1.1\\r\\nHost: {target}\\r\\n\\r\\n".encode())
    head = b""
    while b"\\r\\n\\r\\n" not in head:
        chunk = sock.recv(1024)
        if not chunk:
            break
        head += chunk
    return sock, head.split(b"\\r\\n")[0].decode()
sock, status = connect(sys.argv[1])
print(status)
if " 200 " in status:
    sock.sendall(b"ping")
    print(sock.recv(1024).decode())
print(connect("example.com:443")[1])
"""


def test_proxy_tunnels_allowed_hosts_and_refuses_the_rest(echo_server, monkeypatch):
    refused: list[str] = []
    monkeypatch.setattr(sandbox, "log", refused.append)
    proxy = sandbox.AllowlistProxy(["localhost"], unix=False).start()

    def ask(request: bytes) -> bytes:
        with socket.create_connection(("127.0.0.1", proxy.port), timeout=10) as sock:
            sock.sendall(request)
            return sock.recv(1024)

    with socket.create_connection(("127.0.0.1", proxy.port), timeout=10) as sock:
        sock.sendall(f"CONNECT localhost:{echo_server} HTTP/1.1\r\n\r\n".encode())
        assert sock.recv(1024).startswith(b"HTTP/1.1 200")
        sock.sendall(b"ping")
        assert sock.recv(1024) == b"pong:ping"
    assert ask(b"CONNECT example.com:443 HTTP/1.1\r\n\r\n").startswith(b"HTTP/1.1 403")
    assert ask(b"GET http://localhost/ HTTP/1.1\r\n\r\n").startswith(b"HTTP/1.1 403")
    assert ask(b"CONNECT example.com:443 HTTP/1.1\r\n\r\n").startswith(b"HTTP/1.1 403")
    assert refused == [  # once per host
        "sandbox: no internet for coding agents, refused example.com",
        "sandbox: no internet for coding agents, refused http://localhost/",
    ]


# --- a real sandbox, where this machine starts one ---


@pytest.fixture
def real_sandbox(monkeypatch):
    monkeypatch.delenv(sandbox.ENV_SWITCH, raising=False)
    try:
        if sandbox.backend() is None:
            pytest.skip("no sandbox exists for this operating system")
    except sandbox.SandboxUnavailable as exc:
        pytest.skip(str(exc))


@pytest.fixture
def outside():
    """A path in the home folder, outside everything a sandbox may write."""
    path = Path.home() / f".hillclimb-sandbox-test-{uuid.uuid4().hex}"
    yield path
    if path.is_dir():
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
    else:
        path.unlink(missing_ok=True)


def run_inside(tmp_path: Path, policy: SandboxPolicy, code: str, *args: str) -> tuple[int | None, str, str]:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    out_path, err_path = tmp_path / "out.log", tmp_path / "err.log"
    with out_path.open("w") as out, err_path.open("w") as err:
        result = run_logged(
            [sys.executable, "-c", code, *args], work, 60, out, err,
            env=dict(os.environ), sandbox=policy,
        )
    assert not result.timed_out
    return result.returncode, out_path.read_text(), err_path.read_text()


def test_writes_stay_inside(real_sandbox, tmp_path, outside):
    inside = tmp_path / "work" / "made.txt"
    code = (
        "import sys\n"
        "for path in sys.argv[1:]:\n"
        "    try:\n"
        "        open(path, 'w').write('x'); print('wrote')\n"
        "    except OSError as exc:\n"
        "        print(type(exc).__name__)\n"
    )
    policy = SandboxPolicy().writable(tmp_path / "work")
    code_, out, err = run_inside(tmp_path, policy, code, str(inside), str(outside))
    assert code_ == 0, err
    assert out.split() in (["wrote", "PermissionError"], ["wrote", "OSError"])
    assert inside.read_text() == "x"
    assert not outside.exists()


def test_denied_paths_are_unreadable(real_sandbox, tmp_path, outside):
    outside.mkdir()
    (outside / "key").write_text("secret")
    code = (
        "import sys\n"
        "try:\n"
        "    print(open(sys.argv[1]).read())\n"
        "except OSError as exc:\n"
        "    print(type(exc).__name__)\n"
    )
    policy = SandboxPolicy().writable(tmp_path / "work")
    assert run_inside(tmp_path, policy, code, str(outside / "key"))[1].strip() == "secret"
    hidden = run_inside(tmp_path, policy.unreadable(outside), code, str(outside / "key"))[1].strip()
    assert hidden in {"PermissionError", "FileNotFoundError"}


def test_a_read_only_path_stays_read_only_inside_a_writable_one(real_sandbox, tmp_path):
    """The problem folder is linked into the candidate's (writable) dir and
    may sit under a temp dir; it is still never writable."""
    problem_dir = tmp_path / "work" / "problem"
    problem_dir.mkdir(parents=True)
    code = (
        "import sys\n"
        "for path in sys.argv[1:]:\n"
        "    try:\n"
        "        open(path, 'w').write('x'); print('wrote')\n"
        "    except OSError as exc:\n"
        "        print(type(exc).__name__)\n"
    )
    policy = SandboxPolicy().writable(tmp_path / "work").protected(problem_dir)
    code_, out, err = run_inside(tmp_path, policy, code, str(tmp_path / "work" / "ok.txt"), str(problem_dir / "verify.py"))
    assert code_ == 0, err
    assert out.split()[0] == "wrote" and out.split()[1] in {"PermissionError", "OSError"}
    assert not (problem_dir / "verify.py").exists()


def test_a_two_step_run_keeps_private_and_holdout_paths_from_the_solution(real_sandbox, tmp_path, outside):
    """On a validation run the solution reads neither the scorer's private
    data nor the holdout inputs; the scorer reads the private data but not
    the holdout inputs (its report reaches the agents)."""
    from hillclimb.harness.executor import CommandExecutor

    outside.mkdir()
    (outside / "labels.txt").write_text("7")
    (outside / "holdout.txt").write_text("9")
    reader = (
        "import json, os, pathlib, sys\n"
        "seen = {}\n"
        "for name in ('labels.txt', 'holdout.txt'):\n"
        "    try:\n"
        "        seen[name] = pathlib.Path(sys.argv[1], name).read_text()\n"
        "    except OSError:\n"
        "        seen[name] = None\n"
    )
    solution = tmp_path / "work" / "solution.py"
    solution.parent.mkdir()
    solution.write_text(reader + "pathlib.Path('solution_saw.json').write_text(json.dumps(seen))\n")
    scorer = tmp_path / "scorer.py"
    scorer.write_text(
        reader + "pathlib.Path('scorer_saw.json').write_text(json.dumps(seen))\n"
        "pathlib.Path(os.environ['HILLCLIMB_RESULT']).write_text('1')\n"
    )
    executor = CommandExecutor(
        Path(sys.executable), ["{python}", "{solution}", str(outside)],
        score_argv=["{python}", str(scorer), str(outside)], sandbox=SandboxPolicy(),
        private=(sandbox.real(outside / "labels.txt"),), holdout_inputs=(sandbox.real(outside / "holdout.txt"),),
    )
    result = executor.execute(solution, solution.parent, 60)
    assert result.ok, Path(result.stderr_path).read_text()
    assert json.loads((solution.parent / "solution_saw.json").read_text()) == {"labels.txt": None, "holdout.txt": None}
    assert json.loads((solution.parent / "scorer_saw.json").read_text()) == {"labels.txt": "7", "holdout.txt": None}


def test_no_network_means_none(real_sandbox, tmp_path):
    code = (
        "import socket\n"
        "for target in (('192.0.2.1', 80), ('example.com', 443)):\n"  # TEST-NET-1, a name
        "    try:\n"
        "        socket.create_connection(target, timeout=4); print('connected')\n"
        "    except OSError as exc:\n"
        "        print(type(exc).__name__)\n"
    )
    policy = SandboxPolicy().writable(tmp_path / "work").offline()
    code_, out, err = run_inside(tmp_path, policy, code)
    assert code_ == 0, err
    # refused on the spot: a timeout would be a route out that merely went unanswered
    assert "connected" not in out and "TimeoutError" not in out.split()[0]


def test_the_proxy_is_the_only_way_out(real_sandbox, tmp_path, echo_server):
    """The whole path an agent without internet takes: sandbox -> (on Linux,
    the bridge) -> the engine's proxy -> an allowed host."""
    policy = SandboxPolicy().writable(tmp_path / "work").through_proxy(["localhost"])
    code_, out, err = run_inside(tmp_path, policy, CONNECT_CLIENT, f"localhost:{echo_server}")
    assert code_ == 0, err
    status, answer, refusal = out.strip().splitlines()
    assert " 200 " in status and answer == "pong:ping"
    assert " 403 " in refusal


def test_the_exit_code_comes_through(real_sandbox, tmp_path):
    policy = SandboxPolicy().writable(tmp_path / "work").through_proxy(["localhost"])
    assert run_inside(tmp_path, policy, "raise SystemExit(7)")[0] == 7
    assert run_inside(tmp_path, SandboxPolicy().offline(), "raise SystemExit(3)")[0] == 3


# --- `hillclimb sandbox check` ---


def test_the_check_holds_on_this_machine(real_sandbox, tmp_path, monkeypatch):
    """The hostile probe inside the policies a search uses."""
    from hillclimb.harness.sandbox_check import run_check

    attempts = run_check(Config())
    by_id = {a.id: a for a in attempts}
    assert by_id["own"].outcome == "allowed"
    for name in ("home", "journal", "connect", "resolve", "signal", "proxy-other", "proxy-bypass"):
        assert by_id[name].outcome == "blocked", by_id[name]
    assert by_id["proxy-allowed"].outcome in {"allowed", "skipped"}  # skipped: no internet here
    assert all(a.holds for a in attempts)
    assert not list(Path.home().glob(".hillclimb-sandbox-check-*"))


def test_check_command_without_a_sandbox_exits_1():
    from typer.testing import CliRunner

    from hillclimb.cli import app

    result = CliRunner().invoke(app, ["sandbox", "check"])  # the suite runs with it off
    assert result.exit_code == 1
    assert "sandbox: off" in result.output


def test_check_command_reports_what_got_through(monkeypatch, sandbox_on):
    from typer.testing import CliRunner

    from hillclimb.cli import app
    from hillclimb.harness import sandbox_check
    from hillclimb.harness.sandbox_check import Probe

    attempts = [
        Probe("own", "files", "write to its own candidate folder", "write", "x", "allowed", "allowed"),
        Probe("home", "files", "write to your home folder", "write", "y", "blocked", "blocked"),
    ]
    monkeypatch.setattr(sandbox_check, "run_check", lambda config: attempts)
    result = CliRunner().invoke(app, ["sandbox", "check"])
    assert result.exit_code == 0, result.output
    assert "The sandbox holds: 1 attempts blocked" in result.output

    attempts[1].outcome = "allowed"
    result = CliRunner().invoke(app, ["sandbox", "check"])
    assert result.exit_code == 1
    assert "write to your home folder was allowed" in result.output

    result = CliRunner().invoke(app, ["sandbox", "check", "--json"])
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["sandbox"] == "seatbelt"
    assert [a["holds"] for a in report["attempts"]] == [True, False]


def test_without_a_sandbox_windows_is_pointed_at_wsl2(monkeypatch):
    """Native Windows has no sandbox; WSL2 has a real Linux kernel, where
    bubblewrap works. Every message that says there is none says where to go."""
    monkeypatch.setattr(sandbox.sys, "platform", "win32")
    assert "WSL2" in sandbox.no_sandbox_reason() and "security-and-sandboxes" in sandbox.no_sandbox_reason()
    monkeypatch.setattr(sandbox.sys, "platform", "linux")
    assert sandbox.no_sandbox_reason() == "none exists for this operating system"
