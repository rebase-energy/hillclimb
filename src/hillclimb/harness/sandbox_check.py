"""`hillclimb sandbox check`: the sandbox, shown rather than described.

Runs `sandbox_probe.py` — a script that behaves like a hostile solution —
inside the very policies a search uses, and says for every attempt whether it
got through. Two runs: one as a verifier (no network at all), one as an agent
without internet (only the proxy). Pure apart from the probe itself: whatever
an attempt managed to write outside its folder is removed again.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from hillclimb.harness import sandbox

PROBE = Path(__file__).with_name("sandbox_probe.py")
ALLOWED_HOST = "example.com"  # stands in for the model provider
OTHER_HOST = "example.org"
OUTSIDE_ADDRESS = "1.1.1.1:443"
TIMEOUT_S = 90


@dataclass
class Probe:
    id: str
    group: str  # files | secrets | network | processes
    label: str
    kind: str
    target: str
    expect: str  # blocked | allowed
    outcome: str = "skipped"  # blocked | allowed | skipped
    detail: str = ""

    @property
    def holds(self) -> bool:
        return self.outcome in (self.expect, "skipped")


def _inside_temp(path: Path) -> bool:
    here = sandbox.real(path)
    return any(here == root or here.startswith(root + os.sep) for root in sandbox.temp_dirs())


def _present(path: Path) -> bool:
    try:
        return path.is_file() or (path.is_dir() and any(path.iterdir()))
    except OSError:
        return False


def _run(attempts: list[Probe], policy: sandbox.SandboxPolicy, cwd: Path) -> None:
    spec = [{"id": a.id, "kind": a.kind, "target": a.target} for a in attempts]
    started = sandbox.launch([sys.executable, str(PROBE), json.dumps(spec)], policy)
    done = subprocess.run(
        started.argv, cwd=cwd, env={**os.environ, **started.env},
        capture_output=True, text=True, timeout=TIMEOUT_S,
    )
    try:
        report = json.loads(done.stdout)
    except ValueError as exc:
        raise sandbox.SandboxUnavailable(
            f"the check could not run inside the sandbox: {(done.stderr or '').strip()[-300:] or exc}"
        ) from exc
    for item in attempts:
        answer = report[item.id]
        if answer["error"] == "NoRoute":
            item.outcome, item.detail = "skipped", "no answer from the host: is this machine online?"
        elif answer["done"] and not (item.kind == "read" and answer["seen"] == 0):
            item.outcome = "allowed"
        else:
            # a hidden folder is an empty one under bubblewrap
            item.outcome, item.detail = "blocked", answer["error"] or "hidden"


def run_check(config) -> list[Probe]:
    """Every attempt with its outcome. Raises SandboxUnavailable when the
    sandbox is on and does not start; returns [] when there is none to check
    (`sandbox: off`, or an OS without one)."""
    if not sandbox.active(config):
        return []
    tag = uuid.uuid4().hex[:12]
    home = Path.home()
    hillclimb_dir = getattr(config, "hillclimb_dir", None)
    with tempfile.TemporaryDirectory(prefix="hillclimb-sandbox-check-") as tmp:
        search_dir = Path(tmp) / "searches" / "check"
        candidate_dir = search_dir / "candidates" / "c001"
        candidate_dir.mkdir(parents=True)
        (search_dir / "journal.jsonl").write_text('{"holdout_score": 1.0}\n')
        outside = [home / f".hillclimb-sandbox-check-{tag}"]
        files = [
            Probe("own", "files", "write to its own candidate folder", "write",
                    str(candidate_dir / "output.txt"), "allowed"),
            Probe("home", "files", "write to your home folder", "write", str(outside[0]), "blocked"),
        ]
        if hillclimb_dir is not None and not _inside_temp(Path(hillclimb_dir)):
            outside.append(Path(hillclimb_dir) / f".sandbox-check-{tag}")
            files.append(Probe("dir", "files", "write to the hillclimb dir", "write",
                                 str(outside[-1]), "blocked"))
        secrets = [Probe("journal", "secrets", "read the search's journal (it holds holdout scores)",
                           "read", str(search_dir / "journal.jsonl"), "blocked")]
        known = [("ssh", "read ~/.ssh", home / ".ssh"), ("aws", "read ~/.aws", home / ".aws"),
                 ("claude", "read Claude Code's login and history (~/.claude)", home / ".claude")]
        if hillclimb_dir is not None:
            known.append(("env", "read the hillclimb dir's .env", Path(hillclimb_dir) / ".env"))
        secrets += [
            Probe(name, "secrets", label, "read", str(path), "blocked")
            for name, label, path in known if _present(path)
        ]
        verifier = [
            *files,
            *secrets,
            Probe("connect", "network", f"connect to the internet ({OUTSIDE_ADDRESS})", "connect",
                    OUTSIDE_ADDRESS, "blocked"),
            Probe("resolve", "network", f"look up a name ({ALLOWED_HOST})", "resolve", ALLOWED_HOST, "blocked"),
            Probe("signal", "processes", "signal a process outside the sandbox", "signal",
                    str(os.getpid()), "blocked"),
        ]
        agent = [
            Probe("proxy-other", "agent without internet", f"reach a host that is not allowed ({OTHER_HOST})",
                    "proxy", f"{OTHER_HOST}:443", "blocked"),
            Probe("proxy-bypass", "agent without internet", "connect past the proxy", "connect",
                    OUTSIDE_ADDRESS, "blocked"),
            Probe("proxy-allowed", "agent without internet",
                    f"reach an allowed host ({ALLOWED_HOST} here, the model provider in a search)",
                    "proxy", f"{ALLOWED_HOST}:443", "allowed"),
        ]
        base = sandbox._base(config, search_dir, None).writable(candidate_dir)
        try:
            _run(verifier, base.offline(), candidate_dir)
            _run(agent, base.through_proxy([ALLOWED_HOST]), candidate_dir)
        finally:
            for path in outside:
                path.unlink(missing_ok=True)
        return verifier + agent


def as_dicts(attempts: list[Probe]) -> list[dict]:
    return [{**asdict(a), "holds": a.holds} for a in attempts]
