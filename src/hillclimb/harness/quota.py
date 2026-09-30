"""Subscription limit-window utilization — "how much of my Max plan is used".

Claude Code's /usage screen reads these percentages from an undocumented
OAuth endpoint; `snapshot()` returns the same numbers so the engine can
journal the window state at the start and end of every agent call. The
numbers are ACCOUNT-WIDE: parallel agents, other searches and interactive
sessions all move them, and a window reset mid-call makes a naive delta
negative — consumers must treat per-candidate deltas as best-effort
telemetry, never accounting (the journaled token counts are the ground
truth).

Everything here is best-effort: no credentials, an unreachable or re-shaped
endpoint, or `HILLCLIMB_QUOTA=off` all yield None, and a search never blocks
on it. Successful snapshots are shared for a short TTL so a burst of
parallel candidates starting together costs one fetch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

from hillclimb.harness.candidate import utcnow

USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
_OK_TTL_S = 30.0  # deltas shorter than this collapse to zero — acceptable
_FAIL_TTL_S = 300.0  # no creds / endpoint down: stop retrying for a while
_TOKEN_TTL_S = 900.0  # keychain reads spawn a process; tokens live hours

_lock = threading.Lock()
_snapshot_cache: tuple[float, dict | None] | None = None
_token_cache: tuple[float, str | None] | None = None


def _credentials_file_token() -> str | None:
    path = os.path.join(os.path.expanduser("~"), ".claude", ".credentials.json")
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    oauth = data.get("claudeAiOauth") if isinstance(data, dict) else None
    return (oauth or {}).get("accessToken") or None


def _keychain_token() -> str | None:
    """macOS stores the interactive login under this generic-password item.
    First access from a new binary can raise a keychain consent dialog —
    'Always Allow' makes it permanent."""
    if sys.platform != "darwin":
        return None
    try:
        proc = subprocess.run(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout.strip())
    except ValueError:
        return None
    oauth = data.get("claudeAiOauth") if isinstance(data, dict) else None
    return (oauth or {}).get("accessToken") or None


def _oauth_token() -> str | None:
    global _token_cache
    now = time.monotonic()
    if _token_cache is not None and now - _token_cache[0] < _TOKEN_TTL_S:
        return _token_cache[1]
    token = (
        os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        or _credentials_file_token()
        or _keychain_token()
    )
    _token_cache = (now, token)
    return token


def _fetch() -> dict | None:
    token = _oauth_token()
    if not token:
        return None
    request = urllib.request.Request(
        USAGE_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "anthropic-beta": "oauth-2025-04-20",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    # schema-tolerant: keep every window-shaped entry (a dict with a numeric
    # `utilization`) whatever it is named — five_hour, seven_day, model caps
    snapshot: dict = {"fetched_at": utcnow()}
    for window, value in payload.items():
        if isinstance(value, dict) and isinstance(value.get("utilization"), (int, float)):
            entry: dict = {"utilization": value["utilization"]}
            if value.get("resets_at") is not None:
                entry["resets_at"] = value["resets_at"]
            snapshot[window] = entry
    if len(snapshot) == 1:  # nothing recognized — schema drifted; journal nothing
        return None
    return snapshot


def snapshot() -> dict | None:
    """Current utilization per limit window, e.g.
    {"fetched_at": …, "five_hour": {"utilization": 34, "resets_at": …},
    "seven_day": {…}} — or None when unavailable or disabled."""
    if os.environ.get("HILLCLIMB_QUOTA", "").lower() in {"off", "0", "false"}:
        return None
    global _snapshot_cache
    with _lock:
        if _snapshot_cache is not None:
            age = time.monotonic() - _snapshot_cache[0]
            ttl = _OK_TTL_S if _snapshot_cache[1] is not None else _FAIL_TTL_S
            if age < ttl:
                return _snapshot_cache[1]
        result = _fetch()
        _snapshot_cache = (time.monotonic(), result)
        return result
