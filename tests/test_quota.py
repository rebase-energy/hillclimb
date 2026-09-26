from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from hillclimb.harness import quota


@pytest.fixture(autouse=True)
def _reset_quota_caches(monkeypatch):
    """Each test starts cold: no cached snapshot, no cached token, and the
    conftest-wide HILLCLIMB_QUOTA=off lifted so the module is exercisable."""
    monkeypatch.setattr(quota, "_snapshot_cache", None)
    monkeypatch.setattr(quota, "_token_cache", None)
    monkeypatch.delenv("HILLCLIMB_QUOTA", raising=False)


def fake_response(payload: dict):
    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

    return _Response(json.dumps(payload).encode())


def test_snapshot_normalizes_windows(monkeypatch):
    monkeypatch.setattr(quota, "_oauth_token", lambda: "tok-123")
    seen: dict = {}

    def fake_urlopen(request, timeout=None):
        seen["auth"] = request.get_header("Authorization")
        seen["beta"] = request.get_header("Anthropic-beta")
        return fake_response(
            {
                "five_hour": {"utilization": 34, "resets_at": "2026-08-25T12:00:00Z"},
                "seven_day": {"utilization": 12.5, "resets_at": "2026-08-28T00:00:00Z"},
                "seven_day_opus": {"utilization": 3},
                "extra_usage": {"enabled": False},  # window-shaped it is not
                "plan": "max",
            }
        )

    monkeypatch.setattr(quota.urllib.request, "urlopen", fake_urlopen)
    snap = quota.snapshot()

    assert seen["auth"] == "Bearer tok-123"
    assert seen["beta"] == "oauth-2025-04-20"
    assert snap["five_hour"] == {"utilization": 34, "resets_at": "2026-08-25T12:00:00Z"}
    assert snap["seven_day"]["utilization"] == 12.5
    assert snap["seven_day_opus"] == {"utilization": 3}
    assert "extra_usage" not in snap and "plan" not in snap
    assert snap["fetched_at"]


def test_snapshot_none_when_schema_unrecognized(monkeypatch):
    monkeypatch.setattr(quota, "_oauth_token", lambda: "tok-123")
    monkeypatch.setattr(
        quota.urllib.request, "urlopen", lambda *a, **k: fake_response({"plan": "max"})
    )
    assert quota.snapshot() is None


def test_snapshot_caches_successes(monkeypatch):
    calls = {"n": 0}

    def fake_fetch():
        calls["n"] += 1
        return {"fetched_at": "t", "five_hour": {"utilization": calls["n"]}}

    monkeypatch.setattr(quota, "_fetch", fake_fetch)
    first = quota.snapshot()
    second = quota.snapshot()
    assert first is second  # within the TTL the fetch is shared
    assert calls["n"] == 1


def test_snapshot_caches_failures(monkeypatch):
    calls = {"n": 0}

    def fake_fetch():
        calls["n"] += 1
        return None

    monkeypatch.setattr(quota, "_fetch", fake_fetch)
    assert quota.snapshot() is None
    assert quota.snapshot() is None  # negative-cached, not retried per call
    assert calls["n"] == 1


def test_snapshot_disabled_by_env(monkeypatch):
    monkeypatch.setenv("HILLCLIMB_QUOTA", "off")
    monkeypatch.setattr(
        quota, "_fetch", lambda: pytest.fail("must not fetch when disabled")
    )
    assert quota.snapshot() is None


def test_no_token_means_no_fetch(monkeypatch):
    monkeypatch.setattr(quota, "_oauth_token", lambda: None)
    monkeypatch.setattr(
        quota.urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("must not call the endpoint without a token"),
    )
    assert quota.snapshot() is None


def test_oauth_token_prefers_env_then_credentials_file(monkeypatch, tmp_path: Path):
    creds = tmp_path / ".claude" / ".credentials.json"
    creds.parent.mkdir()
    creds.write_text(json.dumps({"claudeAiOauth": {"accessToken": "file-tok"}}))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        quota, "_keychain_token", lambda: pytest.fail("file token must win")
    )

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "env-tok")
    assert quota._oauth_token() == "env-tok"

    monkeypatch.setattr(quota, "_token_cache", None)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    assert quota._oauth_token() == "file-tok"
