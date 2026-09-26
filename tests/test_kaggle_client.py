from pathlib import Path

import pytest

from hillclimb.integrations.mlebench import kaggle_client
from hillclimb.integrations.mlebench.kaggle_client import (
    KaggleError,
    list_submissions,
    submit_competition,
    wait_for_score,
)

CSV_COMPLETE = """fileName,date,description,status,publicScore,privateScore
submission.csv,2026-07-04 10:00:00,run-x node n003,complete,0.61234,
submission.csv,2026-07-01 09:00:00,older try,complete,0.90000,
"""

CSV_PENDING = """fileName,date,description,status,publicScore,privateScore
submission.csv,2026-07-04 10:00:00,run-x node n003,pending,,
"""

CSV_ERROR = """fileName,date,description,status,publicScore,privateScore
submission.csv,2026-07-04 10:00:00,run-x node n003,error,,
"""


class FakeProc:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


def patch_run(monkeypatch, outputs: list):
    """Each call to subprocess.run pops the next canned output."""
    calls = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        out = outputs.pop(0)
        return out if isinstance(out, FakeProc) else FakeProc(out)

    monkeypatch.setattr(kaggle_client.subprocess, "run", fake_run)
    return calls


def test_submit_builds_command(monkeypatch):
    calls = patch_run(monkeypatch, ["Successfully submitted"])
    submit_competition(Path("kaggle"), "my-comp", Path("sub.csv"), "msg")
    assert calls[0][:3] == ["kaggle", "competitions", "submit"]
    assert "-c" in calls[0] and "my-comp" in calls[0] and "msg" in calls[0]


def test_submit_nonzero_raises(monkeypatch):
    patch_run(monkeypatch, [FakeProc("denied", returncode=1)])
    with pytest.raises(KaggleError, match="failed"):
        submit_competition(Path("kaggle"), "my-comp", Path("sub.csv"), "msg")


def test_list_submissions_parses_csv_with_banner(monkeypatch):
    patch_run(monkeypatch, ["Warning: some banner line\n" + CSV_COMPLETE])
    rows = list_submissions(Path("kaggle"), "my-comp")
    assert len(rows) == 2
    assert rows[0]["publicScore"] == "0.61234"
    assert rows[0]["description"] == "run-x node n003"


def test_wait_for_score_matches_message_and_polls(monkeypatch):
    patch_run(monkeypatch, [CSV_PENDING, CSV_COMPLETE])
    monkeypatch.setattr(kaggle_client.time, "sleep", lambda s: None)
    row = wait_for_score(Path("kaggle"), "my-comp", "run-x node n003", timeout_s=60)
    assert row["publicScore"] == "0.61234"


def test_wait_for_score_error_status_raises(monkeypatch):
    patch_run(monkeypatch, [CSV_ERROR])
    with pytest.raises(KaggleError, match="failed on Kaggle"):
        wait_for_score(Path("kaggle"), "my-comp", "run-x node n003", timeout_s=60)


def test_wait_for_score_timeout(monkeypatch):
    patch_run(monkeypatch, [CSV_PENDING] * 50)
    monkeypatch.setattr(kaggle_client.time, "sleep", lambda s: None)
    clock = iter(range(0, 10_000, 300))
    monkeypatch.setattr(kaggle_client.time, "monotonic", lambda: next(clock))
    with pytest.raises(KaggleError, match="no scored submission"):
        wait_for_score(Path("kaggle"), "my-comp", "run-x node n003", timeout_s=200)
