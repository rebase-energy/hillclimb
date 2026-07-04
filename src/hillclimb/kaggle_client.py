"""Thin subprocess wrappers around the kaggle CLI for late submissions.

Credentials come from the environment (KAGGLE_API_TOKEN / KAGGLE_USERNAME+KEY
or ~/.kaggle/kaggle.json); source them before calling, e.g.
`set -a; source ../agent-work/.env; set +a`.
"""

from __future__ import annotations

import csv
import io
import subprocess
import time
from pathlib import Path


class KaggleError(RuntimeError):
    pass


def _run(kaggle_bin: Path, *args: str) -> str:
    result = subprocess.run(
        [str(kaggle_bin), *args], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise KaggleError(
            f"kaggle {' '.join(args)} failed (rc={result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def submit_competition(kaggle_bin: Path, comp_id: str, submission: Path, message: str) -> None:
    _run(
        kaggle_bin, "competitions", "submit",
        "-c", comp_id, "-f", str(submission), "-m", message,
    )


def list_submissions(kaggle_bin: Path, comp_id: str) -> list[dict]:
    """Rows of `kaggle competitions submissions -v`, newest first. Columns
    include fileName, date, description, status, publicScore, privateScore."""
    output = _run(kaggle_bin, "competitions", "submissions", "-c", comp_id, "-v")
    # the CLI may print a banner line before the CSV header
    lines = output.strip().splitlines()
    start = next(i for i, line in enumerate(lines) if "fileName" in line or "status" in line)
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def wait_for_score(
    kaggle_bin: Path,
    comp_id: str,
    message: str,
    timeout_s: int = 600,
    poll_s: int = 20,
) -> dict:
    """Poll until the submission with our (unique) message is scored.
    Matching by message is robust to the file always being submission.csv."""
    deadline = time.monotonic() + timeout_s
    while True:
        rows = [r for r in list_submissions(kaggle_bin, comp_id)
                if r.get("description") == message]
        if rows:
            row = rows[0]  # newest first
            status = (row.get("status") or "").lower()
            if "complete" in status:
                return row
            if "error" in status:
                raise KaggleError(f"submission failed on Kaggle: {row}")
        if time.monotonic() > deadline:
            raise KaggleError(
                f"no scored submission with message {message!r} after {timeout_s}s"
            )
        time.sleep(poll_s)
