from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from hillclimb.config import Config


def grade_submission(submission: Path, comp_id: str, config: Config) -> dict:
    """Shell out to the mlebench venv's grade-sample and extract its JSON report."""
    mlebench_bin = config.paths.mlebench_python.parent / "mlebench"
    cmd = [str(mlebench_bin), "grade-sample", str(submission), comp_id]
    if config.paths.mlebench_data_dir:
        cmd += ["--data-dir", str(config.paths.mlebench_data_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        raise RuntimeError(f"mlebench grade-sample produced no report:\n{output}")
    return json.loads(match.group(0))
