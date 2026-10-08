"""What `hillclimb problem new` writes, and the Windows edition of the
standard verifier: engine data, not catalog.

`problem/` is a working two-step problem (run.py, score.py) to edit into
your own. Two steps need no bash, so it runs on Windows as it is.
`windows_verifier.py` is the Windows edition of the standard verifier.sh
(run the candidate, drop any stale result, run problem/verify.py): a catalog
problem that ships no verifier.py of its own gets this one on Windows
(tests/test_windows_verifier.py holds every such verifier.sh to that shape
and both editions to one score).
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

SCAFFOLD = "problem"
SCAFFOLD_TOKEN = "__PROBLEM_ID__"
PROBLEM_ID_PATTERN = r"[a-z0-9][a-z0-9._-]*"
WINDOWS_VERIFIER = "windows_verifier.py"


def windows_verifier_path() -> Path:
    return Path(str(resources.files(__package__) / WINDOWS_VERIFIER))


def scaffold_problem(problems_dir: Path, problem_id: str) -> Path:
    """Write a new problem named `problem_id` into `problems_dir` from the
    scaffold. Refuses an id that is not a plain folder name, and a folder
    that already exists (it may hold the user's work)."""
    if not re.fullmatch(PROBLEM_ID_PATTERN, problem_id):
        raise ValueError(
            f"problem id {problem_id!r} must be lowercase letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
    target = problems_dir / problem_id
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    target.mkdir(parents=True)
    for item in sorted((resources.files(__package__) / SCAFFOLD).iterdir(), key=lambda item: item.name):
        if item.is_file():
            (target / item.name).write_text(item.read_text().replace(SCAFFOLD_TOKEN, problem_id))
    return target
