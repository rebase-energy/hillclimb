"""Bundled problems shipped as package data.

Keeping these inside the wheel makes `hillclimb problem get` work from a bare
`pip install hillclimb` with no repository checkout.
"""

from __future__ import annotations

import shutil
import sys
from importlib import resources
from pathlib import Path

DEMO_PROBLEM_ID = "circle-packing"

# The example catalog, in ladder order: construction problems in one shape —
# a small CSV of numbers, an exact verifier, no data, no holdout, no noise —
# each family from the instance a ten-minute run climbs to the one an hour
# does not saturate. Every family is stamped by a generator in problems/
# (`make_<family>.py`) into both problems/ and this package; the two copies
# must stay identical (tests/test_demo.py). circle-packing is the one
# deliberate exception: a lean runtime so a first run starts in seconds.
STARTER_PROBLEM_IDS = (
    DEMO_PROBLEM_ID, "circle-packing-32",
    "heilbronn-11", "heilbronn-14", "heilbronn-17", "heilbronn-convex-13",
    "labs-40", "labs-60",
    "tammes-30", "tammes-50",
    "thomson-50", "thomson-100",
    "autocorr-1", "autocorr-3", "erdos-overlap",
    "kissing-11",
    "golomb-20", "golomb-27",
    "tsp-200",
    "mknap-100-5", "mknap-250-10",
)
# bundled beyond the example set: a data + holdout example, and the terrain
# the scripted `toy` agent walks (the Python SDK's examples run on it)
BUNDLED_PROBLEM_IDS = (*STARTER_PROBLEM_IDS, "knapsack", "fitness-landscape")

# The Windows edition of the standard verifier.sh (run the candidate, drop
# any stale result, run problem/verify.py): a problem that ships no
# verifier.py of its own gets this one on Windows (tests/test_windows_verifier.py
# holds every such verifier.sh to that shape and both editions to one score).
WINDOWS_VERIFIER = "windows_verifier.py"


def demo_problem_resource(problem_id: str = DEMO_PROBLEM_ID):
    """Traversable for one of the bundled problem folders."""
    if problem_id not in BUNDLED_PROBLEM_IDS:
        available = ", ".join(BUNDLED_PROBLEM_IDS)
        raise ValueError(f"no bundled problem {problem_id!r} (available: {available})")
    return resources.files(__package__) / problem_id


def install_demo_problem(
    problems_dir: Path,
    problem_id: str = DEMO_PROBLEM_ID,
    *,
    windows: bool | None = None,
) -> tuple[Path, bool]:
    """Copy a bundled problem into `problems_dir` unless it is already
    there. Returns (problem_dir, created). An existing folder is never
    touched — the user may have edited it.

    The verifier is written for the machine that gets it: verifier.sh on
    macOS/Linux, verifier.py on Windows (the problem's own, else the shared
    WINDOWS_VERIFIER), so a Windows user never needs bash. `windows`
    defaults to this machine."""
    if windows is None:
        windows = sys.platform == "win32"
    # Validate before inspecting the destination: callers should never be able
    # to turn an arbitrary id into a filesystem copy target through this API.
    source_resource = demo_problem_resource(problem_id)
    target = problems_dir / problem_id
    if (target / "problem.yaml").exists():
        return target, False
    problems_dir.mkdir(parents=True, exist_ok=True)
    skip = "verifier.sh" if windows else "verifier.py"
    with resources.as_file(source_resource) as source:
        shutil.copytree(
            source, target, dirs_exist_ok=True,
            ignore=lambda directory, names: {skip} & set(names) if Path(directory) == source else set(),
        )
    if windows:
        if not (target / "verifier.py").exists():
            shared = resources.files(__package__) / WINDOWS_VERIFIER
            (target / "verifier.py").write_bytes(shared.read_bytes())
    else:
        # wheels keep mode bits, but a copy through some installers does not
        (target / "verifier.sh").chmod(0o755)
    return target, True
