"""Windows gets verifier.py, everyone else verifier.sh — and the two agree.

`hillclimb problem get` writes the verifier for the machine that fetches the
problem: verifier.sh on macOS/Linux, verifier.py on Windows (the problem's
own, else the shared `demo/windows_verifier.py`). These tests hold the
Windows editions to their shell originals: same shape, same score, and never
importing the solution (it must run as its own process, or it could reach
the scorer)."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from hillclimb.demo import BUNDLED_PROBLEM_IDS, WINDOWS_VERIFIER, install_demo_problem
from hillclimb.problem import windows_edition

DEMO = Path(__file__).resolve().parent.parent / "src" / "hillclimb" / "demo"
SHARED = DEMO / WINDOWS_VERIFIER

# verifier.sh, comments and blank lines dropped, of every problem the shared
# Windows verifier stands in for
STANDARD_SH = [
    "set -euo pipefail",
    '"$HILLCLIMB_PYTHON" "$HILLCLIMB_SOLUTION"',
    'rm -f "$HILLCLIMB_RESULT"',
    '"$HILLCLIMB_PYTHON" problem/verify.py',
]


def _body(path: Path) -> list[str]:
    lines = [line.split("  #")[0].strip() for line in path.read_text().splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _windows_verifier(problem_id: str) -> Path:
    own = DEMO / problem_id / "verifier.py"
    return own if own.exists() else SHARED


@pytest.mark.parametrize("problem_id", BUNDLED_PROBLEM_IDS)
def test_every_bundled_problem_has_a_windows_verifier(problem_id):
    """A problem whose verifier.sh leaves the standard shape must ship its
    own verifier.py — the shared one would silently score it differently."""
    if not (DEMO / problem_id / "verifier.py").exists():
        assert _body(DEMO / problem_id / "verifier.sh") == STANDARD_SH, (
            f"{problem_id}/verifier.sh is not the standard shape: ship a verifier.py beside it"
        )


@pytest.mark.parametrize("path", sorted({_windows_verifier(p) for p in BUNDLED_PROBLEM_IDS}), ids=lambda p: p.parent.name)
def test_windows_verifiers_never_import_the_solution(path):
    tree = ast.parse(path.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("exec", "eval", "__import__"):
            pytest.fail(f"{path.name} calls {node.func.id}")
    assert imported <= {"os", "subprocess", "sys", "pathlib"}, imported


@pytest.mark.parametrize("windows", [False, True], ids=["posix", "windows"])
@pytest.mark.parametrize("problem_id", ["heilbronn-convex-13", "knapsack"])
def test_problem_get_writes_the_verifier_for_this_os(tmp_path, problem_id, windows):
    problem_dir, created = install_demo_problem(tmp_path, problem_id, windows=windows)
    assert created
    if windows:
        assert not (problem_dir / "verifier.sh").exists()
        assert (problem_dir / "verifier.py").read_bytes() == _windows_verifier(problem_id).read_bytes()
    else:
        assert not (problem_dir / "verifier.py").exists()
        assert os.access(problem_dir / "verifier.sh", os.X_OK)


def test_windows_edition_is_chosen_only_on_windows(tmp_path):
    sh = tmp_path / "verifier.sh"
    sh.write_text("#!/usr/bin/env bash\n")
    assert windows_edition(sh, windows=True) == sh  # no .py edition: bash it is
    (tmp_path / "verifier.py").write_text("")
    assert windows_edition(sh, windows=True) == tmp_path / "verifier.py"
    assert windows_edition(sh, windows=False) == sh
    other = tmp_path / "score.sh"
    assert windows_edition(other, windows=True) == other


def _score(problem_id: str, argv: list[str], tmp_path: Path, tag: str) -> float:
    """Run one verifier edition the way the engine does: in a fresh dir
    holding solution.py (the baseline) and a `problem` link."""
    work = tmp_path / tag
    work.mkdir()
    source = DEMO / problem_id
    if (source / "baseline.py").exists():
        shutil.copy(source / "baseline.py", work / "solution.py")
    else:  # a files-only floor: the sample submission, and a solution that leaves it be
        (work / "solution.py").write_text("")
        meta = yaml.safe_load((source / "problem.yaml").read_text())
        for name, sample in (meta.get("baseline_files") or {"submission.csv": "sample_submission.csv"}).items():
            shutil.copy(source / sample, work / name)
    (work / "problem").symlink_to(source, target_is_directory=True)
    if (source / "data").is_dir():
        (work / "data").symlink_to(source / "data", target_is_directory=True)
    result = work / "eval_result.json"
    env = dict(
        os.environ,
        HILLCLIMB_PYTHON=sys.executable,
        HILLCLIMB_SOLUTION=str(work / "solution.py"),
        HILLCLIMB_RESULT=str(result),
        HILLCLIMB_SPLIT="validation",
    )
    subprocess.run(argv, cwd=work, env=env, check=True, capture_output=True, timeout=600)
    payload = json.loads(result.read_text())
    return payload["score"] if isinstance(payload, dict) else float(payload)


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None, reason="needs bash to run the original")
@pytest.mark.parametrize("problem_id", BUNDLED_PROBLEM_IDS)
def test_both_editions_score_the_baseline_the_same(tmp_path, problem_id):
    sh = _score(problem_id, ["bash", str(DEMO / problem_id / "verifier.sh")], tmp_path, "sh")
    py = _score(problem_id, [sys.executable, str(_windows_verifier(problem_id))], tmp_path, "py")
    assert py == sh
