"""The Tammes starter instances: generated problem dirs (repo + bundled demo
copy), the ring-layout floor, invalid submissions, and exact polyhedra."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hillclimb.config import Config
from hillclimb.problem import load_problem

REPO = Path(__file__).resolve().parent.parent
INSTANCES = (30, 50)
ROOTS = (REPO / "problems", REPO / "src" / "hillclimb" / "demo")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_module(REPO / "problems" / "make_tammes.py", "make_tammes")


def _score(problem_dir: Path, submission: str, cwd: Path) -> float:
    """Run the instance's verify.py on `submission` the way verifier.sh
    does: cwd holds submission.csv, the score lands in $HILLCLIMB_RESULT."""
    cwd.mkdir(parents=True, exist_ok=True)
    (cwd / "submission.csv").write_text(submission)
    result = cwd / "result.json"
    run = subprocess.run(
        [sys.executable, str(problem_dir / "verify.py")],
        cwd=cwd, capture_output=True, text=True,
        env={**os.environ, "HILLCLIMB_RESULT": str(result)},
    )
    assert run.returncode == 0, run.stderr
    return json.loads(result.read_text())["score"]


def _csv(points) -> str:
    return "id,x,y,z\n" + "".join(f"{i},{x},{y},{z}\n" for i, (x, y, z) in enumerate(points))


def test_committed_dirs_match_the_generator(generator, tmp_path):
    """Rerun the generator after editing a template; both copies are committed."""
    for n in INSTANCES:
        generated = generator.stamp(tmp_path, n)
        for root in ROOTS:
            committed = root / f"tammes-{n}"
            names = sorted(p.name for p in committed.iterdir() if p.name != "__pycache__")
            assert names == sorted(p.name for p in generated.iterdir())
            for path in generated.iterdir():
                assert (committed / path.name).read_text() == path.read_text(), f"{committed / path.name} is stale"
    assert (REPO / "problems" / "tammes-30" / "verifier.sh").read_bytes() == (
        REPO / "problems" / "heilbronn-11" / "verifier.sh").read_bytes()


def _ring_floor(n: int, ring_size: int) -> float:
    """The sample's min angle: two neighbours on the ring nearest a pole
    (polar angle theta = pi/(R+1), azimuth step 2*pi/ring_size)."""
    rings = math.ceil(n / ring_size)
    theta = math.pi / (rings + 1)
    cos_angle = math.cos(theta) ** 2 + math.sin(theta) ** 2 * math.cos(2 * math.pi / ring_size)
    return math.degrees(math.acos(cos_angle))


@pytest.mark.parametrize("n", INSTANCES)
def test_instance_loads_and_scores_its_sample(config: Config, generator, tmp_path, n):
    from hillclimb.harness.executor import CommandExecutor

    spec = load_problem(f"tammes-{n}", config)
    assert spec.metric_name == "min-angle-deg" and spec.higher_is_better
    assert spec.time_budget_s == 900
    assert len(spec.chart_baselines) == 1

    candidate_dir = tmp_path / f"tammes-{n}"
    candidate_dir.mkdir()
    (candidate_dir / "problem").symlink_to(spec.problem_dir, target_is_directory=True)
    (candidate_dir / "solution.py").write_text(
        "import shutil\n"
        f'shutil.copy(r"{spec.problem_dir / "sample_submission.csv"}", "submission.csv")\n'
    )
    executor = CommandExecutor(Path(sys.executable), spec.verifier_cmd)
    result = executor.execute(candidate_dir / "solution.py", candidate_dir, timeout_s=120)
    assert result.ok, Path(result.stderr_path).read_text()[-400:]
    assert result.val_score == pytest.approx(_ring_floor(n, generator.RING_SIZE), rel=1e-4)
    best = spec.chart_baselines[next(iter(spec.chart_baselines))]
    assert result.val_score < best
    assert best == pytest.approx(generator.BEST_KNOWN[n][0], abs=1e-6)


@pytest.mark.parametrize("n", INSTANCES)
def test_invalid_submissions_score_the_floor(tmp_path, n):
    problem_dir = REPO / "problems" / f"tammes-{n}"
    good = [(math.cos(k), math.sin(k), 0.1 * k) for k in range(n)]
    cases = {
        "short": _csv(good[:-1]),
        "long": _csv(good + [(0.3, 0.3, 0.3)]),
        "zero-vector": _csv([(0.0, 0.0, 0.0)] + good[1:]),
        "non-finite": _csv([("nan", 0.0, 1.0)] + good[1:]),
        "non-numeric": _csv([("a", "b", "c")] + good[1:]),
        "missing-column": "id,x,y\n" + "".join(f"{i},{x},{y}\n" for i, (x, y, _) in enumerate(good)),
        "bad-ids": "id,x,y,z\n" + "".join(f"{i + 1},{x},{y},{z}\n" for i, (x, y, z) in enumerate(good)),
        "empty": "",
    }
    for name, text in cases.items():
        assert _score(problem_dir, text, tmp_path / name) == 0.0, name
    # a valid but degenerate code: two coincident points score 0 without failing
    assert _score(problem_dir, _csv([good[1]] + good[1:]), tmp_path / "coincident") == 0.0


def test_regular_polyhedra_score_their_exact_angles(generator, tmp_path):
    """Any n stamps; the verifier normalizes, so un-normalized vertices are fine."""
    phi = (1 + math.sqrt(5)) / 2
    icosahedron = [(0, s1, s2 * phi) for s1 in (1, -1) for s2 in (1, -1)]
    icosahedron += [(s1, s2 * phi, 0) for s1 in (1, -1) for s2 in (1, -1)]
    icosahedron += [(s2 * phi, 0, s1) for s1 in (1, -1) for s2 in (1, -1)]
    tetrahedron = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    octahedron = [(1, 0, 0), (-1, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 3), (0, 0, -3)]
    for n, points in ((12, icosahedron), (4, tetrahedron), (6, octahedron)):
        problem_dir = generator.stamp(tmp_path / "problems", n)
        score = _score(problem_dir, _csv(points), tmp_path / f"n{n}")
        assert score == pytest.approx(generator.BEST_KNOWN[n][0], abs=1e-9)
    assert generator.BEST_KNOWN[12][0] == pytest.approx(63.434948822922, abs=1e-9)
