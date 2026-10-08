"""The Thomson starter instances: generated problem dirs (repo + bundled demo
copy), the ring-layout floor, invalid submissions, and exact polyhedra."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hillclimb.config import Config
from hillclimb.problem import load_problem

REPO = Path(__file__).resolve().parent.parent
INSTANCES = (50, 100)
ROOTS = (REPO / "problems",)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_module(REPO / "problems" / "make_thomson.py", "make_thomson")


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


def _energy(points) -> float:
    """Independent Coulomb energy of the normalized points."""
    pts = np.asarray(points, float)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    i, j = np.triu_indices(len(pts), 1)
    return float(np.sum(1.0 / np.linalg.norm(pts[i] - pts[j], axis=1)))


def test_committed_dirs_match_the_generator(generator, tmp_path):
    """Rerun the generator after editing a template; both copies are committed."""
    for n in INSTANCES:
        generated = generator.stamp(tmp_path, n)
        for root in ROOTS:
            committed = root / f"thomson-{n}"
            names = sorted(p.name for p in committed.iterdir() if p.name != "__pycache__")
            assert names == sorted(p.name for p in generated.iterdir())
            for path in generated.iterdir():
                assert (committed / path.name).read_text() == path.read_text(), f"{committed / path.name} is stale"
    assert (REPO / "problems" / "thomson-50" / "verifier.sh").read_bytes() == (
        REPO / "problems" / "heilbronn-11" / "verifier.sh").read_bytes()


@pytest.mark.parametrize("n", INSTANCES)
def test_instance_loads_and_scores_its_sample(config: Config, generator, tmp_path, n):
    from hillclimb.harness.executor import CommandExecutor

    spec = load_problem(f"thomson-{n}", config)
    assert spec.metric_name == "coulomb-energy" and not spec.higher_is_better
    assert len(spec.chart_baselines) == 1

    candidate_dir = tmp_path / f"thomson-{n}"
    candidate_dir.mkdir()
    (candidate_dir / "problem").symlink_to(spec.problem_dir, target_is_directory=True)
    (candidate_dir / "solution.py").write_text(
        "import shutil\n"
        f'shutil.copy(r"{spec.problem_dir / "sample_submission.csv"}", "submission.csv")\n'
    )
    executor = CommandExecutor(Path(sys.executable), spec.verifier_cmd)
    result = executor.execute(candidate_dir / "solution.py", candidate_dir, timeout_s=120)
    assert result.ok, Path(result.stderr_path).read_text()[-400:]
    sample = [(x, y, z) for _, x, y, z in generator.ring_points(n)]
    assert result.val_score == pytest.approx(_energy(sample), rel=1e-9)
    best = spec.chart_baselines[next(iter(spec.chart_baselines))]
    assert result.val_score > best  # lower is better: the floor sits above the best known
    assert best == pytest.approx(generator.BEST_KNOWN[n][0], abs=1e-6)


@pytest.mark.parametrize("n", INSTANCES)
def test_invalid_submissions_score_the_penalty(generator, tmp_path, n):
    problem_dir = REPO / "problems" / f"thomson-{n}"
    penalty = generator.penalty(n)
    good = [(math.cos(k), math.sin(k), 0.1 * k) for k in range(n)]
    cases = {
        "short": _csv(good[:-1]),
        "long": _csv(good + [(0.3, 0.3, 0.3)]),
        "zero-vector": _csv([(0.0, 0.0, 0.0)] + good[1:]),
        "non-finite": _csv([("inf", 0.0, 1.0)] + good[1:]),
        "non-numeric": _csv([("a", "b", "c")] + good[1:]),
        "coincident": _csv([good[1]] + good[1:]),
        "missing-column": "id,x,y\n" + "".join(f"{i},{x},{y}\n" for i, (x, y, _) in enumerate(good)),
        "bad-ids": "id,x,y,z\n" + "".join(f"{i + 1},{x},{y},{z}\n" for i, (x, y, z) in enumerate(good)),
        "empty": "",
    }
    for name, text in cases.items():
        assert _score(problem_dir, text, tmp_path / name) == penalty, name
    assert _score(problem_dir, _csv(good), tmp_path / "valid") == pytest.approx(_energy(good), rel=1e-9)


def test_regular_polyhedra_score_their_known_energies(generator, tmp_path):
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
        assert score == pytest.approx(generator.BEST_KNOWN[n][0], abs=1e-8)
    # the tetrahedron in closed form: 6 pairs at distance sqrt(8/3)
    assert generator.BEST_KNOWN[4][0] == pytest.approx(6 / math.sqrt(8 / 3), abs=1e-9)
