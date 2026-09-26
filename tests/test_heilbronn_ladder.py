"""The Heilbronn difficulty ladder: generated problem dirs, the N-agnostic
shared seed, and the problem-supplied behavioral fingerprint."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hillclimb.config import Config
from hillclimb.problem import load_problem

REPO = Path(__file__).resolve().parent.parent
LEVELS = (11, 14, 17)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_module(REPO / "problems" / "make_heilbronn.py", "make_heilbronn")


def test_committed_dirs_match_the_generator(generator, tmp_path):
    """Rerun the generator after editing a template; the dirs are committed —
    both the repo's problems/ and the wheel's bundled copy."""
    for n in LEVELS:
        generated = generator.stamp(tmp_path, n)
        for root in (REPO / "problems", REPO / "src" / "hillclimb" / "demo"):
            committed = root / f"heilbronn-{n}"
            for path in generated.iterdir():
                assert (committed / path.name).read_text() == path.read_text(), f"{committed / path.name} is stale"


@pytest.mark.parametrize("n", LEVELS)
def test_level_loads_and_scores_its_sample(config: Config, tmp_path, n):
    from hillclimb.executor import CommandExecutor

    spec = load_problem(f"heilbronn-{n}", config)
    assert spec.metric_name == "min-triangle-area" and spec.higher_is_better
    assert spec.time_budget_s == 900
    assert len(spec.chart_baselines) == 1
    assert spec.fingerprint_path == spec.problem_dir / "fingerprint.py"

    candidate_dir = tmp_path / f"heilbronn-{n}"
    candidate_dir.mkdir()
    (candidate_dir / "data").symlink_to(spec.data_dir, target_is_directory=True)
    (candidate_dir / "problem").symlink_to(spec.problem_dir, target_is_directory=True)
    (candidate_dir / "solution.py").write_text(
        "import shutil\n"
        f'shutil.copy(r"{spec.problem_dir / "sample_submission.csv"}", "submission.csv")\n'
    )
    executor = CommandExecutor(Path(sys.executable), spec.verifier_cmd)
    result = executor.execute(candidate_dir / "solution.py", candidate_dir, timeout_s=120)
    assert result.ok, Path(result.stderr_path).read_text()[-400:]
    # points on a parabola with spacing h: the smallest triangle is h^3
    assert result.val_score == pytest.approx((1 / (n - 1)) ** 3, rel=1e-3)
    assert result.val_score < spec.chart_baselines[next(iter(spec.chart_baselines))]


@pytest.mark.parametrize("n", LEVELS)
def test_shared_seed_reads_n_from_the_problem(config: Config, tmp_path, n):
    spec = load_problem(f"heilbronn-{n}", config)
    candidate_dir = tmp_path / "cand"
    candidate_dir.mkdir()
    (candidate_dir / "problem").symlink_to(spec.problem_dir, target_is_directory=True)
    seed = REPO / "hillclimb" / "experiments" / "seeds" / "heilbronn.py"
    run = subprocess.run([sys.executable, str(seed)], cwd=candidate_dir, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    rows = (candidate_dir / "submission.csv").read_text().splitlines()
    assert rows[0] == "id,x,y" and len(rows) == n + 1
    # the seed is exactly the sample: the same neutral start for every arm
    assert (candidate_dir / "submission.csv").read_text() == (spec.problem_dir / "sample_submission.csv").read_text()


@pytest.fixture(scope="module")
def module():
    return _load_module(REPO / "problems" / "heilbronn-11" / "fingerprint.py", "heilbronn_11_fingerprint")


class TestFingerprint:
    @staticmethod
    def _write(path: Path, pts: np.ndarray, ids=None) -> Path:
        ids = list(range(len(pts))) if ids is None else ids
        path.mkdir(parents=True, exist_ok=True)
        lines = ["id,x,y"] + [f"{i},{x:.6f},{y:.6f}" for i, (x, y) in zip(ids, pts)]
        (path / "submission.csv").write_text("\n".join(lines) + "\n")
        return path

    def test_invariant_to_relabeling_and_symmetry(self, module, tmp_path):
        rng = np.random.default_rng(7)
        pts = rng.random((11, 2))
        base = np.asarray(module.fingerprint(self._write(tmp_path / "a", pts)))
        assert base.shape == (165,) and np.all(np.diff(base) >= 0) and base[0] > 0
        permuted = self._write(tmp_path / "b", pts[::-1], ids=list(range(11)))
        rotated = self._write(tmp_path / "c", np.column_stack([1 - pts[:, 1], pts[:, 0]]))
        reflected = self._write(tmp_path / "d", np.column_stack([1 - pts[:, 0], pts[:, 1]]))
        for path in (permuted, rotated, reflected):
            assert np.allclose(module.fingerprint(path), base, atol=1e-6)
        moved = pts.copy()
        moved[3] = (moved[3] + 0.3) % 1.0
        assert not np.allclose(module.fingerprint(self._write(tmp_path / "e", moved)), base, atol=1e-6)

    def test_rejects_what_the_scorer_rejects(self, module, tmp_path):
        rng = np.random.default_rng(1)
        assert module.fingerprint(self._write(tmp_path / "short", rng.random((10, 2)))) is None
        outside = rng.random((11, 2))
        outside[0] = (1.5, 0.5)
        assert module.fingerprint(self._write(tmp_path / "out", outside)) is None
        assert module.fingerprint(tmp_path / "missing") is None
