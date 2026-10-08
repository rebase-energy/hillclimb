"""The kissing-number starter problems: generated dirs (repo + bundled demo
copy), the sample's floor, invalid submissions scoring the floor, and a known
lattice configuration (D_11's 220 minimal vectors) scoring its size."""

from __future__ import annotations

import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import pytest

from hillclimb.config import Config
from hillclimb.problem import load_problem

REPO = Path(__file__).resolve().parent.parent
INSTANCES = (11,)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_module(REPO / "problems" / "make_kissing.py", "make_kissing")


def _score(config: Config, tmp_path: Path, d: int, submission: str) -> float:
    """Run the problem's verifier on `submission` through the executor."""
    from hillclimb.harness.executor import CommandExecutor

    spec = load_problem(f"kissing-{d}", config)
    candidate_dir = tmp_path / "cand"
    candidate_dir.mkdir()
    (candidate_dir / "problem").symlink_to(spec.problem_dir, target_is_directory=True)
    (candidate_dir / "payload.csv").write_text(submission)
    (candidate_dir / "solution.py").write_text('import shutil\nshutil.copy("payload.csv", "submission.csv")\n')
    executor = CommandExecutor(Path(sys.executable), spec.verifier_cmd)
    result = executor.execute(candidate_dir / "solution.py", candidate_dir, timeout_s=120)
    assert result.ok, Path(result.stderr_path).read_text()[-400:]
    return result.val_score


def _csv(d: int, points: list[list[int]], ids=None) -> str:
    ids = range(len(points)) if ids is None else ids
    header = "id," + ",".join(f"c{i}" for i in range(d))
    return "\n".join([header, *(f"{k}," + ",".join(str(c) for c in p) for k, p in zip(ids, points))]) + "\n"


def test_committed_dirs_match_the_generator(generator, tmp_path):
    """Rerun the generator after editing a template; both copies are committed."""
    for d in INSTANCES:
        generated = generator.stamp(tmp_path, d)
        for committed in (REPO / "problems" / f"kissing-{d}",):
            assert sorted(p.name for p in committed.iterdir() if p.name != "__pycache__") == sorted(
                p.name for p in generated.iterdir()
            )
            for path in generated.iterdir():
                assert (committed / path.name).read_bytes() == path.read_bytes(), f"{committed / path.name} is stale"
    assert (REPO / "problems" / "kissing-11" / "verifier.sh").read_bytes() == (
        REPO / "problems" / "heilbronn-11" / "verifier.sh"
    ).read_bytes()


@pytest.mark.parametrize("d", INSTANCES)
def test_sample_scores_at_the_floor(config: Config, tmp_path, d):
    spec = load_problem(f"kissing-{d}", config)
    assert spec.metric_name == "points" and spec.higher_is_better
    assert spec.time_budget_s == 3600
    assert len(spec.chart_baselines) == 1
    sample = (spec.problem_dir / "sample_submission.csv").read_text()
    score = _score(config, tmp_path, d, sample)
    assert score == 2 * d  # the ±e_i configuration
    assert score < spec.chart_baselines[next(iter(spec.chart_baselines))]


@pytest.mark.parametrize("d", INSTANCES)
def test_d_lattice_minimal_vectors_score_their_count(config: Config, tmp_path, d):
    """±e_i ± e_j: norm² 2, pairwise distance² >= 2 — a valid configuration
    of 2d(d-1) points (the D_d lattice's kissing number)."""
    points = []
    for i, j in combinations(range(d), 2):
        for si in (1, -1):
            for sj in (1, -1):
                row = [0] * d
                row[i], row[j] = si, sj
                points.append(row)
    assert _score(config, tmp_path, d, _csv(d, points)) == 2 * d * (d - 1)


def _unit(d: int, i: int, sign: int = 1) -> list[int]:
    row = [0] * d
    row[i] = sign
    return row


@pytest.mark.parametrize(
    "name, make",
    [
        ("wrong ids", lambda d: _csv(d, [_unit(d, 0), _unit(d, 1)], ids=[0, 5])),
        ("too many rows", lambda d: _csv(d, [[k + 1] + [0] * (d - 1) for k in range(2001)])),
        ("origin", lambda d: _csv(d, [_unit(d, 0), [0] * d])),
        ("duplicate point", lambda d: _csv(d, [_unit(d, 0), _unit(d, 0)])),
        ("non-integer", lambda d: _csv(d, [_unit(d, 0), [0.5] + [0] * (d - 1)])),
        ("out of range", lambda d: _csv(d, [_unit(d, 0), [10**9] + [0] * (d - 1)])),
        ("non-finite", lambda d: _csv(d, [_unit(d, 0), ["nan"] + [0] * (d - 1)])),
        ("too close", lambda d: _csv(d, [[2] + [0] * (d - 1), [2, 1] + [0] * (d - 2)])),
        ("missing column", lambda d: "id,c0\n0,1\n"),
        ("garbage", lambda d: "not,a,csv\n\x00"),
    ],
)
@pytest.mark.parametrize("d", INSTANCES)
def test_invalid_submissions_score_zero(config: Config, tmp_path, d, name, make):
    assert _score(config, tmp_path, d, make(d)) == 0.0, name
