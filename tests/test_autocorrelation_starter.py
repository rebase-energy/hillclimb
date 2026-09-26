"""The AlphaEvolve step-function analysis problems (autocorrelation
inequalities + Erdős minimum overlap): generated problem dirs, the wheel's
bundled copies, the sample floors, invalid submissions, exact scoring."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hillclimb.config import Config
from hillclimb.problem import load_problem

REPO = Path(__file__).resolve().parent.parent
KEYS = ("autocorr-1", "autocorr-3", "erdos-overlap")
# key -> (K, sample floor); the indicator scores exactly 2 (tent of height
# 1/2 over (1/2)^2) and the indicator of [0, 1] the trivial overlap bound 1
FLOORS = {"autocorr-1": (600, 2.0), "autocorr-3": (400, 2.0), "erdos-overlap": (400, 1.0)}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve string annotations through here
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_module(REPO / "problems" / "make_autocorrelation.py", "make_autocorrelation")


def csv_of(values) -> str:
    return "id,value\n" + "".join(f"{i},{v!r}\n" for i, v in enumerate(values))


def score(key: str, submission: str, tmp_path: Path) -> float:
    """Run the committed verify.py on one submission text, as the engine would."""
    cwd = tmp_path / "cand"
    cwd.mkdir(exist_ok=True)
    (cwd / "submission.csv").write_text(submission)
    result = cwd / "result.json"
    result.unlink(missing_ok=True)
    env = {**os.environ, "HILLCLIMB_RESULT": str(result)}
    run = subprocess.run(
        [sys.executable, str(REPO / "problems" / key / "verify.py")],
        cwd=cwd, env=env, capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr[-400:]
    return json.loads(result.read_text())["score"]


def test_committed_dirs_match_the_generator(generator, tmp_path):
    """Rerun the generator after editing a template; both copies are committed."""
    assert tuple(generator.DEFAULT_KEYS) == KEYS
    for key in KEYS:
        generated = generator.stamp(tmp_path, key)
        for committed in (REPO / "problems" / key, REPO / "src" / "hillclimb" / "demo" / key):
            for path in generated.iterdir():
                assert (committed / path.name).read_text() == path.read_text(), f"{committed / path.name} is stale"
            assert {p.name for p in committed.iterdir() if p.name != "__pycache__"} == {
                p.name for p in generated.iterdir()
            }


@pytest.mark.parametrize("key", KEYS)
def test_instance_loads_and_scores_its_sample(config: Config, tmp_path, key):
    from hillclimb.executor import CommandExecutor

    spec = load_problem(key, config)
    assert not spec.higher_is_better
    assert spec.time_budget_s == 900
    assert spec.metric_name == {"autocorr-1": "c1-ratio", "autocorr-3": "c3-ratio", "erdos-overlap": "overlap-bound"}[key]
    assert (spec.problem_dir / "verifier.sh").read_bytes() == (REPO / "problems" / "heilbronn-11" / "verifier.sh").read_bytes()
    best = spec.chart_baselines[next(iter(spec.chart_baselines))]

    candidate_dir = tmp_path / key
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
    k, floor = FLOORS[key]
    assert result.val_score == pytest.approx(floor, abs=1e-12)
    assert result.val_score > best  # lower is better: the floor is above the best known


@pytest.mark.parametrize("key", KEYS)
def test_invalid_submissions_score_the_penalty(key, tmp_path):
    k, _ = FLOORS[key]
    penalty = 10000.0
    good = [1.0 if i < k // 2 else 0.0 for i in range(k)] if key == "erdos-overlap" else [1.0] * k
    assert score(key, csv_of(good), tmp_path) != penalty
    assert score(key, csv_of(good[:-1]), tmp_path) == penalty  # wrong row count
    assert score(key, csv_of(good + [0.0]), tmp_path) == penalty
    nonfinite = list(good)
    nonfinite[3] = float("nan")
    assert score(key, csv_of(nonfinite), tmp_path) == penalty
    assert score(key, "id,value\n0,abc\n", tmp_path) == penalty
    assert score(key, "not a csv at all", tmp_path) == penalty
    assert score(key, csv_of([0.0] * k), tmp_path) == penalty  # zero integral / mass
    if key == "autocorr-1":
        bad = list(good)
        bad[0] = -0.5
        assert score(key, csv_of(bad), tmp_path) == penalty  # out of domain: negative
    if key == "autocorr-3":
        signed = [1.0] * (k // 2) + [-1.0] * (k // 2)
        assert score(key, csv_of(signed), tmp_path) == penalty  # ∫f = 0
    if key == "erdos-overlap":
        assert score(key, csv_of([1.5] * (k // 3) + [0.0] * (k - k // 3)), tmp_path) == penalty  # value > 1
        assert score(key, csv_of([0.4] * k), tmp_path) == penalty  # mass 0.8, not 1
        assert score(key, csv_of([0.5 + 1e-6] * k), tmp_path) == penalty  # mass off by 2e-6


class TestKnownConstructions:
    """Closed-form values for two-level step functions: with f = a on the
    left half and b on the right half the ratio is
    4 * max(a^2, b^2, 2|ab|) / (a + b)^2, independent of K."""

    def test_indicator_gives_two_for_any_k(self, generator, tmp_path):
        # the analytic value (tent of height 1/2, integral 1/2) does not depend on the grid
        assert generator.INSTANCES["autocorr-1"].bins != generator.INSTANCES["autocorr-3"].bins
        for key in ("autocorr-1", "autocorr-3"):
            k = FLOORS[key][0]
            assert score(key, csv_of([1.0] * k), tmp_path) == pytest.approx(2.0, abs=1e-12)
            assert score(key, csv_of([0.37] * k), tmp_path) == pytest.approx(2.0, abs=1e-12)  # scale-free

    def test_half_indicator_gives_four(self, tmp_path):
        k = FLOORS["autocorr-1"][0]
        assert score("autocorr-1", csv_of([1.0] * (k // 2) + [0.0] * (k // 2)), tmp_path) == pytest.approx(4.0, abs=1e-12)

    def test_signed_two_level_gives_sixteen(self, tmp_path):
        # a = 1, b = -1/2: max(1, 1/4, 1) = 1 over (1/2)^2
        k = FLOORS["autocorr-3"][0]
        assert score("autocorr-3", csv_of([1.0] * (k // 2) + [-0.5] * (k // 2)), tmp_path) == pytest.approx(16.0, abs=1e-12)

    def test_constant_half_overlap_is_one_half(self, tmp_path):
        k = FLOORS["erdos-overlap"][0]
        assert score("erdos-overlap", csv_of([0.5] * k), tmp_path) == pytest.approx(0.5, abs=1e-12)

    def test_triangle_is_worse_than_the_indicator(self, tmp_path):
        k = FLOORS["autocorr-1"][0]
        x = (np.arange(k) + 0.5) / k - 0.5
        tri = (0.5 - np.abs(x)).tolist()
        value = score("autocorr-1", csv_of(tri), tmp_path)
        assert 2.0 < value < 2.7 and value == pytest.approx(8 / 3, rel=1e-2)


class TestExactness:
    """The closed form (max over the kinks) equals the maximum of the same
    step function refined r-fold — f*f is linear between kinks, so a finer
    grid finds no higher point."""

    def test_autoconvolution_max_is_at_a_kink(self, tmp_path):
        rng = np.random.default_rng(3)
        k, r = FLOORS["autocorr-3"][0], 7
        a = rng.normal(size=k)
        coarse = score("autocorr-3", csv_of(a.tolist()), tmp_path)
        fine = np.repeat(a, r)
        refined = 2 * k * r * np.max(np.abs(np.convolve(fine, fine))) / fine.sum() ** 2
        assert coarse == pytest.approx(refined, rel=1e-12)

    def test_overlap_max_is_at_a_kink(self, tmp_path):
        rng = np.random.default_rng(5)
        k, r = FLOORS["erdos-overlap"][0], 5
        h = rng.random(k)
        h *= 0.5 / h.mean()
        h = np.clip(h, 0, 1)
        h[-1] += 0.5 * k - h.sum()  # exact unit mass
        assert 0 <= h[-1] <= 1
        coarse = score("erdos-overlap", csv_of(h.tolist()), tmp_path)
        fine = np.repeat(h, r)
        refined = (2 / (k * r)) * np.max(np.correlate(fine, 1 - fine, mode="full"))
        assert coarse == pytest.approx(refined, rel=1e-12)
        assert coarse < 1.0
