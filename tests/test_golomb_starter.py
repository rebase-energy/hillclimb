"""The Golomb-ruler starter problems: generated dirs (repo + bundled demo
copy), the greedy sample's floor, invalid submissions scoring the penalty,
and the proven-optimal rulers scoring their length."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from hillclimb.config import Config
from hillclimb.problem import load_problem

REPO = Path(__file__).resolve().parent.parent
INSTANCES = (20, 27)
PENALTY = 10**6

# proven-optimal rulers, https://en.wikipedia.org/wiki/Golomb_ruler (table of optimal rulers)
OPTIMAL = {
    20: [0, 1, 8, 11, 68, 77, 94, 116, 121, 156, 158, 179, 194, 208, 212, 228, 240, 253, 259, 283],
    27: [0, 3, 15, 41, 66, 95, 97, 106, 142, 152, 220, 221, 225, 242, 295, 330, 338, 354, 382, 388, 402,
         415, 486, 504, 523, 546, 553],
}
GREEDY_FLOOR = {20: 474, 27: 969}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_module(REPO / "problems" / "make_golomb.py", "make_golomb")


def _score(config: Config, tmp_path: Path, m: int, submission: str) -> float:
    """Run the problem's verifier on `submission` through the executor."""
    from hillclimb.executor import CommandExecutor

    spec = load_problem(f"golomb-{m}", config)
    candidate_dir = tmp_path / "cand"
    candidate_dir.mkdir()
    (candidate_dir / "problem").symlink_to(spec.problem_dir, target_is_directory=True)
    (candidate_dir / "payload.csv").write_text(submission)
    (candidate_dir / "solution.py").write_text('import shutil\nshutil.copy("payload.csv", "submission.csv")\n')
    executor = CommandExecutor(Path(sys.executable), spec.verifier_cmd)
    result = executor.execute(candidate_dir / "solution.py", candidate_dir, timeout_s=120)
    assert result.ok, Path(result.stderr_path).read_text()[-400:]
    return result.val_score


def _csv(marks, ids=None) -> str:
    ids = range(len(marks)) if ids is None else ids
    return "id,mark\n" + "".join(f"{i},{mark}\n" for i, mark in zip(ids, marks))


def test_committed_dirs_match_the_generator(generator, tmp_path):
    """Rerun the generator after editing a template; both copies are committed."""
    for m in INSTANCES:
        generated = generator.stamp(tmp_path, m)
        for committed in (REPO / "problems" / f"golomb-{m}", REPO / "src" / "hillclimb" / "demo" / f"golomb-{m}"):
            assert sorted(p.name for p in committed.iterdir() if p.name != "__pycache__") == sorted(
                p.name for p in generated.iterdir()
            )
            for path in generated.iterdir():
                assert (committed / path.name).read_bytes() == path.read_bytes(), f"{committed / path.name} is stale"
        assert (REPO / "problems" / f"golomb-{m}" / "verifier.sh").read_bytes() == (
            REPO / "problems" / "heilbronn-11" / "verifier.sh"
        ).read_bytes()


@pytest.mark.parametrize("m", INSTANCES)
def test_sample_scores_at_the_floor(config: Config, tmp_path, m):
    spec = load_problem(f"golomb-{m}", config)
    assert spec.metric_name == "length" and not spec.higher_is_better
    assert spec.time_budget_s == 900
    assert len(spec.chart_baselines) == 1
    sample = (spec.problem_dir / "sample_submission.csv").read_text()
    score = _score(config, tmp_path, m, sample)
    assert score == GREEDY_FLOOR[m]
    assert score > spec.chart_baselines[next(iter(spec.chart_baselines))]


@pytest.mark.parametrize("m", INSTANCES)
def test_optimal_ruler_scores_its_length(config: Config, tmp_path, m):
    spec = load_problem(f"golomb-{m}", config)
    best = spec.chart_baselines[next(iter(spec.chart_baselines))]
    # shuffled row order: the verifier sorts marks itself
    marks = OPTIMAL[m][::-1]
    assert _score(config, tmp_path, m, _csv(marks)) == best == OPTIMAL[m][-1]


@pytest.mark.parametrize(
    "name, make",
    [
        ("wrong row count", lambda m: _csv(OPTIMAL[m][:-1])),
        ("wrong ids", lambda m: _csv(OPTIMAL[m], ids=range(1, m + 1))),
        ("no zero", lambda m: _csv([x + 1 for x in OPTIMAL[m]])),
        ("negative", lambda m: _csv([-1, *OPTIMAL[m][1:]])),
        ("duplicate mark", lambda m: _csv([*OPTIMAL[m][:-1], OPTIMAL[m][-2]])),
        ("repeated difference", lambda m: _csv([*OPTIMAL[m][:-1], OPTIMAL[m][-2] + 1])),
        ("non-integer", lambda m: _csv([*OPTIMAL[m][:-1], 0.5])),
        ("out of range", lambda m: _csv([*OPTIMAL[m][:-1], PENALTY])),
        ("non-finite", lambda m: _csv([*OPTIMAL[m][:-1], "inf"])),
        ("missing column", lambda m: "id,x\n0,0\n"),
        ("garbage", lambda m: "not,a,csv\n\x00"),
    ],
)
@pytest.mark.parametrize("m", INSTANCES)
def test_invalid_submissions_score_the_penalty(config: Config, tmp_path, m, name, make):
    assert _score(config, tmp_path, m, make(m)) == PENALTY, name
