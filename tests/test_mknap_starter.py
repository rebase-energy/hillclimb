"""The multidimensional-knapsack ladder: generated problem dirs (committed
to problems/ and the wheel's demo copy), the empty-knapsack floor, invalid
submissions that must score the penalty rather than crash, a greedy fill
that is valid and worth something, and the reference line agreeing with
OR-Library's mkcbres table."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from hillclimb.config import Config
from hillclimb.problem import load_problem

REPO = Path(__file__).resolve().parent.parent
LEVELS = ((100, 5), (250, 10))
COMMITTED_ROOTS = (REPO / "problems", REPO / "src" / "hillclimb" / "demo")
PENALTY = 0.0
# OR-Library mkcbres: Chu & Beasley's best feasible values for 5.100-00 and 10.250-00
BEST_KNOWN = {(100, 5): 24381, (250, 10): 59187}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_module(REPO / "problems" / "make_mknap.py", "make_mknap")


def _score(spec, candidate_dir: Path, csv_text: str):
    """Run the real verifier contract on a solution that writes `csv_text`."""
    from hillclimb.harness.executor import CommandExecutor

    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "data").symlink_to(spec.data_dir, target_is_directory=True)
    (candidate_dir / "problem").symlink_to(spec.problem_dir, target_is_directory=True)
    (candidate_dir / "solution.py").write_text(
        "from pathlib import Path\n"
        f"Path('submission.csv').write_text({csv_text!r})\n"
    )
    executor = CommandExecutor(Path(sys.executable), spec.verifier_cmd)
    result = executor.execute(candidate_dir / "solution.py", candidate_dir, timeout_s=120)
    assert result.ok, Path(result.stderr_path).read_text()[-400:]
    return result


def _csv(take) -> str:
    return "id,take\n" + "".join(f"{i},{t}\n" for i, t in enumerate(take))


def _instance(spec):
    """values, weights (m lists of n), capacities — from the shipped data."""
    rows = [line.split(",") for line in (spec.data_dir / "items.csv").read_text().strip().splitlines()]
    header, body = rows[0], rows[1:]
    m = len(header) - 2
    values = [int(r[1]) for r in body]
    weights = [[int(r[2 + i]) for r in body] for i in range(m)]
    caps = [int(line.split(",")[1]) for line in (spec.data_dir / "capacities.csv").read_text().strip().splitlines()[1:]]
    return values, weights, caps


def _greedy(values, weights, caps):
    """Items by value per unit of capacity-normalised weight, added while they fit."""
    n, m = len(values), len(weights)
    order = sorted(range(n), key=lambda j: -values[j] / sum(weights[i][j] / caps[i] for i in range(m)))
    take, used = [0] * n, [0] * m
    for j in order:
        if all(used[i] + weights[i][j] <= caps[i] for i in range(m)):
            take[j] = 1
            for i in range(m):
                used[i] += weights[i][j]
    return take


@pytest.mark.parametrize("level", LEVELS)
def test_committed_dirs_match_the_generator(generator, level):
    n, m = level
    for root in COMMITTED_ROOTS:
        problem_dir = root / f"mknap-{n}-{m}"
        assert problem_dir.is_dir(), f"missing {problem_dir}"
        for name, text in generator.files_for(n, m).items():
            assert (problem_dir / name).read_text() == text, f"{problem_dir / name} drifted from make_mknap.py"


@pytest.mark.parametrize("level", LEVELS)
def test_instance_shape_and_reference_line(config: Config, level):
    n, m = level
    spec = load_problem(f"mknap-{n}-{m}", config)
    assert spec.metric_name == "total-value" and spec.higher_is_better
    values, weights, caps = _instance(spec)
    assert len(values) == n and len(weights) == m and len(caps) == m
    assert all(len(w) == n for w in weights)
    # tightness 0.25: each capacity is a quarter of that constraint's total weight, rounded
    for i in range(m):
        assert abs(caps[i] - 0.25 * sum(weights[i])) <= 1
    assert spec.chart_baselines == {"best known (Chu & Beasley 1998)": BEST_KNOWN[level]}


@pytest.mark.parametrize("level", LEVELS)
def test_sample_is_the_empty_knapsack_and_greedy_fills_it(config: Config, tmp_path, level):
    n, m = level
    spec = load_problem(f"mknap-{n}-{m}", config)
    sample = spec.baseline_files["submission.csv"].read_text()
    assert _score(spec, tmp_path / "empty", sample).val_score == 0.0
    values, weights, caps = _instance(spec)
    take = _greedy(values, weights, caps)
    result = _score(spec, tmp_path / "greedy", _csv(take))
    assert result.val_score == sum(v for v, t in zip(values, take) if t)
    assert 0.9 * BEST_KNOWN[level] < result.val_score < BEST_KNOWN[level]
    assert "INVALID" not in Path(result.stdout_path).read_text()


@pytest.mark.parametrize("level", LEVELS)
def test_invalid_submissions_score_the_penalty(config: Config, tmp_path, level):
    n, m = level
    spec = load_problem(f"mknap-{n}-{m}", config)
    cases = {
        "everything": _csv([1] * n),
        "not-binary": _csv([2] + [0] * (n - 1)),
        "text": _csv(["yes"] + [0] * (n - 1)),
        "short": _csv([0] * (n - 1)),
    }
    for name, csv_text in cases.items():
        result = _score(spec, tmp_path / name, csv_text)
        assert result.val_score == PENALTY, name
        assert "INVALID" in Path(result.stdout_path).read_text(), name
