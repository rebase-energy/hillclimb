"""The LABS ladder: generated problem dirs (committed to problems/ and the
wheel's demo copy), the growing-runs floor, invalid submissions that must
score the penalty rather than crash, and the published optimal sequences
scoring their proven energies."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from hillclimb.config import Config
from hillclimb.problem import load_problem

REPO = Path(__file__).resolve().parent.parent
LEVELS = (40, 60)
COMMITTED_ROOTS = (REPO / "problems", REPO / "src" / "hillclimb" / "demo")
PENALTY = 100000.0
# energy of the growing-runs sample (1, 2, 3, ... alternating sign, cut at N)
SAMPLE_ENERGY = {40: 2356, 60: 6882}
# run-length encodings of optimal sequences from Packebusch & Mertens 2016
# (Table 1): the sequence alternates sign at every run boundary
OPTIMAL_RUNS = {40: ("44412112131121313131", 108), 60: ("761112141111131124211322211222", 218)}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_module(REPO / "problems" / "make_labs.py", "make_labs")


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


def _csv(spins) -> str:
    return "id,spin\n" + "".join(f"{i},{s}\n" for i, s in enumerate(spins))


def _decode_runs(code: str) -> list[int]:
    spins, sign = [], 1
    for run in code:
        spins += [sign] * int(run)
        sign = -sign
    return spins


def test_committed_dirs_match_the_generator(generator, tmp_path):
    """Rerun the generator after editing a template; both copies are committed."""
    heilbronn_verifier = (REPO / "problems" / "heilbronn-11" / "verifier.sh").read_text()
    for n in LEVELS:
        generated = generator.stamp(tmp_path, n)
        assert (generated / "verifier.sh").read_text() == heilbronn_verifier
        for root in COMMITTED_ROOTS:
            committed = root / f"labs-{n}"
            for path in generated.iterdir():
                assert (committed / path.name).read_text() == path.read_text(), f"{committed / path.name} is stale"
            assert committed.joinpath("verifier.sh").stat().st_mode & 0o111


@pytest.mark.parametrize("n", LEVELS)
def test_sample_scores_the_growing_runs_floor(config: Config, tmp_path, generator, n):
    spec = load_problem(f"labs-{n}", config)
    assert spec.metric_name == "autocorrelation-energy" and not spec.higher_is_better
    assert spec.time_budget_s == 900
    optimal = spec.chart_baselines[f"optimal ({generator.OPTIMAL_SOURCE})"]
    assert optimal == generator.BEST_KNOWN[n][0]

    result = _score(spec, tmp_path / "sample", (spec.problem_dir / "sample_submission.csv").read_text())
    assert "INVALID" not in Path(result.stdout_path).read_text()
    assert result.val_score == SAMPLE_ENERGY[n] == generator.energy(generator.growing_runs(n))
    assert result.val_score > optimal


@pytest.mark.parametrize("n", LEVELS)
def test_invalid_submissions_score_the_penalty(config: Config, tmp_path, generator, n):
    spec = load_problem(f"labs-{n}", config)
    good = generator.growing_runs(n)
    bad = {
        "wrong-row-count": _csv(good[:-1]),
        "out-of-domain": _csv([2] + good[1:]),
        "zero-spin": _csv([0] + good[1:]),
        "non-finite": _csv(["nan"] + good[1:]),
        "text-spin": _csv(["up"] + good[1:]),
        "duplicate-ids": "id,spin\n" + "".join(f"0,{s}\n" for s in good),
        "not-a-csv": "",
    }
    for name, text in bad.items():
        result = _score(spec, tmp_path / name, text)
        assert result.val_score == PENALTY, name
        assert "INVALID" in Path(result.stdout_path).read_text(), name


@pytest.mark.parametrize("n", LEVELS)
def test_published_optimum_scores_its_proven_energy(config: Config, tmp_path, generator, n):
    spec = load_problem(f"labs-{n}", config)
    code, expected = OPTIMAL_RUNS[n]
    spins = _decode_runs(code)
    assert len(spins) == n and generator.energy(spins) == expected
    result = _score(spec, tmp_path / "optimal", _csv(spins))
    assert "INVALID" not in Path(result.stdout_path).read_text()
    assert result.val_score == expected == spec.chart_baselines[f"optimal ({generator.OPTIMAL_SOURCE})"]
