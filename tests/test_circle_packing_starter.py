"""The circle-packing starter instances: generated problem dirs (committed to
problems/ and the wheel's demo copy), the grid floor, invalid submissions
that must score the floor rather than crash, and AlphaEvolve's published
construction scoring what the paper reports."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from hillclimb.config import Config
from hillclimb.problem import load_problem

REPO = Path(__file__).resolve().parent.parent
INSTANCES = (32,)
COMMITTED_ROOTS = (REPO / "problems", REPO / "src" / "hillclimb" / "demo")

# AlphaEvolve's 32-circle construction (google-deepmind/alphaevolve_results,
# mathematical_results.ipynb, B.12 Construction 2): sum of radii 2.937944526205518
ALPHAEVOLVE_32 = [
    (0.09076163, 0.40381803, 0.090761620923837),
    (0.07310993, 0.92689178, 0.07310821268917801),
    (0.08745017, 0.22570576, 0.087381421261857),
    (0.24855246, 0.30880277, 0.093428060657193),
    (0.4079865, 0.06300614, 0.063006133699386),
    (0.47646318, 0.90136179, 0.09863820013617901),
    (0.89604966, 0.10309934, 0.10309932969006601),
    (0.9066386, 0.68096117, 0.09336139066386),
    (0.08962002, 0.76509474, 0.0895289910471),
    (0.06973669, 0.06965159, 0.06965158303484101),
    (0.40979823, 0.21756451, 0.09156283084371601),
    (0.25742466, 0.88393887, 0.11606111839388701),
    (0.09064689, 0.58506214, 0.090482500951749),
    (0.90294698, 0.30231577, 0.09623644037635501),
    (0.57265603, 0.10585396, 0.105853949414604),
    (0.74007588, 0.40129314, 0.09435083056491601),
    (0.57539962, 0.71183255, 0.115160168483982),
    (0.7367635, 0.21592191, 0.09104997089500201),
    (0.41096972, 0.40263617, 0.093512520648747),
    (0.88664452, 0.88667032, 0.113317128668286),
    (0.57582722, 0.49961748, 0.09705531029446801),
    (0.24962585, 0.49417195, 0.09194421080557799),
    (0.90546338, 0.49309632, 0.094507120549287),
    (0.67381348, 0.90149423, 0.09850576014942301),
    (0.24310147, 0.1077195, 0.10771948922805),
    (0.40815297, 0.5886157, 0.09248833075116601),
    (0.24737889, 0.6771266, 0.090994980900501),
    (0.75801377, 0.7532924, 0.07192969280703),
    (0.73526642, 0.06243992, 0.062439303756069),
    (0.57415412, 0.30715219, 0.095403150459684),
    (0.39239379, 0.75259664, 0.07223814277618501),
    (0.7439361, 0.58879735, 0.093166630683336),
]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_module(REPO / "problems" / "make_circle_packing.py", "make_circle_packing")


def _score(spec, candidate_dir: Path, csv_text: str):
    """Run the real verifier contract on a solution that writes `csv_text`."""
    from hillclimb.executor import CommandExecutor

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


def _csv(rows) -> str:
    return "id,x,y,r\n" + "".join(f"{i},{x!r},{y!r},{r!r}\n" for i, (x, y, r) in enumerate(rows))


def test_committed_dirs_match_the_generator(generator, tmp_path):
    """Rerun the generator after editing a template; both copies are committed."""
    heilbronn_verifier = (REPO / "problems" / "heilbronn-11" / "verifier.sh").read_text()
    for n in INSTANCES:
        generated = generator.stamp(tmp_path, n)
        assert (generated / "verifier.sh").read_text() == heilbronn_verifier
        for root in COMMITTED_ROOTS:
            committed = root / f"circle-packing-{n}"
            for path in generated.iterdir():
                assert (committed / path.name).read_text() == path.read_text(), f"{committed / path.name} is stale"
            assert committed.joinpath("verifier.sh").stat().st_mode & 0o111
    # the hand-written n = 26 problem is not the generator's
    assert not (tmp_path / "circle-packing").exists()


@pytest.mark.parametrize("n", INSTANCES)
def test_sample_scores_the_grid_floor(config: Config, tmp_path, generator, n):
    spec = load_problem(f"circle-packing-{n}", config)
    assert spec.metric_name == "sum-radii" and spec.higher_is_better
    assert spec.time_budget_s == 900
    assert spec.interface_path == spec.problem_dir / "interface.py"
    assert spec.chart_baselines and all(v > 0 for v in spec.chart_baselines.values())

    result = _score(spec, tmp_path / "sample", (spec.problem_dir / "sample_submission.csv").read_text())
    assert "INVALID" not in Path(result.stdout_path).read_text()
    # n equal circles inscribed in the cells of a k x k grid: n / (2k)
    assert result.val_score == pytest.approx(n / (2 * generator.grid_side(n)), rel=1e-9)
    assert result.val_score < min(spec.chart_baselines.values())
    # one instance per circle, decomposing the score
    assert len(result.instance_scores) == n
    assert sum(result.instance_scores.values()) == pytest.approx(result.val_score, abs=1e-4)


@pytest.mark.parametrize("n", INSTANCES)
def test_invalid_submissions_score_the_floor(config: Config, tmp_path, generator, n):
    spec = load_problem(f"circle-packing-{n}", config)
    grid = [(x, y, r) for _, x, y, r in generator.grid_circles(n)]
    bad = {
        "wrong-row-count": _csv(grid[:-1]),
        "outside-the-square": _csv([(0.0, 0.5, 0.1)] + grid[1:]),
        "overlapping": _csv([(grid[1][0], grid[1][1], grid[1][2] + 0.01)] + grid[1:]),
        "negative-radius": _csv([(0.5, 0.5, -0.1)] + grid[1:]),
        "non-finite": _csv([(0.5, 0.5, float("nan"))] + grid[1:]),
        "garbage": "id,x,y,r\n0,a,b,c\n",
        "not-a-csv": "",
    }
    for name, text in bad.items():
        result = _score(spec, tmp_path / name, text)
        assert result.val_score == 0.0, name
        assert "INVALID" in Path(result.stdout_path).read_text(), name


def test_alphaevolve_construction_scores_as_published(config: Config, tmp_path):
    spec = load_problem("circle-packing-32", config)
    result = _score(spec, tmp_path / "alphaevolve", _csv(ALPHAEVOLVE_32))
    assert "INVALID" not in Path(result.stdout_path).read_text()
    assert result.val_score == pytest.approx(2.937944526205518, abs=1e-9)
    assert result.val_score == pytest.approx(spec.chart_baselines["AlphaEvolve 2025"], abs=1e-9)
