"""`Problem` as the thing a climber solves, and `climber.solve` as the sklearn shape."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

import hillclimb as hc
from hillclimb import Climber, Problem, register_agent
from hillclimb.agents import AgentResult
from hillclimb.policies import Greedy
from hillclimb.selectors import Best

QUIET = dict(log=lambda *_: None)

SCORING_PY = '''\
from pathlib import Path


def largest(run_dir: Path) -> float:
    return float((run_dir / "answer.txt").read_text())


def with_metrics(run_dir: Path) -> dict:
    value = float((run_dir / "answer.txt").read_text())
    return {"score": value, "digits": len(str(int(value)))}


def broken(run_dir: Path) -> float:
    raise RuntimeError("the scorer itself is wrong")


def nested():
    def inner(run_dir):
        return 0.0
    return inner
'''


@pytest.fixture
def scoring(tmp_path):
    """Scoring functions in a .py file of their own, as a user would have."""
    path = tmp_path / "scoring.py"
    path.write_text(SCORING_PY)
    spec = importlib.util.spec_from_file_location("scoring_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


@pytest.fixture
def counter(monkeypatch):
    """An agent whose every solution writes a bigger number."""
    monkeypatch.setattr("hillclimb.agents._AGENTS", dict(hc.agents._AGENTS))

    class Counter:
        name = "counter"

        def __init__(self):
            self.n = 0

        def invoke(self, request):
            self.n += 1
            (request.candidate_dir / "solution.py").write_text(
                f'open("answer.txt", "w").write("{self.n * 10}")\n'
            )
            return AgentResult(ok=True)

    register_agent("counter", Counter)
    return "counter"


@pytest.fixture
def folder(config, tmp_path, monkeypatch):
    config.paths.problems_dir = tmp_path / "problems"
    monkeypatch.setattr("hillclimb.api.ensure_runtime_venv", lambda *a, **k: Path(sys.executable))
    return config


def largest_number(scoring, **extra) -> Problem:
    options = dict(
        score=scoring.largest, higher_is_better=True, baseline=0.0,
        description="Write the largest number you can to answer.txt.", output="answer.txt",
    )
    return Problem("largest-number", **{**options, **extra})


# --- defining a problem ---


def test_a_defined_problem_is_written_as_a_folder_a_search_can_use(scoring, tmp_path):
    problem = largest_number(scoring, files={"hint.txt": "bigger is better\n"}, requirements=["numpy"])
    assert problem.defined and problem.id == "largest-number" and "defined here" in repr(problem)
    folder = problem.save(tmp_path / "problems")
    assert folder == tmp_path / "problems" / "largest-number"
    names = {p.name for p in folder.iterdir()}
    assert names == {"problem.yaml", "description.md", "verify.py", "verifier.sh", "verifier.py", "hint.txt", "requirements.txt"}
    meta = yaml.safe_load((folder / "problem.yaml").read_text())
    assert meta["metric"] == "score" and meta["higher_is_better"] and meta["baseline"] == 0.0
    assert meta["output_artifacts"] == ["answer.txt"] and meta["requirements"] == "requirements.txt"
    assert meta["score_function"] == f"{tmp_path / 'scoring.py'}:largest"
    assert "answer.txt" in (folder / "description.md").read_text()
    assert (folder / "verifier.sh").stat().st_mode & 0o111
    assert f"SOURCE = {str(tmp_path / 'scoring.py')!r}" in (folder / "verify.py").read_text()
    # written again when the definition changes; refused on a folder that is not ours
    largest_number(scoring, baseline=1.0).save(tmp_path / "problems")
    assert yaml.safe_load((folder / "problem.yaml").read_text())["baseline"] == 1.0
    foreign = tmp_path / "problems" / "theirs"
    foreign.mkdir()
    (foreign / "problem.yaml").write_text("metric: x\n")
    with pytest.raises(FileExistsError, match="not one hillclimb.Problem wrote"):
        Problem("theirs", score=scoring.largest, description="d").save(tmp_path / "problems")


def test_a_definition_that_cannot_work_fails_at_once(scoring):
    with pytest.raises(ValueError, match="top level of a .py file"):
        Problem("p", score=lambda d: 0.0, description="d")
    with pytest.raises(ValueError, match="top level of a .py file"):
        Problem("p", score=scoring.nested(), description="d")
    with pytest.raises(ValueError, match="description"):
        Problem("p", score=scoring.largest)
    with pytest.raises(ValueError, match="plain name"):
        Problem("emflow://x", score=scoring.largest, description="d")
    with pytest.raises(TypeError):
        Problem("p", score="verify.sh", description="d")
    with pytest.raises(ValueError, match="exists already"):
        Problem("fitness-landscape").save()


def test_an_existing_problem_resolves_and_a_bundled_one_is_fetched(folder):
    assert not Problem("heilbronn-11").defined
    assert Problem("emflow://gefcom2014:solar").id == "gefcom2014-solar"
    problem = Problem("fitness-landscape")
    assert problem.resolve(folder) == "fitness-landscape"
    assert (folder.paths.problems_dir / "fitness-landscape" / "problem.yaml").exists()
    spec = problem.spec(folder)
    assert spec.problem_id == "fitness-landscape" and spec.higher_is_better
    assert "terrain" in problem.description
    with pytest.raises(FileNotFoundError):
        Problem("no-such-problem").spec(folder)


# --- solving one ---


def test_climber_solve_leaves_the_result_on_the_climber(scoring, counter, folder):
    climber = Climber(select=Best(num_drafts=2, ensemble=False), policy=Greedy(tune_budget=0))
    with pytest.raises(RuntimeError, match="no search yet"):
        climber.best  # noqa: B018
    assert climber.search(largest_number(scoring), agent=counter, max_evaluations=4, learning=False, config=folder, **QUIET) is climber

    assert climber.result.state == "done" and climber.result is climber.session.outcome
    assert climber.best.val_score == 40.0 and climber.selected.candidate_id == climber.best.candidate_id
    assert [(s.candidate_id, s.score) for s in climber.history] == [
        ("c000", 0.0), ("c001", 10.0), ("c002", 20.0), ("c003", 30.0), ("c004", 40.0),
    ]
    assert climber.solution == 'open("answer.txt", "w").write("40")\n' == climber.result.source("c004")
    assert climber.params == {} and climber.spend.evaluations == 4
    frame = climber.to_frame()
    assert list(frame.index) == [c.candidate_id for c in climber.candidates] == ["c000", "c001", "c002", "c003", "c004"]
    assert frame["status"].eq("passing").all()
    assert (folder.paths.problems_dir / "largest-number" / "verify.py").exists()
    # the climber is still the definition it was, and solves again
    assert climber.search(largest_number(scoring), agent=counter, max_evaluations=1, learning=False, config=folder, **QUIET).best.val_score == 10.0


def test_hc_run_takes_a_problem_too(scoring, counter, folder):
    outcome = hc.run(largest_number(scoring), agent=counter, max_evaluations=2, learning=False, config=folder, **QUIET)
    assert outcome.state == "done" and outcome.best.val_score == 20.0
    assert yaml.safe_load((outcome.run_dir / "spec.yaml").read_text())["problems"][0]["target"].endswith("largest-number")


def test_a_scorer_may_report_metrics_and_a_broken_one_makes_a_buggy_candidate(scoring, counter, folder):
    rich = Problem("rich", score=scoring.with_metrics, description="Write a number to answer.txt.", output="answer.txt")
    climber = Climber(select=Best(num_drafts=1, ensemble=False), policy=Greedy(tune_budget=0))
    climber.search(rich, agent=counter, max_evaluations=1, learning=False, config=folder, **QUIET)
    assert climber.best.val_score == 10.0 and climber.best.metrics == {"digits": 2}

    broken = Problem("broken", score=scoring.broken, description="Write a number to answer.txt.", output="answer.txt")
    climber = Climber(select=Best(num_drafts=1, ensemble=False, debug=False), policy=Greedy(tune_budget=0))
    climber.search(broken, agent=counter, max_evaluations=1, learning=False, config=folder, **QUIET)
    candidate = climber.candidates[-1]
    assert candidate.status == "buggy" and "the scorer itself is wrong" in candidate.last_replicate.stdout_tail


def test_a_budget_sets_every_limit_and_the_spec_records_them(scoring, counter, folder):
    from hillclimb import Budget

    budget = Budget(wall_clock="5m", evaluations=2, tokens=1_000_000, cost_usd=1.5)
    assert repr(budget) == "Budget(wall_clock='5m', evaluations=2, tokens=1000000, cost_usd=1.5)"
    assert Budget.of("10m") == Budget(wall_clock="10m") and Budget.of(90).seconds == 90 and Budget.of(None) == Budget()
    for bad in ({"evaluations": 0}, {"tokens": 2.5}, {"cost_usd": -1}, {"wall_clock": "soon"}):
        with pytest.raises(ValueError):
            Budget(**bad)
    with pytest.raises(TypeError):
        Budget.of(["10m"])

    climber = Climber(select=Best(num_drafts=2, ensemble=False), policy=Greedy(tune_budget=0))
    climber.search(largest_number(scoring), budget=budget, agent=counter, learning=False, config=folder, **QUIET)
    assert climber.spend.evaluations == 2 and climber.result.meta.budget_s == 300
    entry = yaml.safe_load((climber.result.run_dir / "spec.yaml").read_text())["problems"][0]
    assert entry["budget"] == "300s"
    assert entry["set"] == [
        "learning.enabled=false", "budget.max_evaluations=2", "budget.max_tokens=1000000", "budget.max_cost_usd=1.5",
    ]
    # the shorthand lays over the Budget
    climber.search(largest_number(scoring), budget=Budget(wall_clock="5m"), max_evaluations=1, agent=counter, learning=False, config=folder, **QUIET)
    assert climber.spend.evaluations == 1
