from __future__ import annotations

from tests.factories import trial as mk_trial

import sys
from pathlib import Path

import pytest

from hillclimb.candidate import Candidate
from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.control import apply_prune, resync_best
from hillclimb.dirs import create_candidate_dir, create_search_dir
from hillclimb.evaluation import CandidateEvaluator
from hillclimb.executor import CommandExecutor
from hillclimb.journal import Journal
from hillclimb.integrations.gepa.searcher import GEPASearcher
from hillclimb.problem import ProblemSpec
from hillclimb.runtime import RUN_SOLUTION
from hillclimb.search import GreedySearcher
from tests.gepa_fakes import FakeGEPADriver


def json_problem(tmp_path, task):
    return task.model_copy(
        update={
            "output_artifacts": ["submission.json"],
            "verifier_cmd": [
                "{python}",
                str(RUN_SOLUTION),
                "{solution}",
                "--require",
                "submission.json",
            ],
        }
    )


def json_solution(score: float) -> str:
    return (
        'from pathlib import Path\n'
        'Path("submission.json").write_text("{}")\n'
        f'print("val_score: {score}")\n'
    )


def test_problem_rejects_unsafe_artifact_paths(task):
    with pytest.raises(ValueError, match="file name"):
        task.model_copy(update={"output_artifacts": ["../secret"]}).model_validate(
            {**task.model_dump(), "output_artifacts": ["../secret"]}
        )


def test_multi_trial_hoists_declared_json_artifact(tmp_path, task, config):
    problem = json_problem(tmp_path, task)
    config.search.n_replicates = 2
    config.search.replicate_mode = "serial"
    search_dir = create_search_dir(tmp_path / "runs" / "r", "s")
    candidate_dir = create_candidate_dir(
        search_dir, "c001", problem.data_dir, problem.problem_dir
    )
    solution = candidate_dir / "solution.py"
    solution.write_text(
        'from pathlib import Path\nPath("submission.json").write_text("{}")\nprint("val_score: 0.5")\n'
    )
    candidate = Candidate(candidate_id="c001", operator="draft", candidate_dir=str(candidate_dir))
    evaluator = CandidateEvaluator(
        executor=CommandExecutor(Path(sys.executable), problem.verifier_cmd),
        problem=problem,
        config=config,
    )

    assert evaluator.run_trial(candidate, solution, candidate_dir, 30)[1]
    assert (candidate_dir / "submission.json").read_text() == "{}"


def test_resync_best_uses_declared_json_artifact(tmp_path):
    search_dir = create_search_dir(tmp_path / "runs" / "r", "s")
    journal = Journal(search_dir / "journal.jsonl")

    def add(cid, score, operator="draft"):
        directory = search_dir / "candidates" / cid
        directory.mkdir()
        (directory / "solution.py").write_text(f"# {cid}\n")
        (directory / "submission.json").write_text(f'{{"candidate": "{cid}"}}')
        candidate = Candidate(
            candidate_id=cid,
            operator=operator,
            candidate_dir=str(directory),
            status="ok",
            trials=[] if score is None else [mk_trial(val_score=score)],
        )
        journal.candidate_result(candidate)
        return candidate

    add("c000", None, "baseline")
    add("c001", 0.5)
    add("c002", 0.9)
    assert resync_best(
        search_dir, journal, True, "rank-blend", ["submission.json"]
    ) == "c002"
    assert "c002" in (search_dir / "best" / "submission.json").read_text()

    apply_prune(journal, "c002")
    assert resync_best(
        search_dir, journal, True, "rank-blend", ["submission.json"]
    ) == "c001"
    assert "c001" in (search_dir / "best" / "submission.json").read_text()


def test_greedy_runner_preserves_json_artifact(tmp_path, task, config):
    problem = json_problem(tmp_path, task).model_copy(
        update={"baseline_text": json_solution(0.1)}
    )
    config.search.num_drafts = 1
    config.ensemble.enabled = False
    backend = FakeBackend()
    backend.queue(script=json_solution(0.8), notes="json draft\n")
    search_dir = create_search_dir(tmp_path / "runs" / "greedy", "s")
    searcher = GreedySearcher(
        problem=problem,
        config=config,
        journal=Journal(search_dir / "journal.jsonl"),
        backend=backend,
        executor=CommandExecutor(Path(sys.executable), problem.verifier_cmd),
        budget=BudgetManager(3600),
        search_dir=search_dir,
        max_candidates=2,
        log=lambda *_: None,
    )

    selected = searcher.run()

    assert selected is not None and selected.val_score == 0.8
    assert (search_dir / "best" / "submission.json").read_text() == "{}"


def test_gepa_runner_preserves_json_artifact(tmp_path, task, config):
    problem = json_problem(tmp_path, task).model_copy(
        update={"baseline_text": json_solution(0.1)}
    )
    config.search.policy = "gepa"
    backend = FakeBackend()
    backend.queue(script=json_solution(0.8), notes="json improvement\n")
    search_dir = create_search_dir(tmp_path / "runs" / "gepa", "s")
    searcher = GEPASearcher(
        problem=problem,
        config=config,
        journal=Journal(search_dir / "journal.jsonl"),
        backend=backend,
        executor=CommandExecutor(Path(sys.executable), problem.verifier_cmd),
        budget=BudgetManager(3600),
        search_dir=search_dir,
        driver=FakeGEPADriver(steps=1),
        log=lambda *_: None,
    )

    selected = searcher.run()

    assert selected is not None and selected.val_score == 0.8
    assert (search_dir / "best" / "submission.json").read_text() == "{}"
