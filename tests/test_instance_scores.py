"""The per-instance verifier contract: the reserved `instances` key flows
ExecResult -> Trial -> Candidate as `instance_scores`, aggregated per key by
median, and never leaks into Trial.metrics."""

from __future__ import annotations

import json

from hillclimb.candidate import Candidate, Trial
from hillclimb.dirs import create_candidate_dir, create_search_dir
from hillclimb.evaluation import CandidateEvaluator, eval_result_for
from hillclimb.executor import result_instances, result_metrics
from hillclimb.journal import Journal
from tests.conftest import executor_for


# --- result parsing ---


def test_result_instances_parses_the_reserved_key():
    payload = {"score": 1.5, "instances": {"z1": 1.0, "z2": 0.5}}
    assert result_instances(payload) == {"z1": 1.0, "z2": 0.5}


def test_result_instances_is_advisory():
    assert result_instances(None) == {}
    assert result_instances({"score": 1.0}) == {}
    assert result_instances({"instances": [1.0, 2.0]}) == {}  # wrong shape degrades
    nan = float("nan")
    assert result_instances({"instances": {"a": True, "b": nan, "c": "x", "d": 2}}) == {"d": 2.0}


def test_instances_never_leak_into_metrics():
    payload = {"score": 1.5, "runtime_s": 3.0, "instances": {"z1": 1.0}}
    assert result_metrics(payload) == {"runtime_s": 3.0}


# --- Trial / Candidate schema ---


def test_trial_round_trips_with_and_without_instances():
    with_field = Trial(val_score=0.5, instance_scores={"z1": 0.2})
    assert Trial.model_validate(with_field.model_dump()).instance_scores == {"z1": 0.2}
    # journals written before the field replay unchanged
    dumped = Trial(val_score=0.5).model_dump()
    dumped.pop("instance_scores")
    assert Trial.model_validate(dumped).instance_scores == {}


def test_old_journal_lines_replay(tmp_path):
    line = {
        "event": "candidate_result",
        "candidate_id": "c001",
        "operator": "draft",
        "status": "ok",
        "candidate_dir": "/tmp/x",
        "trials": [{"val_score": 0.5, "submission_ok": True}],
    }
    path = tmp_path / "journal.jsonl"
    path.write_text(json.dumps(line) + "\n")
    journal = Journal(path)
    assert journal.get("c001").instance_scores == {}


def test_candidate_instance_scores_is_per_key_median():
    candidate = Candidate(
        candidate_id="c1",
        operator="draft",
        status="ok",
        candidate_dir="/tmp/x",
        trials=[
            Trial(val_score=1.0, submission_ok=True, instance_scores={"a": 1.0, "b": 5.0}),
            Trial(val_score=1.1, submission_ok=True, instance_scores={"a": 3.0, "b": 4.0}),
            Trial(val_score=1.2, submission_ok=True, instance_scores={"a": 2.0}),
            Trial(val_score=None, instance_scores={"a": 99.0}),  # unscored trial ignored
        ],
    )
    assert candidate.instance_scores == {"a": 2.0, "b": 4.5}


# --- end-to-end through the evaluator ---

INSTANCES_SCRIPT = """\
import json, os, shutil
shutil.copy("data/sample_submission.csv", "submission.csv")
payload = {{"score": {score}, "split": "validation",
           "instances": {{"z1": {z1}, "z2": {z2}}}}}
with open(os.environ["HILLCLIMB_RESULT"], "w") as f:
    json.dump(payload, f)
print("val_score: {score}")
"""


def test_instances_flow_from_verifier_to_eval_result(tmp_path, task, config):
    """A self-reporting solution that writes the object form with `instances`;
    the field must arrive on the Trial, the Candidate median, and EvalResult."""
    search_dir = create_search_dir(tmp_path / "runs" / "r", "s")
    candidate_dir = create_candidate_dir(search_dir, "c001", task.data_dir, task.problem_dir)
    (candidate_dir / "solution.py").write_text(
        INSTANCES_SCRIPT.format(score=1.5, z1=1.0, z2=0.5)
    )
    candidate = Candidate(candidate_id="c001", operator="draft", candidate_dir=str(candidate_dir))
    evaluator = CandidateEvaluator(executor=executor_for(task), problem=task, config=config)

    ok = evaluator.run_trials(candidate, candidate_dir / "solution.py", candidate_dir, 30)

    assert ok
    assert candidate.trials[0].instance_scores == {"z1": 1.0, "z2": 0.5}
    assert candidate.instance_scores == {"z1": 1.0, "z2": 0.5}
    candidate.status = "ok"
    assert eval_result_for(candidate).instance_scores == {"z1": 1.0, "z2": 0.5}
