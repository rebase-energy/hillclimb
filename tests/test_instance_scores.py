"""The per-instance verifier contract: the reserved `instances` key flows
ExecResult -> Replicate -> Trial -> Candidate as `instance_scores`, aggregated
per key by median, and never leaks into Replicate.metrics."""

from __future__ import annotations

import json

from hillclimb.candidate import Candidate, Replicate
from hillclimb.dirs import create_candidate_dir, create_search_dir
from hillclimb.evaluation import CandidateEvaluator, eval_result_for
from hillclimb.executor import result_instances, result_metrics
from hillclimb.journal import Journal
from tests.conftest import executor_for
from tests.factories import trial as mk_trial


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


def test_replicate_round_trips_with_and_without_instances():
    with_field = Replicate(val_score=0.5, instance_scores={"z1": 0.2})
    assert Replicate.model_validate(with_field.model_dump()).instance_scores == {"z1": 0.2}
    # journals written before the field replay unchanged
    dumped = Replicate(val_score=0.5).model_dump()
    dumped.pop("instance_scores")
    assert Replicate.model_validate(dumped).instance_scores == {}


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
        trials=[mk_trial(replicates=[
            Replicate(val_score=1.0, submission_ok=True, instance_scores={"a": 1.0, "b": 5.0}),
            Replicate(val_score=1.1, submission_ok=True, instance_scores={"a": 3.0, "b": 4.0}),
            Replicate(val_score=1.2, submission_ok=True, instance_scores={"a": 2.0}),
            Replicate(val_score=None, instance_scores={"a": 99.0}),  # unscored replicate ignored
        ])],
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

    trial, ok = evaluator.run_trial(candidate, candidate_dir / "solution.py", candidate_dir, 30)

    assert ok
    assert trial.replicates[0].instance_scores == {"z1": 1.0, "z2": 0.5}
    assert trial.instance_scores == {"z1": 1.0, "z2": 0.5}
    assert candidate.instance_scores == {"z1": 1.0, "z2": 0.5}
    candidate.status = "ok"
    assert eval_result_for(candidate).instance_scores == {"z1": 1.0, "z2": 0.5}


# --- GEPA's view: a fixed vocabulary, failed instances, never new ones ---


def _bridge(*, higher_is_better=True):
    """The evaluator bridge's instance-key state alone: the two methods under
    test read only these attributes."""
    from types import SimpleNamespace

    from hillclimb.integrations.gepa.evaluator import GEPAEvaluatorBridge

    bridge = GEPAEvaluatorBridge.__new__(GEPAEvaluatorBridge)
    bridge.instance_keys = None
    bridge._min_valid_fitness = None
    bridge.problem = SimpleNamespace(higher_is_better=higher_is_better)
    bridge.params = SimpleNamespace(failure_fitness=-1e100)
    return bridge


def _result(instances, score=1.0, valid=True):
    from hillclimb.evaluation import EvalResult

    return EvalResult(candidate_id="c001", score=score, valid=valid, instance_scores=instances)


def test_gepa_tolerates_missing_instances_but_not_new_ones():
    import pytest

    from hillclimb.integrations.gepa.evaluator import InstanceKeyMismatch

    bridge = _bridge()
    bridge._check_instance_keys(_result({"t1/z1": 0.5, "t1/z2": 0.7}))
    assert bridge.instance_keys == ("t1/z1", "t1/z2")
    bridge._check_instance_keys(_result({"t1/z1": 0.6}))  # left t1/z2 unscored: fine
    bridge._check_instance_keys(_result({}))  # buggy candidate: fine
    with pytest.raises(InstanceKeyMismatch, match="t9/z9"):
        bridge._check_instance_keys(_result({"t1/z1": 0.6, "t9/z9": 0.1}))


def test_gepa_scores_a_missing_instance_as_a_failure():
    bridge = _bridge(higher_is_better=False)
    bridge._observe(_result({"t1/z1": 0.5}, score=0.5))
    partial = _result({"t1/z1": 0.4}, score=0.4)
    assert bridge.instance_fitness(partial, "t1/z1") == -0.4  # lower is better: negated once
    assert bridge.instance_fitness(partial, "t1/z2") == bridge._failure_fitness() < -0.5
    # a verifier that reports no instances at all keeps the aggregate fallback
    assert bridge.instance_fitness(_result({}, score=0.3), "t1/z2") == -0.3
