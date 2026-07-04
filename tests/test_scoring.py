import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss, roc_auc_score

from hillclimb.scoring import ScoringError, score


def frame(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


def test_accuracy_bool_strings():
    answers = frame(id=[1, 2, 3, 4], target=[True, False, True, True])
    preds = frame(id=[1, 2, 3, 4], target=["True", "False", "False", "True"])
    assert score("accuracy", answers, preds, "id") == 0.75


def test_auc_roc_matches_sklearn():
    y = [0, 1, 1, 0, 1]
    p = [0.1, 0.8, 0.4, 0.3, 0.9]
    answers = frame(id=list(range(5)), target=y)
    preds = frame(id=list(range(5)), target=p)
    assert score("auc-roc", answers, preds, "id") == pytest.approx(roc_auc_score(y, p))


def test_multi_class_log_loss_label_vs_class_columns():
    answers = frame(id=[1, 2, 3], author=["EAP", "HPL", "MWS"])
    preds = frame(id=[1, 2, 3], EAP=[0.8, 0.1, 0.2], HPL=[0.1, 0.7, 0.2], MWS=[0.1, 0.2, 0.6])
    expected = log_loss(["EAP", "HPL", "MWS"],
                        np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6]]),
                        labels=["EAP", "HPL", "MWS"])
    assert score("multi-class-log-loss", answers, preds, "id") == pytest.approx(expected)


def test_multi_class_log_loss_unknown_label():
    answers = frame(id=[1], author=["UNKNOWN"])
    preds = frame(id=[1], EAP=[1.0])
    with pytest.raises(ScoringError, match="missing from prediction columns"):
        score("multi-class-log-loss", answers, preds, "id")


def test_mean_column_wise_rmsle_two_targets():
    answers = frame(id=[1, 2], a=[1.0, 2.0], b=[3.0, 4.0])
    preds = frame(id=[1, 2], a=[1.0, 2.0], b=[3.0, 4.0])
    assert score("mean-column-wise-rmsle", answers, preds, "id") == 0.0
    preds_neg = frame(id=[1, 2], a=[-1.0, 2.0], b=[3.0, 4.0])
    with pytest.raises(ScoringError, match="non-negative"):
        score("mean-column-wise-rmsle", answers, preds_neg, "id")


def test_rmse():
    answers = frame(id=[1, 2], fare=[10.0, 20.0])
    preds = frame(id=[1, 2], fare=[12.0, 18.0])
    assert score("rmse", answers, preds, "id") == pytest.approx(2.0)


def test_nrmse():
    answers = frame(id=[1, 2], net_load_kwh=[-10.0, 20.0])
    preds = frame(id=[1, 2], net_load_kwh=[-8.0, 22.0])
    # rmse 2.0 / mean(|y|) 15.0
    assert score("nrmse", answers, preds, "id") == pytest.approx(2.0 / 15.0)


def test_nrmse_zero_denominator():
    answers = frame(id=[1, 2], y=[0.0, 0.0])
    preds = frame(id=[1, 2], y=[1.0, -1.0])
    with pytest.raises(ScoringError, match="NRMSE undefined"):
        score("nrmse", answers, preds, "id")


def test_nrmse_is_lower_better():
    from hillclimb.scoring import LOWER_IS_BETTER

    assert "nrmse" in LOWER_IS_BETTER


def test_alignment_by_id_not_order():
    answers = frame(id=[1, 2], target=[0, 1])
    preds = frame(id=[2, 1], target=[1, 0])  # reversed order, correct by id
    assert score("accuracy", answers, preds, "id") == 1.0


def test_missing_ids_and_duplicates():
    answers = frame(id=[1, 2, 3], target=[0, 1, 0])
    with pytest.raises(ScoringError, match="missing"):
        score("accuracy", answers, frame(id=[1, 2], target=[0, 1]), "id")
    with pytest.raises(ScoringError, match="duplicate"):
        score("accuracy", answers, frame(id=[1, 1, 2, 3], target=[0, 0, 1, 0]), "id")
    with pytest.raises(ScoringError, match="no id column"):
        score("accuracy", answers, frame(other=[1], target=[0]), "id")


def test_unknown_metric():
    with pytest.raises(ScoringError, match="Unknown metric"):
        score("nope", frame(id=[1], t=[0]), frame(id=[1], t=[0]), "id")


def test_single_target_column_name_mismatch_ok():
    answers = frame(id=[1, 2], requester_received_pizza=[0, 1])
    preds = frame(id=[1, 2], prediction=[0.2, 0.9])
    assert score("auc-roc", answers, preds, "id") == 1.0
