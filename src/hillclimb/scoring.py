from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    roc_auc_score,
)


class ScoringError(Exception):
    """Prediction file can't be scored (missing ids, wrong columns, bad values)."""


def _rmsle_column(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if (y_pred < 0).any() or (y_true < 0).any():
        raise ScoringError("RMSLE requires non-negative values")
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def _binary_series(s: pd.Series) -> np.ndarray:
    """Normalize a binary target column (bool, 0/1, or True/False strings)."""
    if pd.api.types.is_bool_dtype(s):
        return s.to_numpy().astype(int)
    if not pd.api.types.is_numeric_dtype(s):
        mapped = s.astype(str).str.lower().map({"true": 1, "false": 0})
        if not mapped.isna().any():
            return mapped.to_numpy().astype(int)
        return s.astype(str).to_numpy()  # non-boolean string labels
    return s.to_numpy()


def _accuracy(y_true: pd.DataFrame, pred: pd.DataFrame) -> float:
    t, p = y_true.iloc[:, 0], pred.iloc[:, 0]
    if pd.api.types.is_float_dtype(p) and not pd.api.types.is_float_dtype(t):
        # threshold probabilistic predictions for hard-label accuracy
        p = (p > 0.5).astype(int)
    return float(accuracy_score(_binary_series(t), _binary_series(p)))


def _auc_roc(y_true: pd.DataFrame, pred: pd.DataFrame) -> float:
    return float(roc_auc_score(_binary_series(y_true.iloc[:, 0]), pred.iloc[:, 0].astype(float)))


def _auroc_multi_label(y_true: pd.DataFrame, pred: pd.DataFrame) -> float:
    scores = [
        roc_auc_score(_binary_series(y_true[c]), pred[c].astype(float)) for c in y_true.columns
    ]
    return float(np.mean(scores))


def _multi_class_log_loss(y_true: pd.DataFrame, pred: pd.DataFrame) -> float:
    """y_true: single label column; pred: one probability column per class."""
    labels = y_true.iloc[:, 0].astype(str)
    classes = list(pred.columns)
    unknown = set(labels) - set(classes)
    if unknown:
        raise ScoringError(f"labels missing from prediction columns: {sorted(unknown)[:5]}")
    probs = pred[classes].astype(float).to_numpy()
    return float(log_loss(labels, probs, labels=classes))


def _rmse(y_true: pd.DataFrame, pred: pd.DataFrame) -> float:
    return float(
        np.sqrt(mean_squared_error(y_true.iloc[:, 0].astype(float), pred.iloc[:, 0].astype(float)))
    )


def _mean_column_wise_rmsle(y_true: pd.DataFrame, pred: pd.DataFrame) -> float:
    scores = [
        _rmsle_column(y_true[c].astype(float).to_numpy(), pred[c].astype(float).to_numpy())
        for c in y_true.columns
    ]
    return float(np.mean(scores))


def _micro_f1(y_true: pd.DataFrame, pred: pd.DataFrame) -> float:
    return float(
        f1_score(y_true.iloc[:, 0].astype(str), pred.iloc[:, 0].astype(str), average="micro")
    )


METRICS = {
    "accuracy": _accuracy,
    "auc-roc": _auc_roc,
    "auroc-multi-label": _auroc_multi_label,
    "multi-class-log-loss": _multi_class_log_loss,
    "rmse": _rmse,
    "root-mean-squared-error": _rmse,
    "mean-column-wise-rmsle": _mean_column_wise_rmsle,
    "micro-f1": _micro_f1,
}

LOWER_IS_BETTER = {
    "multi-class-log-loss",
    "rmse",
    "root-mean-squared-error",
    "mean-column-wise-rmsle",
}


def score(metric: str, answers: pd.DataFrame, predictions: pd.DataFrame, id_col: str) -> float:
    """Score predictions against answers, aligned by id column.

    `answers` holds the id column + target column(s) as they appear in
    train.csv; `predictions` is in submission format (id + prediction
    columns). For class-probability metrics the target/prediction column
    names differ by design; otherwise they must overlap.
    """
    if metric not in METRICS:
        raise ScoringError(f"Unknown metric: {metric!r} (known: {sorted(METRICS)})")
    if id_col not in predictions.columns:
        raise ScoringError(f"prediction file has no id column {id_col!r}")
    if predictions[id_col].duplicated().any():
        raise ScoringError("prediction file has duplicate ids")

    answers = answers.set_index(answers[id_col].astype(str)).drop(columns=[id_col])
    predictions = predictions.set_index(predictions[id_col].astype(str)).drop(columns=[id_col])
    missing = answers.index.difference(predictions.index)
    if len(missing):
        raise ScoringError(f"predictions missing {len(missing)} holdout ids (e.g. {list(missing[:3])})")
    predictions = predictions.reindex(answers.index)

    if metric != "multi-class-log-loss":
        # align prediction columns to targets by name where they correspond 1:1
        if set(answers.columns) <= set(predictions.columns):
            predictions = predictions[list(answers.columns)]
        elif len(answers.columns) == 1 and len(predictions.columns) == 1:
            pass  # single-target: accept whatever the prediction column is named
        else:
            raise ScoringError(
                f"prediction columns {list(predictions.columns)[:6]} don't cover targets {list(answers.columns)}"
            )
    try:
        return METRICS[metric](answers, predictions)
    except ScoringError:
        raise
    except Exception as e:  # sklearn/numpy errors → uniform contract failure
        raise ScoringError(f"scoring failed: {e}") from e
