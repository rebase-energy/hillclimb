from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANSWERS_FILENAME = "holdout-answers.csv"
DATA_VIEW_DIRNAME = "data-view"


@dataclass
class HoldoutInfo:
    data_view: Path      # what agents see instead of the raw public dir
    answers_path: Path   # hidden labels, outside the data view
    id_col: str
    target_cols: list[str]
    n_holdout: int


def infer_targets(train: pd.DataFrame, sample: pd.DataFrame) -> tuple[str, list[str]] | None:
    """Return (id_col, target_cols) or None if inference fails (→ disable)."""
    id_col = sample.columns[0]
    if id_col not in train.columns:
        return None
    # direct case: submission's prediction columns exist in train
    direct = [c for c in sample.columns[1:] if c in train.columns]
    if direct:
        return id_col, direct
    # class-columns case: one train column whose values are submission column names
    class_names = set(sample.columns[1:])
    for col in train.columns:
        if col == id_col or pd.api.types.is_numeric_dtype(train[col]):
            continue
        values = set(train[col].astype(str).unique())
        if values <= class_names:
            return id_col, [col]
    return None


def _holdout_mask(
    train: pd.DataFrame, target_cols: list[str], fraction: float, seed: int
) -> np.ndarray:
    """Row mask for the holdout. Stratified by target when it looks like a
    classification label, so every class is represented (a random split on
    e.g. 99 classes × 10 rows misses a third of the classes and makes the
    holdout metric mostly noise)."""
    rng = np.random.default_rng(seed)
    n = len(train)
    mask = np.zeros(n, dtype=bool)
    target = train[target_cols[0]]
    is_classification = len(target_cols) == 1 and (
        not pd.api.types.is_numeric_dtype(target) or target.nunique() <= max(20, int(0.05 * n))
    )
    if is_classification:
        for _, group in train.groupby(target_cols[0], observed=True):
            take = max(1, int(round(len(group) * fraction)))
            picked = rng.choice(group.index.to_numpy(), size=take, replace=False)
            mask[train.index.get_indexer(picked)] = True
    else:
        n_holdout = max(1, int(round(n * fraction)))
        mask[rng.choice(n, size=n_holdout, replace=False)] = True
    return mask


def build_data_view(
    public_dir: Path,
    run_dir: Path,
    sample_submission: Path,
    fraction: float,
    seed: int,
) -> HoldoutInfo | None:
    """Create runs/<id>/data-view/ with a hidden holdout carved from train.csv.

    Returns None (holdout disabled, callers fall back to the raw public dir)
    when there is no train.csv or target inference fails. Idempotent: an
    existing complete data-view is reused so `resume` keeps node lineage.
    """
    data_view = run_dir / DATA_VIEW_DIRNAME
    answers_path = run_dir / ANSWERS_FILENAME
    train_path = public_dir / "train.csv"
    if not train_path.exists():
        logger.warning("holdout disabled: no train.csv in %s", public_dir)
        return None

    train = pd.read_csv(train_path)
    sample = pd.read_csv(sample_submission, nrows=50)
    inferred = infer_targets(train, sample)
    if inferred is None:
        logger.warning("holdout disabled: could not infer id/target columns for %s", public_dir)
        return None
    id_col, target_cols = inferred

    if answers_path.exists() and (data_view / "holdout.csv").exists():
        answers = pd.read_csv(answers_path)
        return HoldoutInfo(data_view, answers_path, id_col, target_cols, len(answers))

    mask = _holdout_mask(train, target_cols, fraction, seed)

    data_view.mkdir(parents=True, exist_ok=True)
    for entry in public_dir.iterdir():
        if entry.name == "train.csv":
            continue
        link = data_view / entry.name
        if not link.exists():
            link.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())

    train[~mask].to_csv(data_view / "train.csv", index=False)
    train[mask].drop(columns=target_cols).to_csv(data_view / "holdout.csv", index=False)
    train[mask][[id_col, *target_cols]].to_csv(answers_path, index=False)
    n_holdout = int(mask.sum())
    logger.info(
        "holdout: %d of %d train rows hidden (targets: %s)", n_holdout, len(train), target_cols
    )
    return HoldoutInfo(data_view, answers_path, id_col, target_cols, n_holdout)
