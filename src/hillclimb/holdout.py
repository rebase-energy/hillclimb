from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ANSWERS_FILENAME = "holdout-answers.csv"
DATA_VIEW_DIRNAME = "data-view"
META_FILENAME = "data-view-meta.json"


class HoldoutOverride(BaseModel):
    """Task-YAML `holdout:` block. Overrides column inference and split
    strategy; anything left None falls back to inference / config."""

    strategy: Literal["random", "time-tail"] = "random"
    time_col: str | None = None          # required for time-tail
    id_col: str | None = None            # with target_cols, bypasses infer_targets
    target_cols: list[str] | None = None
    group_col: str | None = None         # per-group tail (e.g. one series per site)
    drop_cols: list[str] = Field(default_factory=list)  # hidden from holdout.csv, NOT scored
    fraction: float | None = None        # overrides config.holdout.fraction


@dataclass
class HoldoutInfo:
    data_view: Path      # what agents see instead of the raw public dir
    answers_path: Path   # hidden labels, outside the data view
    id_col: str
    target_cols: list[str]
    n_holdout: int
    strategy: str = "random"
    group_col: str | None = None
    time_cutoff: str | None = None  # earliest held-out timestamp (global tail only)


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


def _time_tail_mask(
    train: pd.DataFrame, time_col: str, fraction: float, group_col: str | None = None
) -> np.ndarray:
    """Mask the chronological tail: the last `fraction` of rows by time_col
    order (per group when group_col is given). A contiguous block, not random
    rows — random rows in a time series are trivially interpolable from their
    neighbors, which makes the holdout score meaningless."""
    if time_col not in train.columns:
        raise ValueError(f"time-tail holdout: no column {time_col!r} in train.csv")
    # Parse into a local only — the frame's original timestamp strings must
    # survive byte-identical, because scoring aligns holdout ids as strings.
    parsed = pd.to_datetime(train[time_col], errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"time-tail holdout: unparseable timestamps in {time_col!r}")

    mask = np.zeros(len(train), dtype=bool)
    groups = train.groupby(group_col, observed=True) if group_col else [(None, train)]
    for _, group in groups:
        order = parsed.loc[group.index].sort_values().index
        take = max(1, int(round(len(group) * fraction)))
        mask[train.index.get_indexer(order[-take:])] = True
    return mask


def _resolve_columns(
    train: pd.DataFrame,
    sample: pd.DataFrame,
    override: HoldoutOverride | None,
) -> tuple[str, list[str]] | None:
    """Explicit problem-YAML columns beat inference — the seam for problems whose
    submission id doesn't exist in train (e.g. row_id keyed to timestamps)."""
    if override and override.id_col and override.target_cols:
        cols = {override.id_col, *override.target_cols, *override.drop_cols}
        missing = cols - set(train.columns)
        if missing:
            logger.warning("holdout override names columns not in train.csv: %s", sorted(missing))
            return None
        return override.id_col, override.target_cols
    return infer_targets(train, sample)


def build_data_view(
    public_dir: Path,
    search_dir: Path,
    sample_submission: Path,
    fraction: float,
    seed: int,
    override: HoldoutOverride | None = None,
) -> HoldoutInfo | None:
    """Create runs/<id>/data-view/ with a hidden holdout carved from train.csv.

    Returns None (holdout disabled, callers fall back to the raw public dir)
    when there is no train.csv or target inference fails. Idempotent: an
    existing complete data-view is reused so `resume` keeps candidate lineage.
    """
    data_view = search_dir / DATA_VIEW_DIRNAME
    answers_path = search_dir / ANSWERS_FILENAME
    train_path = public_dir / "train.csv"
    if not train_path.exists():
        logger.warning("holdout disabled: no train.csv in %s", public_dir)
        return None

    train = pd.read_csv(train_path)
    sample = pd.read_csv(sample_submission, nrows=50)
    resolved = _resolve_columns(train, sample, override)
    if resolved is None:
        logger.warning("holdout disabled: could not infer id/target columns for %s", public_dir)
        return None
    id_col, target_cols = resolved

    strategy = override.strategy if override else "random"
    group_col = override.group_col if override else None
    drop_cols = override.drop_cols if override else []
    if override and override.fraction:
        fraction = override.fraction
    meta = {
        "strategy": strategy,
        "fraction": fraction,
        "seed": seed,
        "id_col": id_col,
        "target_cols": target_cols,
        "group_col": group_col,
        "drop_cols": drop_cols,
    }

    meta_path = search_dir / META_FILENAME
    if answers_path.exists() and (data_view / "holdout.csv").exists():
        # Never rebuild an existing view: earlier candidates were scored against the
        # old answers, so a rebuild mid-run would corrupt the lineage.
        if meta_path.exists():
            recorded = json.loads(meta_path.read_text())
            if {k: recorded.get(k) for k in meta} != meta:
                logger.warning(
                    "data-view was built with different holdout params; reusing existing split"
                )
        answers = pd.read_csv(answers_path)
        info = HoldoutInfo(data_view, answers_path, id_col, target_cols, len(answers))
        info.strategy = strategy
        info.group_col = group_col
        if (
            strategy == "time-tail"
            and not group_col
            and override
            and override.time_col == id_col
        ):
            ids = answers[id_col]
            info.time_cutoff = str(ids.loc[pd.to_datetime(ids).idxmin()])
        return info

    if strategy == "time-tail":
        if not (override and override.time_col):
            logger.warning("holdout disabled: time-tail strategy requires time_col")
            return None
        try:
            mask = _time_tail_mask(train, override.time_col, fraction, group_col)
        except ValueError as e:
            logger.warning("holdout disabled: %s", e)
            return None
        # Scoring aligns by id, so held-out ids must be unique and must not
        # also appear in the visible train (duplicates elsewhere — e.g. a
        # DST fall-back hour mid-series — are the agent's problem, not ours).
        held_ids = train.loc[mask, id_col]
        if held_ids.duplicated().any() or held_ids.isin(train.loc[~mask, id_col]).any():
            logger.warning(
                "holdout disabled: held-out %r values are not unique/disjoint", id_col
            )
            return None
    else:
        mask = _holdout_mask(train, target_cols, fraction, seed)

    data_view.mkdir(parents=True, exist_ok=True)
    for entry in public_dir.iterdir():
        if entry.name == "train.csv":
            continue
        link = data_view / entry.name
        if not link.exists():
            link.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())

    hidden = [c for c in {*target_cols, *drop_cols} if c in train.columns]
    train[~mask].to_csv(data_view / "train.csv", index=False)
    train[mask].drop(columns=hidden).to_csv(data_view / "holdout.csv", index=False)
    train[mask][[id_col, *target_cols]].to_csv(answers_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2))
    n_holdout = int(mask.sum())
    logger.info(
        "holdout: %d of %d train rows hidden (%s; targets: %s)",
        n_holdout, len(train), strategy, target_cols,
    )
    info = HoldoutInfo(data_view, answers_path, id_col, target_cols, n_holdout)
    info.strategy = strategy
    info.group_col = group_col
    if strategy == "time-tail" and not group_col:
        held = train.loc[mask, override.time_col]
        info.time_cutoff = str(held.loc[pd.to_datetime(held).idxmin()])
    return info
