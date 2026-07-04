from pathlib import Path

import pandas as pd
import pytest

from hillclimb.holdout import HoldoutOverride, build_data_view, infer_targets


@pytest.fixture
def public_dir(tmp_path: Path) -> Path:
    d = tmp_path / "public"
    d.mkdir()
    n = 100
    pd.DataFrame({
        "id": range(n),
        "feature": [i * 0.1 for i in range(n)],
        "target": [i % 2 == 0 for i in range(n)],
    }).to_csv(d / "train.csv", index=False)
    pd.DataFrame({"id": range(n, n + 20), "feature": [0.5] * 20}).to_csv(d / "test.csv", index=False)
    pd.DataFrame({"id": range(n, n + 20), "target": [False] * 20}).to_csv(
        d / "sample_submission.csv", index=False
    )
    (d / "description.md").write_text("desc")
    (d / "images").mkdir()
    return d


def build(public_dir, run_dir, fraction=0.1, seed=42):
    return build_data_view(
        public_dir, run_dir, public_dir / "sample_submission.csv", fraction, seed
    )


def test_split_disjoint_and_complete(public_dir, tmp_path):
    info = build(public_dir, tmp_path / "run")
    visible = pd.read_csv(info.data_view / "train.csv")
    holdout = pd.read_csv(info.data_view / "holdout.csv")
    answers = pd.read_csv(info.answers_path)

    assert len(visible) + len(holdout) == 100
    assert len(holdout) == 10  # fraction respected
    assert set(visible["id"]).isdisjoint(set(holdout["id"]))
    assert "target" not in holdout.columns          # labels dropped from view
    assert list(answers.columns) == ["id", "target"]
    assert set(answers["id"]) == set(holdout["id"])
    assert list(visible.columns) == ["id", "feature", "target"]  # schema preserved


def test_answers_outside_data_view(public_dir, tmp_path):
    info = build(public_dir, tmp_path / "run")
    assert info.answers_path.parent == tmp_path / "run"
    assert not (info.data_view / info.answers_path.name).exists()


def test_other_files_symlinked(public_dir, tmp_path):
    info = build(public_dir, tmp_path / "run")
    assert (info.data_view / "test.csv").is_symlink()
    assert (info.data_view / "description.md").is_symlink()
    assert (info.data_view / "images").is_dir()
    assert not (info.data_view / "train.csv").is_symlink()  # replaced, not linked


def test_deterministic_and_idempotent(public_dir, tmp_path):
    info1 = build(public_dir, tmp_path / "run1")
    ids1 = set(pd.read_csv(info1.answers_path)["id"])
    info2 = build(public_dir, tmp_path / "run2")
    assert ids1 == set(pd.read_csv(info2.answers_path)["id"])  # same seed
    # reuse: rebuilding the same run dir must not re-split
    before = (info1.data_view / "train.csv").stat().st_mtime_ns
    info1b = build(public_dir, tmp_path / "run1")
    assert (info1b.data_view / "train.csv").stat().st_mtime_ns == before
    info3 = build(public_dir, tmp_path / "run3", seed=7)
    assert ids1 != set(pd.read_csv(info3.answers_path)["id"])  # different seed


def test_class_columns_case(tmp_path):
    d = tmp_path / "public"
    d.mkdir()
    pd.DataFrame({
        "id": range(30),
        "text": ["x"] * 30,
        "author": (["EAP", "HPL", "MWS"] * 10),
    }).to_csv(d / "train.csv", index=False)
    pd.DataFrame({"id": [99], "EAP": [0.3], "HPL": [0.3], "MWS": [0.4]}).to_csv(
        d / "sample_submission.csv", index=False
    )
    info = build_data_view(d, tmp_path / "run", d / "sample_submission.csv", 0.2, 42)
    assert info.target_cols == ["author"]
    holdout = pd.read_csv(info.data_view / "holdout.csv")
    assert "author" not in holdout.columns
    assert "text" in holdout.columns


def test_auto_disable_no_train_csv(tmp_path):
    d = tmp_path / "public"
    d.mkdir()
    (d / "train.json").write_text("[]")
    pd.DataFrame({"id": [1], "t": [0]}).to_csv(d / "sampleSubmission.csv", index=False)
    assert build_data_view(d, tmp_path / "run", d / "sampleSubmission.csv", 0.1, 42) is None


def test_auto_disable_unmatchable_targets(tmp_path):
    d = tmp_path / "public"
    d.mkdir()
    pd.DataFrame({"weird_key": [1], "label": ["a"]}).to_csv(d / "train.csv", index=False)
    pd.DataFrame({"id": [1], "prediction": [0]}).to_csv(d / "sample_submission.csv", index=False)
    assert build_data_view(d, tmp_path / "run", d / "sample_submission.csv", 0.1, 42) is None


def test_infer_targets_direct_multi():
    train = pd.DataFrame({"id": [1], "f": [0.1], "formation": [0.2], "bandgap": [1.1]})
    sample = pd.DataFrame({"id": [9], "formation": [0.0], "bandgap": [0.0]})
    assert infer_targets(train, sample) == ("id", ["formation", "bandgap"])


@pytest.fixture
def ts_public_dir(tmp_path: Path) -> Path:
    """Dutch-shaped fixture: submission id (row_id) does NOT exist in train,
    train is a single time series with shuffled row order."""
    d = tmp_path / "ts-public"
    d.mkdir()
    ts = pd.date_range("2025-05-01", periods=200, freq="h").strftime("%Y-%m-%d %H:%M:%S")
    train = pd.DataFrame({"timestamp_utc": ts, "net_load_kwh": [(-1) ** i * i * 0.5 for i in range(200)]})
    train.sample(frac=1, random_state=0).to_csv(d / "train.csv", index=False)
    pd.DataFrame({"row_id": range(48), "timestamp_utc": ts[:48]}).to_csv(d / "test.csv", index=False)
    pd.DataFrame({"row_id": range(48), "net_load_kwh": [0.0] * 48}).to_csv(
        d / "sample_submission.csv", index=False
    )
    (d / "description.md").write_text("desc")
    return d


DUTCH_OVERRIDE = HoldoutOverride(
    strategy="time-tail",
    time_col="timestamp_utc",
    id_col="timestamp_utc",
    target_cols=["net_load_kwh"],
    fraction=0.1,
)


def build_ts(public_dir, run_dir, override=DUTCH_OVERRIDE):
    return build_data_view(
        public_dir, run_dir, public_dir / "sample_submission.csv", 0.3, 42, override=override
    )


def test_time_tail_is_contiguous_tail(ts_public_dir, tmp_path):
    info = build_ts(ts_public_dir, tmp_path / "run")
    assert info.n_holdout == 20  # override fraction 0.1 beats the 0.3 passed in
    visible = pd.read_csv(info.data_view / "train.csv")
    holdout = pd.read_csv(info.data_view / "holdout.csv")
    answers = pd.read_csv(info.answers_path)
    assert pd.to_datetime(visible["timestamp_utc"]).max() < pd.to_datetime(holdout["timestamp_utc"]).min()
    assert list(answers.columns) == ["timestamp_utc", "net_load_kwh"]
    # Dutch case: holdout.csv is the bare timestamp column — intentional,
    # there are no other feature columns to show.
    assert list(holdout.columns) == ["timestamp_utc"]
    assert info.strategy == "time-tail"
    assert info.time_cutoff == holdout["timestamp_utc"].min()


def test_time_tail_ignores_row_order(ts_public_dir, tmp_path):
    info = build_ts(ts_public_dir, tmp_path / "run")
    answers = pd.read_csv(info.answers_path)
    # the tail by time, not by row position (fixture rows are shuffled)
    expected = pd.date_range("2025-05-01", periods=200, freq="h")[-20:].strftime("%Y-%m-%d %H:%M:%S")
    assert set(answers["timestamp_utc"]) == set(expected)


def test_time_tail_grouped(tmp_path):
    d = tmp_path / "public"
    d.mkdir()
    t1 = pd.date_range("2018-01-01", periods=100, freq="h").strftime("%Y-%m-%d %H:%M:%S")
    t2 = pd.date_range("2018-06-01", periods=100, freq="h").strftime("%Y-%m-%d %H:%M:%S")
    pd.DataFrame({
        "time": list(t1) + list(t2),
        "load": range(200),
        "residual_load": range(200),
        "dataset_id": [1] * 100 + [2] * 100,
    }).to_csv(d / "train.csv", index=False)
    pd.DataFrame({"time": ["2018-12-01 00:00:00"], "residual_load": [0.0]}).to_csv(
        d / "sample_submission.csv", index=False
    )
    override = HoldoutOverride(
        strategy="time-tail", time_col="time", id_col="time",
        target_cols=["residual_load"], group_col="dataset_id",
        drop_cols=["load"], fraction=0.1,
    )
    info = build_data_view(d, tmp_path / "run", d / "sample_submission.csv", 0.1, 42, override=override)
    answers = pd.read_csv(info.answers_path)
    holdout = pd.read_csv(info.data_view / "holdout.csv")
    assert len(answers) == 20  # 10 per group
    # each group loses its own tail
    held = set(answers["time"])
    assert set(t1[-10:]) <= held and set(t2[-10:]) <= held
    # drop_cols hidden from the agent-facing holdout but not scored
    assert "load" not in holdout.columns
    assert list(answers.columns) == ["time", "residual_load"]
    # grouped tails have no single cutoff
    assert info.time_cutoff is None
    assert info.group_col == "dataset_id"


def test_override_rescues_failed_inference(ts_public_dir, tmp_path):
    # without override: row_id not in train → disabled (regression pin)
    assert build_ts(ts_public_dir, tmp_path / "run-a", override=None) is None
    assert build_ts(ts_public_dir, tmp_path / "run-b") is not None


def test_override_with_missing_columns_disables(ts_public_dir, tmp_path):
    bad = DUTCH_OVERRIDE.model_copy(update={"target_cols": ["nope"]})
    assert build_ts(ts_public_dir, tmp_path / "run", override=bad) is None


def test_time_tail_duplicate_in_tail_disables(ts_public_dir, tmp_path):
    train = pd.read_csv(ts_public_dir / "train.csv")
    latest = train.loc[[pd.to_datetime(train["timestamp_utc"]).idxmax()]]
    pd.concat([train, latest]).to_csv(ts_public_dir / "train.csv", index=False)
    assert build_ts(ts_public_dir, tmp_path / "run") is None


def test_time_tail_duplicate_mid_series_is_fine(ts_public_dir, tmp_path):
    """A DST-style duplicated hour far from the tail must not disable the
    holdout — only held-out ids need to be unique and disjoint."""
    train = pd.read_csv(ts_public_dir / "train.csv")
    earliest = train.loc[[pd.to_datetime(train["timestamp_utc"]).idxmin()]]
    pd.concat([train, earliest]).to_csv(ts_public_dir / "train.csv", index=False)
    info = build_ts(ts_public_dir, tmp_path / "run")
    assert info is not None
    answers = pd.read_csv(info.answers_path)
    assert not answers["timestamp_utc"].duplicated().any()


def test_time_tail_idempotent_and_param_change_keeps_split(ts_public_dir, tmp_path):
    info1 = build_ts(ts_public_dir, tmp_path / "run")
    before = (info1.data_view / "train.csv").stat().st_mtime_ns
    info2 = build_ts(ts_public_dir, tmp_path / "run")
    assert (info2.data_view / "train.csv").stat().st_mtime_ns == before
    # different fraction on an existing run dir: old split reused, not rebuilt
    changed = DUTCH_OVERRIDE.model_copy(update={"fraction": 0.2})
    info3 = build_ts(ts_public_dir, tmp_path / "run", override=changed)
    assert info3.n_holdout == info1.n_holdout


def test_stratified_split_covers_all_classes(tmp_path):
    d = tmp_path / "public"
    d.mkdir()
    rows = ["id,feature,species"]
    for cls in range(20):
        for i in range(10):
            rows.append(f"{cls * 10 + i},{i * 0.1},class_{cls}")
    (d / "train.csv").write_text("\n".join(rows) + "\n")
    cols = ",".join(f"class_{c}" for c in range(20))
    (d / "sample_submission.csv").write_text(f"id,{cols}\n999,{','.join(['0.05'] * 20)}\n")
    info = build_data_view(d, tmp_path / "run", d / "sample_submission.csv", 0.1, 42)
    answers = pd.read_csv(info.answers_path)
    assert answers["species"].nunique() == 20  # every class represented
    assert len(answers) == 20  # 10% of each 10-row class = 1 each
