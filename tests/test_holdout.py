from pathlib import Path

import pandas as pd
import pytest

from hillclimb.holdout import build_data_view, infer_targets


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
