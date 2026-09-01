"""The spaces vocabulary: check/describe/sample and interface.py loading."""

import numpy as np
import pandas as pd
import pytest

from hillclimb import spaces


@pytest.fixture
def packing_table() -> spaces.Table:
    return spaces.Table(
        "submission.csv",
        columns={
            "id": spaces.Int(low=0, high=25, unique=True),
            "x": spaces.Float(low=0.0, high=1.0),
            "y": spaces.Float(low=0.0, high=1.0),
            "r": spaces.Float(low=0.0),
        },
        n_rows=26,
    )


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "id": range(26),
        "x": [0.5] * 26,
        "y": [0.5] * 26,
        "r": [0.01] * 26,
    })


def write(tmp_path, frame) -> None:
    frame.to_csv(tmp_path / "submission.csv", index=False)


class TestTable:
    def test_valid_artifact_passes(self, tmp_path, packing_table):
        write(tmp_path, valid_frame())
        assert packing_table.check(tmp_path) == []

    def test_missing_file(self, tmp_path, packing_table):
        (violation,) = packing_table.check(tmp_path)
        assert violation.path == "submission.csv"
        assert "not found" in violation.message

    def test_unparseable_file(self, tmp_path, packing_table):
        (tmp_path / "submission.csv").mkdir()  # unreadable as a CSV file
        violations = packing_table.check(tmp_path)
        assert violations and "CSV" in violations[0].expected

    def test_missing_and_extra_columns(self, tmp_path, packing_table):
        frame = valid_frame().drop(columns=["r"]).assign(bogus=1)
        write(tmp_path, frame)
        messages = {v.path: v.message for v in packing_table.check(tmp_path)}
        assert messages["submission.csv[r]"] == "missing column"
        assert "extra column" in messages["submission.csv[bogus]"]

    def test_extra_columns_ignore(self, tmp_path):
        table = spaces.Table("submission.csv", columns={"x": spaces.Float()},
                             extra_columns="ignore")
        write(tmp_path, pd.DataFrame({"x": [1.0], "bogus": [1]}))
        assert table.check(tmp_path) == []

    def test_row_count(self, tmp_path, packing_table):
        write(tmp_path, valid_frame().head(10))
        assert any("10 rows" in v.message for v in packing_table.check(tmp_path))

    def test_row_range(self, tmp_path):
        table = spaces.Table("submission.csv", columns={"x": spaces.Float()},
                             n_rows=(2, None))
        write(tmp_path, pd.DataFrame({"x": [1.0]}))
        assert any("at least 2 rows" in v.expected for v in table.check(tmp_path))

    def test_bounds_and_dtype(self, tmp_path, packing_table):
        frame = valid_frame()
        frame.loc[3, "x"] = 1.7
        frame.loc[5, "r"] = -0.2
        frame["id"] = frame["id"].astype(float)
        write(tmp_path, frame)
        by_path = {}
        for violation in packing_table.check(tmp_path):
            by_path.setdefault(violation.path, violation)
        assert "above high=1" in by_path["submission.csv[x]"].message
        assert "below low=0" in by_path["submission.csv[r]"].message
        # pandas re-infers int on read, so dtype survives only via NaN presence:
        # a missing id makes the column float -> integer-typed violation
        frame = valid_frame()
        frame.loc[7, "id"] = None
        write(tmp_path, frame)
        assert any(
            "not integer-typed" in v.message
            for v in packing_table.check(tmp_path) if v.path == "submission.csv[id]"
        )

    def test_unique_and_values(self, tmp_path):
        table = spaces.Table("t.csv", columns={
            "id": spaces.Int(values=range(5), unique=True),
        })
        pd.DataFrame({"id": [0, 1, 1, 9]}).to_csv(tmp_path / "t.csv", index=False)
        messages = [v.message for v in table.check(tmp_path)]
        assert any("outside the allowed set" in m for m in messages)
        assert any("duplicate" in m for m in messages)

    def test_nan_policy(self, tmp_path):
        strict = spaces.Table("t.csv", columns={"x": spaces.Float()})
        lax = spaces.Table("t.csv", columns={"x": spaces.Float(allow_nan=True)})
        pd.DataFrame({"x": [1.0, None]}).to_csv(tmp_path / "t.csv", index=False)
        assert any("NaN" in v.message for v in strict.check(tmp_path))
        assert lax.check(tmp_path) == []

    def test_check_value_on_frame(self, packing_table):
        assert packing_table.check_value(valid_frame()) == []
        assert packing_table.check_value("not a frame")[0].message == "value is not a DataFrame"

    def test_file_none_is_check_value_only(self, tmp_path):
        table = spaces.Table(None, columns={"x": spaces.Float()})
        (violation,) = table.check(tmp_path)
        assert "check_value" in violation.message
        assert table.check_value(pd.DataFrame({"x": [1.0]})) == []
        with pytest.raises(spaces.InterfaceError):
            table.sample(tmp_path)

    def test_sample_then_check_roundtrip(self, tmp_path, packing_table):
        packing_table.sample(tmp_path, seed=7)
        assert packing_table.check(tmp_path) == []

    def test_sample_is_deterministic(self, tmp_path, packing_table):
        packing_table.sample(tmp_path, seed=7)
        first = (tmp_path / "submission.csv").read_text()
        packing_table.sample(tmp_path, seed=7)
        assert (tmp_path / "submission.csv").read_text() == first


class TestArray:
    def test_check_and_sample(self, tmp_path):
        space = spaces.Array("weights.npy", dtype="float", shape=(None, 3),
                             low=0.0, high=1.0)
        space.sample(tmp_path, seed=3)
        assert space.check(tmp_path) == []

    def test_violations(self, tmp_path):
        space = spaces.Array("weights.npy", dtype="float", shape=(2, 3), high=1.0)
        assert "not found" in space.check(tmp_path)[0].message
        np.save(tmp_path / "weights.npy", np.ones((2, 2), dtype=int) * 5)
        messages = [v.message for v in space.check(tmp_path)]
        assert "wrong dtype" in messages
        assert "wrong shape" in messages
        assert any("above high=1" in m for m in messages)


class TestCallable:
    def test_valid_module(self, tmp_path):
        (tmp_path / "solution.py").write_text("def pack(items, capacity):\n    return []\n")
        space = spaces.Callable("pack", params=("items", "capacity"))
        assert space.check(tmp_path) == []

    def test_violations(self, tmp_path):
        space = spaces.Callable("pack", params=("items", "capacity"))
        assert "not found" in space.check(tmp_path)[0].message
        (tmp_path / "solution.py").write_text("pack = 3\n")
        assert "not callable" in space.check(tmp_path)[0].message
        (tmp_path / "solution.py").write_text("def pack(items):\n    return []\n")
        (violation,) = space.check(tmp_path)
        assert violation.message == "wrong parameters"
        assert violation.actual == "(items)"
        (tmp_path / "solution.py").write_text("raise RuntimeError('boom')\n")
        assert "failed: boom" in space.check(tmp_path)[0].message

    def test_sample_unsupported(self, tmp_path):
        with pytest.raises(NotImplementedError):
            spaces.Callable("pack").sample(tmp_path)


class TestDict:
    def test_aggregates_with_prefixes(self, tmp_path):
        space = spaces.Dict({
            "table": spaces.Table("t.csv", columns={"x": spaces.Float()}),
            "fn": spaces.Callable("pack"),
        })
        paths = [v.path for v in space.check(tmp_path)]
        assert paths == ["table.t.csv", "fn.solution.pack"]

    def test_sample_delegates(self, tmp_path):
        space = spaces.Dict({
            "table": spaces.Table("t.csv", columns={"x": spaces.Float()}),
        })
        space.sample(tmp_path, seed=1)
        assert (tmp_path / "t.csv").exists()


class TestInterfaceModule:
    def write_interface(self, tmp_path, body: str):
        path = tmp_path / "interface.py"
        path.write_text(body)
        return path

    def test_load_and_describe(self, tmp_path):
        path = self.write_interface(tmp_path, (
            "from hillclimb import spaces\n"
            "output = spaces.Table('submission.csv',"
            " columns={'x': spaces.Float(low=0.0, high=1.0)}, n_rows=3)\n"
            "inputs = {'train': spaces.Table('data/train.csv',"
            " columns={'x': spaces.Float()})}\n"
        ))
        module = spaces.load_interface(path)
        text = spaces.describe_interface(module)
        assert text == spaces.describe_interface(module)  # deterministic
        assert "### Output" in text and "`submission.csv`" in text
        assert "exactly 3 rows" in text and "in [0, 1]" in text
        assert "### Inputs" in text and "train" in text

    def test_load_failures(self, tmp_path):
        with pytest.raises(spaces.InterfaceError, match="not found"):
            spaces.load_interface(tmp_path / "interface.py")
        path = self.write_interface(tmp_path, "raise ValueError('bad')\n")
        with pytest.raises(spaces.InterfaceError, match="bad"):
            spaces.load_interface(path)
        path = self.write_interface(tmp_path, "x = 1\n")
        with pytest.raises(spaces.InterfaceError, match="neither"):
            spaces.load_interface(path)

    def test_main_exit_codes(self, tmp_path, capsys):
        table = spaces.Table("t.csv", columns={"x": spaces.Float()})
        assert spaces.main(table, [str(tmp_path)]) == 1
        assert "interface: t.csv" in capsys.readouterr().out
        pd.DataFrame({"x": [1.0]}).to_csv(tmp_path / "t.csv", index=False)
        assert spaces.main(table, [str(tmp_path)]) == 0
        assert "interface: OK" in capsys.readouterr().out
