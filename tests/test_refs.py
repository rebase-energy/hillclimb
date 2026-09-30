"""Module references (`modules/refs.py`): one resolver for every slot —
a registry name, a `.py` file, or `module:Class` — and the file scope that
lets a climber's local files share one package."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hillclimb.modules import refs
from hillclimb.modules.refs import ClimberLoadError, FileScope, resolve_ref

POLICY = (
    "class Mine:\n"
    "    name = 'mine'\n"
    "    def propose(self, view):\n        return None\n"
    "    def observe(self, view, candidate):\n        pass\n"
)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_the_three_forms(tmp_path):
    by_name = resolve_ref("random", "tuner")
    assert (by_name.form, by_name.label) == ("name", "random")
    assert by_name.target.__name__ == "RandomSearch"

    by_module = resolve_ref("hillclimb.modules.tuners.random_search:RandomSearch", "tuner")
    assert by_module.form == "module" and by_module.target is by_name.target
    assert by_module.label == "RandomSearch"

    file = write(tmp_path / "mine.py", POLICY)
    by_file = resolve_ref("mine.py", "policy", base_dir=tmp_path)
    assert (by_file.form, by_file.label, by_file.path) == ("file", "mine", file.resolve())
    assert by_file.target.__name__ == "Mine"
    assert resolve_ref(f"{file}:Mine", "policy").target is by_file.target  # same bytes, same class object


def test_a_file_names_its_pick_by_attribute_or_by_being_the_only_one(tmp_path):
    two = write(tmp_path / "two.py", POLICY + POLICY.replace("Mine", "Other"))
    with pytest.raises(ClimberLoadError, match="must define exactly one policy class .*Mine.*Other.* or set POLICY"):
        resolve_ref(str(two), "policy")
    assert resolve_ref(f"{two}:Other", "policy").target.__name__ == "Other"
    write(tmp_path / "picked.py", POLICY + POLICY.replace("Mine", "Other") + "POLICY = Other\n")
    assert resolve_ref("picked.py", "policy", base_dir=tmp_path).target.__name__ == "Other"
    write(tmp_path / "factory.py", POLICY + "def POLICY(params=None):\n    return Mine()\n")
    assert callable(resolve_ref("factory.py", "policy", base_dir=tmp_path).target)


def test_errors_name_the_file_and_the_fix(tmp_path):
    with pytest.raises(ClimberLoadError, match="policy file not found: .*nope.py"):
        resolve_ref("nope.py", "policy", base_dir=tmp_path)
    broken = write(tmp_path / "broken.py", "import definitely_not_a_module\n")
    with pytest.raises(ClimberLoadError, match="broken.py failed to import: ModuleNotFoundError"):
        resolve_ref(str(broken), "policy")
    ok = write(tmp_path / "ok.py", POLICY)
    with pytest.raises(ClimberLoadError, match="ok.py defines no Missing"):
        resolve_ref(f"{ok}:Missing", "policy")
    with pytest.raises(ClimberLoadError, match="cannot import tuner 'not.a.module:X'"):
        resolve_ref("not.a.module:X", "tuner")
    with pytest.raises(ClimberLoadError, match=r"unknown tuner 'grid' \(available: optuna, random, a path to a .py file"):
        resolve_ref("grid", "tuner")
    with pytest.raises(ClimberLoadError, match="is not a GraphModule subclass"):
        resolve_ref("hillclimb.modules.tuners.random_search:RandomSearch", "graph")
    with pytest.raises(ValueError, match="would read as a file or module:Class"):
        refs.register("tuner", "bad.py", object)


def test_a_climbers_files_share_one_package_and_import_each_other(tmp_path):
    write(tmp_path / "c" / "helpers.py", "NUM = 3\n")
    write(tmp_path / "c" / "policy.py", "from .helpers import NUM\n" + POLICY)
    write(
        tmp_path / "c" / "ops" / "cross.py",
        "from ..helpers import NUM\nfrom hillclimb.sdk import Operator, Attempt\n"
        "class Cross(Operator):\n    name, role = 'cross', 'combine'\n"
        "    def prepare(self, ctx):\n        return Attempt(prompt=str(NUM))\n",
    )
    scope = FileScope([tmp_path / "c" / "policy.py", tmp_path / "c" / "ops" / "cross.py"])
    assert scope.root == (tmp_path / "c").resolve()
    assert [scope.relative(f).as_posix() for f in scope.files] == ["helpers.py", "ops/cross.py", "policy.py"]
    policy = resolve_ref(str(tmp_path / "c" / "policy.py"), "policy", scope=scope)
    cross = resolve_ref(str(tmp_path / "c" / "ops" / "cross.py"), "operator", scope=scope)
    # one package: both files see the SAME helpers module
    helpers = sys.modules[f"{scope.package}.helpers"]
    assert sys.modules[policy.target.__module__].NUM is helpers.NUM
    assert cross.target.__module__ == f"{scope.package}.ops.cross"


def test_identity_follows_the_bytes_not_the_place(tmp_path):
    for root in ("a", "b"):
        write(tmp_path / root / "helpers.py", "NUM = 3\n")
        write(tmp_path / root / "policy.py", "from .helpers import NUM\n" + POLICY)
    a, b = FileScope([tmp_path / "a" / "policy.py"]), FileScope([tmp_path / "b" / "policy.py"])
    assert a.digest == b.digest  # a snapshot hashes like its source
    write(tmp_path / "b" / "helpers.py", "NUM = 4\n")  # a file only reached by import is part of it
    edited = FileScope([tmp_path / "b" / "policy.py"])
    assert edited.digest != a.digest and edited.package != a.package
    # two versions live side by side in one process
    old = resolve_ref(str(tmp_path / "a" / "policy.py"), "policy", scope=a).target
    new = resolve_ref(str(tmp_path / "b" / "policy.py"), "policy", scope=edited).target
    assert sys.modules[old.__module__].NUM == 3 and sys.modules[new.__module__].NUM == 4


def test_file_refs_split_and_anchor(tmp_path):
    assert refs.split_file_ref("dir/mine.py:Class") == ("dir/mine.py", "Class")
    assert refs.split_file_ref("C:/x/mine.py:Class") == ("C:/x/mine.py", "Class")  # a drive letter is not a class
    assert refs.split_file_ref("C:/x/mine.py") == ("C:/x/mine.py", "")
    assert refs.is_file_ref("a.py") and refs.is_file_ref("a.py:B") and not refs.is_file_ref("pkg.mod:Cls")
    assert refs.is_module_ref("pkg.mod:Cls") and not refs.is_module_ref("greedy")
    assert refs.anchor_ref("mine.py:Class", tmp_path) == f"{tmp_path / 'mine.py'}:Class"
    assert refs.anchor_ref("greedy", tmp_path) == "greedy"
    assert refs.anchor_ref("pkg.mod:Cls", tmp_path) == "pkg.mod:Cls"


def test_a_tuner_can_be_a_file(tmp_path):
    from hillclimb.modules.tuners import get_tuner

    write(
        tmp_path / "fixed.py",
        "class Fixed:\n"
        "    def __init__(self, params=None):\n        self.params = dict(params or {})\n"
        "    def ask(self, space, history, *, higher_is_better, seed):\n        return {'k': self.params['k']}\n",
    )
    tuner = get_tuner("fixed.py", {"k": 7}, base_dir=tmp_path)
    assert tuner.name == "fixed" and tuner.ask(None, [], higher_is_better=True, seed=0) == {"k": 7}
    write(tmp_path / "bad.py", "class NotATuner:\n    pass\nTUNER = NotATuner\n")
    with pytest.raises(ClimberLoadError, match="has no ask"):
        get_tuner("bad.py", base_dir=tmp_path)


def test_construct_passes_only_what_the_signature_takes():
    class Takes:
        def __init__(self, params=None):
            self.params = params

    class Bare:
        pass

    offered = {"params": {"a": 1}, "log": print}
    assert refs.construct(Takes, offered).params == {"a": 1}
    assert isinstance(refs.construct(Bare, offered), Bare)
    assert refs.construct(lambda **kw: kw, offered) == offered
