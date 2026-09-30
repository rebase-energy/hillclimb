"""Package layout rules (docs/package-layout-plan.md): who may import whom.

`harness/` and `modules/` never import the tui or the cli; `tui/` never
imports the cli; a climber module under `modules/` is held to the sdk-only
rule in test_sdk_imports; only climber modules import the sdk eagerly. Each
rule is vacuous while a package does not exist and bites the moment it does.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.test_sdk_imports import CLIMBER_MODULES, SRC, hillclimb_imports

HARNESS, MODULES, TUI = SRC / "harness", SRC / "modules", SRC / "tui"

# package -> import prefixes its files may never name
FORBIDDEN = {
    HARNESS: ("hillclimb.tui", "hillclimb.cli"),
    MODULES: ("hillclimb.tui", "hillclimb.cli"),
    TUI: ("hillclimb.cli",),
}

# files under modules/<kind>/ that are the contract or harness-side glue, not climber code
# (`compat.py`: what pre-0.6 snapshots still name — harness-side, never a template)
NOT_IMPLEMENTATIONS = {"__init__.py", "base.py", "check.py", "compute.py", "compat.py"}
EXCHANGE_KINDS = ("policies", "selectors", "operators", "tuners", "similarity")  # memory/ joins with its ABC

# the only files that may import hillclimb.sdk at module top level
SDK_EAGER_IMPORTERS = set(CLIMBER_MODULES) | {"climbers/gepa/evaluator.py"}


def _py_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py")) if root.is_dir() else []


def _names(prefixes: tuple[str, ...], name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _top_level_hillclimb_imports(path: Path) -> set[str]:
    """Only the module body — not function bodies, not `if TYPE_CHECKING:` blocks."""
    found: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "hillclimb":
                found.update(f"hillclimb.{alias.name}" for alias in node.names)
            else:
                found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.split(".")[0] == "hillclimb")
    return found


@pytest.mark.parametrize("package", [HARNESS, MODULES, TUI], ids=lambda p: p.name)
def test_import_direction(package: Path):
    offenders = [
        f"{path.relative_to(SRC)}: imports {name}"
        for path in _py_files(package)
        for name in sorted(hillclimb_imports(path))
        if _names(FORBIDDEN[package], name)
    ]
    assert not offenders, "\n".join(offenders)


def test_every_module_implementation_is_held_to_the_sdk_rule():
    """A file under modules/<kind>/ that is not the contract is climber code:
    it must be listed in test_sdk_imports.CLIMBER_MODULES, so the sdk-only
    rule cannot be dodged by adding a file."""
    missing = [
        str(path.relative_to(SRC))
        for kind in EXCHANGE_KINDS
        for path in _py_files(MODULES / kind)
        if path.name not in NOT_IMPLEMENTATIONS and str(path.relative_to(SRC)) not in CLIMBER_MODULES
    ]
    assert not missing, "add to tests/test_sdk_imports.CLIMBER_MODULES:\n" + "\n".join(missing)


def test_only_module_implementations_import_the_sdk_eagerly():
    """The sdk re-exports lazily from harness/ and modules/*/base.py; a
    harness file importing the sdk at the top would be a cycle waiting to
    happen. Climber code (and the gepa loop's own modules) may."""
    offenders = [
        str(path.relative_to(SRC))
        for path in sorted(SRC.rglob("*.py"))
        if "demo" not in path.parts and "sdk" not in path.parts
        and str(path.relative_to(SRC)) not in SDK_EAGER_IMPORTERS
        and "hillclimb.sdk" in _top_level_hillclimb_imports(path)
    ]
    assert not offenders, "\n".join(offenders)


def test_package_inits_under_harness_and_modules_import_nothing():
    """harness/__init__ importing core would make every `hillclimb.harness.x`
    import load the whole engine — and close the harness <-> modules cycle."""
    inits = [HARNESS / "__init__.py", MODULES / "__init__.py", MODULES / "memory" / "__init__.py"]
    offenders = [str(p.relative_to(SRC)) for p in inits if p.is_file() and hillclimb_imports(p)]
    assert not offenders, offenders


# the flat top level is the public surface and nothing else
FLAT = {"__init__", "_moved", "api", "benchmark_providers", "climber", "config",
        "connect", "experiment", "meta", "problem", "project", "spaces"}


def test_top_level_is_only_the_public_surface():
    stray = sorted(p.stem for p in SRC.glob("*.py") if p.stem not in FLAT)
    assert not stray, f"move into harness/, modules/ or tui/: {stray}"


# --- the cli package ---

CLI = SRC / "cli"
CLI_BY_NAME_FROM_COMMON = {"say", "warn", "fail", "_m", "legend", "next_steps"}  # pure output; nothing patches them


def test_cli_modules_use_absolute_imports_and_reach_common_through_the_module():
    """`from hillclimb.cli.common import load_config` binds at import time and
    a test's patch on hillclimb.cli.common never reaches it; commands call
    `common.load_config()`. Relative imports would blur hillclimb.cli.experiment
    against hillclimb.experiment."""
    offenders = []
    for path in _py_files(CLI):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                offenders.append(f"{path.name}: relative import")
            if node.module == "hillclimb.cli.common" and path.name != "__init__.py":
                bad = [a.name for a in node.names if a.name not in CLI_BY_NAME_FROM_COMMON]
                if bad:
                    offenders.append(f"{path.name}: imports {bad} by name — call them as common.<name>()")
    assert not offenders, "\n".join(offenders)


def test_cli_leaves_import_nothing_from_the_package():
    for name in ("_app.py", "common.py"):
        inside = [m for m in hillclimb_imports(CLI / name) if m.startswith("hillclimb.cli")]
        assert not inside, f"{name} imports {inside}"


def test_nothing_outside_the_cli_package_imports_it():
    offenders = [
        str(path.relative_to(SRC))
        for path in sorted(SRC.rglob("*.py"))
        if "cli" not in path.parts and "demo" not in path.parts
        and any(m == "hillclimb.cli" or m.startswith("hillclimb.cli.") for m in hillclimb_imports(path))
    ]
    assert not offenders, offenders
