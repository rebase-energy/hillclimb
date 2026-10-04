"""Runtime-venv requirements, shipped as package data so they survive
installation into site-packages (no repo-root assumptions)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

# generic verifier for self-reported problems (see run_solution.py)
RUN_SOLUTION = Path(__file__).parent / "run_solution.py"
# the opt-in machine-learning stack (torch, xgboost, lightgbm, …): MLE-bench
# competitions run on it; the default csv runtime stays lean
ML_REQUIREMENTS = Path(__file__).parent / "requirements-ml.txt"

KINDS = ("csv", "emflow")


def requirements_resource(kind: str = "csv"):
    """Traversable for the kind's requirements file (use resources.as_file
    to get a real path for subprocesses). The evaluator kind runs in the csv
    runtime unless the problem ships its own requirements file."""
    if kind == "evaluator":
        kind = "csv"
    if kind not in KINDS:
        raise ValueError(f"Unknown runtime kind {kind!r}; expected one of {KINDS}")
    return resources.files(__package__) / f"requirements-{kind}.txt"


def ensure_interface_shim() -> Path:
    """A PYTHONPATH dir that makes `from hillclimb import spaces` work inside
    the runtime venvs, where hillclimb itself is not installed (they stay
    lean; the convention for hillclimb-owned runtime code is invoke-by-path).

    Contains a generated two-line `hillclimb/__init__.py` stub — NOT the real
    one, whose lazy attributes would dangle without the full package — plus a
    verbatim copy of spaces.py. Content-keyed like the venvs, so editing
    spaces.py (or upgrading hillclimb) re-materializes automatically; the
    build lands in a temp dir first and renames into place, so concurrent
    searches can only ever see a complete shim."""
    import hashlib
    import shutil
    import tempfile

    source = (Path(__file__).parent.parent / "spaces.py").read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:12]
    from hillclimb.project import machine_cache_dir

    shim = machine_cache_dir() / "interface-shim" / digest
    if (shim / "hillclimb" / "spaces.py").exists():
        return shim
    shim.parent.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(dir=shim.parent, prefix=".build-"))
    try:
        package = build / "hillclimb"
        package.mkdir()
        (package / "__init__.py").write_text(
            '"""Runtime-venv shim: only `hillclimb.spaces` lives here."""\n'
        )
        (package / "spaces.py").write_bytes(source)
        try:
            build.rename(shim)
        except OSError:
            pass  # a concurrent search won the race; its shim is identical
    finally:
        shutil.rmtree(build, ignore_errors=True)
    return shim


def _parse_requirements(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]


def runtime_packages(kind: str = "csv", requirements_file: Path | None = None) -> list[str]:
    """Package names for the prompt's 'Available packages' line: the kind's
    packaged requirement set, or the problem's own requirements file when it
    pins one (per-problem venv)."""
    if requirements_file is not None:
        return _parse_requirements(Path(requirements_file).read_text())
    pkgs = _parse_requirements(requirements_resource(kind).read_text())
    if kind == "emflow":
        pkgs.append("emflow")  # installed separately from the configured source
    return pkgs
