"""Frozen problem unit tests and their framework-neutral command runner."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from hillclimb.harness.candidate import UnitTestResult
from hillclimb.harness.executor import (
    PARAMS_FILE,
    confined,
    prepend_pythonpath,
    run_logged,
    scrubbed_env,
    verifier_env,
)
from hillclimb.harness.oscompat import env_path, link_dir, lock_file
from hillclimb.problem import ProblemSpec, UnitTestSpec

BUNDLES_DIR = "test-bundles"
MANIFEST_FILE = "manifest.json"
TESTS_DIR = "tests"
VISIBLE_TESTS_DIR = "unit_tests"
_IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_TAIL_CHARS = 4000


class UnitTestInfrastructureError(RuntimeError):
    """The frozen suite or its runner is unavailable; not a candidate bug."""


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _IGNORED_DIRS or name == ".DS_Store"}


def _validate_links(root: Path) -> None:
    resolved_root = root.resolve()
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        try:
            path.resolve(strict=True).relative_to(resolved_root)
        except (OSError, ValueError) as exc:
            raise ValueError(f"unit-test symlink escapes its root: {path}") from exc


def _digest(root: Path, command: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(command, separators=(",", ":")).encode())
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        if any(part in _IGNORED_DIRS for part in path.relative_to(root).parts):
            continue
        relative = path.relative_to(root).as_posix()
        if relative.endswith("/.DS_Store") or relative == ".DS_Store":
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_manifest(bundle: Path) -> dict:
    try:
        payload = json.loads((bundle / MANIFEST_FILE).read_text())
    except (OSError, ValueError) as exc:
        raise UnitTestInfrastructureError(f"invalid frozen unit-test bundle: {bundle}") from exc
    if not isinstance(payload, dict):
        raise UnitTestInfrastructureError(f"invalid frozen unit-test manifest: {bundle}")
    return payload


def freeze_for_run(problem: ProblemSpec, run_dir: Path) -> UnitTestSpec | None:
    """Return the run-frozen suite, creating it once per problem under a lock."""
    source = problem.unit_tests
    if source is None:
        return None
    if source.sha256 is not None:
        verify_frozen(source)
        return source

    root = run_dir / BUNDLES_DIR
    root.mkdir(parents=True, exist_ok=True)
    bundle = root / problem.problem_id
    lock_path = root / f".{problem.problem_id}.lock"
    with lock_path.open("a+") as lock:
        lock_file(lock)
        if not (bundle / MANIFEST_FILE).exists():
            _validate_links(source.root)
            temporary = Path(tempfile.mkdtemp(prefix=f".{problem.problem_id}-", dir=root))
            try:
                copied = temporary / TESTS_DIR
                shutil.copytree(source.root, copied, symlinks=False, ignore=_ignore)
                sha256 = _digest(copied, source.command)
                (temporary / MANIFEST_FILE).write_text(
                    json.dumps(
                        {"command": source.command, "sha256": sha256},
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
                os.replace(temporary, bundle)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        manifest = _read_manifest(bundle)
    try:
        frozen = UnitTestSpec(
            root=bundle / TESTS_DIR,
            command=list(manifest.get("command") or []),
            sha256=str(manifest.get("sha256") or ""),
        )
    except (TypeError, ValueError) as exc:
        raise UnitTestInfrastructureError(
            f"invalid frozen unit-test manifest: {bundle}"
        ) from exc
    verify_frozen(frozen)
    return frozen


def restore_frozen(
    problem: ProblemSpec,
    run_dir: Path,
    *,
    bundle_path: str | None,
    command: list[str],
    sha256: str | None,
) -> ProblemSpec:
    """Bind a reloaded live ProblemSpec to the immutable suite in search metadata."""
    if bundle_path is None:
        if hasattr(problem, "unit_tests"):
            problem.unit_tests = None
        return problem
    bundle = (run_dir / bundle_path).resolve()
    try:
        bundle.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise UnitTestInfrastructureError("unit-test bundle path escapes the run dir") from exc
    root = bundle / TESTS_DIR
    try:
        frozen = UnitTestSpec(root=root, command=command, sha256=sha256)
    except ValueError as exc:
        raise UnitTestInfrastructureError(
            f"invalid frozen unit-test metadata: {bundle}"
        ) from exc
    verify_frozen(frozen)
    problem.unit_tests = frozen
    return problem


def bundle_relative(spec: UnitTestSpec, run_dir: Path) -> str:
    return str(spec.root.parent.relative_to(run_dir))


def verify_frozen(spec: UnitTestSpec) -> None:
    if spec.sha256 is None:
        raise UnitTestInfrastructureError("unit-test suite was not frozen for this run")
    if not spec.root.is_dir():
        raise UnitTestInfrastructureError(f"frozen unit-test directory is missing: {spec.root}")
    try:
        actual = _digest(spec.root, spec.command)
    except OSError as exc:
        raise UnitTestInfrastructureError(
            f"cannot read frozen unit-test bundle: {spec.root}"
        ) from exc
    if actual != spec.sha256:
        raise UnitTestInfrastructureError(
            f"frozen unit-test bundle changed ({spec.sha256[:12]} -> {actual[:12]})"
        )


def copy_visible_root(root: Path, candidate_dir: Path) -> None:
    """Give a coding agent a disposable copy; authoritative evaluation ignores it."""
    target = candidate_dir / VISIBLE_TESTS_DIR
    if not target.exists():
        shutil.copytree(root, target, symlinks=False, ignore=_ignore)


class UnitTestRunner:
    def __init__(
        self,
        python: Path,
        spec: UnitTestSpec,
        pythonpath: str | None = None,
        sandbox=None,  # harness.sandbox.SandboxPolicy: what a test run is confined to
    ):
        self.sandbox = sandbox
        self.python = python.absolute()
        self.spec = spec
        self.pythonpath = pythonpath

    def run(self, solution: Path, candidate_dir: Path, timeout_s: float) -> UnitTestResult:
        try:
            return self._run(solution, candidate_dir, timeout_s)
        except UnitTestInfrastructureError:
            raise
        except OSError as exc:
            raise UnitTestInfrastructureError(
                f"cannot prepare or read the unit-test run: {exc}"
            ) from exc

    def _run(self, solution: Path, candidate_dir: Path, timeout_s: float) -> UnitTestResult:
        verify_frozen(self.spec)
        work_dir = candidate_dir / ".hillclimb-test-run"
        if work_dir.exists():
            raise UnitTestInfrastructureError(
                f"unit-test work directory already exists: {work_dir}"
            )
        work_dir.mkdir()
        for link_name in ("data", "problem"):
            source = candidate_dir / link_name
            if source.exists():
                link_dir(work_dir / link_name, source.resolve())
        for source in [solution, *candidate_dir.glob("candidate_*.py")]:
            if source.exists():
                shutil.copy(source, work_dir / source.name)
        source_params = candidate_dir / PARAMS_FILE
        if source_params.exists():
            shutil.copy(source_params, work_dir / PARAMS_FILE)
        test_root = work_dir / VISIBLE_TESTS_DIR
        shutil.copytree(self.spec.root, test_root, symlinks=False, ignore=_ignore)
        stdout_path = candidate_dir / "tests_stdout.log"
        stderr_path = candidate_dir / "tests_stderr.log"
        result_path = candidate_dir / "unit_test_result.json"
        run_solution = work_dir / solution.name
        params_path = work_dir / PARAMS_FILE
        env = scrubbed_env()
        env.update(
            verifier_env(
                self.python,
                run_solution.absolute(),
                result_path,
                "tests",
                params=params_path if params_path.exists() else None,
            )
        )
        env["HILLCLIMB_TESTS"] = env_path(test_root.absolute())
        prepend_pythonpath(env, self.pythonpath)
        values = {
            "python": str(self.python),
            "solution": str(run_solution.absolute()),
            "tests": str(test_root.absolute()),
        }
        argv = [token.format(**values) for token in self.spec.command]
        started = time.monotonic()
        try:
            with stdout_path.open("w") as out, stderr_path.open("w") as err:
                run = run_logged(
                    argv, work_dir, max(0.001, timeout_s), out, err, env,
                    sandbox=confined(self.sandbox, work_dir),
                )
        except OSError as exc:
            raise UnitTestInfrastructureError(
                f"cannot start unit-test command {argv[0]!r}: {exc}"
            ) from exc
        duration = time.monotonic() - started
        return UnitTestResult(
            passed=run.returncode == 0 and not run.timed_out,
            returncode=run.returncode,
            duration_s=duration,
            cpu_s=run.cpu_s,
            timed_out=run.timed_out,
            stdout_tail=stdout_path.read_text(errors="replace")[-_TAIL_CHARS:],
            stderr_tail=stderr_path.read_text(errors="replace")[-_TAIL_CHARS:],
        )
