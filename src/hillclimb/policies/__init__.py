"""Policy registry: name -> factory, mirroring backends.get_backend — plus
file policies, the seam an edited exploration process is loaded through.

`search.policy` names either a registry entry (`greedy`, `openevolve`) or
a Python file: any value ending in `.py`, relative to the folder holding
the hillclimb dir (the same anchor as `paths.runs_dir`), else the CWD. The
file exposes its policy as a module attribute `POLICY` (a class or a
factory called `POLICY(params, complexity_start=...)`) or, failing that,
as the one class it defines with `propose` and `observe`. Its identity is
the sha256 of its bytes (`policy_sha256`, recorded in `SearchMeta`) — the
name in the record stays the path as written, so `resume` reloads it from
the same place.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Callable

from hillclimb.policies.greedy import GreedyPolicy
from hillclimb.policy import SearchPolicy

POLICY_FILE_SUFFIX = ".py"
POLICY_ATTR = "POLICY"


def _make_greedy(params: dict, *, complexity_start: int = 0) -> GreedyPolicy:
    return GreedyPolicy(complexity_start=complexity_start, params=params)


def _make_openevolve(params: dict, *, complexity_start: int = 0) -> SearchPolicy:
    from hillclimb.policies.openevolve import OpenEvolvePolicy  # optional extra

    return OpenEvolvePolicy(params=params, complexity_start=complexity_start)


_POLICIES: dict[str, Callable[..., SearchPolicy]] = {
    "greedy": _make_greedy,
    "openevolve": _make_openevolve,
}


def is_policy_file(name: str) -> bool:
    return name.endswith(POLICY_FILE_SUFFIX)


def policy_path(name: str, base_dir: Path | None = None) -> Path | None:
    """The file a policy name points at (None for a registry name). A
    relative path is anchored at `base_dir` when given."""
    if not is_policy_file(name):
        return None
    path = Path(name).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def policy_sha256(name: str, base_dir: Path | None = None) -> str | None:
    """Identity of a file policy: sha256 of its bytes. None for a registry
    name; a missing file is an error here, not a None — the search record
    must never claim a policy it could not read."""
    path = policy_path(name, base_dir)
    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy_file(path: Path) -> Callable[..., SearchPolicy]:
    """Import the file and return a factory `(params, *, complexity_start)`
    for the policy it exposes. Import errors and a missing/ambiguous class
    surface as ValueError naming the file, never as a bare traceback deep
    in engine start-up."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"policy file not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    module_name = f"hillclimb_policy_{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import policy file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses/pickling look modules up by name
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        del sys.modules[module_name]
        raise ValueError(f"policy file {path} failed to import: {type(exc).__name__}: {exc}") from exc
    target = getattr(module, POLICY_ATTR, None)
    if target is None:
        classes = [
            obj for obj in vars(module).values()
            if inspect.isclass(obj)
            and obj.__module__ == module_name
            and callable(getattr(obj, "propose", None))
            and callable(getattr(obj, "observe", None))
        ]
        if len(classes) != 1:
            raise ValueError(
                f"policy file {path} must expose exactly one policy class (found "
                f"{[c.__name__ for c in classes]}) or set {POLICY_ATTR} = <class or factory>"
            )
        target = classes[0]
    if not callable(target):
        raise ValueError(f"{POLICY_ATTR} in {path} is not a class or factory: {target!r}")

    def factory(params: dict, *, complexity_start: int = 0) -> SearchPolicy:
        kwargs: dict = {}
        try:
            accepted = inspect.signature(target).parameters
        except (TypeError, ValueError):
            accepted = {}
        takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values())
        if "params" in accepted or takes_any:
            kwargs["params"] = params
        if "complexity_start" in accepted or takes_any:
            kwargs["complexity_start"] = complexity_start
        policy = target(**kwargs)
        for attr in ("propose", "observe"):
            if not callable(getattr(policy, attr, None)):
                raise ValueError(f"policy from {path} has no {attr}() — not a SearchPolicy")
        if not getattr(policy, "name", None):
            policy.name = path.stem
        if getattr(policy, "params", None) is None:
            policy.params = dict(params)
        return policy

    return factory


def get_policy(
    name: str,
    params: dict | None = None,
    *,
    complexity_start: int = 0,
    base_dir: Path | None = None,
) -> SearchPolicy:
    """A policy by registry name or file path (`base_dir` anchors a relative
    path — pass `policy_base_dir(config)`)."""
    path = policy_path(name, base_dir)
    if path is not None:
        return load_policy_file(path)(params or {}, complexity_start=complexity_start)
    if name not in _POLICIES:
        raise ValueError(
            f"Unknown policy: {name} (available: {', '.join(sorted(_POLICIES))}, or a path to a .py file)"
        )
    return _POLICIES[name](params or {}, complexity_start=complexity_start)


def policy_base_dir(config) -> Path | None:
    """Where a relative policy path resolves from: the folder holding the
    hillclimb dir (like `paths.runs_dir`), None when no dir is known."""
    hillclimb_dir = getattr(config, "hillclimb_dir", None)
    return hillclimb_dir.parent if hillclimb_dir is not None else None


def policy_label(name: str) -> str:
    """A short display name: the registry name, or a file's stem."""
    return Path(name).stem if is_policy_file(name) else name
