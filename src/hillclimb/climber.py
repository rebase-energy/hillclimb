"""Climbers: the shareable unit of "how to hillclimb".

A climber bundles the exchangeable modules of a search — a `SearchPolicy`
(or, for climbers that own their control flow, a `SearchLoop`), the operators
it may use with their prompts, a memory, a tuner, similarity scores — behind
one manifest. Everything else is the harness, the same for every climber.

A climber reference (`--climber`, `climber:` in config.yaml) is one of

- a bundled name            greedy | openevolve | gepa
- a directory               holding `climber.yaml` (+ its own .py files, `prompts/`)
- one `.py` file            a one-file climber: the single SearchPolicy or
                            SearchLoop it defines, plus any Operator subclasses

Inside a manifest a module is named `file.py[:Class]` (relative to the
climber's directory) or `package.module:Class`. A directory climber's files
are imported as one digest-named package, so they may import each other and
several versions of a climber can live in one process (fleets, tests).

Identity is `Climber.sha256`: a hash over the climber's files (the manifest
included). The manifest has every key from day one; `routing` is reserved and
refused — which model runs is the user's choice, never a climber's.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hillclimb.loop import PolicyLoop, SearchLoop
from hillclimb.operators import Operator
from hillclimb.operators.builtin import BUILTIN_OPERATORS

MANIFEST = "climber.yaml"
BUNDLED_DIR = Path(__file__).parent / "climbers"
DEFAULT_OPERATORS = tuple(cls.name for cls in BUILTIN_OPERATORS)
# templates a climber's prompts/ may never shadow: they are the problem's
# contract and the harness's own passes, the same for every climber
HARNESS_TEMPLATE_PREFIXES = ("contract_",)
HARNESS_TEMPLATES = frozenset(
    {"holdout_clause", "report_clause", "params_cue", "tools_cue", "distill", "consolidate", "paper"}
)


class ClimberLoadError(ValueError):
    """The climber cannot be loaded; the message names the file and the fix."""


class ClimberManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    # exactly one of the two: WHAT to try next, or the whole control flow
    policy: str | None = None
    loop: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    # operator names (built-in) or `file.py:Class` / `module:Class`, each
    # optionally with params: `- draft: {retrieval: true}`
    operators: list[str | dict[str, dict[str, Any]]] = Field(default_factory=lambda: list(DEFAULT_OPERATORS))
    memory: Literal["knowledge-graph", "none"] = "knowledge-graph"
    tuner: str = "random"
    tuner_params: dict[str, Any] = Field(default_factory=dict)
    similarity: list[str] = Field(default_factory=list)
    prompts: str | None = None  # a dir next to the manifest that shadows built-in operator templates by name
    # a loop whose state must never meet a holdout value asks for `after`;
    # asking can only ever tighten what the user's config allows
    holdout_timing: Literal["after"] | None = None

    @model_validator(mode="before")
    @classmethod
    def _reserved(cls, data):
        if isinstance(data, dict) and "routing" in data:
            raise ValueError(
                "`routing` is reserved: which backend and model run is the user's "
                "choice (config.yaml `routing:`), never a climber's"
            )
        return data

    @model_validator(mode="after")
    def _one_brain(self) -> ClimberManifest:
        if (self.policy is None) == (self.loop is None):
            raise ValueError("name exactly one of `policy:` (what to try next) or `loop:` (the whole control flow)")
        return self


class OperatorSet:
    """The operators ONE search may use: name -> (class, params). Built per
    search from its climber, so two climbers in one process never see each
    other's operators."""

    def __init__(self, entries: Mapping[str, tuple[type[Operator], Mapping]]):
        self._entries = dict(entries)

    @classmethod
    def builtin(cls, params: Mapping[str, Mapping] | None = None) -> OperatorSet:
        params = params or {}
        return cls({op.name: (op, params.get(op.name, {})) for op in BUILTIN_OPERATORS})

    def names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def get(self, name: str) -> Operator:
        try:
            operator_cls, params = self._entries[name]
        except KeyError:
            raise ValueError(f"Unknown operator {name!r}. Available: {sorted(self._entries)}") from None
        return operator_cls(params)

    def role_of(self, name: str) -> str | None:
        entry = self._entries.get(name)
        return entry[0].role if entry is not None else None


@dataclass
class Climber:
    ref: str  # as the user wrote it
    manifest: ClimberManifest
    root: Path | None  # the climber's directory (None for a one-file climber)
    sha256: str
    source: Path  # the manifest, or the one file
    _package: str | None = field(default=None, repr=False)
    _module: types.ModuleType | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def is_loop(self) -> bool:
        return self.manifest.loop is not None

    @property
    def prompts_dir(self) -> Path | None:
        if self.root is None or not self.manifest.prompts:
            return None
        path = self.root / self.manifest.prompts
        return path if path.is_dir() else None

    # --- building the modules ---

    def resolved_params(self, overlay: Mapping | None = None) -> dict:
        """The manifest's params with the user's `climber_params` on top."""
        return {**self.manifest.params, **dict(overlay or {})}

    def build_loop(self, *, params: Mapping | None = None, complexity_start: int = 0,
                   parallelism: int = 1, log=print) -> SearchLoop:
        merged = self.resolved_params(params)
        offered = {"params": merged, "complexity_start": complexity_start, "parallelism": parallelism, "log": log}
        if self.manifest.loop is not None:
            loop = _construct(self._resolve(self.manifest.loop, "loop"), offered, self.source)
            if not isinstance(loop, SearchLoop):
                raise ClimberLoadError(f"{self.source}: `loop:` must name a SearchLoop, got {type(loop).__name__}")
            return loop
        policy = _construct(self._resolve(self.manifest.policy, "policy"), offered, self.source)
        for method in ("propose", "observe"):
            if not callable(getattr(policy, method, None)):
                raise ClimberLoadError(f"{self.source}: the policy has no {method}() — not a SearchPolicy")
        if not getattr(policy, "name", None):
            policy.name = self.name
        if getattr(policy, "params", None) is None:
            policy.params = dict(merged)
        return PolicyLoop(policy)

    def operator_set(self) -> OperatorSet:
        builtin = {op.name: op for op in BUILTIN_OPERATORS}
        entries: dict[str, tuple[type[Operator], Mapping]] = {}
        for item in self.manifest.operators:
            ref, params = (item, {}) if isinstance(item, str) else next(iter(item.items()))
            operator_cls = builtin.get(ref) or self._resolve(ref, "operators")
            if not (inspect.isclass(operator_cls) and issubclass(operator_cls, Operator)):
                raise ClimberLoadError(f"{self.source}: operator {ref!r} is not an Operator subclass")
            entries[operator_cls.name] = (operator_cls, params or {})
        for extra in _classes(self._module, Operator) if self._module is not None else ():
            entries.setdefault(extra.name, (extra, {}))  # a one-file climber's own operators
        return OperatorSet(entries)

    def lint_prompts(self) -> list[str]:
        """Problems with the climber's prompts dir (empty when clean): a
        harness-owned template shadowed, or a token nothing fills."""
        from hillclimb.prompts.render import TEMPLATE_SUFFIX, lint_overrides

        directory = self.prompts_dir
        if directory is None:
            return []
        problems = [
            f"{path.name}: harness-owned template — the problem's contract is the same for every climber"
            for path in sorted(directory.glob(f"*{TEMPLATE_SUFFIX}"))
            if path.stem in HARNESS_TEMPLATES or path.stem.startswith(HARNESS_TEMPLATE_PREFIXES)
        ]
        return problems + lint_overrides(directory)

    # --- module resolution ---

    def _resolve(self, ref: str, key: str):
        if ref.endswith(".py") or ".py:" in ref:
            file_name, _, attr = ref.partition(":")
            module = self._import_file(file_name)
            if attr:
                target = getattr(module, attr, None)
                if target is None:
                    raise ClimberLoadError(f"{self.source}: `{key}: {ref}` — {file_name} defines no {attr}")
                return target
            base = SearchLoop if key == "loop" else None
            return _only_class(module, key, self.source, base)
        if ":" in ref:
            module_name, _, attr = ref.partition(":")
            try:
                return getattr(importlib.import_module(module_name), attr)
            except (ImportError, AttributeError) as exc:
                raise ClimberLoadError(f"{self.source}: `{key}: {ref}` cannot be imported: {exc}") from exc
        raise ClimberLoadError(
            f"{self.source}: `{key}: {ref}` — name a file (`policy.py` or `policy.py:Class`) "
            "or an importable `package.module:Class`"
        )

    def _import_file(self, file_name: str) -> types.ModuleType:
        if self._module is not None and self.root is None:
            return self._module  # the one file IS the climber
        path = (self.root / file_name).resolve()
        if self.root.resolve() not in path.parents or not path.is_file():
            raise ClimberLoadError(f"{self.source}: {file_name} is not a file inside the climber's directory")
        package = self._ensure_package()
        relative = path.relative_to(self.root.resolve()).with_suffix("")
        try:
            return importlib.import_module(package + "." + ".".join(relative.parts))
        except Exception as exc:  # noqa: BLE001 — an author's import error, reported with its file
            raise ClimberLoadError(f"{path} failed to import: {type(exc).__name__}: {exc}") from exc

    def _ensure_package(self) -> str:
        if self._package is None:
            if BUNDLED_DIR in self.root.resolve().parents or self.root.resolve() == BUNDLED_DIR:
                self._package = "hillclimb.climbers." + self.root.name  # a real package: one class object for everyone
            else:
                self._package = f"hillclimb_climber_{_identifier(self.name)}_{self.sha256[:12]}"
                if self._package not in sys.modules:
                    package = types.ModuleType(self._package)
                    package.__path__ = [str(self.root.resolve())]
                    sys.modules[self._package] = package
        return self._package


# --- loading ---


def bundled_climbers() -> list[str]:
    return sorted(p.parent.name for p in BUNDLED_DIR.glob(f"*/{MANIFEST}"))


def load_climber(ref: str, base_dir: Path | None = None) -> Climber:
    """Resolve a climber reference (see the module docstring). Every failure
    is a `ClimberLoadError` naming the file and the fix — never a traceback from
    deep inside engine start-up."""
    if ref in bundled_climbers():
        return _load_dir(ref, BUNDLED_DIR / ref)
    path = Path(ref).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if path.is_dir():
        return _load_dir(ref, path)
    if path.suffix == ".py":
        return _load_file(ref, path)
    raise ClimberLoadError(
        f"Unknown climber: {ref} (bundled: {', '.join(bundled_climbers())}; "
        f"or a directory holding {MANIFEST}; or one .py file)"
    )


def _load_dir(ref: str, root: Path) -> Climber:
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        raise ClimberLoadError(f"{root} holds no {MANIFEST}")
    try:
        data = yaml.safe_load(manifest_path.read_text()) or {}
        data.setdefault("name", root.name)
        manifest = ClimberManifest.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        raise ClimberLoadError(f"{manifest_path}: {exc}") from exc
    return Climber(ref=ref, manifest=manifest, root=root, sha256=tree_sha256(root), source=manifest_path)


def _load_file(ref: str, path: Path) -> Climber:
    if not path.is_file():
        raise ClimberLoadError(f"climber file not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    module_name = f"hillclimb_climber_{_identifier(path.stem)}_{digest[:12]}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ClimberLoadError(f"cannot import climber file: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # dataclasses/pickling look modules up by name
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            del sys.modules[module_name]
            raise ClimberLoadError(f"climber file {path} failed to import: {type(exc).__name__}: {exc}") from exc
    is_loop = bool(_classes(module, SearchLoop))
    key = "loop" if is_loop else "policy"
    manifest = ClimberManifest(name=path.stem, **{key: path.name})
    return Climber(ref=ref, manifest=manifest, root=None, sha256=digest, source=path, _module=module)


def tree_sha256(root: Path) -> str:
    """Identity of a climber directory: every file's relative path and bytes,
    sorted; caches and dotfiles are not part of what a climber IS."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


# --- helpers ---

POLICY_ATTR = "POLICY"


def _identifier(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def _classes(module: types.ModuleType, base: type) -> list[type]:
    return [
        obj for obj in vars(module).values()
        if inspect.isclass(obj) and obj.__module__ == module.__name__ and issubclass(obj, base) and obj is not base
    ]


def _only_class(module: types.ModuleType, key: str, source: Path, base: type | None):
    """The one policy (duck-typed: propose + observe), or the one SearchLoop,
    a file defines; `POLICY = <class or factory>` names it explicitly."""
    explicit = getattr(module, POLICY_ATTR, None)
    if key == "policy" and explicit is not None:
        if not callable(explicit):
            raise ClimberLoadError(f"{POLICY_ATTR} in {source} is not a class or factory: {explicit!r}")
        return explicit
    if base is not None:
        found = _classes(module, base)
    else:
        found = [
            obj for obj in vars(module).values()
            if inspect.isclass(obj) and obj.__module__ == module.__name__
            and callable(getattr(obj, "propose", None)) and callable(getattr(obj, "observe", None))
        ]
    if len(found) != 1:
        kind = "SearchLoop subclass" if base is not None else "policy class (propose + observe)"
        raise ClimberLoadError(
            f"{source} must define exactly one {kind} (found {[c.__name__ for c in found]})"
            + (f" or set {POLICY_ATTR} = <class or factory>" if key == "policy" else "")
        )
    return found[0]


def _construct(target, offered: Mapping[str, Any], source: Path):
    """Call a class or factory with whichever of the offered keyword
    arguments its signature accepts (`params`, `complexity_start`,
    `parallelism`, `log`)."""
    if not callable(target):
        raise ClimberLoadError(f"{source}: {target!r} is not a class or factory")
    try:
        accepted = inspect.signature(target).parameters
    except (TypeError, ValueError):
        accepted = {}
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values())
    kwargs = {name: value for name, value in offered.items() if takes_any or name in accepted}
    return target(**kwargs)
