"""Module references: the ONE way a name in a config becomes a class.

Every exchangeable module of a climber — the policy or loop, a selector, the
operators, the tuner, the memory and its graph — and every similarity score
is named the same three ways:

- a registry name            `greedy`, `optuna`, `knowledge-graph`
- a file                     `mine.py` or `mine.py:Class`
- an importable class        `package.module:Class`

`resolve_ref(ref, kind)` turns any of them into the class (or factory) and
says which form it was. A file that names no class exposes its pick through
the kind's module attribute (`POLICY = ...`) or by defining exactly one class
of the kind.

Local files are imported through a `FileScope`: the files one climber
reaches (its refs plus everything they import relatively), as ONE synthetic
package rooted at their common directory and named by the digest of their
bytes. So a climber's files share class objects, `from .helpers import x`
works, and two versions of the same file coexist in one process (a fleet, a
test, a resumed search next to a live edit).

This module imports only the standard library (and `_moved`), so every
package under `modules/` can use it without a cycle.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import os
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hillclimb._moved import modernize

FILE_SUFFIX = ".py"
KNOBS = "knobs"  # the `**knobs` of a module's constructor: its params, by keyword


def with_knobs(params, knobs: Mapping, known: Mapping | None = None, who: str = "") -> Any:
    """A module's params with keyword knobs laid over them. Without knobs
    the params come back AS GIVEN (the same object: a caller may hold a live
    mapping); with a `known` set, a knob that is not in it is a TypeError —
    a typo in `Greedy(num_draft=3)` fails where it was written."""
    if not knobs:
        return params if params is not None else {}
    if known is not None:
        unknown = sorted(set(knobs) - set(known))
        if unknown:
            raise TypeError(f"{who} has no param {', '.join(map(repr, unknown))} (it has: {', '.join(sorted(known))})")
    return {**dict(params or {}), **knobs}


class ClimberLoadError(ValueError):
    """A module (or the climber naming it) cannot be loaded; the message
    names the file and the fix."""


# --- the kinds -----------------------------------------------------------------


@dataclass
class Kind:
    """One module slot: how a file's pick is recognised and what may be named."""

    name: str  # the key it is written under: `operator_policy`, `tuner`, `graph`
    noun: str  # how messages call it
    attr: str | None = None  # module attribute that names a file's pick explicitly
    base: str | None = None  # `module:Class` every pick must subclass (None = duck-typed)
    duck: tuple[str, ...] = ()  # methods a duck-typed pick must have
    home: str | None = None  # the package whose import registers the built-ins
    # registry name -> the class, or a lazy `module:Attr`
    registry: dict[str, Any] = field(default_factory=dict)

    def load(self) -> "Kind":
        """Make sure the built-ins are registered (their package does it on import)."""
        if self.home is not None and self.home not in sys.modules:
            importlib.import_module(self.home)
        return self

    def base_class(self) -> type | None:
        return _import_attr(self.base) if self.base else None

    def describe(self) -> str:
        base = self.base_class()
        if base is not None:
            return f"{base.__name__} subclass"
        return f"{self.noun} class ({' + '.join(self.duck)})"

    def matches(self, obj: Any) -> bool:
        if not inspect.isclass(obj):
            return False
        base = self.base_class()
        if base is not None:
            return issubclass(obj, base) and obj is not base
        return all(callable(getattr(obj, method, None)) for method in self.duck)


KINDS: dict[str, Kind] = {
    kind.name: kind
    for kind in (
        Kind("operator_policy", "operator policy", attr="POLICY", duck=("propose", "observe"),
             home="hillclimb.modules.policies"),
        Kind("loop", "loop", attr="LOOP", base="hillclimb.harness.loop:Loop", home=None),
        Kind("selector_policy", "selector policy", attr="SELECTOR",
             base="hillclimb.modules.selectors.base:SelectorPolicy", home="hillclimb.modules.selectors"),
        Kind("operator", "operator", base="hillclimb.modules.operators.base:Operator",
             home="hillclimb.modules.operators"),
        Kind("tuner", "tuner", attr="TUNER", duck=("ask",), home="hillclimb.modules.tuners"),
        Kind("memory", "memory", attr="MEMORY", base="hillclimb.modules.memory.base:Memory",
             home="hillclimb.modules.memory.files"),
        Kind("graph", "graph module", attr="KNOWLEDGE_GRAPH", base="hillclimb.modules.memory.base:GraphModule",
             home="hillclimb.modules.memory.graphs"),
        Kind("similarity", "similarity score", attr="SIMILARITY_SCORE",
             base="hillclimb.modules.similarity.base:SimilarityScore", home="hillclimb.modules.similarity"),
    )
}


# the two decisions were the kinds `policy` and `select` until 0.7; a
# `refs.register("policy", ...)` in a user's file still lands
KIND_ALIASES = {"policy": "operator_policy", "select": "selector_policy"}


def kind_of(kind: str | Kind) -> Kind:
    if isinstance(kind, Kind):
        return kind
    try:
        return KINDS[KIND_ALIASES.get(kind, kind)]
    except KeyError:
        raise ClimberLoadError(f"unknown module kind {kind!r} (known: {', '.join(sorted(KINDS))})") from None


def register(kind: str, name: str, target: Any) -> None:
    """Add a registry name for a kind: the class, or a lazy `module:Attr`.
    A registry name never looks like a file or a dotted path, so the three
    forms stay unambiguous."""
    if not name:
        raise ValueError(f"a {kind_of(kind).noun} needs a name to be registered under")
    if name.endswith(FILE_SUFFIX) or ":" in name:
        raise ValueError(f"{kind_of(kind).noun} name {name!r} would read as a file or module:Class")
    kind_of(kind).registry[name] = target


def registered(kind: str) -> dict[str, Any]:
    """The kind's registry, every lazy entry resolved to its class."""
    slot = kind_of(kind).load()
    return {name: _registered_target(slot, name) for name in slot.registry}


def registered_names(kind: str) -> list[str]:
    return sorted(kind_of(kind).load().registry)


def _registered_target(slot: Kind, name: str) -> Any:
    target = slot.registry[name]
    if isinstance(target, str):
        try:
            return _import_attr(target)
        except (ImportError, AttributeError) as exc:
            raise ClimberLoadError(f"{slot.noun} {name!r} cannot be imported: {exc}") from exc
    return target


def _import_attr(spec: str) -> Any:
    module_name, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module_name), attr)


# --- the three forms -------------------------------------------------------------


def is_file_ref(ref: str) -> bool:
    return ref.endswith(FILE_SUFFIX) or f"{FILE_SUFFIX}:" in ref


def is_module_ref(ref: str) -> bool:
    return ":" in ref and not is_file_ref(ref)


def split_file_ref(ref: str) -> tuple[str, str]:
    """`dir/mine.py:Class` -> (`dir/mine.py`, `Class`); the class may be empty.
    Splits at the `.py:` boundary, so a Windows drive letter is left alone."""
    marker = f"{FILE_SUFFIX}:"
    if marker in ref:
        head, _, attr = ref.rpartition(marker)
        return head + FILE_SUFFIX, attr
    return ref, ""


def ref_path(ref: str, base_dir: Path | None = None) -> Path | None:
    """The file a ref names, anchored at `base_dir` when relative; None for
    a registry name or `module:Class`."""
    if not is_file_ref(ref):
        return None
    path = Path(split_file_ref(ref)[0]).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path


def anchor_ref(ref: str, base_dir: Path | None) -> str:
    """A file ref with its path made absolute (other forms come back as they
    are) — so a block read from one file means the same thing anywhere."""
    path = ref_path(ref, base_dir)
    if path is None:
        return ref
    attr = split_file_ref(ref)[1]
    text = str(path if path.is_absolute() else path.resolve())
    return f"{text}:{attr}" if attr else text


# --- local files: closure, identity, import ---------------------------------------


def _relative_imports(path: Path) -> list[Path]:
    """The files a module reaches through RELATIVE imports, as they resolve
    on disk. (An absolute `import sibling` is not followed: the directory is
    not on `sys.path`, so it would not import either.)"""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []  # the import reports it, with the file's name
    found: list[Path] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        base = path.parent
        for _ in range(node.level - 1):
            base = base.parent
        target = base.joinpath(*node.module.split(".")) if node.module else base
        candidates = [target.with_suffix(FILE_SUFFIX), target / "__init__.py"] if node.module else []
        for alias in node.names:  # `from .pkg import submodule` / `from . import sibling`
            candidates += [target / f"{alias.name}{FILE_SUFFIX}", target / alias.name / "__init__.py"]
        found += [candidate for candidate in candidates if candidate.is_file()]
    return found


def source_closure(paths: Sequence[Path]) -> list[Path]:
    """Every file the given files reach: themselves, what they import
    relatively (transitively), and the `__init__.py` of each package on the
    way down from their common directory. Sorted, resolved, existing only."""
    seen: dict[Path, None] = {}
    queue = [Path(p).resolve() for p in paths if Path(p).is_file()]
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen[path] = None
        queue += [p.resolve() for p in _relative_imports(path)]
        if not queue:  # a fixpoint so far: add the package inits between the root and each file
            root = _common_root(list(seen))
            for member in list(seen):
                for parent in member.relative_to(root).parents:
                    init = root / parent / "__init__.py"
                    if init.is_file() and init.resolve() not in seen:
                        queue.append(init.resolve())
    return sorted(seen)


def _common_root(files: Sequence[Path]) -> Path:
    return Path(os.path.commonpath([str(f.parent) for f in files]))


def closure_sha256(files: Sequence[Path], root: Path) -> str:
    """Identity of a set of local files: each one's path below `root` and its
    bytes. Independent of where `root` is, so a snapshot hashes like its source."""
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


SCOPE_PACKAGE_PREFIX = "hillclimb_climber_"  # the synthetic packages climber files are imported as

# how many climber files are being imported right now (a file that starts a
# search at import time would start it again when the search imports it)
_IMPORTING = 0


def importing() -> bool:
    return _IMPORTING > 0


@dataclass
class _Component:
    """One synthetic package: files that belong together, rooted at their
    common directory, hashed by their place below it and their bytes."""

    root: Path
    files: list[Path]
    digest: str
    package: str = ""


def _one_package(files: Sequence[Path], root: Path) -> bool:
    """Can every file be imported below ONE root? Not from the filesystem's
    root, and only through directories that are module names: a user's file
    beside a catalog file in `…/lib/python3.12/site-packages/…` is not."""
    if root == Path(root.anchor):
        return False
    return all(part.isidentifier() for f in files for part in f.relative_to(root).parts[:-1])


def _connected(entries: Sequence[Path]) -> list[list[Path]]:
    """The entries grouped with what they reach through relative imports:
    two entries whose closures share a file are one group."""
    groups: list[set[Path]] = []
    for entry in entries:
        closure = set(source_closure([entry]))
        for group in [g for g in groups if g & closure]:
            closure |= group
            groups.remove(group)
        groups.append(closure)
    return [sorted(g) for g in groups]


def _components(entries: Sequence[Path]) -> list[_Component]:
    closure = source_closure(entries)
    if not closure:
        return []
    root = _common_root(closure)
    groups = [closure] if _one_package(closure, root) else _connected(entries)
    components = []
    for files in groups:
        group_root = _common_root(files)
        components.append(_Component(group_root, files, closure_sha256(files, group_root)))
    # ordered by content, so a snapshot (other roots, same bytes) numbers them alike
    components.sort(key=lambda c: c.digest)
    for component in components:
        # the package is named by WHERE the files are as well as what they hold: two
        # byte-identical copies (`climber get greedy` twice, under two names) are two
        # modules with their own `__file__`, not one cached under the first's path —
        # while `digest`, the identity, stays a function of the bytes alone
        located = hashlib.sha256(f"{component.root}\0{component.digest}".encode()).hexdigest()
        component.package = f"{SCOPE_PACKAGE_PREFIX}{located[:12]}"
    return components


class FileScope:
    """The local files one climber (or one standalone ref) reaches, imported
    as synthetic packages. `files` are the entry points; the closure, its
    roots and its digest follow from them. Files that sit together under one
    importable root (a climber folder) are ONE package, as they read; files
    far apart on disk — a user's policy subclassing a catalog class that
    lives in site-packages — are one package EACH, rooted where they are,
    never one rooted at `/`. A one-package scope hashes and numbers its files
    exactly as before, so every recorded identity stands."""

    def __init__(self, files: Sequence[Path]):
        missing = [Path(f) for f in files if not Path(f).is_file()]
        if missing:
            raise ClimberLoadError(f"file not found: {missing[0]}")
        self.entries = [Path(f).resolve() for f in files]
        self.components = _components(self.entries)
        self.files = sorted(f for component in self.components for f in component.files)
        if not self.components:
            self.digest = hashlib.sha256().hexdigest()
        elif len(self.components) == 1:
            self.digest = self.components[0].digest
        else:
            self.digest = hashlib.sha256("\0".join(c.digest for c in self.components).encode()).hexdigest()

    @property
    def root(self) -> Path:
        """The one root, or the directory every component sits below."""
        if len(self.components) == 1:
            return self.components[0].root
        return _common_root(self.files) if self.files else Path.cwd()

    @property
    def package(self) -> str:
        """The one package's name (a multi-root scope: `component_of(path).package`)."""
        if self.components:
            return self.components[0].package
        located = hashlib.sha256(f"{Path.cwd()}\0{self.digest}".encode()).hexdigest()
        return f"{SCOPE_PACKAGE_PREFIX}{located[:12]}"

    def component_of(self, path: Path) -> tuple[int, _Component]:
        resolved = Path(path).resolve()
        for index, component in enumerate(self.components):
            if resolved in component.files or component.root in resolved.parents:
                return index, component
        raise ClimberLoadError(f"{resolved} is not among this climber's files ({self.root})")

    def relative(self, path: Path) -> Path:
        """Where `path` sits in the scope: below the one root, or
        `<component>/<below its root>` when there are several."""
        index, component = self.component_of(path)
        below = Path(path).resolve().relative_to(component.root)
        return below if len(self.components) == 1 else Path(str(index)) / below

    def _ensure_package(self, component: _Component) -> None:
        if component.package not in sys.modules:
            package = types.ModuleType(component.package)
            package.__path__ = [str(component.root)]
            sys.modules[component.package] = package

    def import_file(self, path: Path, noun: str = "climber") -> types.ModuleType:
        path = Path(path).resolve()
        if not path.is_file():
            raise ClimberLoadError(f"{noun} file not found: {path}")
        try:
            _, component = self.component_of(path)
        except ClimberLoadError:
            raise ClimberLoadError(f"{noun} file {path} is outside {self.root}") from None
        relative = path.relative_to(component.root).with_suffix("")
        self._ensure_package(component)
        global _IMPORTING
        _IMPORTING += 1
        try:
            return importlib.import_module(".".join((component.package, *relative.parts)))
        except Exception as exc:  # noqa: BLE001 — an author's import error, reported with its file
            hint = requirements_hint(component.root, exc)
            raise ClimberLoadError(f"{noun} file {path} failed to import: {type(exc).__name__}: {exc}{hint}") from exc
        finally:
            _IMPORTING -= 1


def requirements_hint(root: Path, exc: BaseException) -> str:
    """The install line for a library a climber's files import and the
    environment lacks: a `requirements.txt` beside them names what they need
    beyond hillclimb, as a problem's names its own. Empty for anything else —
    a missing sibling, a typo in hillclimb's name, no requirements file."""
    if not isinstance(exc, ModuleNotFoundError):
        return ""
    missing = (exc.name or "").split(".")[0]
    if not missing or missing == "hillclimb" or missing.startswith(SCOPE_PACKAGE_PREFIX):
        return ""
    requirements = root / "requirements.txt"
    if not requirements.is_file():
        return ""
    return f" — this climber's requirements: pip install -r {requirements}"


# --- resolving -------------------------------------------------------------------


@dataclass(frozen=True)
class Resolved:
    target: Any  # the class, or a factory
    form: str  # name | file | module
    ref: str
    path: Path | None = None  # the file, for the file form
    module: types.ModuleType | None = None  # the imported file

    @property
    def label(self) -> str:
        """A short display name: the registry name, a file's stem (the class
        named after the colon, when `file.py:Class` names one — one file may
        hold both policies), a class's name."""
        if self.form == "name":
            return self.ref
        if self.form == "file":
            _, attr = split_file_ref(self.ref)
            if attr:
                return attr
            return self.path.stem if self.path is not None else self.ref
        return getattr(self.target, "__name__", self.ref)


def pick(module: types.ModuleType, kind: str | Kind, source: Path):
    """The one thing of `kind` a file defines: what its explicit attribute
    names (`POLICY = ...`), else its only class of the kind."""
    slot = kind_of(kind)
    if slot.attr is not None and getattr(module, slot.attr, None) is not None:
        explicit = getattr(module, slot.attr)
        if slot.base is not None and not slot.matches(explicit):
            raise ClimberLoadError(f"{slot.attr} in {source} is not a {slot.describe()}: {explicit!r}")
        if not callable(explicit):
            raise ClimberLoadError(f"{slot.attr} in {source} is not a class or factory: {explicit!r}")
        return explicit
    found = [
        obj for obj in vars(module).values()
        if inspect.isclass(obj) and obj.__module__ == module.__name__ and slot.matches(obj)
    ]
    if len(found) != 1:
        raise ClimberLoadError(
            f"{slot.noun} file {source} must define exactly one {slot.describe()} "
            f"(found {[c.__name__ for c in found]})"
            + (f" or set {slot.attr} = <class{'' if slot.base else ' or factory'}>" if slot.attr else "")
        )
    return found[0]


class NoClimber(ClimberLoadError):
    """Nothing names a climber: the folder's runs/config.yaml has no `climber:`
    and the call gave none. The message says how to fetch one."""


NO_CLIMBER_HINT = (
    "no climber: fetch one from the catalog with `hillclimb climber get greedy`, "
    "or name your own (`--climber <file.py>`, `climber:` in runs/config.yaml, "
    "`hc.catalog.climber('greedy')` in Python)"
)


def _recorded(slot: Kind, ref: str, form: str, legacy: bool) -> Resolved | None:
    """A name or module path a RECORD holds from before 0.9, when the bundled
    climbers were engine code: resolved to the catalog file that holds the
    class today — only when `legacy` says this is a record (a snapshot, a run
    folder), never for a new config."""
    if not legacy:
        return None
    from hillclimb import catalog

    entry = catalog.RECORDED.get((slot.name, ref)) if form == "name" else catalog.RECORDED_MODULES.get(modernize(ref))
    if entry is None:
        return None
    path, attr = catalog.recorded_file(entry)
    module = FileScope([path]).import_file(path, slot.noun)
    target = getattr(module, attr)
    return Resolved(_checked(target, slot, ref), form, ref, path=path.resolve(), module=module)


def _catalog_hint(slot: Kind, ref: str) -> str:
    from hillclimb import catalog

    if ref in catalog.RECORDED_PRESETS or any(name == ref for kind, name in catalog.RECORDED):
        return (
            f"{ref!r} is a catalog climber, not a name the engine knows: `hillclimb climber get {ref}` (CLI) "
            f"or `hc.catalog.climber({ref!r})` (Python)"
        )
    return (
        f"unknown {slot.noun} {ref!r} (available: {', '.join(sorted(slot.registry)) or 'none registered'}, "
        "a path to a .py file, or module:Class)"
    )


def resolve_ref(
    ref: str,
    kind: str | Kind,
    *,
    base_dir: Path | None = None,
    scope: FileScope | None = None,
    legacy: bool = False,
) -> Resolved:
    """The class (or factory) a ref names. `base_dir` anchors a relative
    file; `scope` is the climber's `FileScope` when the file is one of
    several that must share a package (a standalone file gets its own);
    `legacy` lets a name or module path a record holds from before 0.9
    (`greedy`, `hillclimb.climbers.greedy.policy:Greedy`) find the catalog
    file that holds the class today."""
    slot = kind_of(kind)
    if not isinstance(ref, str) or not ref:
        raise ClimberLoadError(f"`{slot.name}:` needs a name, a .py file or module:Class (got {ref!r})")
    if is_file_ref(ref):
        path = ref_path(ref, base_dir)
        attr = split_file_ref(ref)[1]
        if not path.is_file():
            raise ClimberLoadError(f"{slot.noun} file not found: {path}")
        module = (scope or FileScope([path])).import_file(path, slot.noun)
        if attr:
            target = getattr(module, attr, None)
            if target is None:
                raise ClimberLoadError(f"`{slot.name}: {ref}` — {path.name} defines no {attr}")
        else:
            target = pick(module, slot, path)
        return Resolved(_checked(target, slot, ref), "file", ref, path=path.resolve(), module=module)
    if ":" in ref:
        module_name, _, attr = modernize(ref).partition(":")  # a ref recorded before a move
        try:
            target = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            recorded = _recorded(slot, ref, "module", legacy)
            if recorded is not None:
                return recorded
            raise ClimberLoadError(f"cannot import {slot.noun} {ref!r}: {exc}") from exc
        return Resolved(_checked(target, slot, ref), "module", ref)
    if ref in slot.load().registry:
        return Resolved(_checked(_registered_target(slot, ref), slot, ref), "name", ref)
    recorded = _recorded(slot, ref, "name", legacy)
    if recorded is not None:
        return recorded
    raise ClimberLoadError(_catalog_hint(slot, ref))


def _checked(target: Any, slot: Kind, ref: str):
    if slot.base is not None:
        if not slot.matches(target):
            raise ClimberLoadError(f"{ref!r} is not a {slot.describe()}")
    elif not callable(target):
        raise ClimberLoadError(f"{ref!r} is not a class or factory: {target!r}")
    return target


def construct(target, offered: Mapping[str, Any], source: Any = ""):
    """Call a class or factory with whichever of the offered keyword
    arguments its signature accepts (`params`, `parallelism`, `log`, ...)."""
    if not callable(target):
        raise ClimberLoadError(f"{source}: {target!r} is not a class or factory")
    try:
        accepted = inspect.signature(target).parameters
    except (TypeError, ValueError):
        accepted = {}
    # `**knobs` is how a person composing in Python sets params by keyword
    # (`Greedy(num_drafts=3)`); it never takes what the loader offers
    takes_any = any(
        p.kind is inspect.Parameter.VAR_KEYWORD and p.name != KNOBS for p in accepted.values()
    )
    kwargs = {name: value for name, value in offered.items() if takes_any or name in accepted}
    return target(**kwargs)
