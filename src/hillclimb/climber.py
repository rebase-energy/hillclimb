"""Climbers: HOW to hillclimb, as one block of config.

A climber is the exchangeable half of a search — a `Policy` (or, for
climbers that own their control flow, a `Loop`), the operators it may use
with their prompts, a tuner, a memory. Everything else is the harness, the
same for every climber.

A climber is DEFINED where the run is defined: the `climber:` block of a run
spec entry, or of `hillclimb.yaml` as the folder's default. `ClimberSpec` is
that block:

    climber:
      policy: greedy                 # xor `loop:`
      params: {num_drafts: 5}
      operators: [draft, debug, improve, crossover.py:Crossover]
      operator_params: {draft: {retrieval: false}}
      tuner: optuna
      tuner_params: {seed: 7}
      memory: files
      prompts: prompts/

Every module is named the same three ways (`modules/refs.py`): a registry
name, a `.py` file (`mine.py` or `mine.py:Class`), or `package.module:Class`.
A bare string is shorthand: a preset's name (`greedy | openevolve | gepa`) or
one `.py` file — the single policy or loop it defines, plus any `Operator`
subclasses in it. Defaults live on the classes, so `{policy: greedy}` and
`{loop: gepa}` are complete climbers.

`resolve_climber(spec)` gives the `Climber`: the spec with its local files
imported as one package (`refs.FileScope`) and its identity, `sha256` — the
block plus the bytes of every local file it reaches and of its prompts.
`routing` is reserved and refused: which model runs is the user's choice,
never a climber's.

A search runs — and resumes — from its snapshot, `<search_dir>/climber/`:
the resolved block as `climber.yaml` beside copies of the local files and
prompts. `load_snapshot` also reads what 0.4/0.5 wrote there (a manifest or
one file), and `climber show` turns such a directory into a block.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hillclimb.harness.loop import Loop, PolicyLoop
from hillclimb.modules import refs
from hillclimb.modules.memory.base import DEFAULT_GRAPH, GraphModule, MemoryKind
from hillclimb.modules.operators import Operator
from hillclimb.modules.operators.builtin import BUILTIN_OPERATORS
from hillclimb.modules.refs import ClimberLoadError, FileScope, Resolved

MANIFEST = "climber.yaml"  # the snapshot's block (and the pre-0.6 manifest's file name)
SNAPSHOT_DIRNAME = "climber"
SNAPSHOT_VERSION = 2
SNAPSHOT_FILES = "files"  # the local files, below their common root
SNAPSHOT_PROMPTS = "prompts"
DEFAULT_POLICY = "greedy"
DEFAULT_OPERATORS = tuple(cls.name for cls in BUILTIN_OPERATORS)
# templates a climber's prompts/ may never shadow: they are the problem's
# contract and the harness's own passes, the same for every climber
HARNESS_TEMPLATE_PREFIXES = ("contract_",)
HARNESS_TEMPLATES = frozenset(
    {"holdout_clause", "report_clause", "params_cue", "params_cue_climber", "tools_cue", "distill", "consolidate", "paper"}
)

# a bare name stands for one of these blocks
PRESETS: dict[str, dict[str, Any]] = {
    "greedy": {"policy": "greedy"},
    "openevolve": {"policy": "openevolve"},
    "gepa": {"loop": "gepa"},
}

# keys a pre-0.6 manifest carried that a block does not
_GONE = {
    "description": "a block has no `description`: say it in a YAML comment",
    "similarity": "similarity scores are a viewer's setting (`similarity.scores` in hillclimb.yaml), not a climber's",
    "holdout_timing": "a loop declares it on its class (`holdout_timing = \"after\"`); otherwise it is the user's `holdout.timing`",
}


def climber_label(ref: str) -> str:
    """A short display name for a climber named by a string: a preset's name
    as it is, a one-file climber's stem."""
    if refs.is_file_ref(ref):
        return Path(refs.split_file_ref(ref)[0]).stem
    return ref.rpartition(":")[2] if refs.is_module_ref(ref) else ref


def climber_base_dir(config) -> Path | None:
    """Where a relative climber (or graph, or similarity-score) path resolves
    from: the hillclimb dir, like `paths.runs_dir`; None when no dir is known."""
    return getattr(config, "hillclimb_dir", None)


def presets() -> list[str]:
    return sorted(PRESETS)


def bundled_climbers() -> list[str]:
    """The presets (the name predates them)."""
    return presets()


def expand_name(ref: str) -> dict[str, Any]:
    """The block a bare string stands for: a preset, or one file / one class."""
    if ref in PRESETS:
        return json.loads(json.dumps(PRESETS[ref]))  # a copy nobody can edit the preset through
    if refs.is_file_ref(ref) or refs.is_module_ref(ref):
        return {"policy": ref}  # a file that defines a Loop is recognised when it is resolved
    raise ValueError(
        f"Unknown climber: {ref} (presets: {', '.join(presets())}; or one .py file; or a full "
        "`climber:` block). A directory holding climber.yaml is the pre-0.6 form: "
        f"`hillclimb climber show {ref}` prints it as a block."
    )


class ClimberSpec(BaseModel):
    """The `climber:` block — the same shape in a run spec and in hillclimb.yaml."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None  # a label (experiment names, the watch column); not part of identity
    # exactly one of the two: WHAT to try next, or the whole control flow
    policy: str | None = None
    loop: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    # which operators: names or refs, each optionally with params
    # (`- draft: {retrieval: true}`). None = what the policy/loop declares,
    # else the built-in four — plus any Operator its own file defines
    operators: list[str | dict[str, dict[str, Any] | None]] | None = None
    # params by operator NAME, laid over the list's: what `--set
    # climber.operators.draft.retrieval=false` edits
    operator_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tuner: str = "random"
    tuner_params: dict[str, Any] = Field(default_factory=dict)
    memory: MemoryKind = "files"  # `knowledge-graph` (pre-0.4) still loads
    # the graph module over the memory: `knowledge-graph`, a file, or module:Class
    graph: str = DEFAULT_GRAPH
    prompts: str | None = None  # a dir whose templates shadow the built-in operator templates by name

    @model_validator(mode="before")
    @classmethod
    def _shorthand_and_refusals(cls, data):
        if isinstance(data, str):
            return expand_name(data)
        if not isinstance(data, dict):
            return data
        if "routing" in data:
            raise ValueError(
                "`routing` is reserved: which agent and model run is the user's "
                "choice (hillclimb.yaml `routing:`), never a climber's"
            )
        for key, advice in _GONE.items():
            if key in data:
                raise ValueError(f"`{key}`: {advice}")
        return data

    @model_validator(mode="after")
    def _one_brain(self) -> ClimberSpec:
        if self.policy is not None and self.loop is not None:
            raise ValueError("name one of `policy:` (what to try next) or `loop:` (the whole control flow), not both")
        if self.policy is None and self.loop is None:
            self.policy = DEFAULT_POLICY
        return self

    # --- reading it ---

    @property
    def brain(self) -> str:
        return self.loop if self.loop is not None else self.policy

    @property
    def label(self) -> str:
        """What views call it: its `name`, else the policy's or loop's."""
        return self.name or climber_label(self.brain)

    def operator_items(self) -> list[tuple[str, dict[str, Any]]] | None:
        """`operators:` as (ref, params) pairs; None when the block leaves them to the policy/loop."""
        if self.operators is None:
            return None
        items = []
        for entry in self.operators:
            if isinstance(entry, str):
                items.append((entry, {}))
            else:
                if len(entry) != 1:
                    raise ValueError(f"operators: write one operator per entry (got {sorted(entry)})")
                ref, params = next(iter(entry.items()))
                items.append((ref, dict(params or {})))
        return items

    def module_refs(self) -> list[str]:
        """Every module this block names."""
        return [self.brain, self.tuner, self.graph, *(ref for ref, _ in self.operator_items() or [])]

    def map_refs(self, change: Callable[[str], str]) -> ClimberSpec:
        """A copy with every module ref passed through `change`."""
        update: dict[str, Any] = {
            "loop" if self.loop is not None else "policy": change(self.brain),
            "tuner": change(self.tuner),
            "graph": change(self.graph),
        }
        if self.operators is not None:
            update["operators"] = [
                change(entry) if isinstance(entry, str) else {change(ref): params for ref, params in entry.items()}
                for entry in self.operators
            ]
        return self.model_copy(update=update)

    def anchored(self, base_dir: Path | None) -> ClimberSpec:
        """A copy whose file refs and `prompts` dir are absolute — relative
        ones resolve from `base_dir`, the folder of the file the block was
        written in — so the block means the same thing wherever it travels."""
        spec = self.map_refs(lambda ref: refs.anchor_ref(ref, base_dir))
        if spec.prompts:
            prompts = Path(spec.prompts).expanduser()
            if not prompts.is_absolute():
                prompts = (Path(base_dir) / prompts) if base_dir is not None else prompts.resolve()
            spec = spec.model_copy(update={"prompts": str(prompts)})
        return spec

    def file_paths(self) -> list[Path]:
        """The local files the block names (after `anchored`)."""
        return [path for path in (refs.ref_path(ref) for ref in self.module_refs()) if path is not None]

    def block(self) -> dict[str, Any]:
        """The block as it is written down: every set key, nothing that is None."""
        return self.model_dump(exclude_none=True)


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
    """A resolved `ClimberSpec`: its modules importable, its identity known."""

    spec: ClimberSpec  # anchored: every file ref absolute
    scope: FileScope  # its local files, one package
    sha256: str
    ref: str = ""  # how it was named, when it was named by a string
    # a pre-0.6 manifest could ask for `holdout_timing: after`
    legacy_holdout_timing: str | None = None
    _brain: Resolved | None = field(default=None, repr=False)
    _brain_kind: str = field(default="", repr=False)

    @property
    def name(self) -> str:
        return self.spec.label

    @property
    def source(self) -> str:
        """What to call it in a message: its file, else its name."""
        path = refs.ref_path(self.spec.brain)
        return str(path) if path is not None else self.name

    # --- the policy or loop ---

    @property
    def brain(self) -> Resolved:
        if self._brain is None:
            ref = self.spec.brain
            kind = "loop" if self.spec.loop is not None else "policy"
            if kind == "policy" and refs.is_file_ref(ref) and not refs.split_file_ref(ref)[1]:
                # a bare file: it is a loop climber when a Loop is what it defines
                module = self.scope.import_file(refs.ref_path(ref), "climber")
                explicit = getattr(module, refs.KINDS["policy"].attr, None)
                if explicit is None and _classes(module, Loop):
                    kind = "loop"
            resolved = refs.resolve_ref(ref, kind, scope=self.scope)
            if kind == "policy" and inspect.isclass(resolved.target) and issubclass(resolved.target, Loop):
                kind = "loop"
            self._brain, self._brain_kind = resolved, kind
        return self._brain

    @property
    def is_loop(self) -> bool:
        return self.brain is not None and self._brain_kind == "loop"

    @property
    def description(self) -> str:
        """The first line of what the policy or loop says about itself."""
        target = self.brain.target
        doc = (getattr(target, "__doc__", None) if "__doc__" in vars(target) else None) or (
            getattr(inspect.getmodule(target), "__doc__", None) or ""
        )
        return doc.strip().splitlines()[0] if doc.strip() else ""

    @property
    def holdout_timing(self) -> str | None:
        """`after` when the loop's state must never meet a holdout value —
        asking can only ever tighten what the user's config allows."""
        return getattr(self.brain.target, "holdout_timing", None) or self.legacy_holdout_timing

    @property
    def prompts_dir(self) -> Path | None:
        if not self.spec.prompts:
            return None
        path = Path(self.spec.prompts)
        return path if path.is_dir() else None

    # --- building the modules ---

    def resolved_params(self, overlay: Mapping | None = None) -> dict:
        """The block's params with an overlay on top."""
        return {**self.spec.params, **dict(overlay or {})}

    def build_loop(self, *, params: Mapping | None = None, complexity_start: int = 0,
                   parallelism: int = 1, log=print) -> Loop:
        merged = self.resolved_params(params)
        offered = {"params": merged, "complexity_start": complexity_start, "parallelism": parallelism, "log": log}
        target = self.brain.target
        if self.is_loop:
            loop = refs.construct(target, offered, self.source)
            if not isinstance(loop, Loop):
                raise ClimberLoadError(f"{self.source}: `loop:` must name a Loop, got {type(loop).__name__}")
            return loop
        policy = refs.construct(target, offered, self.source)
        for method in ("propose", "observe"):
            if not callable(getattr(policy, method, None)):
                raise ClimberLoadError(f"{self.source}: the policy has no {method}() — not a Policy")
        if not getattr(policy, "name", None):
            policy.name = self.name
        if getattr(policy, "params", None) is None:
            policy.params = dict(merged)
        return PolicyLoop(policy)

    def operator_set(self) -> OperatorSet:
        """The operators this climber's search may use. The block's list;
        else what the policy/loop declares (`operators = (...)` on the class);
        else the built-in four. A one-file climber's own Operator subclasses
        are always in. `operator_params` lie over each by name."""
        items = self.spec.operator_items()
        if items is None:
            declared = getattr(self.brain.target, "operators", None)
            items = [(entry, {}) for entry in (declared if declared is not None else DEFAULT_OPERATORS)]
        entries: dict[str, tuple[type[Operator], Mapping]] = {}
        for ref, params in items:
            if inspect.isclass(ref):  # a class names its operators by class
                operator_cls = ref
            else:
                operator_cls = refs.resolve_ref(ref, "operator", scope=self.scope).target
            if not (inspect.isclass(operator_cls) and issubclass(operator_cls, Operator)):
                raise ClimberLoadError(f"{self.source}: operator {ref!r} is not an Operator subclass")
            entries[operator_cls.name] = (operator_cls, params)
        if self.brain.module is not None:  # a one-file climber's own operators
            for extra in _classes(self.brain.module, Operator):
                entries.setdefault(extra.name, (extra, {}))
        for name, overlay in self.spec.operator_params.items():
            if name not in entries:
                raise ClimberLoadError(
                    f"climber.operators.{name}: this climber has no operator {name!r} (it has {sorted(entries)})"
                )
            operator_cls, params = entries[name]
            entries[name] = (operator_cls, {**params, **overlay})
        return OperatorSet(entries)

    def tuner(self):
        """The tuner the block names, built with its `tuner_params`."""
        from hillclimb.modules.tuners import get_tuner

        return get_tuner(self.spec.tuner, self.spec.tuner_params, scope=self.scope)

    def graph_module(self) -> GraphModule:
        """The climber's graph module (`graph:`): the built-in by registry
        name, a local file (so it travels with the snapshot), or an
        importable class — keyed so that an edited file is a different
        builder and graph.json is rebuilt."""
        from hillclimb.modules.memory.graphs import graph_key

        ref = self.spec.graph
        resolved = refs.resolve_ref(ref, "graph", scope=self.scope)
        module = resolved.target()
        if resolved.path is not None:
            # the file's NAME, not where it is: a snapshot's copy is the same builder
            attr = refs.split_file_ref(ref)[1]
            module.key = graph_key(resolved.path.name + (f":{attr}" if attr else ""), resolved.path)
        else:
            module.key = graph_key(ref)
        if not module.name:
            module.name = resolved.label  # type: ignore[misc]
        return module

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


# --- resolving ---------------------------------------------------------------------


def as_spec(value: Any) -> ClimberSpec:
    """A `ClimberSpec` from whatever named it: a spec, a block, a bare string."""
    if isinstance(value, ClimberSpec):
        return value
    try:
        return ClimberSpec.model_validate(value)
    except ValueError as exc:
        raise ClimberLoadError(_first_error(exc)) from exc


def _first_error(exc: Exception) -> str:
    """pydantic's report without its header — the message an author can act on."""
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    lines = []
    for error in errors():
        where = ".".join(str(part) for part in error.get("loc", ()))
        message = error.get("msg", "").removeprefix("Value error, ")
        lines.append(f"{where}: {message}" if where else message)
    return "; ".join(lines) or str(exc)


def resolve_climber(spec: Any, base_dir: Path | None = None, *, ref: str = "") -> Climber:
    """The `Climber` a block defines. Relative file refs resolve from
    `base_dir`. Every failure is a `ClimberLoadError` naming the file and the
    fix — never a traceback from deep inside engine start-up. The modules
    themselves are imported when they are first asked for."""
    spec = as_spec(spec).anchored(base_dir)
    try:
        files = spec.file_paths()
    except ValueError as exc:
        raise ClimberLoadError(str(exc)) from exc
    for path in files:
        if not path.is_file():
            raise ClimberLoadError(f"climber file not found: {path}")
    scope = FileScope(files)
    return Climber(spec=spec, scope=scope, sha256=identity(spec, scope), ref=ref)


def identity(spec: ClimberSpec, scope: FileScope) -> str:
    """What a climber IS: its block (without the label), the bytes of every
    local file it reaches, and its prompts. Files count by their place below
    the common root, so a snapshot has the identity of what it was taken of."""

    def portable(ref: str) -> str:
        path = refs.ref_path(ref)
        if path is None:
            return ref
        attr = refs.split_file_ref(ref)[1]
        relative = scope.relative(path).as_posix()
        return f"{relative}:{attr}" if attr else relative

    prompts = Path(spec.prompts) if spec.prompts else None
    blob = json.dumps(
        {
            "climber": spec.map_refs(portable).model_dump(exclude={"name", "prompts"}),
            "files": scope.digest if scope.files else None,
            "prompts": tree_sha256(prompts) if prompts is not None and prompts.is_dir() else None,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def load_climber(ref: str, base_dir: Path | None = None) -> Climber:
    """The climber a STRING names: a preset, one `.py` file, or — the pre-0.6
    form — a directory holding `climber.yaml`."""
    if ref in PRESETS:
        return resolve_climber(ref, ref=ref)
    path = Path(refs.split_file_ref(ref)[0]).expanduser() if not refs.is_module_ref(ref) else None
    if path is not None and not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    if path is not None and path.is_dir():
        return _load_legacy_dir(ref, path)
    if refs.is_file_ref(ref) or refs.is_module_ref(ref):
        return resolve_climber(ref, base_dir, ref=ref)
    raise ClimberLoadError(
        f"Unknown climber: {ref} (bundled: {', '.join(presets())}; "
        f"or one .py file; or a pre-0.6 directory holding {MANIFEST})"
    )


# --- the pre-0.6 manifest ------------------------------------------------------------


def spec_from_legacy_manifest(data: Mapping[str, Any], name: str) -> tuple[ClimberSpec, str | None]:
    """A 0.4/0.5 `climber.yaml` as a block. `description` and `similarity`
    carried nothing a search used and are dropped; `holdout_timing` is handed
    back beside the spec (a loop declares it on its class now)."""
    data = dict(data)
    data.setdefault("name", name)  # a snapshot dir is always called `climber`
    for dropped in ("description", "similarity"):
        data.pop(dropped, None)
    timing = data.pop("holdout_timing", None)
    if timing not in (None, "after"):
        raise ValueError(f"holdout_timing: {timing!r} (only `after` was ever accepted)")
    if (data.get("policy") is None) == (data.get("loop") is None):
        raise ValueError("name exactly one of `policy:` (what to try next) or `loop:` (the whole control flow)")
    return ClimberSpec.model_validate(data), timing


def _load_legacy_dir(ref: str, root: Path, name: str | None = None) -> Climber:
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        raise ClimberLoadError(f"{root} holds no {MANIFEST}")
    try:
        data = yaml.safe_load(manifest_path.read_text()) or {}
        spec, timing = spec_from_legacy_manifest(data, name or root.name)
    except Exception as exc:  # noqa: BLE001
        raise ClimberLoadError(f"{manifest_path}: {_first_error(exc)}") from exc
    try:
        climber = resolve_climber(spec, root, ref=ref)
    except ClimberLoadError as exc:
        raise ClimberLoadError(f"{manifest_path}: {exc}") from exc
    climber.legacy_holdout_timing = timing
    return climber


# --- snapshots ------------------------------------------------------------------------


def snapshot_climber(climber: Climber, search_dir: Path) -> Path:
    """Write the climber into `<search_dir>/climber/`: its block as
    `climber.yaml`, beside copies of its local files (`files/`) and prompts
    (`prompts/`). A search runs — and resumes — from this copy, so editing
    the live files (a person iterating, or a search that improves climbers)
    never changes a search that has already started. Idempotent: an existing
    snapshot is kept."""
    target = search_dir / SNAPSHOT_DIRNAME
    if target.exists():
        return target
    target.mkdir(parents=True)
    scope = climber.scope
    for path in scope.files:
        copy = target / SNAPSHOT_FILES / scope.relative(path)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)

    def in_snapshot(ref: str) -> str:
        path = refs.ref_path(ref)
        if path is None:
            return ref
        attr = refs.split_file_ref(ref)[1]
        relative = (Path(SNAPSHOT_FILES) / scope.relative(path)).as_posix()
        return f"{relative}:{attr}" if attr else relative

    spec = climber.spec.map_refs(in_snapshot)
    if climber.prompts_dir is not None:
        shutil.copytree(
            climber.prompts_dir, target / SNAPSHOT_PROMPTS,
            ignore=lambda _dir, names: [n for n in names if n == "__pycache__" or n.startswith(".")],
        )
        spec = spec.model_copy(update={"prompts": SNAPSHOT_PROMPTS})
    else:
        spec = spec.model_copy(update={"prompts": None})
    block = {"snapshot": SNAPSHOT_VERSION, **spec.block()}
    (target / MANIFEST).write_text(
        "# The climber this search runs, as resolved when it started. Paths are relative to this folder.\n"
        + yaml.safe_dump(block, sort_keys=False)
    )
    return target


def load_snapshot(search_dir: Path, name: str | None = None) -> Climber | None:
    """The climber a search was started with, from its snapshot (None when
    the search predates snapshots). Reads the 0.6 form and what 0.4/0.5 left
    there: a manifest with its files, or one `.py` file."""
    root = search_dir / SNAPSHOT_DIRNAME
    if not root.is_dir():
        return None
    manifest_path = root / MANIFEST
    if manifest_path.is_file():
        try:
            data = yaml.safe_load(manifest_path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ClimberLoadError(f"{manifest_path}: {exc}") from exc
        if isinstance(data, dict) and data.get("snapshot") == SNAPSHOT_VERSION:
            data = {key: value for key, value in data.items() if key != "snapshot"}
            try:
                return resolve_climber(data, root, ref=str(root))
            except ClimberLoadError as exc:
                raise ClimberLoadError(f"{manifest_path}: {exc}") from exc
        return _load_legacy_dir(str(root), root, name=name)
    files = sorted(root.glob("*.py"))
    if len(files) != 1:
        raise ClimberLoadError(f"{root} is neither a climber snapshot nor a one-file climber")
    return resolve_climber(str(files[0]), ref=str(files[0]))


def tree_sha256(root: Path) -> str:
    """Identity of a directory: every file's relative path and bytes,
    sorted; caches and dotfiles are not part of it."""
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


def _classes(module, base: type) -> list[type]:
    return [
        obj for obj in vars(module).values()
        if inspect.isclass(obj) and obj.__module__ == module.__name__ and issubclass(obj, base) and obj is not base
    ]
