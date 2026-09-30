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
      select: map-elites             # which candidate the policy expands
      select_params: {num_islands: 3}
      operators: [draft, debug, improve, crossover.py:Crossover]
      operator_params: {draft: {retrieval: false}}
      tuner: optuna
      tuner_params: {seed: 7}
      memory: files
      memory_params: {max_cards: 3, claims: true}
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
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from hillclimb.harness.loop import Loop, PolicyLoop
from hillclimb.modules import refs
from hillclimb.modules.memory.base import GraphModule
from hillclimb.modules.operators import Operator
from hillclimb.modules.operators.builtin import BUILTIN_OPERATORS
from hillclimb.modules.refs import ClimberLoadError, FileScope, Resolved
from hillclimb.modules.spec import (  # noqa: F401 — the block's home is modules/spec.py
    DEFAULT_POLICY,
    PRESETS,
    ClimberSpec,
    block_error,
    climber_label,
    expand_name,
    presets,
)

MANIFEST = "climber.yaml"  # the snapshot's block (and the pre-0.6 manifest's file name)
SNAPSHOT_DIRNAME = "climber"
SNAPSHOT_VERSION = 2
SNAPSHOT_FILES = "files"  # the local files, below their common root
SNAPSHOT_PROMPTS = "prompts"
DEFAULT_OPERATORS = tuple(cls.name for cls in BUILTIN_OPERATORS)
# templates a climber's prompts/ may never shadow: they are the problem's
# contract and the harness's own passes, the same for every climber
HARNESS_TEMPLATE_PREFIXES = ("contract_",)
HARNESS_TEMPLATES = frozenset(
    {"holdout_clause", "report_clause", "params_cue", "params_cue_climber", "tools_cue", "distill", "consolidate", "paper"}
)

def climber_base_dir(config) -> Path | None:
    """Where a relative climber (or graph, or similarity-score) path resolves
    from: the hillclimb dir, like `paths.runs_dir`; None when no dir is known."""
    return getattr(config, "hillclimb_dir", None)


def bundled_climbers() -> list[str]:
    """The presets (the name predates them)."""
    return presets()


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


class NotPortableError(ClimberLoadError):
    """The climber holds a block that exists only as a class in this process
    (defined in a notebook or a REPL): it runs here, but cannot be written
    down, snapshotted for a resume, or handed to another engine."""


LIVE = "live"  # `live:ClassName`: a block that is a class of this process and nothing else


class Climber:
    """A climber: the block (`ClimberSpec`) with its modules importable and
    its identity known.

    Compose one from building blocks — names, classes or instances:

        import hillclimb as hc

        climber = hc.Climber(
            policy=hc.policies.Greedy(num_drafts=3),
            select=hc.selectors.MapElites(num_islands=2),
            operators=[hc.operators.Draft(retrieval=False), hc.operators.Debug(), MyCrossover],
            tuner="optuna",
            memory=hc.memory.FilesMemory(max_cards=1),
        )
        hc.run("heilbronn-11", climber=climber, budget="10m")
        climber.to_spec()          # the same climber as the block a run config takes

    A preset's name stands for its block: `hc.Climber("openevolve",
    params={"num_drafts": 5})`. An instance stands for its class and params (`Greedy(num_drafts=3)` is
    `policy: greedy` + `params: {num_drafts: 3}`): a search always builds its
    own. A class is written down the most portable way it can be — a
    registry name, `module:Class`, else `its_file.py:Class` — and a class
    that exists only in this process (a notebook cell) still runs here, but
    makes the climber not `portable`: `to_spec()`, resume and detached
    engines refuse it, saying why.

    `resolve_climber(block)` (or `Climber.from_spec`) is the same thing
    starting from a block.
    """

    def __init__(self, policy=None, *, loop=None, select=None, operators=None, tuner=None,
                 memory=None, params: Mapping | None = None, prompts=None, name: str | None = None,
                 base_dir: Path | None = None):
        preset: dict[str, Any] = {}
        if isinstance(policy, str) and policy in PRESETS and loop is None:
            # `Climber("openevolve", params=...)`: a preset's name stands for
            # its whole block, and the other arguments lie over it
            preset, policy = expand_name(policy), None
        block, live = _compose(
            policy=policy, loop=loop, select=select, operators=operators, tuner=tuner,
            memory=memory, params=params, prompts=prompts, name=name,
        )
        if preset:
            block = {**preset, **block, "params": {**preset.get("params", {}), **block.get("params", {})}}
            if not block["params"]:
                del block["params"]
        self._setup(as_spec(block).anchored(base_dir if base_dir is not None else Path.cwd()), live=live)

    @classmethod
    def from_spec(cls, spec: Any, base_dir: Path | None = None, *, ref: str = "") -> Climber:
        """The climber a block defines (relative file refs resolve from `base_dir`)."""
        climber = cls.__new__(cls)
        climber._setup(as_spec(spec).anchored(base_dir), ref=ref)
        return climber

    def _setup(self, spec: ClimberSpec, *, ref: str = "", live: Mapping[str, type] | None = None) -> None:
        try:
            files = spec.file_paths()
        except ValueError as exc:
            raise ClimberLoadError(str(exc)) from exc
        for path in files:
            if not path.is_file():
                raise ClimberLoadError(f"climber file not found: {path}")
        self.spec = spec  # anchored: every file ref absolute
        self.scope = FileScope(files)  # its local files, one package
        self.ref = ref  # how it was named, when it was named by a string
        # blocks that are classes of this process only: `live:Name` -> the class
        self._live: dict[str, type] = dict(live or {})
        # a pre-0.6 manifest could ask for `holdout_timing: after`
        self.legacy_holdout_timing: str | None = None
        self._brain: Resolved | None = None
        self._brain_kind = ""
        self.sha256 = identity(spec, self.scope, self._live)

    def __repr__(self) -> str:
        return f"Climber({self.spec.block()!r}, sha256={self.sha256[:12]!r})"

    def __deepcopy__(self, memo) -> Climber:
        return self  # a climber is not edited after it is built (and holds modules)

    @property
    def name(self) -> str:
        return self.spec.label

    @property
    def portable(self) -> bool:
        """Can it be written down and rebuilt elsewhere? False when a block
        exists only as a class in this process."""
        return not self._live

    def to_spec(self) -> ClimberSpec:
        """The block a run config takes for this climber."""
        if self._live:
            raise NotPortableError(
                f"climber {self.name}: {', '.join(sorted(self._live))} "
                f"exist{'s' if len(self._live) == 1 else ''} only in this process — put the "
                "class in a .py file (or an importable module) to write the climber down, "
                "resume it, or run it in a detached engine"
            )
        return self.spec

    def write(self, path: Path | str) -> Path:
        """Write the climber as a `climber:` block — the top of a run spec
        (add `problems:`), or what goes into hillclimb.yaml."""
        path = Path(path)
        path.write_text(yaml.safe_dump({"climber": self.to_spec().block()}, sort_keys=False))
        return path

    def _resolve(self, ref: str, kind: str) -> Resolved:
        if ref in self._live:
            return Resolved(self._live[ref], LIVE, ref)
        return refs.resolve_ref(ref, kind, scope=self.scope)

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
            resolved = self._resolve(ref, kind)
            if kind == "policy" and inspect.isclass(resolved.target) and issubclass(resolved.target, Loop):
                kind = "loop"
            self._brain, self._brain_kind = resolved, kind
        return self._brain

    @property
    def is_loop(self) -> bool:
        return self.brain is not None and self._brain_kind == "loop"

    @property
    def description(self) -> str:
        """One line on what it does: what the policy or loop says about
        itself — and, for a policy over a selector the block names, what
        that selector picks."""
        text = _doc_line(self.brain.target)
        if self.spec.select and not self.is_loop:
            picks = _doc_line(self._resolve(self.spec.select, "select").target)
            return f"{climber_label(self.spec.brain)} over {climber_label(self.spec.select)}: {picks}"
        return text

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

    def _known_params(self) -> dict | None:
        """The knobs the policy/loop declares (`Policy.defaults()`), or None
        when it takes free-form params."""
        declared = getattr(self.brain.target, "defaults", None)
        return dict(declared()) if callable(declared) else None

    def selector(self):
        """The selector the policy expands with: the block's `select:`, else
        the policy class's own default. None for a loop, and for a policy
        that names none."""
        from hillclimb.modules.selectors import get_selector

        if self.is_loop:
            return None
        ref = self.spec.select or getattr(self.brain.target, "default_selector", None)
        if ref is None:
            if self.spec.select_params:
                raise ClimberLoadError(f"{self.source}: `select_params` without a selector to give them to")
            return None
        if ref in self._live:
            return self._live[ref](dict(self.spec.select_params))
        return get_selector(ref, self.spec.select_params, scope=self.scope)

    def build_loop(self, *, params: Mapping | None = None, priors: Mapping | None = None,
                   parallelism: int = 1, log=print) -> Loop:
        """The loop this climber runs: its `loop:`, or a `PolicyLoop` over
        its `policy:` (with its selector). `priors` are param values memory
        learned (a draft-complexity offset): they sit UNDER the block's
        params and reach only a policy that declares the knob. `params` is
        an overlay for callers that hold a resolved climber."""
        target = self.brain.target
        known = self._known_params()
        merged = self.resolved_params(params)
        if known is not None:
            if getattr(target, "strict_params", False):
                unknown = sorted(set(merged) - set(known))
                if unknown:
                    raise ClimberLoadError(
                        f"climber.params: {climber_label(self.spec.brain)} has no param {', '.join(map(repr, unknown))} "
                        f"(it has: {', '.join(sorted(known))}). A selector's settings go in `select_params`."
                    )
            merged = {**{k: v for k, v in (priors or {}).items() if k in known}, **merged}
        offered = {"params": merged, "parallelism": parallelism, "log": log}
        if self.is_loop:
            loop = refs.construct(target, offered, self.source)
            if not isinstance(loop, Loop):
                raise ClimberLoadError(f"{self.source}: `loop:` must name a Loop, got {type(loop).__name__}")
            return loop
        selector = self.selector()
        if selector is not None:
            if self.spec.select is not None and not _accepts(target, "selector"):
                raise ClimberLoadError(
                    f"{self.source}: `select: {self.spec.select}` — this policy takes no selector "
                    "(its constructor has no `selector` argument)"
                )
            offered["selector"] = selector
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
                operator_cls = self._resolve(ref, "operator").target
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

        if self.spec.tuner in self._live:
            return self._live[self.spec.tuner](dict(self.spec.tuner_params))
        return get_tuner(self.spec.tuner, self.spec.tuner_params, scope=self.scope)

    def memory(self):
        """The memory the block names, built with its `memory_params`."""
        from hillclimb.modules.memory.files import get_memory

        if self.spec.memory in self._live:
            memory = self._live[self.spec.memory](dict(self.spec.memory_params))
            memory.scope = self.scope
            return memory
        return get_memory(self.spec.memory, self.spec.memory_params, scope=self.scope)

    def graph_module(self) -> GraphModule:
        """The graph module that indexes the climber's memory (a setting of
        the memory: `memory_params.graph`); the built-in for a memory that
        has none, so reading the knowledge never depends on the climber."""
        from hillclimb.modules.memory.base import DEFAULT_GRAPH
        from hillclimb.modules.memory.graphs import get_graph

        return self.memory().graph_module() or get_graph(DEFAULT_GRAPH)

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
    """A `ClimberSpec` from whatever named it: a spec, a block, a bare
    string, or a composed `Climber` (which must be portable)."""
    if isinstance(value, ClimberSpec):
        return value
    if isinstance(value, Climber):
        return value.to_spec()
    try:
        return ClimberSpec.model_validate(value)
    except ValueError as exc:
        raise ClimberLoadError(block_error(exc)) from exc


def resolve_climber(spec: Any, base_dir: Path | None = None, *, ref: str = "") -> Climber:
    """The `Climber` a block defines. Relative file refs resolve from
    `base_dir`. Every failure is a `ClimberLoadError` naming the file and the
    fix — never a traceback from deep inside engine start-up. The modules
    themselves are imported when they are first asked for."""
    if isinstance(spec, Climber):
        return spec
    return Climber.from_spec(spec, base_dir, ref=ref)


def identity(spec: ClimberSpec, scope: FileScope, live: Mapping[str, type] | None = None) -> str:
    """What a climber IS: its block (without the label), the bytes of every
    local file it reaches, and its prompts. Files count by their place below
    the common root, so a snapshot has the identity of what it was taken of.
    A class that exists only in this process counts by its source."""

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
            **({"live": {ref: _live_source(cls) for ref, cls in sorted(live.items())}} if live else {}),
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
        raise ClimberLoadError(f"{manifest_path}: {block_error(exc)}") from exc
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
    for ref, cls in climber._live.items():
        # a class of the launching process: its source is kept for the
        # record, but the snapshot cannot rebuild the climber
        copy = target / LIVE / f"{cls.__name__}.py"
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_text(_live_source(cls))
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
    block = {"snapshot": SNAPSHOT_VERSION, **({"portable": False} if climber._live else {}), **spec.block()}
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
            if data.get("portable") is False:
                raise NotPortableError(
                    f"{manifest_path}: this search ran a climber composed from classes that existed "
                    "only in the process that started it (their source is under climber/live/ for "
                    "the record); it cannot be rebuilt — not resumed, not loaded here"
                )
            data = {key: value for key, value in data.items() if key not in ("snapshot", "portable")}
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


# --- composing in Python ---------------------------------------------------------------


def _live_source(cls: type) -> str:
    try:
        return inspect.getsource(cls)
    except (OSError, TypeError):
        return f"# source not available: {cls!r}\n"


def _named(value: Any, kind: str) -> tuple[str | None, dict, type | None]:
    """How a building block given in Python is written in a block:
    (ref, its params, None) — or (None, params, the class) when the class
    exists only in this process. An instance stands for its class and params."""
    if isinstance(value, str):
        return value, {}, None
    cls = value if inspect.isclass(value) else type(value)
    params = {} if inspect.isclass(value) else dict(getattr(value, "params", None) or {})
    spelled = f"{cls.__module__}:{cls.__name__}"
    for name, target in refs.kind_of(kind).load().registry.items():  # 1. a registry name
        if target is cls or target == spelled:
            return name, params, None
    nested = "<locals>" in cls.__qualname__ or "." in cls.__qualname__
    if not nested and cls.__module__ != "__main__" and _importable(cls):
        return spelled, params, None  # 2. importable anywhere this package is
    try:
        source = inspect.getsourcefile(cls)
    except (OSError, TypeError):
        source = None
    if not nested and source and Path(source).is_file():
        return f"{Path(source).resolve()}:{cls.__name__}", params, None  # 3. its file
    return None, params, cls  # 4. only here


def _importable(cls: type) -> bool:
    """Would `module:Class` find this class in a fresh process? Its top-level
    package must be on the import path (not merely in `sys.modules`: a file
    a climber loaded is there too) and the name must lead back to the class."""
    from importlib.machinery import PathFinder

    try:
        if PathFinder.find_spec(cls.__module__.partition(".")[0]) is None:
            return False
    except (ImportError, ValueError):
        return False
    return getattr(sys.modules.get(cls.__module__), cls.__name__, None) is cls


def _compose(*, policy, loop, select, operators, tuner, memory, params, prompts, name) -> tuple[dict, dict[str, type]]:
    """The block — and the classes that exist only in this process — for
    building blocks given as names, classes or instances."""
    if policy is not None and loop is not None:
        raise ClimberLoadError("name one of `policy` (what to try next) or `loop` (the whole control flow), not both")
    block: dict[str, Any] = {}
    live: dict[str, type] = {}
    if name:
        block["name"] = name

    def written(value: Any, kind: str) -> tuple[str, dict]:
        ref, its_params, cls = _named(value, kind)
        if cls is not None:
            ref = f"{LIVE}:{cls.__name__}"
            live[ref] = cls
        return ref, its_params

    def place(slot: str, value: Any, kind: str, params_key: str) -> None:
        if value is None:
            return
        block[slot], its_params = written(value, kind)
        if its_params:
            block[params_key] = its_params

    place("loop" if loop is not None else "policy", loop if loop is not None else policy,
          "loop" if loop is not None else "policy", "params")
    if select is None and policy is not None and not isinstance(policy, (str, type)):
        select = getattr(policy, "_selector", None)  # `Greedy(selector=MapElites())` brings its own
    place("select", select, "select", "select_params")
    place("tuner", tuner, "tuner", "tuner_params")
    place("memory", memory, "memory", "memory_params")
    if operators is not None:
        entries: list = []
        for operator in operators:
            ref, its_params = written(operator, "operator")
            entries.append({ref: its_params} if its_params else ref)
        block["operators"] = entries
    if params:
        block["params"] = {**block.get("params", {}), **dict(params)}
    if prompts is not None:
        block["prompts"] = str(prompts)
    return block, live


# --- helpers ---


def _doc_line(target) -> str:
    """The first line of a class's own docstring, else of its module's."""
    doc = (getattr(target, "__doc__", None) if "__doc__" in vars(target) else None) or (
        getattr(inspect.getmodule(target), "__doc__", None) or ""
    )
    return doc.strip().splitlines()[0] if doc.strip() else ""


def _accepts(target, name: str) -> bool:
    """Does calling `target` take a keyword argument `name`?"""
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def _classes(module, base: type) -> list[type]:
    return [
        obj for obj in vars(module).values()
        if inspect.isclass(obj) and obj.__module__ == module.__name__ and issubclass(obj, base) and obj is not base
    ]
