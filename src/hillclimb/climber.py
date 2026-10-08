"""Climbers: HOW to hillclimb, as one block of config.

A climber is the exchangeable half of a search — a selector policy and an
operator policy (or, for climbers that own their control flow, a `Loop`),
the operators it may use with their prompts, a tuner, a memory. Everything
else is the harness, the same for every climber.

A climber is DEFINED where the run is defined: the `climber:` block of a run
spec entry, or of `runs/config.yaml` as the folder's default. `ClimberSpec` is
that block:

    climber:
      selector_policy: map-elites    # π_sel: which candidate to build on next
      selector_params: {num_islands: 3, num_drafts: 5}
      operator_policy: greedy        # π_op: which operator to use on it; xor `loop:`
      params: {tune_budget: 4}
      operators: [draft, debug, improve]
      operator_params: {draft: {retrieval: false}}
      tuner: optuna
      tuner_params: {seed: 7}
      memory: files
      memory_params: {max_cards: 3, claims: true}
      prompts: prompts/

Every module is named the same three ways (`modules/refs.py`): a registry
name, a `.py` file (`mine.py` or `mine.py:Class`), or `package.module:Class`.
A bare string is shorthand: one `.py` file — the whole `Climber(...)` it
builds, or the single operator policy or loop it defines plus any
`Operator` subclasses in it — or a climber folder (`policy.py` inside).
Defaults live on the classes, so `{operator_policy: mine.py}` is a
complete climber. The engine ships none: `hillclimb climber get greedy`
fetches one from the catalog (`hillclimb.catalog`).

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
from hillclimb.modules.refs import NO_CLIMBER_HINT, ClimberLoadError, FileScope, NoClimber, Resolved
from hillclimb.modules.spec import (  # noqa: F401 — the block's home is modules/spec.py
    ClimberSpec,
    block_error,
    climber_label,
    expand_name,
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

    def kind_of(self, name: str) -> str | None:
        entry = self._entries.get(name)
        return entry[0].kind if entry is not None else None


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
            selector_policy=hc.selectors.MapElites(num_islands=2, num_drafts=3),  # π_sel: which node, or none
            operator_policy=hc.policies.Greedy(),                                 # π_op: which operator for it
            operators=[hc.operators.Draft(retrieval=False), hc.operators.Debug(), hc.operators.Improve()],
            tuner="optuna",
            memory=hc.memory.FilesMemory(max_cards=1),
        )
        hc.run("heilbronn-11", climber=climber, budget="10m")
        climber.to_spec()          # the same climber as the block a run config takes

    A catalog climber's classes come from `hc.catalog.module("greedy")`
    (`Greedy`, `Best`); a bare name like `Climber("greedy")` is refused. An instance stands for its class and params (`Best(num_drafts=3)` is
    `selector_policy: best` + `selector_params: {num_drafts: 3}`): a search always builds its
    own. A class is written down the most portable way it can be — a
    registry name, `module:Class`, else `its_file.py:Class` — and a class
    that exists only in this process (a notebook cell) still runs here, but
    makes the climber not `portable`: `to_spec()`, resume and detached
    engines refuse it, saying why.

    `resolve_climber(block)` (or `Climber.from_spec`) is the same thing
    starting from a block.
    """

    def __init__(self, preset=None, *, selector_policy=None, operator_policy=None, operators=None, tuner=None,
                 memory=None, loop=None, params: Mapping | None = None, selector_params: Mapping | None = None,
                 prompts_dir=None, name: str | None = None, base_dir: Path | None = None,
                 holdout: Mapping | None = None,
                 select=None, policy=None, select_params: Mapping | None = None, prompts=None):
        # the keywords are in the order one step runs them: π_sel, π_op, the
        # operator, a tune trial of the result, the memory the operator reads.
        # `prompts_dir=` names the templates' directory (a `prompts/` beside a
        # climber file needs no naming). `select=`, `policy=` and
        # `select_params=` are the pre-0.7 spellings, `prompts=` the pre-0.9 one.
        if prompts is not None:
            raise TypeError("Climber(): `prompts=` is the old spelling of `prompts_dir=`")
        if select is not None:
            if selector_policy is not None:
                raise TypeError("Climber(): `select=` is the old spelling of `selector_policy=`; give one")
            selector_policy = select
        if policy is not None:
            if operator_policy is not None:
                raise TypeError("Climber(): `policy=` is the old spelling of `operator_policy=`; give one")
            operator_policy = policy
        if select_params:
            selector_params = {**dict(select_params), **dict(selector_params or {})}
        block: dict[str, Any] = {}
        if isinstance(preset, str) and not (refs.is_file_ref(preset) or refs.is_module_ref(preset)) and loop is None:
            # `Climber("greedy")` named a preset once; the bundled climbers are a catalog now
            raise ClimberLoadError(
                f"{preset!r} is not a climber the engine knows: `hc.catalog.climber({preset!r})` loads the catalog's, "
                "or give a .py file"
            )
        if preset is not None:
            if operator_policy is not None:
                raise TypeError(
                    "Climber(): the operator policy is given twice (positionally and as `operator_policy=`)"
                )
            operator_policy = preset  # `Climber(Greedy())`: an operator policy, positionally
        preset_block, block = block, {}
        block, live = _compose(
            operator_policy=operator_policy, loop=loop, selector_policy=selector_policy, operators=operators,
            tuner=tuner, memory=memory, params=params, prompts=prompts_dir, name=name,
        )
        if selector_params:
            block["selector_params"] = {**block.get("selector_params", {}), **dict(selector_params)}
        if holdout:
            # how the best is picked on a problem with a hidden split
            # (selection, top_k, timing): the climber's to say
            block["holdout"] = dict(holdout)
        if preset_block:
            block = {
                **preset_block, **block,
                "params": {**preset_block.get("params", {}), **block.get("params", {})},
                "selector_params": {**preset_block.get("selector_params", {}), **block.get("selector_params", {})},
            }
            for key in ("params", "selector_params"):
                if not block[key]:
                    del block[key]
        self._setup(as_spec(block).anchored(base_dir if base_dir is not None else Path.cwd()), live=live)

    @classmethod
    def from_spec(cls, spec: Any, base_dir: Path | None = None, *, ref: str = "", legacy_names: bool = False) -> Climber:
        """The climber a block defines (relative file refs resolve from
        `base_dir`). `legacy_names`: the block is a RECORD's (a snapshot, a
        run folder) and may name the bundled climbers of before 0.9 by their
        registry names; those resolve to the catalog's files."""
        climber = cls.__new__(cls)
        climber._setup(as_spec(spec, base_dir, legacy=legacy_names).anchored(base_dir), ref=ref, legacy_names=legacy_names)
        return climber

    def _setup(
        self, spec: ClimberSpec, *, ref: str = "", live: Mapping[str, type] | None = None, legacy_names: bool = False
    ) -> None:
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
        self.legacy_names = legacy_names  # a record's block: pre-0.9 names find the catalog
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
        (add `problems:`), or what goes into runs/config.yaml."""
        path = Path(path)
        path.write_text(yaml.safe_dump({"climber": self.to_spec().block()}, sort_keys=False))
        return path

    # --- searching a problem, and stepping through it ---
    #
    # `search` runs one search; `start` opens one to drive by hand. Both go
    # through `hillclimb.api.Search`, which the climber only forwards to, so
    # the climber stays the definition it is: nothing here touches its spec
    # or its identity. What the search found reads back off the climber.

    _search = None  # the search `search` or `start` opened last

    def search(self, problem, **options) -> Climber:
        """Search `problem` for a solution, here, to the end of the budget,
        and return self with what it found on it — the sklearn shape,
        `model.fit(data)`:

            problem = Problem("fitness-landscape")
            budget = Budget(evaluations=30)
            climber = Climber(selector_policy=Best(num_drafts=3), operator_policy=Greedy())
            climber.search(problem, budget=budget)
            climber.best           # the best candidate
            climber.solution       # the source that ships
            climber.history        # every time the best score rose
            climber.to_frame()     # one pandas row per candidate
            climber.result         # the whole SearchOutcome

        `problem` is a `Problem` or a problem id; `options` are what
        `hillclimb.run` takes (`budget` — a `Budget` or "10m" —, `agent`,
        `model`, `learning`, `holdout`, `seed_from`, `name`, `config`, `log`, `hints`)."""
        self.start(problem, **options).finish()
        return self

    def start(self, problem, **options):
        """Open one search on `problem` to drive by hand, and return it.
        Takes what `search` takes:

            climber.start("fitness-landscape", budget=Budget(evaluations=30))
            climber.select()                # π_sel: which node(s), or None
            action = climber.propose()      # π_op on it: the Action
            outcome = climber.run(action)   # run it (or an Action of your own)
            outcome = climber.step()        # both in one call
            climber.finish()                # let the climber run the rest
            result = climber.close()        # the SearchOutcome `search` leaves on the climber

        One search at a time: close this one before starting the next."""
        from hillclimb import api

        if self._search is not None and self._search.outcome is None:
            raise RuntimeError(
                f"this climber is still on search {self._search.ref}: close() it before starting "
                "another (a second Climber can run one alongside only in another process)"
            )
        self._search = api.start(problem, climber=self, **options)
        return self._search

    @property
    def session(self):
        """The `Search` underneath: the one `search` or `start` opened last
        (open, or since closed)."""
        if self._search is None:
            raise RuntimeError("no search yet: climber.search(problem, ...) or climber.start(problem, ...) opens one")
        return self._search

    def select(self):
        """π_sel alone: which node(s) the next attempt would start from in
        the open search (None: a root step); nothing runs."""
        return self.session.select()

    def propose(self):
        """What the climber would do next in the open search: the selector
        policy's node(s), the operator policy's operator on them, as an
        Action; nothing runs."""
        return self.session.propose()

    def run(self, action):
        """Run one action in the open search and return its Outcome."""
        return self.session.run(action)

    def step(self):
        """One move: `run(propose())`. None when there was nothing to run."""
        return self.session.step()

    @property
    def state(self):
        """The `SearchState` the operator policy sees in the open search."""
        return self.session.state

    def finish(self):
        """Let the climber's own loop run the open search to its end."""
        return self.session.finish()

    def close(self):
        """Settle the open search where it stands; returns its SearchOutcome."""
        return self.session.close()

    # --- what the search found ---

    @property
    def result(self):
        """The search as a `SearchOutcome`: settled once it is over, a live
        reading while it is open."""
        return self.session.result

    @property
    def candidates(self):
        """Every candidate of the search so far."""
        search = self.session
        return search.candidates if search.outcome is None else search.outcome.candidates

    @property
    def best(self):
        """The best scored candidate of the search, by validation score."""
        search = self.session
        return search.best if search.outcome is None else search.outcome.best

    @property
    def selected(self):
        """The candidate that ships (the best, unless a holdout split says otherwise)."""
        return self.result.selected

    @property
    def history(self):
        """Every time the best-so-far moved: `(minutes, candidate_id, score)`."""
        return self.result.history

    @property
    def spend(self):
        """Evaluations, tokens, cost and seconds the search used."""
        return self.result.spend

    @property
    def solution(self):
        """The source that ships (`best/solution.py`)."""
        return self.result.solution

    @property
    def params(self):
        """The parameter values the selected candidate ran with."""
        return self.result.params

    def to_frame(self):
        """One pandas row per candidate of the search."""
        return self.result.to_frame()

    def _resolve(self, ref: str, kind: str) -> Resolved:
        if ref in self._live:
            return Resolved(self._live[ref], LIVE, ref)
        return refs.resolve_ref(ref, kind, scope=self.scope, legacy=self.legacy_names)

    @property
    def source(self) -> str:
        """What to call it in a message: its file, else its name."""
        path = refs.ref_path(self.spec.brain)
        return str(path) if path is not None else self.name

    # --- the operator policy or loop ---

    @property
    def brain(self) -> Resolved:
        if self._brain is None:
            ref = self.spec.brain
            kind = "loop" if self.spec.loop is not None else "operator_policy"
            if kind == "operator_policy" and refs.is_file_ref(ref) and not refs.split_file_ref(ref)[1]:
                # a bare file: it is a loop climber when a Loop is what it defines
                module = self.scope.import_file(refs.ref_path(ref), "climber")
                explicit = getattr(module, refs.KINDS["operator_policy"].attr, None)
                if explicit is None and _classes(module, Loop):
                    kind = "loop"
            resolved = self._resolve(ref, kind)
            if kind == "operator_policy" and inspect.isclass(resolved.target) and issubclass(resolved.target, Loop):
                kind = "loop"
            self._brain, self._brain_kind = resolved, kind
        return self._brain

    @property
    def is_loop(self) -> bool:
        return self.brain is not None and self._brain_kind == "loop"

    @property
    def description(self) -> str:
        """One line on what it does: what the operator policy or loop says
        about itself — and, for an operator policy over a selector policy the
        block names, what that selector policy picks."""
        text = _doc_line(self.brain.target)
        if not self.is_loop and (self.spec.selector_policy or self._own_selector() is not None):
            own = self._own_selector()
            if own is not None:
                return f"{self._label(self.spec.brain, 'operator_policy')} over {own.__name__}: {_doc_line(own)}"
            ref = self._selector_ref()
            picks = _doc_line(self._resolve(ref, "selector_policy").target)
            return f"{self._label(self.spec.brain, 'operator_policy')} over {self._label(ref, 'selector_policy')}: {picks}"
        return text

    def _label(self, ref: str, kind: str) -> str:
        """What to call a module in a sentence: its registry name, else the
        class's own name (a file's stem says nothing when one file holds both
        policies)."""
        resolved = self._resolve(ref, kind)
        if resolved.form == "name":
            return resolved.ref
        return getattr(resolved.target, "__name__", None) or resolved.label

    @property
    def holdout_timing(self) -> str | None:
        """When the hidden split is scored: the block's `holdout.timing`,
        else `after` when the loop's state must never meet a holdout value
        (its class says so), else what an old record said; None = inline."""
        if self.spec.holdout is not None and self.spec.holdout.timing:
            return self.spec.holdout.timing
        return getattr(self.brain.target, "holdout_timing", None) or self.legacy_holdout_timing

    @property
    def prompts_dir(self) -> Path | None:
        if not self.spec.prompts:
            return None
        path = Path(self.spec.prompts)
        return path if path.is_dir() else None

    @property
    def module(self):
        """The module a one-file climber IS, when it was named by a file:
        its classes to subclass or compose with (`hc.catalog.module("greedy")
        .Greedy`). None for a climber built from registry names."""
        return self.brain.module

    # --- building the modules ---

    def resolved_params(self, overlay: Mapping | None = None) -> dict:
        """The block's params with an overlay on top."""
        return {**self.spec.params, **dict(overlay or {})}

    def _known_params(self) -> dict | None:
        """The knobs the operator policy/loop declares (`OperatorPolicy.defaults()`), or None
        when it takes free-form params."""
        declared = getattr(self.brain.target, "defaults", None)
        return dict(declared()) if callable(declared) else None

    def _own_selector(self):
        """The selector policy the operator policy's own file defines, when
        the block names none: `SELECTOR = <class>`, else the one
        `SelectorPolicy` subclass written there (one file, both policies —
        what `hillclimb climber get` writes and a meta-problem's candidate
        is). None when the brain is not a file, or the file defines none."""
        if self.is_loop or self.spec.selector_policy:
            return None
        from hillclimb.modules.selectors.base import SelectorPolicy

        module = self.brain.module
        if module is not None:
            explicit = getattr(module, refs.KINDS["selector_policy"].attr, None)
            if explicit is not None:
                return explicit
            found = _classes(module, SelectorPolicy)
            if len(found) > 1:
                raise ClimberLoadError(
                    f"{self.source}: defines {len(found)} selector policies ({', '.join(c.__name__ for c in found)}); "
                    f"set SELECTOR = <class>, or name one in `selector_policy:`"
                )
            if found:
                return found[0]
        # a policy that subclasses one written in a climber file (`class Mine(
        # hc.catalog.module("greedy").Greedy)`) inherits the selector written
        # beside its base: the two halves of that file go together
        target = self.brain.target
        for base in (inspect.getmro(target)[1:] if inspect.isclass(target) else ()):
            base_module = sys.modules.get(base.__module__)
            if base_module is None or not base.__module__.startswith(refs.SCOPE_PACKAGE_PREFIX):
                continue
            explicit = getattr(base_module, refs.KINDS["selector_policy"].attr, None)
            found = [explicit] if explicit is not None else _classes(base_module, SelectorPolicy)
            if found:
                return found[0]
        return None

    def _selector_ref(self) -> str | None:
        """How the selector policy (π_sel) is named: the block's
        `selector_policy:`, else the one its own file defines (as
        `<file>:<Class>`), else the operator policy class's `default_selector`.
        None for a loop, and for an operator policy that names none."""
        if self.is_loop:
            return None
        if self.spec.selector_policy:
            return self.spec.selector_policy
        own = self._own_selector()
        if own is not None:
            try:
                return f"{Path(inspect.getsourcefile(own)).resolve()}:{own.__name__}"
            except TypeError:
                return f"{self.brain.path}:{own.__name__}"
        return getattr(self.brain.target, "default_selector", None)

    def _selector_target(self):
        """The selector policy's class (or factory), or None — see `_selector_ref`."""
        own = self._own_selector()
        if own is not None:
            return own
        ref = self._selector_ref()
        return None if ref is None else self._resolve(ref, "selector_policy").target

    def selector(self, extra: Mapping | None = None):
        """The selector policy (π_sel) the loop asks first (`_selector_ref`:
        the block's, the file's own, the class's default). None for a loop,
        and for an operator policy that names none. `extra` lays more
        settings over the block's `selector_params` (the schedule knobs a
        caller handed `build_loop`)."""
        from hillclimb.modules.selectors import get_selector

        if self.is_loop:
            return None
        params = {**self.spec.selector_params, **dict(extra or {})}
        own = self._own_selector()
        if own is not None:
            # the file's own class, built directly: it is already imported with
            # the brain, in the brain's package, wherever the file sits
            selector = refs.construct(own, {"params": params}, self.source)
            if not getattr(selector, "name", None):
                selector.name = own.__name__
            return selector
        ref = self._selector_ref()
        if ref is None:
            if params:
                raise ClimberLoadError(
                    f"{self.source}: `selector_params` without a selector policy to give them to"
                )
            return None
        if ref in self._live:
            return self._live[ref](params)
        return get_selector(ref, params, scope=self.scope, legacy=self.legacy_names)

    def build_loop(self, *, params: Mapping | None = None, priors: Mapping | None = None,
                   parallelism: int = 1, log=print) -> Loop:
        """The loop this climber runs: its `loop:`, or a `PolicyLoop` over
        its `operator_policy:` (with its selector policy). `priors` are param
        values memory learned (a draft-complexity offset): they sit UNDER the
        block's params and reach only an operator policy that declares the
        knob. `params` is an overlay for callers that hold a resolved climber."""
        from hillclimb.modules.spec import SCHEDULE_KNOBS

        target = self.brain.target
        known = self._known_params()
        merged = self.resolved_params(params)
        # the schedule is the selector policy's: knobs handed to the operator
        # policy (every block before 0.7, a caller's overlay, a tuned
        # `--set climber.params.<knob>`) reach it there — the fixed schedule
        # names, plus whatever the selector policy class itself declares
        selector_target = None if self.is_loop else self._selector_target()
        declared = getattr(selector_target, "defaults", None)
        selector_known = set(declared()) if callable(declared) else set()
        schedule = {
            k: merged.pop(k) for k in list(merged)
            if (k in SCHEDULE_KNOBS or k in selector_known) and not (known and k in known)
        }
        if known is not None:
            if getattr(target, "strict_params", False):
                unknown = sorted(set(merged) - set(known))
                if unknown:
                    raise ClimberLoadError(
                        f"climber.params: {climber_label(self.spec.brain)} has no param {', '.join(map(repr, unknown))} "
                        f"(it has: {', '.join(sorted(known))}). A selector policy's settings go in `selector_params`."
                    )
            merged = {**{k: v for k, v in (priors or {}).items() if k in known}, **merged}
        offered = {"params": merged, "parallelism": parallelism, "log": log}
        if self.is_loop:
            loop = refs.construct(target, offered, self.source)
            if not isinstance(loop, Loop):
                raise ClimberLoadError(f"{self.source}: `loop:` must name a Loop, got {type(loop).__name__}")
            return loop
        selector = self.selector(schedule)
        if selector is not None:
            named = self.spec.selector_policy is not None or self._own_selector() is not None
            if named and not _accepts(target, "selector"):
                raise ClimberLoadError(
                    f"{self.source}: `selector_policy: {self._selector_ref()}` — this operator policy "
                    "takes no selector policy (its constructor has no `selector` argument)"
                )
            offered["selector"] = selector
        policy = refs.construct(target, offered, self.source)
        for method in ("propose", "observe"):
            if not callable(getattr(policy, method, None)):
                raise ClimberLoadError(
                    f"{self.source}: the operator policy has no {method}() — not an OperatorPolicy"
                )
        if not getattr(policy, "name", None):
            policy.name = self.name
        if getattr(policy, "params", None) is None:
            policy.params = dict(merged)
        return PolicyLoop(policy)

    def operator_set(self) -> OperatorSet:
        """The operators this climber's search may use. The block's list;
        else what the operator policy/loop declares (`operators = (...)` on the class);
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


def as_spec(value: Any, base_dir: Path | None = None, *, legacy: bool = False) -> ClimberSpec:
    """A `ClimberSpec` from whatever named it: a spec, a block, a bare
    string (a file, a climber folder looked for under `base_dir`), or a
    composed `Climber` (which must be portable). None names nothing: the
    folder has no climber (`NoClimber`, which says how to fetch one).
    `legacy`: a record's value, which may name a preset of before 0.9."""
    if value is None:
        raise NoClimber(NO_CLIMBER_HINT)
    if isinstance(value, ClimberSpec):
        return value
    if isinstance(value, Climber):
        return value.to_spec()
    try:
        return ClimberSpec.model_validate(value, context={"base_dir": base_dir, "legacy": legacy})
    except ValueError as exc:
        raise ClimberLoadError(block_error(exc)) from exc


def resolve_climber(spec: Any, base_dir: Path | None = None, *, ref: str = "", legacy_names: bool = False) -> Climber:
    """The `Climber` a block defines. Relative file refs resolve from
    `base_dir`. Every failure is a `ClimberLoadError` naming the file and the
    fix — never a traceback from deep inside engine start-up. The modules
    themselves are imported when they are first asked for. `legacy_names`:
    the block is a record's (see `Climber.from_spec`)."""
    if isinstance(spec, Climber):
        return spec
    return Climber.from_spec(spec, base_dir, ref=ref, legacy_names=legacy_names)


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
    # the hashed block keeps the key spellings identity was first taken with
    # (0.7 renamed `policy`/`select`/`select_params`), so a search's recorded
    # identity survives the rename and a resume does not call it a change
    from hillclimb.modules.spec import RENAMED_BLOCK_KEYS

    spelled_as_hashed = {new: old for old, new in RENAMED_BLOCK_KEYS.items()}
    block = {
        spelled_as_hashed.get(key, key): value
        for key, value in spec.canonical().map_refs(portable).model_dump(
            # `holdout` came in 0.9: a climber without one keeps the identity it had
            exclude={"name", "prompts", *(() if spec.holdout else ("holdout",))}
        ).items()
    }
    blob = json.dumps(
        {
            "climber": block,
            "files": scope.digest if scope.files else None,
            "prompts": tree_sha256(prompts) if prompts is not None and prompts.is_dir() else None,
            **({"live": {ref: _live_source(cls) for ref, cls in sorted(live.items())}} if live else {}),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def load_climber(ref: str, base_dir: Path | None = None) -> Climber:
    """The climber a STRING names: one `.py` file, or a climber
    folder — a directory holding `policy.py` (`hillclimb climber get` writes
    one: the file builds the whole climber, the folder names it) or a
    pre-0.9 `climber.yaml` (the pre-0.6 manifest form reads the same way)."""
    path = Path(refs.split_file_ref(ref)[0]).expanduser() if not refs.is_module_ref(ref) else None
    if path is not None and not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    if path is not None and path.is_dir():
        if not (path / MANIFEST).is_file() and (path / FOLDER_POLICY).is_file():
            return resolve_climber(ref, base_dir, ref=ref)  # `spec.folder_block`: the folder's policy.py
        return _load_climber_dir(ref, path)
    if refs.is_file_ref(ref) or refs.is_module_ref(ref):
        return resolve_climber(ref, base_dir, ref=ref)
    from hillclimb.catalog import climber_names

    raise ClimberLoadError(
        f"Unknown climber: {ref} (one .py file, or a climber folder holding {FOLDER_POLICY}; "
        f"the catalog's — {', '.join(climber_names()) or 'none'} — are fetched with `hillclimb climber get <name>`)"
    )


# --- the pre-0.6 manifest ------------------------------------------------------------


def spec_from_legacy_manifest(data: Mapping[str, Any], name: str) -> tuple[ClimberSpec, str | None]:
    """A folder's `climber.yaml` as a block — the 0.4/0.5 manifest, or the
    block a climber folder holds. `description` and `similarity` carried
    nothing a search used and are dropped; `holdout_timing` is handed back
    beside the spec (a loop declares it on its class now)."""
    data = dict(data)
    data.setdefault("name", name)  # a snapshot dir is always called `climber`
    for dropped in ("description", "similarity"):
        data.pop(dropped, None)
    timing = data.pop("holdout_timing", None)
    if timing not in (None, "after"):
        raise ValueError(f"holdout_timing: {timing!r} (only `after` was ever accepted)")
    brain = data.get("operator_policy") if data.get("operator_policy") is not None else data.get("policy")
    if (brain is None) == (data.get("loop") is None):
        raise ValueError(
            "name exactly one of `operator_policy:` (the operator policy) or `loop:` (the whole control flow)"
        )
    return ClimberSpec.model_validate(data), timing


def _load_climber_dir(ref: str, root: Path, name: str | None = None) -> Climber:
    """A climber folder: its manifest resolved with the folder as the base of
    every relative file ref and of `prompts:`."""
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        raise ClimberLoadError(f"{root} holds no {MANIFEST}")
    try:
        data = yaml.safe_load(manifest_path.read_text()) or {}
        spec, timing = spec_from_legacy_manifest(data, name or root.name)
    except Exception as exc:  # noqa: BLE001
        raise ClimberLoadError(f"{manifest_path}: {block_error(exc)}") from exc
    try:
        climber = resolve_climber(spec, root, ref=ref, legacy_names=True)  # a manifest is a record
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
                return resolve_climber(data, root, ref=str(root), legacy_names=True)  # a record: pre-0.9 names resolve
            except ClimberLoadError as exc:
                raise ClimberLoadError(f"{manifest_path}: {exc}") from exc
        return _load_climber_dir(str(root), root, name=name)
    files = sorted(root.glob("*.py"))
    if len(files) != 1:
        raise ClimberLoadError(f"{root} is neither a climber snapshot nor a one-file climber")
    return resolve_climber(str(files[0]), ref=str(files[0]))


# --- climber folders --------------------------------------------------------------------
# `hillclimb climber get <name>` copies a catalog climber (`catalog.install_climber`):
# the folder IS a climber — its `policy.py` builds the whole `Climber(...)` —
# and, when the catalog folder brought no prompts/ of its own, gets one here:
# every template its operators render, with a README of what fills them.

FOLDER_POLICY = "policy.py"
FOLDER_README = "README.md"


def operator_templates(climber: Climber) -> list[str]:
    """The templates the climber's operators render (`Operator.templates`),
    in operator order, each once — what its prompts/ holds when it is a folder."""
    operators = climber.operator_set()
    seen: dict[str, None] = {}
    for name in operators.names():
        for template in operators.get(name).templates:
            seen.setdefault(template, None)
    return list(seen)


def write_prompts_folder(climber: Climber, target: Path, name: str) -> list[str]:
    """Write `target/` (a climber's prompts dir): the built-in templates its
    operators render, byte for byte, plus README.md. Returns the files
    written, relative to `target`. Refuses when a template an operator
    renders is not a built-in one."""
    from hillclimb.prompts.render import TEMPLATE_DIR, TEMPLATE_SUFFIX

    templates = operator_templates(climber)
    missing = [t for t in templates if not (TEMPLATE_DIR / f"{t}{TEMPLATE_SUFFIX}").is_file()]
    if missing:
        raise ClimberLoadError(f"{climber.name}: no built-in template {', '.join(missing)}")
    target.mkdir(parents=True)
    written: list[str] = []
    for template in templates:
        shutil.copy2(TEMPLATE_DIR / f"{template}{TEMPLATE_SUFFIX}", target / f"{template}{TEMPLATE_SUFFIX}")
        written.append(f"{template}{TEMPLATE_SUFFIX}")
    (target / FOLDER_README).write_text(prompts_guide(climber, name, templates))
    written.append(FOLDER_README)
    return written


def _spelled_out(climber: Climber) -> tuple[dict, dict]:
    """(params, selector_params) with every default filled in, read off the
    classes' `DEFAULTS` with the block's overrides on top — no selector is
    built, so a copy needs no optional extra to be written."""
    known = climber._known_params() or {}
    params = {**known, **climber.spec.params}
    declared = getattr(climber._selector_target(), "defaults", None)
    selector_params = {**(declared() if callable(declared) else {}), **climber.spec.selector_params}
    return params, selector_params


def prompts_guide(climber: Climber, name: str, templates: list[str]) -> str:
    """prompts/README.md: how a prompt is made from a template, when each
    template is used (the climber's schedule, numbers filled in), and what
    fills every token the templates carry (`operators.builtin.TOKEN_GUIDE`)."""
    from hillclimb.modules.operators.builtin import TOKEN_GUIDE
    from hillclimb.prompts.render import TEMPLATE_DIR, TEMPLATE_SUFFIX, tokens_in

    params, selector_params = _spelled_out(climber)
    operators = climber.operator_set()
    policy_cls = climber.brain.target
    selector_cls = climber._selector_target()
    lines = [
        f"# The prompts of the `{name}` climber",
        "",
        "Each file here is a template: the words are the climber's, the `{{tokens}}` are",
        "context the harness fills in for one attempt — the problem, the candidate the",
        "attempt builds on, what earlier attempts tried, what memory knows from other",
        "searches. `../policy.py` is the climber: its selector policy decides which",
        "candidate the next attempt starts from, its operator policy which operator",
        "makes it; the operator renders its template; the harness adds the contract.",
        "Edit a template and the next run climbs with it (`hillclimb climber check`",
        "lints the tokens). A template deleted here falls back to the built-in one.",
        "",
        "## When each template is used",
        "",
    ]
    lines += _schedule_lines(policy_cls, selector_cls, params, selector_params)
    lines += ["", "| operator | kind | template(s) | what it does |", "| --- | --- | --- | --- |"]
    for op_name in operators.names():
        operator = operators.get(op_name)
        doc = (inspect.getdoc(type(operator)) or "").split("\n\n")[0].replace("\n", " ").strip()
        shown = ", ".join(f"`{t}{TEMPLATE_SUFFIX}`" for t in operator.templates) or "—"
        lines.append(f"| `{op_name}` | {operator.kind} | {shown} | {doc} |")
    lines += [
        "",
        "A tune job (the policy's `tune_budget`) re-runs a candidate's own code with other",
        "`params.json` values; no coding agent, no prompt.",
        "",
        "## What fills the tokens",
        "",
        "| token | filled with | empty when |",
        "| --- | --- | --- |",
    ]
    seen: dict[str, None] = {}
    for template in templates:
        text = (TEMPLATE_DIR / f"{template}{TEMPLATE_SUFFIX}").read_text()
        for token in sorted(tokens_in(text), key=text.index):
            seen.setdefault(token, None)
    seen.pop("contract", None)
    seen["contract"] = None  # the harness's, last
    for token in seen:
        filled, empty = TOKEN_GUIDE[token]
        lines.append(f"| `{{{{{token}}}}}` | {filled} | {empty} |")
    lines += [
        "",
        "## What stays the harness's",
        "",
        "`{{contract}}` is filled from the harness's own templates (`contract_*.md`,",
        "`holdout_clause.md`, `params_cue.md`, …): how the solution is run and scored.",
        "They are the same for every climber and cannot be shadowed from here; a",
        "candidate's `prompt.md` under its run shows the whole prompt as it was sent.",
        "",
    ]
    return "\n".join(lines)


def _schedule_lines(policy_cls, selector_cls, params: dict, selector_params: dict) -> list[str]:
    """The climber's schedule in words. The greedy shape — a selector with
    the bundled schedule knobs over an operator policy that tunes — is
    spelled out with its numbers; anything else gets the two docstrings and
    the resolved knobs, which is what it decides from."""
    doc = (inspect.getdoc(policy_cls) or "").split("\n\n")[0].replace("\n", " ").strip()
    picks = (inspect.getdoc(selector_cls) or "").split("\n\n")[0].replace("\n", " ").strip() if selector_cls else ""
    selector_name = getattr(selector_cls, "__name__", "none")
    head = (
        f"The selector policy, `{selector_name}`, and the operator policy, `{policy_cls.__name__}` "
        f"(both in `../policy.py`): {picks} {doc}".rstrip()
    )
    knobs = f"Knobs (the classes' `DEFAULTS`, as resolved): `selector_params: {json.dumps(selector_params)}`, `params: {json.dumps(params)}`."
    schedule_knobs = {"num_drafts", "max_debug_depth", "ensemble_reserve_fraction", "ensemble_top_k", "debug", "ensemble"}
    if callable(getattr(policy_cls, "tune_now", None)) and schedule_knobs <= set(selector_params) and "tune_budget" in params:
        drafts = selector_params["num_drafts"]
        depth = selector_params["max_debug_depth"]
        reserve = selector_params["ensemble_reserve_fraction"]
        top_k = selector_params["ensemble_top_k"]
        tune = params["tune_budget"]
        return [
            f"{head} In order, at every step:",
            "",
            f"1. a candidate that failed gets `debug.md` (`debug: {json.dumps(selector_params['debug'])}`, "
            f"up to {depth} fixes deep);",
            f"2. in the last {reserve:.0%} of the budget, the top {top_k} get combined with `ensemble.md`"
            f" (`ensemble: {json.dumps(selector_params['ensemble'])}`);",
            f"3. until {drafts} roots are scored, a fresh `draft.md` (`num_drafts`);",
            f"4. a scored candidate that declared `params.json` is tuned first ({tune} extra trials, `tune_budget`);",
            f"5. otherwise the candidate `{selector_name}` selects gets `improve.md`: {picks}".rstrip(":"),
            "",
            knobs,
        ]
    return [head, "", knobs]


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


def _compose(*, operator_policy, loop, selector_policy, operators, tuner, memory, params, prompts,
             name) -> tuple[dict, dict[str, type]]:
    """The block — and the classes that exist only in this process — for
    building blocks given as names, classes or instances."""
    if operator_policy is not None and loop is not None:
        raise ClimberLoadError(
            "name one of `operator_policy` (which operator to apply next) or `loop` (the whole control flow), not both"
        )
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

    place("loop" if loop is not None else "operator_policy", loop if loop is not None else operator_policy,
          "loop" if loop is not None else "operator_policy", "params")
    if selector_policy is None and operator_policy is not None and not isinstance(operator_policy, (str, type)):
        selector_policy = getattr(operator_policy, "_selector", None)  # `Greedy(selector=MapElites())` brings its own
    place("selector_policy", selector_policy, "selector_policy", "selector_params")
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
