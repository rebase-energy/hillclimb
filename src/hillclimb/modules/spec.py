"""The `climber:` block — `ClimberSpec` — and the presets a bare name stands for.

Kept apart from `hillclimb.climber` (which resolves a block into classes) so
the config schema can hold one without importing the harness: this module
needs only `modules/refs.py`, the memory kinds and pydantic.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hillclimb.modules import refs
from hillclimb.modules.memory.base import MemoryKind

DEFAULT_POLICY = "greedy"

# a bare name stands for one of these blocks
PRESETS: dict[str, dict[str, Any]] = {
    "greedy": {"operator_policy": "greedy"},
    # quality-diversity search: greedy's schedule over the MAP-Elites
    # selector policy, without the moves OpenEvolve has no counterpart for
    "openevolve": {
        "name": "openevolve", "operator_policy": "greedy", "selector_policy": "map-elites",
        "selector_params": {"ensemble": False}, "params": {"tune_budget": 0},
    },
    "gepa": {"loop": "gepa"},
}

# The two decisions were `policy:` and `select:` (its knobs `select_params:`)
# until 0.7; they are named after the RSI framework now — the selector policy
# (π_sel) picks the node, the operator policy (π_op) picks the operator — and a
# block, a `--set climber.…`, a snapshot or a record in either spelling loads
# (the new spelling wins on a clash).
RENAMED_BLOCK_KEYS = {
    "policy": "operator_policy",
    "select": "selector_policy",
    "select_params": "selector_params",
}


def renamed_block_keys(data: dict) -> dict:
    """The block with the pre-0.7 key spellings moved to the current ones."""
    if not any(old in data for old in RENAMED_BLOCK_KEYS):
        return data
    data = dict(data)
    for old, new in RENAMED_BLOCK_KEYS.items():
        if old in data:
            value = data.pop(old)
            if new not in data:
                data[new] = value
            elif isinstance(value, dict) and isinstance(data[new], dict):
                data[new] = {**value, **data[new]}
    return data

# keys a pre-0.6 manifest carried that a block does not
_GONE = {
    "description": "a block has no `description`: say it in a YAML comment",
    "similarity": "similarity scores are a viewer's setting (`similarity.scores` in hillclimb.yaml), not a climber's",
    "holdout_timing": "a loop declares it on its class (`holdout_timing = \"after\"`); otherwise it is the user's `holdout.timing`",
}


# the schedule — which node, or none — is the SELECTOR POLICY's (π_sel) since
# 0.7; a block from before wrote these knobs under the operator policy's `params`
SCHEDULE_KNOBS = (
    "num_drafts", "max_debug_depth", "debug",
    "ensemble", "ensemble_reserve_fraction", "ensemble_top_k", "ensemble_max_attempts",
)
# what a policy's params could hold in 0.5 that was the schedule's; in an
# `openevolve` block, anything else was a MAP-Elites setting
_OPENEVOLVE_SCHEDULE_KNOBS = (
    *SCHEDULE_KNOBS, "complexity_start", "tune_budget", "tune_gate", "tune_parallel", "tune_burst",
)


def schedule_to_selector(data: dict) -> dict:
    """A block that sets the schedule under `params` (how every block did
    before 0.7) means the same schedule under `selector_params`, where the
    selector policy reads it. `selector_params` wins where both say something."""
    params = data.get("params")
    if not isinstance(params, dict) or not any(key in params for key in SCHEDULE_KNOBS):
        return data
    data = dict(data)
    moved = {key: value for key, value in params.items() if key in SCHEDULE_KNOBS}
    kept = {key: value for key, value in params.items() if key not in SCHEDULE_KNOBS}
    if kept:
        data["params"] = kept
    else:
        del data["params"]
    data["selector_params"] = {**moved, **(data.get("selector_params") or {})}
    return data


def block_from_05(data: dict) -> dict:
    """The `climber:` block hillclimb.yaml held in 0.4/0.5 named a climber
    (`ref:`) and what the user laid over it; `operators:` was that overlay,
    a mapping by operator name; `graph:` sat beside `memory:`. Same settings,
    older shape: the `ref` is expanded and the rest laid over it."""
    if "graph" in data:
        # the graph module is a setting of the memory it indexes
        data = dict(data)
        graph = data.pop("graph")
        if graph is not None:
            data["memory_params"] = {"graph": graph, **(data.get("memory_params") or {})}
    if "ref" not in data and not isinstance(data.get("operators"), dict):
        return data
    data = dict(data)
    ref = data.pop("ref", None)
    base = expand_name(ref) if ref is not None else {}
    overlay = data.pop("operators", None)
    merged = dict(base)
    for key, value in data.items():
        if value is None:  # 0.5 wrote None for "whatever the climber says"
            continue
        merged[key] = {**base.get(key, {}), **value} if key == "params" and isinstance(value, dict) else value
    if ref == "openevolve" and isinstance(data.get("params"), dict):
        # 0.5's openevolve policy kept MAP-Elites' settings among its own
        # params; they are the selector policy's now
        theirs = {k: v for k, v in data["params"].items() if k not in _OPENEVOLVE_SCHEDULE_KNOBS}
        merged["params"] = {k: v for k, v in merged["params"].items() if k not in theirs}
        merged["selector_params"] = {**theirs, **(merged.get("selector_params") or {})}
    if isinstance(overlay, dict):
        if overlay:
            merged["operator_params"] = {**overlay, **(merged.get("operator_params") or {})}
    elif overlay is not None:
        merged["operators"] = overlay
    return merged


def climber_label(ref: str) -> str:
    """A short display name for a climber named by a string: a preset's name
    as it is, a one-file climber's stem."""
    if refs.is_file_ref(ref):
        return Path(refs.split_file_ref(ref)[0]).stem
    return ref.rpartition(":")[2] if refs.is_module_ref(ref) else ref


def presets() -> list[str]:
    return sorted(PRESETS)


def expand_name(ref: str) -> dict[str, Any]:
    """The block a bare string stands for: a preset, or one file / one class."""
    if ref in PRESETS:
        return json.loads(json.dumps(PRESETS[ref]))  # a copy nobody can edit the preset through
    if refs.is_file_ref(ref) or refs.is_module_ref(ref):
        return {"operator_policy": ref}  # a file that defines a Loop is recognised when it is resolved
    raise ValueError(
        f"Unknown climber: {ref} (presets: {', '.join(presets())}; or one .py file; or a full "
        "`climber:` block). A directory holding climber.yaml is the pre-0.6 form: "
        f"`hillclimb climber show {ref}` prints it as a block."
    )


class ClimberSpec(BaseModel):
    """The `climber:` block — the same shape in a run spec and in hillclimb.yaml."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None  # a label (experiment names, the watch column); not part of identity
    # exactly one of the two: the operator policy (π_op: which operator to
    # apply to what the selector policy chose), or the whole control flow
    operator_policy: str | None = None
    loop: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)  # the operator policy's knobs
    # the selector policy (π_sel): which candidate the next attempt starts
    # from, or none (`best` unless the operator policy says otherwise;
    # `map-elites` for a quality-diversity archive). Only with `operator_policy:`
    selector_policy: str | None = None
    selector_params: dict[str, Any] = Field(default_factory=dict)
    # which operators: names or refs, each optionally with params
    # (`- draft: {retrieval: true}`). None = what the policy/loop declares,
    # else the built-in four — plus any Operator its own file defines
    operators: list[str | dict[str, dict[str, Any] | None]] | None = None
    # params by operator NAME, laid over the list's: what `--set
    # climber.operators.draft.retrieval=false` edits
    operator_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tuner: str = "random"
    tuner_params: dict[str, Any] = Field(default_factory=dict)
    # what the search knows from other searches and leaves for the next:
    # `files` (the knowledge/ directory), `none`, a file, or module:Class.
    # Its settings — for `files`: max_cards, claims, skills, the `graph`
    # module that indexes it, ... — are `memory_params`
    memory: MemoryKind = "files"  # `knowledge-graph` (pre-0.4) still loads
    memory_params: dict[str, Any] = Field(default_factory=dict)
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
                "`routing` is reserved: which coding agent and model run is the user's "
                "choice (hillclimb.yaml `routing:`), never a climber's"
            )
        for key, advice in _GONE.items():
            if key in data:
                raise ValueError(f"`{key}`: {advice}")
        return schedule_to_selector(renamed_block_keys(block_from_05(data)))

    @model_validator(mode="after")
    def _one_brain(self) -> ClimberSpec:
        if self.operator_policy is not None and self.loop is not None:
            raise ValueError(
                "name one of `operator_policy:` (which operator to apply next) or `loop:` "
                "(the whole control flow), not both"
            )
        if self.operator_policy is None and self.loop is None:
            self.operator_policy = DEFAULT_POLICY
        if self.loop is not None and (self.selector_policy is not None or self.selector_params):
            raise ValueError(
                "`selector_policy:` picks what an operator policy expands; a `loop:` does its own selection"
            )
        return self

    # --- reading it ---

    @property
    def brain(self) -> str:
        return self.loop if self.loop is not None else self.operator_policy

    @property
    def label(self) -> str:
        """What views call it: its `name`, else the operator policy's or loop's."""
        return self.name or climber_label(self.brain)

    def operator_items(self) -> list[tuple[str, dict[str, Any]]] | None:
        """`operators:` as (ref, params) pairs; None when the block leaves them to the operator policy/loop."""
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
        return [
            self.brain, *([self.selector_policy] if self.selector_policy else []), self.tuner, self.memory,
            *([self.graph] if self.graph else []),
            *(ref for ref, _ in self.operator_items() or []),
        ]

    @property
    def graph(self) -> str | None:
        """The graph module the block names (a setting of its memory), when it names one."""
        ref = self.memory_params.get("graph")
        return ref if isinstance(ref, str) else None

    def map_refs(self, change: Callable[[str], str]) -> ClimberSpec:
        """A copy with every module ref passed through `change`."""
        update: dict[str, Any] = {
            "loop" if self.loop is not None else "operator_policy": change(self.brain),
            "tuner": change(self.tuner),
            "memory": change(self.memory),
        }
        if self.graph is not None:
            update["memory_params"] = {**self.memory_params, "graph": change(self.graph)}
        if self.selector_policy is not None:
            update["selector_policy"] = change(self.selector_policy)
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
        """The block as it is written down: every module it names (defaults
        included, so it reads whole), without the keys that say nothing — a
        None, an empty mapping of params — and with the schedule's knobs where
        the selector policy reads them, however they were set."""
        data = schedule_to_selector(self.model_dump(exclude_none=True))
        return {key: value for key, value in data.items() if value != {}}

    def canonical(self) -> ClimberSpec:
        """The same block, normalized: what identity and a snapshot are taken of."""
        return ClimberSpec.model_validate(self.block())


def block_error(exc: Exception) -> str:
    """What is wrong with a block, without pydantic's header: the message an author can act on."""
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    lines = []
    for error in errors():
        where = ".".join(str(part) for part in error.get("loc", ()))
        message = error.get("msg", "").removeprefix("Value error, ")
        lines.append(f"{where}: {message}" if where else message)
    return "; ".join(lines) or str(exc)
