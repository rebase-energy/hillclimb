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
from hillclimb.modules.memory.base import DEFAULT_GRAPH, MemoryKind

DEFAULT_POLICY = "greedy"

# a bare name stands for one of these blocks
PRESETS: dict[str, dict[str, Any]] = {
    "greedy": {"policy": "greedy"},
    # quality-diversity search: greedy's schedule over the MAP-Elites
    # selector, without the moves OpenEvolve has no counterpart for
    "openevolve": {
        "name": "openevolve", "policy": "greedy", "select": "map-elites",
        "params": {"ensemble": False, "tune_budget": 0},
    },
    "gepa": {"loop": "gepa"},
}

# keys a pre-0.6 manifest carried that a block does not
_GONE = {
    "description": "a block has no `description`: say it in a YAML comment",
    "similarity": "similarity scores are a viewer's setting (`similarity.scores` in hillclimb.yaml), not a climber's",
    "holdout_timing": "a loop declares it on its class (`holdout_timing = \"after\"`); otherwise it is the user's `holdout.timing`",
}


# what a policy's params could hold in 0.5 that was the schedule's; in an
# `openevolve` block, anything else was a MAP-Elites setting
_OPENEVOLVE_SCHEDULE_KNOBS = (
    "num_drafts", "max_debug_depth", "debug", "complexity_start",
    "ensemble", "ensemble_reserve_fraction", "ensemble_top_k", "ensemble_max_attempts",
    "tune_budget", "tune_gate", "tune_parallel", "tune_burst",
)


def _from_05_config(data: dict) -> dict:
    """The `climber:` block hillclimb.yaml held in 0.4/0.5 named a climber
    (`ref:`) and what the user laid over it; `operators:` was that overlay,
    a mapping by operator name. Same settings, older shape: the `ref` is
    expanded and the rest laid over it."""
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
        # params; they are the selector's now
        theirs = {k: v for k, v in data["params"].items() if k not in _OPENEVOLVE_SCHEDULE_KNOBS}
        merged["params"] = {k: v for k, v in merged["params"].items() if k not in theirs}
        merged["select_params"] = {**theirs, **(merged.get("select_params") or {})}
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
    # which candidate a policy expands next (`best` unless the policy says
    # otherwise; `map-elites` for a quality-diversity archive). Only with `policy:`
    select: str | None = None
    select_params: dict[str, Any] = Field(default_factory=dict)
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
        return _from_05_config(data)

    @model_validator(mode="after")
    def _one_brain(self) -> ClimberSpec:
        if self.policy is not None and self.loop is not None:
            raise ValueError("name one of `policy:` (what to try next) or `loop:` (the whole control flow), not both")
        if self.policy is None and self.loop is None:
            self.policy = DEFAULT_POLICY
        if self.loop is not None and (self.select is not None or self.select_params):
            raise ValueError("`select:` picks what a policy expands; a `loop:` does its own selection")
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
        return [
            self.brain, *([self.select] if self.select else []), self.tuner, self.graph,
            *(ref for ref, _ in self.operator_items() or []),
        ]

    def map_refs(self, change: Callable[[str], str]) -> ClimberSpec:
        """A copy with every module ref passed through `change`."""
        update: dict[str, Any] = {
            "loop" if self.loop is not None else "policy": change(self.brain),
            "tuner": change(self.tuner),
            "graph": change(self.graph),
        }
        if self.select is not None:
            update["select"] = change(self.select)
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
        None, an empty mapping of params."""
        return {key: value for key, value in self.model_dump(exclude_none=True).items() if value != {}}


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
