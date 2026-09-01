"""The agentic GEPA proposer: gepa asks for a mutation of solution.py, a
routed hillclimb coding agent performs it in a scratch dir under
SEARCH_DIR/gepa/proposals/. The edited source is the proposal — the agent
communicates by files (the invoke_knowledge_agent precedent), no response
text is needed."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from hillclimb.backends.base import OperatorRequest, OperatorResult
from hillclimb.budget import BudgetManager
from hillclimb.candidate import BackendInfo
from hillclimb.config import Config
from hillclimb.problem import ProblemSpec
from hillclimb.routing import BackendPool, Router
from hillclimb.slots import MachineSlots

COMPONENT = "solution.py"
FEEDBACK_CAP = 16_384  # bytes of serialized reflective feedback


def source_hash(source: str) -> str:
    """Stable identity of a proposal: sha256 over newline-normalized text."""
    normalized = "\n".join(source.splitlines()).strip() + "\n"
    return hashlib.sha256(normalized.encode()).hexdigest()


class ProposerError(Exception):
    """The backend produced no usable solution.py; carries the accounting of
    the failed call so its burn is still journaled."""

    def __init__(self, message: str, result: OperatorResult | None = None):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class ProposalRecord:
    """What the evaluator needs to journal a proposal's lineage and cost."""

    hash: str
    parent_hash: str
    backend: BackendInfo
    scratch_dir: str


class ProposalBridge:
    """Hash-keyed handoff from proposer to evaluator (lock-guarded so a
    future parallel engine cannot corrupt it)."""

    def __init__(self) -> None:
        self._records: dict[str, ProposalRecord] = {}
        self._lock = threading.Lock()

    def record(self, record: ProposalRecord) -> None:
        with self._lock:
            self._records[record.hash] = record

    def pop(self, key: str) -> ProposalRecord | None:
        with self._lock:
            return self._records.pop(key, None)


PROMPT_TEMPLATE = """\
# Improve solution.py

You are one mutation step of an automated search. The current `solution.py`
in this directory is a working solution to the problem described in
`./problem/description.md` (data in `./data/`). Your job: edit `solution.py`
IN PLACE so it scores better.

Objective: {metric} — {direction}.

## Evaluation feedback on the current solution

{feedback}

## Rules

- Edit `solution.py` in place; it must remain a self-contained script that
  writes the same outputs the verifier expects.
- Inspect the current code and the problem/data files before changing anything.
- Make one focused, concrete improvement addressing the feedback above.
- Do not invent, read, or optimize against holdout data — only the
  validation evaluation above exists for you.
{extra}
Remaining search budget: {remaining}.
"""


class GEPAProposer:
    """gepa's custom_candidate_proposer: (candidate, reflective_dataset,
    components_to_update) -> {"solution.py": new_source}."""

    def __init__(
        self,
        *,
        problem: ProblemSpec,
        config: Config,
        router: Router,
        backends: BackendPool,
        budget: BudgetManager,
        search_dir: Path,
        bridge: ProposalBridge,
        checkpoint: Callable[[], None],
        abort: threading.Event | None = None,
        slots: MachineSlots | None = None,
        knowledge_context: str | None = None,
        reference_note: str = "",
        on_phase: Callable[[str], None] | None = None,
        on_failure: Callable[[ProposerError], None] | None = None,
        log=print,
    ):
        self.problem = problem
        self.config = config
        self.router = router
        self.backends = backends
        self.budget = budget
        self.search_dir = search_dir
        self.bridge = bridge
        self.checkpoint = checkpoint
        self.abort = abort
        self.slots = slots
        self.knowledge_context = knowledge_context
        self.reference_note = reference_note
        self.on_phase = on_phase or (lambda phase: None)
        self.on_failure = on_failure or (lambda exc: None)
        self.log = log
        self._seq = 0

    def __call__(self, candidate, reflective_dataset, components_to_update) -> dict[str, str]:
        try:
            return self._propose(candidate, reflective_dataset, components_to_update)
        except ProposerError as exc:
            # gepa swallows proposer exceptions and loops on — the owner's
            # on_failure hook is where burn accounting and the consecutive-
            # failure park rule live
            self.on_failure(exc)
            raise

    def _propose(self, candidate, reflective_dataset, components_to_update) -> dict[str, str]:
        if set(components_to_update) != {COMPONENT} or set(candidate) != {COMPONENT}:
            raise ProposerError(
                f"GEPA MVP mutates only {COMPONENT}; got components={list(components_to_update)}"
            )
        self.checkpoint()  # control queue, wall budget, cost ceiling
        parent_source = candidate[COMPONENT]
        scratch = self._scratch_dir()
        self._materialize(scratch, parent_source, reflective_dataset)
        route = self.router.resolve("gepa")
        backend = self.backends.get(route.backend, route.backend_auth)
        request = OperatorRequest(
            operator="improve",
            prompt=self._prompt(reflective_dataset),
            candidate_dir=scratch,
            timeout_s=min(
                self.config.budget.agent_timeout_s, max(60, int(self.budget.remaining()))
            ),
            model=route.model,
        )
        (scratch / "prompt.md").write_text(request.prompt)
        slot = None
        if self.slots is not None:
            slot = self.slots.acquire(
                abort=self.abort,
                should_stop=self.budget.should_stop,
                on_wait=lambda: self.on_phase("waiting-slot"),
            )
            if slot is None and self.slots.limit > 0:
                raise ProposerError("aborted waiting for a machine agent slot")
        self.on_phase("agent")
        try:
            result = backend.invoke(request)
        finally:
            if slot is not None:
                slot.release()
        info = BackendInfo(
            name=route.backend,
            model=route.model,
            model_id=result.model_id,
            session_id=result.session_id,
            cost_usd=result.cost_usd,
            num_turns=result.num_turns,
            total_tokens=result.total_tokens,
            token_usage=result.token_usage,
            quota_start=result.quota_start,
            quota_end=result.quota_end,
            agent_duration_s=result.duration_s,
            error_kind=result.error_kind,
        )
        if not result.ok:
            raise ProposerError(
                f"agent call failed ({result.error_kind}): {result.error_message[:200]}",
                result=result,
            )
        new_source = self._read_back(scratch, parent_source, result)
        record = ProposalRecord(
            hash=source_hash(new_source),
            parent_hash=source_hash(parent_source),
            backend=info,
            scratch_dir=str(scratch),
        )
        self.bridge.record(record)
        return {COMPONENT: new_source}

    # --- pieces ---

    def _scratch_dir(self) -> Path:
        root = self.search_dir / "gepa" / "proposals"
        root.mkdir(parents=True, exist_ok=True)
        while True:
            self._seq += 1
            scratch = root / f"p{self._seq:04d}"
            try:
                scratch.mkdir()
            except FileExistsError:
                continue  # resumed search: skip past prior proposals
            return scratch

    def _materialize(self, scratch: Path, parent_source: str, reflective_dataset) -> None:
        (scratch / COMPONENT).write_text(parent_source)
        for link_name, target in (("data", self.problem.data_dir), ("problem", self.problem.problem_dir)):
            link = scratch / link_name
            if not link.exists():
                link.symlink_to(target.resolve(), target_is_directory=True)
        (scratch / "feedback.json").write_text(self._feedback_json(reflective_dataset))

    def _feedback_json(self, reflective_dataset) -> str:
        try:
            text = json.dumps(reflective_dataset, indent=2, default=str)
        except (TypeError, ValueError):
            text = json.dumps({"feedback": str(reflective_dataset)[:FEEDBACK_CAP]})
        return text[:FEEDBACK_CAP]

    def _prompt(self, reflective_dataset) -> str:
        extra = ""
        if self.knowledge_context:
            extra += f"\n## Prior experience\n\n{self.knowledge_context}\n"
        if self.reference_note:
            extra += f"\n{self.reference_note}\n"
        direction = "higher is better" if self.problem.higher_is_better else "lower is better"
        return PROMPT_TEMPLATE.format(
            metric=self.problem.metric_name,
            direction=direction,
            feedback=self._feedback_json(reflective_dataset),
            extra=extra,
            remaining=self.budget.remaining_str(),
        )

    def _read_back(self, scratch: Path, parent_source: str, result: OperatorResult) -> str:
        solution = scratch / COMPONENT
        if not solution.exists():
            raise ProposerError("agent produced no solution.py", result=result)
        if scratch.resolve() not in solution.resolve().parents:
            raise ProposerError("solution.py escaped the proposal dir", result=result)
        new_source = solution.read_text()
        if not new_source.strip():
            raise ProposerError("agent produced an empty solution.py", result=result)
        if source_hash(new_source) == source_hash(parent_source):
            raise ProposerError("agent returned the parent source unchanged", result=result)
        return new_source
