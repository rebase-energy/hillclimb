from __future__ import annotations

import threading

from hillclimb.backends.base import OperatorRequest, OperatorResult


class FakeBackend:
    """Test-only backend: replays a queue of scripted responses and records
    every request so tests can assert on prompts and call order.

    Thread-safe. Responses queued with `operator=` are matched to requests of
    that operator (first match wins); unkeyed responses stay strict FIFO —
    ordering-sensitive serial tests keep working unchanged."""

    name = "fake"

    def __init__(self, responses: list[dict] | None = None):
        self.responses = list(responses or [])
        self.requests: list[OperatorRequest] = []
        self._lock = threading.Lock()

    def queue(
        self,
        script: str | None = None,
        notes: str = "",
        result: dict | None = None,
        operator: str | None = None,
        **result_kwargs,
    ) -> None:
        merged = {**(result or {}), **result_kwargs}
        self.responses.append(
            {"script": script, "notes": notes, "result": merged, "operator": operator}
        )

    def _pop_response(self, request: OperatorRequest) -> dict:
        for index, response in enumerate(self.responses):
            if response.get("operator") in (None, request.operator):
                return self.responses.pop(index)
        raise AssertionError(
            f"FakeBackend queue exhausted (no response for operator {request.operator!r})"
        )

    def invoke(self, request: OperatorRequest) -> OperatorResult:
        with self._lock:
            self.requests.append(request)
            response = self._pop_response(request)
        if response.get("script") is not None:
            (request.workspace / "solution.py").write_text(response["script"])
        if response.get("notes"):
            (request.workspace / "notes.md").write_text(response["notes"])
        return OperatorResult(**{"ok": True, **response.get("result", {})})


class GateBackend(FakeBackend):
    """FakeBackend whose invokes block until released — the workhorse for
    concurrency tests. Each invoke registers an Event in `gates` (indexed by
    arrival order) and waits on it; `release(i)` lets call i proceed.
    Honors an abort event like the real backend."""

    def __init__(self, responses: list[dict] | None = None, abort: threading.Event | None = None):
        super().__init__(responses)
        self.abort = abort
        self.gates: list[threading.Event] = []
        self.started = threading.Semaphore(0)  # released once per arrived invoke

    def release(self, index: int) -> None:
        self.gates[index].set()

    def release_all(self) -> None:
        for gate in self.gates:
            gate.set()

    def invoke(self, request: OperatorRequest) -> OperatorResult:
        with self._lock:
            gate = threading.Event()
            self.gates.append(gate)
        self.started.release()
        while not gate.wait(timeout=0.05):
            if self.abort is not None and self.abort.is_set():
                return OperatorResult(
                    ok=False, error_kind="aborted", error_message="aborted in gate"
                )
        return super().invoke(request)
