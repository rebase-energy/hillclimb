from __future__ import annotations

from hillclimb.backends.base import OperatorRequest, OperatorResult


class FakeBackend:
    """Test-only backend: replays a queue of scripted responses and records
    every request so tests can assert on prompts and call order."""

    name = "fake"

    def __init__(self, responses: list[dict] | None = None):
        self.responses = list(responses or [])
        self.requests: list[OperatorRequest] = []

    def queue(
        self,
        script: str | None = None,
        notes: str = "",
        result: dict | None = None,
        **result_kwargs,
    ) -> None:
        merged = {**(result or {}), **result_kwargs}
        self.responses.append({"script": script, "notes": notes, "result": merged})

    def invoke(self, request: OperatorRequest) -> OperatorResult:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("FakeBackend queue exhausted")
        response = self.responses.pop(0)
        if response.get("script") is not None:
            (request.workspace / "solution.py").write_text(response["script"])
        if response.get("notes"):
            (request.workspace / "notes.md").write_text(response["notes"])
        return OperatorResult(**{"ok": True, **response.get("result", {})})
