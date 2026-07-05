from __future__ import annotations

from hillclimb.backends.base import OperatorBackend, OperatorRequest, OperatorResult


def get_backend(name: str, auth: str = "subscription") -> OperatorBackend:
    if name == "claude-code":
        from hillclimb.backends.claude_code import ClaudeCodeBackend

        return ClaudeCodeBackend(auth=auth)
    if name == "dummy":
        from hillclimb.backends.dummy import DummyBackend

        return DummyBackend()
    raise ValueError(f"Unknown backend: {name}")


__all__ = ["OperatorBackend", "OperatorRequest", "OperatorResult", "get_backend"]
