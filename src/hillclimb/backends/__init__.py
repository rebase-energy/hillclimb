from __future__ import annotations

from typing import Callable

from hillclimb.backends.base import OperatorBackend, OperatorRequest, OperatorResult


def _make_claude_code(auth: str) -> OperatorBackend:
    from hillclimb.backends.claude_code import ClaudeCodeBackend

    return ClaudeCodeBackend(auth=auth)


def _make_dummy(auth: str) -> OperatorBackend:
    from hillclimb.backends.dummy import DummyBackend

    return DummyBackend()


def _make_codex(auth: str) -> OperatorBackend:
    from hillclimb.backends.codex_cli import CodexCliBackend

    return CodexCliBackend(auth=auth)


_BACKENDS: dict[str, Callable[[str], OperatorBackend]] = {
    "claude-code": _make_claude_code,
    "codex": _make_codex,
    "dummy": _make_dummy,
}


def get_backend(name: str, auth: str = "subscription") -> OperatorBackend:
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend: {name} (available: {', '.join(sorted(_BACKENDS))})")
    return _BACKENDS[name](auth)


__all__ = ["OperatorBackend", "OperatorRequest", "OperatorResult", "get_backend"]
