from __future__ import annotations

from pathlib import Path
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


def _make_pi(auth: str, models_file: Path | None = None) -> OperatorBackend:
    from hillclimb.backends.pi_cli import PiCliBackend

    return PiCliBackend(auth=auth, models_file=models_file)


_BACKENDS: dict[str, Callable[[str], OperatorBackend]] = {
    "claude-code": _make_claude_code,
    "codex": _make_codex,
    "dummy": _make_dummy,
    "pi": _make_pi,
}


def get_backend(
    name: str,
    auth: str = "subscription",
    *,
    pi_models_file: Path | None = None,
) -> OperatorBackend:
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend: {name} (available: {', '.join(sorted(_BACKENDS))})")
    if name == "pi":
        return _make_pi(auth, pi_models_file)
    return _BACKENDS[name](auth)


__all__ = ["OperatorBackend", "OperatorRequest", "OperatorResult", "get_backend"]
