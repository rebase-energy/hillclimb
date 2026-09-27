from __future__ import annotations

from pathlib import Path
from typing import Callable

from hillclimb.agents.base import Agent, OperatorRequest, OperatorResult


def _make_claude_code(auth: str) -> Agent:
    from hillclimb.agents.claude_code import ClaudeCodeAgent

    return ClaudeCodeAgent(auth=auth)


def _make_dummy(auth: str) -> Agent:
    from hillclimb.agents.dummy import DummyAgent

    return DummyAgent()


def _make_codex(auth: str) -> Agent:
    from hillclimb.agents.codex_cli import CodexCliAgent

    return CodexCliAgent(auth=auth)


def _make_pi(auth: str, models_file: Path | None = None) -> Agent:
    from hillclimb.agents.pi_cli import PiCliAgent

    return PiCliAgent(auth=auth, models_file=models_file)


_AGENTS: dict[str, Callable[[str], Agent]] = {
    "claude-code": _make_claude_code,
    "codex": _make_codex,
    "dummy": _make_dummy,
    "pi": _make_pi,
}


def get_agent(
    name: str,
    auth: str = "subscription",
    *,
    pi_models_file: Path | None = None,
) -> Agent:
    if name not in _AGENTS:
        raise ValueError(f"Unknown agent: {name} (available: {', '.join(sorted(_AGENTS))})")
    if name == "pi":
        return _make_pi(auth, pi_models_file)
    return _AGENTS[name](auth)


__all__ = ["Agent", "OperatorRequest", "OperatorResult", "get_agent"]
