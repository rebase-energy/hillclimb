from __future__ import annotations

from pathlib import Path
from typing import Callable

from hillclimb.agents.base import Agent, AgentRequest, AgentResult


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


def _make_toy(auth: str) -> Agent:
    from hillclimb.agents.toy import ToyAgent

    return ToyAgent()


_AGENTS: dict[str, Callable[[str], Agent]] = {
    "claude-code": _make_claude_code,
    "codex": _make_codex,
    "dummy": _make_dummy,
    "pi": _make_pi,
    "toy": _make_toy,
}
# the names every hillclimb process knows; anything else was registered here
_BUILTIN = frozenset(_AGENTS)


def register_agent(name: str, factory: Callable[[], Agent], *, replace: bool = False) -> None:
    """Make `factory()` — a class or function returning an object with
    `name` and `invoke(request) -> AgentResult` — the agent called `name`,
    in THIS process: `hc.run(..., agent=name)` and `climber.start(...,
    agent=name)` find it, a detached engine does not. A search builds its
    own instance per route."""
    if name in _BUILTIN:
        raise ValueError(f"{name!r} is a built-in agent: register yours under another name")
    if name in _AGENTS and not replace:
        raise ValueError(f"agent {name!r} is already registered (pass replace=True to swap it)")
    if not callable(factory):
        raise TypeError(f"register_agent({name!r}, ...) takes a class or a function returning the agent")
    _AGENTS[name] = lambda auth: factory()


def agent_names() -> tuple[str, ...]:
    return tuple(sorted(_AGENTS))


def is_builtin(name: str) -> bool:
    return name in _BUILTIN


# the CLI each built-in coding agent drives, and how to get it
AGENT_CLIS = {"claude-code": "claude", "codex": "codex", "pi": "pi"}


class AgentCLIMissing(RuntimeError):
    """A coding agent whose CLI is not installed: found before a search
    detaches, not three failed operator calls into it. The CLI prints it as
    one line."""


def require_agent_clis(names) -> None:
    """Raise AgentCLIMissing for the first named agent whose CLI is not on PATH."""
    import shutil

    from hillclimb.connect import INSTALL_HINT

    for name in dict.fromkeys(names):
        binary = AGENT_CLIS.get(name)
        if binary and shutil.which(binary) is None:
            raise AgentCLIMissing(
                f"the {name} coding agent needs the `{binary}` CLI, which is not on PATH. "
                f"Install it ({INSTALL_HINT[binary]}), then `hillclimb connect {binary}`; "
                "or pick another with --agent"
            )


def get_agent(
    name: str,
    auth: str = "subscription",
    *,
    pi_models_file: Path | None = None,
) -> Agent:
    if name not in _AGENTS:
        raise ValueError(
            f"Unknown agent: {name} (available: {', '.join(agent_names())}; an agent registered "
            "with register_agent exists only in the process that registered it)"
        )
    if name == "pi":
        return _make_pi(auth, pi_models_file)
    return _AGENTS[name](auth)


__all__ = [
    "Agent", "AgentRequest", "AgentResult", "agent_names", "get_agent", "is_builtin", "register_agent",
]
