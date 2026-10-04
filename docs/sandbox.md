# The sandbox

hillclimb starts two kinds of process that run code nobody has read: the
coding agents, which work with their permission prompts off, and the verifier, which
runs the `solution.py` a coding agent wrote. Both run as you. The sandbox is what
keeps a mistake, or a prompt injection, from costing you anything outside
the search.

It is on by default and needs no container:

| OS | what enforces it | install |
|---|---|---|
| macOS | `sandbox-exec` (Seatbelt) | nothing, it is part of macOS |
| Linux | [bubblewrap](https://github.com/containers/bubblewrap) | `sudo apt install bubblewrap` (`dnf`, `pacman` alike) |
| Windows | bubblewrap inside WSL2 (a real Linux kernel) | in WSL2's Ubuntu: `sudo apt install bubblewrap`; natively none exists, hillclimb runs unsandboxed and says so, pointing to WSL2. `scripts/wsl2-check.sh` checks a machine |

`hillclimb connect` shows whether the sandbox starts on this machine, and
`hillclimb sandbox check` shows that it holds: it runs a script that behaves
like a hostile solution inside the sandbox a search uses (writes outside its
folder, reads of `~/.ssh` and the journal, connections out, a signal to a
process outside) and lists what happened to every attempt. Exit code 1 when
one got through; `--json` for scripts.

## What a sandboxed process can do

| | allowed | blocked |
|---|---|---|
| write | its candidate's folder, the temp dirs, the caches of `huggingface`, `torch` and `matplotlib` | everything else: your home folder, the repository, the search's journal and `best/`, the runtime venvs, and always the problem's folder and data (even inside a temp dir) |
| read | most of the disk | `~/.ssh`, `~/.aws`, `~/.gnupg`, cloud and GitHub logins, browser profiles, shell history, `.env` files, the other coding agents' logins, the search's journal and its holdout runs, container runtime sockets, and the problem's `private:` paths (only its scorer reads those, see `docs/problems.md`) |
| network, coding agents | everything, or with `allow_internet_for_agents: false` only their model provider | the rest |
| network, verifier | nothing, or with the problem's `allow_internet_during_solution: true` everything | the rest, name lookups included |
| processes | its own children | signalling anything outside |

A coding agent additionally writes to its own operator home under
`~/.cache/hillclimb/` (Claude Code's `claude-home`, pi's isolated home and
session folder). Your own `~/.claude`, `~/.claude.json`, `~/.codex` and
`~/.agents` are unreadable to every process a search starts (see
`docs/agents.md`, Operator homes).

## Per coding agent

| coding agent | how it is confined |
|---|---|
| `claude-code` | the whole process runs inside the sandbox |
| `pi` | the whole process runs inside the sandbox |
| `codex` | by its own sandbox (`--sandbox workspace-write`): writes stay in the candidate's folder and its commands have no network. hillclimb's sandbox is not put round it, because macOS allows no sandbox inside a sandbox. Codex can therefore still read files the table above lists as blocked |
| `dummy` | runs no model and no tools |

## Coding agents without internet

```yaml
allow_internet_for_agents: false
```

The coding agents then reach their model provider and nothing else. Everything
they and their tools send goes through a small proxy inside the engine that
lets HTTPS through to the provider's hosts and refuses the rest; the engine's
log names each host it refused. Claude Code's web search and fetch and its
MCP servers are off, codex's web search is off, and the draft prompt drops
its web-research cue. This needs the sandbox: with `sandbox: off`, or on
Windows, a search with claude-code or pi does not start.

## Settings

```yaml
sandbox: off              # run everything unsandboxed

sandbox:                  # or adjust it
  write: [scratch]        # more writable paths, relative to the hillclimb dir
  deny_read: [~/private]  # more unreadable paths
  allow_hosts: [bedrock-runtime.eu-north-1.amazonaws.com]   # more hosts for coding agents without internet
  local_ports: [8000]     # localhost ports left open when the network is off
```

`HILLCLIMB_SANDBOX=off` in the environment switches it off for one command,
which is what a CI container that cannot start one needs.

## When it does not start

A search does not start when the sandbox is on and cannot be started; the
message says why.

- **bubblewrap is not installed**: install it with your package manager.
- **Ubuntu 24.04 and later** keep the kernel feature bubblewrap uses behind
  AppArmor: `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`.
- **Inside a container**, the container has to allow user namespaces. If it
  does not, the container is the sandbox: set `sandbox: off`.
- **Inside another sandbox** on macOS, for example when a coding agent with
  its own sandbox starts hillclimb: run hillclimb from a plain terminal, or
  set `sandbox: off` and rely on the outer one.

## What it is not

The sandbox is the operating system's process confinement, not a virtual
machine. It stops writes outside the candidate's folder, reads of your keys
and traffic to hosts you did not allow. It does not protect against a flaw
in the kernel, and these stay open:

- A coding agent can read its own login, because it needs it.
- A sandboxed process can still read most of your files. Add what is private
  to `sandbox.deny_read`.
- A coding agent without internet still talks to its model provider, so what it
  can read can reach that provider.
- A meta-problem's verifier is not sandboxed; the searches it starts are.
- A benchmark provider's once-per-search official grading runs unsandboxed.
