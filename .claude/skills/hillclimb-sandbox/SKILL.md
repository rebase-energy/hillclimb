---
name: hillclimb-sandbox
description: Run `sandbox check` to see what gets blocked locally.
category: security
---

Run `hillclimb sandbox check` to verify the sandbox starts on your platform and to see what it blocks locally. Control network access separately: `allow_internet_for_agents: false` (default true) blocks agents except to their model provider (routed through a local AllowlistProxy); `allow_internet_during_solution: true` in the problem's YAML enables verifier network (default disabled).

Sandbox is on by default and uses sandbox-exec on macOS, bubblewrap on Linux. Disable with `sandbox: off` if needed. The preflight check at search start fails fast if the sandbox is unavailable on the platform.

See `docs/sandbox.md` and `harness/sandbox.py` for the full enforcement table, platform-specific notes, and backend implementations.