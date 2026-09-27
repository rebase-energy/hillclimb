#!/usr/bin/env bash
# hillclimb verifier for a meta-problem: the candidate is a climber, so the
# engine's own interpreter runs inner searches with it and reports the gap
# they closed. Never $HILLCLIMB_PYTHON — the runtime venv has no hillclimb —
# and without the verifier's PYTHONPATH, which is that venv's `hillclimb.spaces`
# shim and would shadow the real package.
set -euo pipefail

env -u PYTHONPATH "$HILLCLIMB_ENGINE_PYTHON" -m hillclimb.cli meta evaluate \
  --climber "$HILLCLIMB_SOLUTION" --spec problem/meta.yaml --result "$HILLCLIMB_RESULT"
