#!/usr/bin/env bash
# hillclimb verifier: the evaluator imports solution.py and owns the score.
# `--holdout` (passed by the engine for final selection) flows through.
set -euo pipefail
exec "$HILLCLIMB_PYTHON" problem/verify.py "$@"
