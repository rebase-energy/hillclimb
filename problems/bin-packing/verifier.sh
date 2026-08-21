#!/usr/bin/env bash
# hillclimb verifier: the evaluator drives solution.py itself and scores it.
# `--holdout` (passed by the engine for the hidden split) flows straight through.
set -euo pipefail
exec "$HILLCLIMB_PYTHON" problem/evaluate.py "$@"
