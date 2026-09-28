#!/usr/bin/env bash
# The website quickstart, step by step, with the dummy agent (no login, no
# tokens). Run from an empty folder with hillclimb on PATH.
set -euxo pipefail
hillclimb --help
hillclimb init
hillclimb problem get heilbronn-convex-13
hillclimb verify heilbronn-convex-13
hillclimb run heilbronn-convex-13 --budget 20s --agent dummy --parallel-agents 2 --no-detach
hillclimb summit
test -f solution.py
