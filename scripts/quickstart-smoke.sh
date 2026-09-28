#!/usr/bin/env bash
# The website quickstart, step by step, with the dummy agent (no login, no
# tokens). Run from an empty folder with hillclimb on PATH. `--backend` is
# the spelling every version accepts (newer ones also take `--agent`); a CLI
# that detaches `run` by default is kept in the foreground so summit sees
# the finished search.
set -euxo pipefail
hillclimb --help
hillclimb init
hillclimb problem get heilbronn-convex-13
# the verifier is written for this OS: verifier.py on Windows (no bash
# needed to score), verifier.sh everywhere else
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) want=verifier.py other=verifier.sh ;;
  *) want=verifier.sh other=verifier.py ;;
esac
test -f "problems/heilbronn-convex-13/$want"
test ! -e "problems/heilbronn-convex-13/$other"
hillclimb verify heilbronn-convex-13
run_help=$(COLUMNS=200 hillclimb run --help)
extra=()
if grep -q -- --no-detach <<<"$run_help"; then extra+=(--no-detach); fi
hillclimb run heilbronn-convex-13 --budget 20s --backend dummy ${extra[@]+"${extra[@]}"}
hillclimb summit
test -f solution.py
