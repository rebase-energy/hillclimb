#!/usr/bin/env bash
# Does the sandbox hold on this machine? Run from an empty folder with
# hillclimb on PATH, on an OS that has a sandbox (macOS, Linux, WSL2). With
# the dummy agent: no login, no tokens. Exits 1 at the first thing that is
# not as it should be. CI runs it on Linux and macOS; scripts/wsl2-check.sh
# runs it on Windows (WSL2).
set -euo pipefail

say() { printf '\n### %s\n' "$*"; }
die() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

say "sandbox check: a hostile script run inside the sandbox"
hillclimb sandbox check || die "hillclimb sandbox check: an attempt got through, or no sandbox starts here"

say "a problem with hidden holdout data (knapsack), scored on both splits"
hillclimb init >/dev/null
hillclimb problem get knapsack >/dev/null
scores=$(hillclimb verify knapsack --holdout 2>&1) || die "hillclimb verify knapsack: $scores"
echo "$scores" | tail -2
grep -q "97.4241" <<<"$scores" || die "knapsack validation score is not 97.4241"
grep -q "97.5494" <<<"$scores" || die "knapsack holdout score is not 97.5494"

say "a hostile solution in a real search: every attempt must be blocked"
cat > hostile.py <<'PY'
import os, pathlib, socket, sys
attempts = [
    ("read the holdout data", lambda: open("problem/holdout/seed.txt").read()),
    ("change the scorer", lambda: open("problem/verify.py", "a").close()),
    ("write to the home folder", lambda: open(os.path.expanduser("~/hillclimb-escaped.txt"), "w").close()),
    ("reach the internet", lambda: socket.create_connection(("1.1.1.1", 443), timeout=3)),
]
for label, attempt in attempts:
    try:
        attempt()
        print(f"ALLOWED: {label}", file=sys.stderr)
    except Exception as exc:
        print(f"BLOCKED: {label} ({type(exc).__name__})", file=sys.stderr)
def select_items(items, capacity):
    return []
PY
hillclimb run knapsack --agent dummy --set budget.max_evaluations=1 --seed-from hostile.py \
  --no-holdout --no-learning --no-detach >run.log 2>&1 || { cat run.log; die "the search did not finish"; }
log=$(ls runs/*/searches/*/candidates/c001/exec_stderr.log)
grep -E "^(ALLOWED|BLOCKED):" "$log" | sort
[ "$(grep -c '^BLOCKED:' "$log")" = 4 ] || die "the hostile solution got through (see above)"
! grep -q '^ALLOWED:' "$log" || die "the hostile solution got through (see above)"
test ! -e "$HOME/hillclimb-escaped.txt" || { rm -f "$HOME/hillclimb-escaped.txt"; die "a file was written to the home folder"; }

say "PASS: the sandbox holds here"
