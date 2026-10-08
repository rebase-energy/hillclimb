"""hillclimb verifier: run the candidate, then score what it produced.

The Windows edition of this problem's verifier.sh, the same steps without
bash. `hillclimb problem get` writes it as verifier.py on Windows only.
The solution is never imported here: it runs as its own process, so it can
touch neither the scorer nor the score.
"""

import os
import subprocess
import sys
from pathlib import Path

PYTHON = os.environ["HILLCLIMB_PYTHON"]


def run(*args: str) -> None:
    """One step; a failing step ends the verifier with its exit code (bash's `set -e`)."""
    code = subprocess.call([PYTHON, *args])
    if code != 0:
        sys.exit(code)


run(os.environ["HILLCLIMB_SOLUTION"])

# Only the trusted scorer may write the result consumed by hillclimb.
Path(os.environ["HILLCLIMB_RESULT"]).unlink(missing_ok=True)
run("problem/verify.py")
