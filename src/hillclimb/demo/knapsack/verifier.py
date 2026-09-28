"""hillclimb verifier: the evaluator imports solution.py and owns the score.

The Windows edition of verifier.sh, the same step without bash.
`hillclimb problem get` ships it on Windows only. `--holdout` (passed by the
engine for final selection) flows through.
"""

import os
import subprocess
import sys

sys.exit(subprocess.call([os.environ["HILLCLIMB_PYTHON"], "problem/verify.py", *sys.argv[1:]]))
