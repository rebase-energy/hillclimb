"""Standalone bridge from solution.py/submission.json to an Arena verifier.

This file runs inside hillclimb's managed runtime venv, where hillclimb itself
is not installed.  Keep it standard-library-only: downloaded verifiers may
use the runtime's numpy/scipy stack, but the bridge must not add dependencies.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def _load_evaluate(verifier: Path):
    spec = importlib.util.spec_from_file_location("_hillclimb_einstein_verifier", verifier)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load verifier: {verifier}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluate = getattr(module, "evaluate", None)
    if not callable(evaluate):
        raise TypeError(f"{verifier} does not define callable evaluate(data)")
    return evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("solution", type=Path)
    parser.add_argument("--verifier", required=True, type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    result = args.result or Path(os.environ.get("HILLCLIMB_RESULT", "eval_result.json"))

    proc = subprocess.run([sys.executable, str(args.solution.absolute())], check=False)
    if proc.returncode != 0:
        return proc.returncode
    submission = Path("submission.json")
    if not submission.is_file():
        print("Einstein verifier: solution.py did not write submission.json", file=sys.stderr)
        return 3
    try:
        data = json.loads(submission.read_text())
    except (OSError, ValueError) as exc:
        print(f"Einstein verifier: invalid submission.json: {exc}", file=sys.stderr)
        return 4
    if not isinstance(data, dict):
        print("Einstein verifier: submission.json must contain a JSON object", file=sys.stderr)
        return 4
    try:
        score = _load_evaluate(args.verifier.absolute())(data)
    except Exception as exc:  # verifier assertion/errors are candidate failures
        print(f"Einstein verifier failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 5
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        print(f"Einstein verifier returned non-numeric score: {score!r}", file=sys.stderr)
        return 6
    score = float(score)
    if not math.isfinite(score):
        print(f"Einstein verifier returned non-finite score: {score!r}", file=sys.stderr)
        return 6
    result.write_text(
        json.dumps(
            {
                "split": os.environ.get("HILLCLIMB_SPLIT", "validation"),
                "score": score,
                "source": "einsteinarena-verifier",
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
