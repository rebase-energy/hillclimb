"""Evaluate a submission module on one split of an emflow problem.

Runs inside the emflow runtime venv (stdlib + emflow only) — invoked BY PATH
by the orchestrator, never imported; hillclimb itself is not installed there.

usage: eval_runner.py SOLUTION_PY --problem NAME [--split validation|holdout]
                      [--result-json eval_result.json]
                      [--verify] [--name NAME] [--metadata-json '{"n_trials": 12}']

Prints `val_score: <float>` as the final stdout line (the orchestrator's
score contract) and writes a result JSON next to the cwd. --verify runs the
official emflow Verifier instead (scorecard + leaderboard row).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("solution")
    ap.add_argument("--problem", required=True)
    ap.add_argument("--split", default="validation", choices=["validation", "holdout"])
    ap.add_argument("--result-json", default="eval_result.json")
    ap.add_argument("--verify", action="store_true",
                    help="official Verifier run (scorecard + leaderboard row)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--metadata-json", default=None)
    args = ap.parse_args()

    solution = Path(args.solution).absolute()
    # ensemble candidates import their inputs as modules (candidate_1.py, ...)
    sys.path.insert(0, str(solution.parent))

    import emflow as ef

    if args.verify:
        from emflow.run.verifier import Verifier

        model = ef.load_submission(solution)
        metadata = json.loads(args.metadata_json) if args.metadata_json else None
        verifier = Verifier(problem=args.problem, split=args.split)
        result = verifier.verify(model, name=args.name or solution.stem, metadata=metadata)
    else:
        result = ef.evaluate(args.problem, solution, split=args.split)

    payload = {
        "problem": args.problem,
        "split": args.split,
        "objective": result.objective,
        "score": result.score,
        "n_origins": result.n_origins,
        "n_scored": result.n_scored,
        "model": result.model,
    }
    Path(args.result_json).write_text(json.dumps(payload, indent=2))
    print(f"val_score: {result.score}")


if __name__ == "__main__":
    main()
