"""Draw a solution with its problem's `plot.py` and save the figure.

Standalone by construction, like run_solution.py: it runs inside the problem's
runtime venv (the verifier's interpreter, with the problem's packages and
matplotlib), which has no hillclimb installed. `hillclimb plot` and
`hillclimb summit --plot` start it.

The problem's contract:

    def plot(solution_dir, ax):       # a pathlib.Path, and matplotlib Axes
        ...draw the solution's output files (submission.csv, …) on ax...
        return "11 points · smallest triangle 0.037037"   # optional caption

It reads what the solution WROTE, never runs solution.py. The caption goes
under the title and is echoed on stdout as `caption: <text>` for the CLI.

    plot_solution.py <plot.py> <solution_dir> <out.png> <title>
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    plot_path, solution_dir, out, title = Path(argv[0]), Path(argv[1]), Path(argv[2]), argv[3]
    import matplotlib

    matplotlib.use("Agg")  # a file, never a window: the CLI opens the PNG
    import matplotlib.pyplot as plt

    spec = importlib.util.spec_from_file_location("problem_plot", plot_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    draw = getattr(module, "plot", None)
    if not callable(draw):
        print(f"{plot_path} defines no plot(solution_dir, ax) function", file=sys.stderr)
        return 2
    fig, ax = plt.subplots(figsize=(7, 7))
    caption = draw(solution_dir, ax) or ""
    ax.set_title(title + (f"\n{caption}" if caption else ""), fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"caption: {caption}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
