"""Copy the deterministic starting configuration into the candidate output."""

from pathlib import Path
import shutil


shutil.copy(Path("problem") / "sample_submission.csv", "submission.csv")
