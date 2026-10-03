"""`hillclimb plot` / `summit --plot`: a solution drawn by its problem's plot.py
(matplotlib, run in the problem's runtime venv like its verifier)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import typer

pytest.importorskip("matplotlib")

from hillclimb.runtime.plot_solution import main as plot_main  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PROBLEMS = REPO / "problems"


def _solution(tmp_path: Path, problem: str) -> Path:
    """A solution dir holding the problem's sample as its submission.csv."""
    solution = tmp_path / "solution"
    solution.mkdir()
    shutil.copy(PROBLEMS / problem / "sample_submission.csv", solution / "submission.csv")
    return solution


@pytest.mark.parametrize(
    "problem, caption",
    [
        ("heilbronn-11", "11 points · smallest triangle "),
        ("heilbronn-14", "14 points · smallest triangle "),
        ("heilbronn-17", "17 points · smallest triangle "),
        ("circle-packing", "26 circles · sum of radii "),
        ("circle-packing-32", "32 circles · sum of radii "),
    ],
)
def test_bundled_plots_draw_their_sample(tmp_path, capsys, problem, caption):
    out = tmp_path / "plot.png"
    code = plot_main([str(PROBLEMS / problem / "plot.py"), str(_solution(tmp_path, problem)), str(out), problem])
    assert code == 0 and out.read_bytes()[:4] == b"\x89PNG"
    assert f"caption: {caption}" in capsys.readouterr().out


def test_a_plot_without_a_plot_function_is_refused(tmp_path):
    no_function = tmp_path / "plot.py"
    no_function.write_text("X = 1\n")
    assert plot_main([str(no_function), str(tmp_path), str(tmp_path / "p.png"), "t"]) == 2


# --- the commands ---


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """A hillclimb dir with heilbronn-11 in problems/ and a summited solution;
    this interpreter (which has matplotlib) stands in for the runtime venv."""
    from hillclimb.cli import common

    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    root = tmp_path / "proj"
    (root / "problems").mkdir(parents=True)
    (root / "hillclimb.yaml").write_text("{}\n")
    shutil.copytree(PROBLEMS / "heilbronn-11", root / "problems" / "heilbronn-11",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(PROBLEMS / "heilbronn-11" / "sample_submission.csv", root / "submission.csv")
    monkeypatch.chdir(root)
    monkeypatch.setattr("hillclimb.api.ensure_runtime_venv", lambda *a, **k: Path(sys.executable))
    opened = []
    monkeypatch.setattr(common, "open_file", opened.append)
    return root, opened


def test_plot_draws_the_folders_solution_beside_it(folder, capsys):
    from hillclimb.cli.views import plot

    root, opened = folder
    plot(target=None, candidate=None, problem=None, out=None, no_open=False)
    png = root / "solution.png"
    assert png.read_bytes()[:4] == b"\x89PNG" and opened == [png]
    out = capsys.readouterr().out
    assert "heilbronn-11 · this folder: 11 points · smallest triangle" in out and "solution.png" in out


def test_plot_says_when_the_problem_ships_no_plot(folder, capsys):
    from hillclimb.cli.views import plot

    root, opened = folder
    (root / "problems" / "heilbronn-11" / "plot.py").unlink()
    with pytest.raises(typer.Exit):
        plot(target=None, candidate=None, problem=None, out=None, no_open=False)
    assert "ships no plot.py" in capsys.readouterr().out and not opened


def test_a_crashing_plot_shows_why(folder, capsys):
    from hillclimb.cli.views import plot

    root, opened = folder
    (root / "submission.csv").unlink()
    with pytest.raises(typer.Exit):
        plot(target=None, candidate=None, problem=None, out=None, no_open=True)
    err = capsys.readouterr().err
    assert "Cannot plot this folder" in err and "submission.csv" in err
