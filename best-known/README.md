# Best known solutions

The best construction we have for each example problem, as a submission
the problem's own `verify.py` accepts, with where it came from.

This folder is reference material for people. It is not part of a
problem: `hillclimb problem get <id>` copies `src/hillclimb/demo/<id>/`
into your `problems/` and never this, so a search cannot read the answer.
The website's playgrounds show these behind their **optimal** /
**best known** button, and the chart's reference lines in each
`problem.yaml` (`chart_baselines`) should agree with the values here.

## Layout

| Path | What |
| --- | --- |
| `<problem>.csv` | the construction, in that problem's submission format (`problems/<problem>/interface.py`) |
| `heilbronn-<n>.csv` | the Heilbronn ladder for every n from 3 to 16; 11, 14 and 17 are catalog problems, the rest are here for completeness |
| `index.json` | one entry per file: the value, its status, who found it, where the coordinates come from, the reference |
| `check.py` | re-scores every file with its problem's verifier and compares with the index |

`index.json` fields:

| Field | Meaning |
| --- | --- |
| `problem` | the catalog problem id, or `null` when no catalog problem has that size |
| `file` | the CSV next to the index |
| `value` | the score `verify.py` prints for the file (or, without a verifier, the source's value) |
| `status` | `optimal` (proven), `best known` (the record as far as we know), or `published` (a published construction that the record has since passed) |
| `who` | who found the construction, and when |
| `coordinates` | where these exact numbers were taken from, with the license |
| `reference` | the paper or page to cite |
| `url` | where to look |
| `note` | anything else worth knowing |

## Adding one

1. Write the construction as `<problem>.csv` in the submission format.
2. Add its entry to `index.json`.
3. Run the check; it must print `ok` for the file:

       uv run python best-known/check.py

Only put a value in `status: best known` if it is at least as good as
every line in the problem's `chart_baselines`, and update those lines
when a record moves.

## Sources

- Heilbronn triangles in the unit square, n = 3 … 16: N. Sudermann-Merx,
  *From Computational Certification to Exact Coordinates: Heilbronn's
  Triangle Problem on the Unit Square*, 2026; coordinates from
  [github.com/spiralulam/heilbronn](https://github.com/spiralulam/heilbronn)
  (MIT). n ≤ 9 are certified optimal there; the records for n ≥ 10 are
  Comellas & Yebra (2002), Karpov, and Beyleveld.
- 32 circles in the unit square: AlphaEvolve (Novikov et al., 2025),
  the construction transcribed in `tests/test_circle_packing_starter.py`.
  The best known since is Georgiev et al. (2025), 2.939572, whose
  coordinates we do not have.
- Golomb rulers with 20 and 27 marks: the proven optimal rulers
  (Garry, Vanderschel et al. 1997; distributed.net OGR-27, 2014).
- Low-autocorrelation binary sequences of length 40 and 60: the
  proven optimal sequences of Packebusch & Mertens (2016).
- Heilbronn convex 13: the winning submission of a hillclimb search
  (2026-08-24), scoring 0.0309372, above AlphaEvolve's published
  0.0309369.
