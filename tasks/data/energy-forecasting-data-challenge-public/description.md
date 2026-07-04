# Energy Forecasting Data Challenge

Forecast the 15-minute **residual load** of 11 independent sites for the 7 days
following each site's training history.

## Data

- `train.csv` — `time`, `P` (PV production), `Gb(i)` / `Gd(i)` (beam / diffuse
  irradiance on the plane of array), `H_sun` (sun height), `T2m` (temperature),
  `WS10m` (wind speed), `load`, `residual_load`, `dataset_id`. ~88,700 rows.
- `test.csv` — `time`, `Gb(i)`, `Gd(i)`, `H_sun`, `T2m`, `WS10m`, `dataset_id`:
  the rows to forecast. Weather is **given** for the forecast period; the load
  columns are not.
- `sample_submission.csv` — `time`, `residual_load`.

Structure that matters:

- Each `dataset_id` (1–11) is a separate site with its own **contiguous ~84-day
  training block followed immediately by its 7-day test window** (672 rows at
  15-minute resolution).
- The 11 blocks occupy **non-overlapping, sequential time ranges** (2018-01-01
  through 2020-10-17), so the submission can be keyed by `time` alone. One
  quirk: blocks 4 and 8 each contain a DST fall-back hour with duplicated
  timestamps in train (2018-10-28 and 2019-10-27, 02:15–03:00). `time` in
  `test.csv` has no duplicates.
- `residual_load = load − P` conceptually: residual load is consumption net of
  PV production, so it dips (and can drop sharply) when irradiance is high.

Per-site models and a single pooled model with `dataset_id` as a feature are
both viable. **Use only the files in `./data` — do not download external
data.**

## Submission format

`submission.csv` with columns `time,residual_load`, covering every row of
`test.csv` (all 11 sites, 7,392 rows), `time` values copied exactly.

## Evaluation

**RMSE** (root mean squared error), lower is better.
