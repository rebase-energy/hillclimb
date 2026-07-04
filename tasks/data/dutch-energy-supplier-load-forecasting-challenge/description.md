# Dutch Energy Supplier Load Forecasting Challenge

Forecast the 15-minute **net load** (kWh) of a Dutch energy supplier's portfolio
for the period **2025-09-01 00:00 through 2025-09-24 21:45 UTC** (2,296
intervals), given roughly four months of history.

Net load is consumption minus behind-the-meter generation: it turns strongly
**negative** around midday on sunny days (solar export) and peaks in the
evening. Timestamps are UTC; daily and weekly shape follows local time
(Europe/Amsterdam, UTC+2 in this period), so derive calendar features from
local time.

## Data

- `train.csv` — `timestamp_utc`, `net_load_kwh`. 15-minute resolution,
  2025-05-08 22:00 to 2025-08-31 23:45 UTC, 11,048 rows, a single continuous
  series.
- `test.csv` — `row_id`, `timestamp_utc`: the 2,296 intervals to forecast.
- `sample_submission.csv` — `row_id`, `net_load_kwh`.

There are no exogenous feature columns in the provided files. Per the
competition rules:

- **You may (and should) fetch historical weather data from the Open-Meteo
  archive API** (https://archive-api.open-meteo.com) for the Netherlands for
  both the training and forecast periods. The portfolio is aggregated Dutch
  households and offices; a representative location (e.g. De Bilt, 52.1°N
  5.18°E) or a blend of major NL cities is reasonable. Useful variables:
  temperature, shortwave radiation / direct+diffuse irradiance, cloud cover,
  wind speed. Cache the fetch to a local file so reruns are fast and robust.
- **The model must be multivariate and weather-driven.**
- **Lag features of the target (`net_load_kwh`) are FORBIDDEN** — in the real
  setting the target arrives with a three-day delay. Do not feed lagged,
  rolled, or aggregated past values of net load into the model as features.
  Calendar features (hour, weekday, holidays) and weather are fair game.
- No other external data.

## Submission format

`submission.csv` with columns `row_id,net_load_kwh` — one row per test row.
The submission is keyed by `row_id`, not by timestamp: **map each forecast to
its `row_id` through `test.csv`'s `timestamp_utc` column**.

## Evaluation

**NRMSE** (normalized root mean squared error), lower is better. On a fixed
evaluation set NRMSE ranks identically to RMSE, so optimize RMSE.
