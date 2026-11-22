# Model design brief for Skillfront Australia v0

Build a model that produces deterministic 24-hour weather forecasts over Australia. Use Skillfront to measure whether your forecasts improve on persistence and climatology. Aim for lower MAE and RMSE on the variables your design targets, and report where either baseline still wins. Skillfront handles verification and scoring. Your model owns training and inference.

```text
Your model ─────────────→ forecast file ────────┐
                                               ├→ Skillfront → JSON scorecard
ERA5 snapshot → scoring grid → truth/baselines ─┘
```

## Forecast task

- Forecast from 00:00 UTC on each of 28 starts, covering the first seven days of January, April, July, and October 2022. The valid time is the next day at 00:00 UTC.
- Fit and tune using data before 2022. For each forecast, use only inputs available by its initialization time. Keep verification values and benchmark scores out of model fitting and tuning.
- Verification uses ERA5 reanalysis. Persistence repeats ERA5 at initialization, including its preceding 24-hour rainfall total. Climatology uses the 1990–2019 ERA5 average for the forecast's valid day of year and hour.

## Choose your grid

The default is **0.25° latitude × 0.25° longitude**. You can design for a larger regular grid. Pass its spacing with `--grid-degrees` and, if its centres are offset, its anchor with `--grid-origin LAT LON`. Skillfront uses cells whose centres fall on Australian land and area-averages ERA5 and the baselines over each coarse cell's full footprint, including water inside coastal cells.

Write predictions on the selected benchmark grid. Use the same size, origin, and cases when comparing model scores.

## Deliver a forecast Skillfront can read

Export a deterministic **NetCDF3** file containing all four fields. Each field must have dimensions `init_time`, `lead_time`, and `cell`. Copy the benchmark template's coordinate values and order, including `latitude` and `longitude` along `cell`. The `lead_time` coordinate must decode to a 24-hour timedelta. If you include `valid_time`, it must equal `init_time + lead_time`. Supply finite numeric values for all cases and cells.

| Variable name | Exact `units` attribute | Target |
| --- | --- | --- |
| `temperature_2m` | `degC` | Air temperature at 2 m at valid time |
| `wind_speed_10m` | `m s-1` | Wind speed at 10 m at valid time |
| `mean_sea_level_pressure` | `hPa` | Mean sea-level pressure at valid time |
| `precipitation_24h` | `mm` | Rainfall accumulated over the 24 hours ending at valid time |

From the Skillfront checkout, run `uv run --locked --python 3.12 skillfront benchmark` to populate `data/australia-v0-native.nc` and produce the default baseline scorecard. Use the template coordinates to assemble your export. This example uses the native grid. For a coarse export, pass the same `step` and `origin` to `prepare_benchmark_grid` that you'll use when scoring.

```python
import xarray as xr

from skillfront.benchmark import benchmark_cases
from skillfront.data import prepare_benchmark_grid

native = xr.load_dataset("data/australia-v0-native.nc", engine="scipy")
_, baselines = benchmark_cases(prepare_benchmark_grid(native))
template = baselines["persistence"]
forecast = xr.Dataset(coords=template.coords)
```

Populate `forecast` with your model's predictions and the units above, then write it with `forecast.to_netcdf("my-model.nc", engine="scipy")`.

## Run and hand back the results

```sh
uv run --locked --python 3.12 skillfront benchmark \
  --forecast my-model.nc --scorecard runs/my-model.json
```

Add the matching grid options to this command for a coarse forecast. The scorecard reports area-weighted MAE, RMSE, and signed bias for each variable and model. Bias is forecast minus ERA5.

Hand back the forecast file, the scorecard, and the command you used to reproduce the run. Include the model version, training data and cutoff, inference inputs, and a short per-variable comparison with both baselines. Report any missing coverage or forecast errors in the handoff.

See the [Skillfront README](README.md) for the coarse-grid command and benchmark implementation details.
