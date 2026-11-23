<img src="assets/icon.svg" width="64" height="64" alt="">

# Skillfront

Skillfront benchmarks weather forecasts against ERA5. Global v0 scores 24-hour global forecasts for four surface variables and compares them with persistence and climatology. Run it on ERA5's native 0.25° grid or on your model's coarser regular global grid.

Building a model? Give your team or agents the [model design brief](MODEL_DESIGN_BRIEF.md) for the forecast contract, evaluation rules, and deliverables.

## Quickstart

With `uv` installed, clone the repository and run the baselines.

```sh
git clone https://github.com/zac-dougo/skillfront.git
cd skillfront
uv run --locked --python 3.12 skillfront benchmark
```

The first run downloads the public global ERA5 and climatology slices (~1 GB) and caches them in `data/global-v0-native.nc`. Subsequent runs reuse that snapshot. The scorecard at `runs/global-v0-0.25deg.json` records baseline scores, grid settings, forecast start times, source URLs, and the snapshot's SHA-256 hash. The snapshot and generated scorecards stay local to each checkout.

## Global v0

The benchmark verifies forecasts initialized at **00:00 UTC** for the first seven days of January, April, July, and October 2022. It verifies each forecast **24 hours later** across the full global grid (`721 × 1440` at native 0.25°).

| Forecast field | Units | Target |
| --- | --- | --- |
| `temperature_2m` | `degC` | Air temperature at 2 m at valid time |
| `wind_speed_10m` | `m s-1` | Wind speed at 10 m at valid time |
| `mean_sea_level_pressure` | `hPa` | Mean sea-level pressure at valid time |
| `precipitation_24h` | `mm` | Rainfall accumulated over the 24 hours ending at valid time |

Persistence repeats ERA5 at initialization, including the previous 24-hour rainfall total. Climatology uses the 1990–2019 ERA5 average for the target day of year and hour. The evaluator reports spherical-area-weighted MAE, RMSE, and signed bias, defined as forecast minus ERA5.

Fit and tune models on data before 2022. For each forecast, use only inputs available by its initialization time.

## Score a coarser grid

Set `--grid-degrees` to your model's cell size in both latitude and longitude (must evenly divide 360°). Centres default to multiples of that spacing (`[-90, 90] × [0, 360)`). Pass `--grid-origin LAT LON` if the centres are offset, as long as the latitude footprints cover `[-90, 90]` (for example, half-step offset grids like `3°` with `-1.5 1.5` or `5.625°` with `2.8125 0`).

```sh
uv run --locked --python 3.12 skillfront benchmark --grid-degrees 3 --grid-origin -1.5 1.5
```

Skillfront conservatively area-averages ERA5 verification and the baselines over each model cell's spherical footprint, wrapping longitude across the `0° / 360°` seam and clamping polar caps at `±90°`. Use the same grid size and origin when comparing scores.

## Score a model

Export a deterministic NetCDF3 file (`NETCDF3_64BIT` recommended at native resolution) with the four fields above on `init_time × lead_time × latitude × longitude`. Match the benchmark's dates, 24-hour lead, coordinates, and units. The [model design brief](MODEL_DESIGN_BRIEF.md#deliver-a-forecast-skillfront-can-read) shows how to generate the coordinate template.

```sh
uv run --locked --python 3.12 skillfront benchmark \
  --forecast my-model.nc --scorecard runs/my-model.json
```

For a coarse model, add the same grid options you used to produce its forecast.

## Run the tests

```sh
uv run --locked --python 3.12 python -m unittest discover -s tests -v
```

## Data sources

- [WeatherBench 2 archive and climatology](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)
- [Copernicus Climate Change Service ERA5](https://doi.org/10.24381/cds.adbb2d47)
