<img src="assets/icon.svg" width="64" height="64" alt="">

# Skillfront

Skillfront benchmarks weather forecasts against ERA5. Australia v0 scores 24-hour forecasts for four surface variables and compares them with persistence and climatology. Run it on ERA5's native 0.25° grid or on your model's coarser grid.

Building a model? Give your team or agents the [model design brief](MODEL_DESIGN_BRIEF.md) for the forecast contract, evaluation rules, and deliverables.

## Quickstart

With `uv` installed, clone the repository and run the baselines.

```sh
git clone https://github.com/zac-dougo/skillfront.git
cd skillfront
uv run --locked --python 3.12 skillfront benchmark
```

The first run downloads public ERA5 and climatology data and caches the regional slice in `data/australia-v0-native.nc`. Subsequent runs reuse that snapshot. The scorecard at `runs/australia-v0-0.25deg.json` records baseline scores, grid settings, forecast start times, source URLs, and the snapshot's SHA-256 hash. The snapshot and generated scorecards stay local to each checkout.

## Australia v0

The benchmark verifies forecasts initialized at **00:00 UTC** for the first seven days of January, April, July, and October 2022. It verifies each forecast **24 hours later** over cells whose centres fall inside the Natural Earth outline of mainland Australia and Tasmania.

| Forecast field | Units | Target |
| --- | --- | --- |
| `temperature_2m` | `degC` | Air temperature at 2 m at valid time |
| `wind_speed_10m` | `m s-1` | Wind speed at 10 m at valid time |
| `mean_sea_level_pressure` | `hPa` | Mean sea-level pressure at valid time |
| `precipitation_24h` | `mm` | Rainfall accumulated over the 24 hours ending at valid time |

Persistence repeats ERA5 at initialization, including the previous 24-hour rainfall total. Climatology uses the 1990–2019 ERA5 average for the target day of year and hour. The evaluator reports area-weighted MAE, RMSE, and signed bias, defined as forecast minus ERA5.

Fit and tune models on data before 2022. For each forecast, use only inputs available by its initialization time.

## Score a coarser grid

Set `--grid-degrees` to your model's cell size in both latitude and longitude. Centres default to multiples of that spacing. Pass `--grid-origin LAT LON` if the centres are offset.

```sh
uv run --locked --python 3.12 skillfront benchmark --grid-degrees 3 --grid-origin -1.5 1.5
```

Skillfront area-averages ERA5 verification and the baselines over each model cell's footprint, including water inside coastal cells. Rainfall stays in millimetres averaged over that footprint. The requested cells must fit inside the native snapshot. Use the same grid size and origin when comparing scores.

## Score a model

Export a deterministic NetCDF3 file with the four fields above on `init_time × lead_time × cell`. Match the benchmark's dates, 24-hour lead, cell coordinates, and units. The [model design brief](MODEL_DESIGN_BRIEF.md#deliver-a-forecast-skillfront-can-read) shows how to generate the coordinate template.

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
- [Natural Earth country boundaries](https://github.com/nvkelso/natural-earth-vector/tree/v5.1.2)
