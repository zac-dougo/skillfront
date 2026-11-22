import numpy as np
import xarray as xr

from skillfront.data import LEAD_HOURS, PROTOCOL, VARIABLES, initialization_times


FORECAST_DIMS = ("init_time", "lead_time", "cell")


def benchmark_cases(data: xr.Dataset) -> tuple[xr.Dataset, dict[str, xr.Dataset]]:
    """Align truth and baseline forecasts to the fixed v0 initialization times."""
    if data.attrs.get("protocol") != PROTOCOL:
        raise ValueError(f"Expected a prepared {PROTOCOL} dataset")

    init = xr.DataArray(initialization_times(), dims="init_time")
    lead = np.timedelta64(LEAD_HOURS, "h")
    valid = init + lead

    persistence = data[list(VARIABLES)].sel(time=init).drop_vars("time")
    verification = data[list(VARIABLES)].sel(time=valid).drop_vars("time")
    climate_names = {f"climatology_{name}": name for name in VARIABLES}
    climate = data[list(climate_names)].sel(dayofyear=valid.dt.dayofyear).drop_vars("dayofyear")
    climate = climate.rename(climate_names)

    def align(fields: xr.Dataset) -> xr.Dataset:
        return (
            fields.assign_coords(init_time=init)
            .expand_dims(lead_time=[lead])
            .transpose(*FORECAST_DIMS)
            .assign_coords(valid_time=(("init_time", "lead_time"), valid.values[:, None]))
        )

    return align(verification), {"persistence": align(persistence), "climatology": align(climate)}


def score_forecast(forecast: xr.Dataset, verification: xr.Dataset) -> dict:
    """Score a complete deterministic forecast against the same dates and cells as the baselines."""
    coordinates = [*FORECAST_DIMS, "latitude", "longitude"]
    if "valid_time" in forecast.coords:
        coordinates.append("valid_time")
    for name in coordinates:
        actual = forecast.coords.get(name)
        expected = verification.coords[name]
        if actual is None or actual.dims != expected.dims or not np.array_equal(actual, expected):
            raise ValueError(f"Forecast coordinate {name!r} must match the benchmark")

    weights = np.cos(np.deg2rad(verification.latitude))
    scores = {}
    for name, (_, unit, _, _) in VARIABLES.items():
        if name not in forecast.data_vars:
            raise ValueError(f"Missing forecast variable {name!r}")
        field = forecast[name]
        if set(field.dims) != set(FORECAST_DIMS):
            raise ValueError(f"Forecast variable {name!r} must have dimensions {FORECAST_DIMS}")
        if field.attrs.get("units") != unit:
            raise ValueError(f"Forecast variable {name!r} must use units {unit!r}")
        if field.dtype.kind not in "fiu" or not np.isfinite(field).all().item():
            raise ValueError(f"Forecast variable {name!r} must contain finite numbers")
        truth = verification[name]
        if not np.isfinite(truth).all().item():
            raise ValueError(f"Verification variable {name!r} must contain finite numbers")

        error = field.astype(np.float64) - truth
        scores[name] = {
            "units": unit,
            "mae": abs(error).weighted(weights).mean().item(),
            "rmse": np.sqrt((error**2).weighted(weights).mean()).item(),
            "bias": error.weighted(weights).mean().item(),
        }
    return scores
