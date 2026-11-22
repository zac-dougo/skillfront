import numpy as np
import xarray as xr


NATIVE_GRID_DEGREES = 0.25
DEFAULT_GRID_ORIGIN = (0.0, 0.0)


def crop_native(data: xr.Dataset) -> xr.Dataset:
    """Keep Australia and enough surrounding ocean to average larger cells."""
    coordinates = {}
    for name in ("latitude", "longitude"):
        values = data[name].values.astype(np.float64)
        snapped = NATIVE_GRID_DEGREES * np.round(values / NATIVE_GRID_DEGREES)
        # The archive's float32 coordinates can be a few millionths of a degree off-grid.
        if not np.allclose(values, snapped, rtol=0, atol=1e-4):
            raise ValueError(f"The archive {name} does not lie on the native 0.25-degree grid")
        coordinates[name] = (name, snapped)
    return data.assign_coords(coordinates).sortby(["latitude", "longitude"]).sel(
        latitude=slice(-60, 5), longitude=slice(100, 170)
    )


def _axis_weights(source: xr.DataArray, targets: np.ndarray, step: float) -> xr.DataArray:
    axis = source.dims[0]
    native_half = NATIVE_GRID_DEGREES / 2
    centres = source.values[None, :]
    low = np.maximum(centres - native_half, targets[:, None] - step / 2)
    high = np.minimum(centres + native_half, targets[:, None] + step / 2)
    if axis == "latitude":
        low, high = np.sin(np.deg2rad(low)), np.sin(np.deg2rad(high))
    overlap = np.maximum(high - low, 0)
    return xr.DataArray(
        overlap / overlap.sum(axis=1, keepdims=True),
        dims=(axis, f"source_{axis}"),
        coords={axis: targets, f"source_{axis}": source.values},
    )


def average_to_cells(data: xr.Dataset, grid: xr.Dataset, step: float) -> xr.Dataset:
    """Area-average native fields over each target footprint, then select target cells."""
    if not np.isfinite(step) or step < NATIVE_GRID_DEGREES:
        raise ValueError("Grid size must be at least 0.25 degrees")

    selectors = {name: grid[name] for name in ("latitude", "longitude")}
    axes = {name: np.unique(values) for name, values in selectors.items()}
    for name, targets in axes.items():
        source = data[name].values
        if (
            source.size < 2
            or not np.allclose(np.diff(source), NATIVE_GRID_DEGREES)
            or source[0] - NATIVE_GRID_DEGREES / 2 > targets[0] - step / 2 + 1e-8
            or source[-1] + NATIVE_GRID_DEGREES / 2 < targets[-1] + step / 2 - 1e-8
        ):
            raise ValueError(f"The native {name} grid does not cover the requested cell footprints")

    if step == NATIVE_GRID_DEGREES and all(np.isin(v, data[k]).all() for k, v in axes.items()):
        return data.sel(selectors)

    weights = {name: _axis_weights(data[name], targets, step) for name, targets in axes.items()}
    renamed = data.rename({name: f"source_{name}" for name in axes})
    averaged = {}
    for name, field in renamed.data_vars.items():
        by_lon = xr.dot(field, weights["longitude"], dim="source_longitude")
        averaged[name] = xr.dot(by_lon, weights["latitude"], dim="source_latitude").assign_attrs(field.attrs)
    return xr.Dataset(averaged, attrs=data.attrs).sel(selectors)
