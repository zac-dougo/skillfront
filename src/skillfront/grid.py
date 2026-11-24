import numpy as np
import xarray as xr

NATIVE_GRID_DEGREES = 0.25
DEFAULT_GRID_ORIGIN = (0.0, 0.0)
_NATIVE_LATITUDES = np.arange(-90.0, 90.0 + NATIVE_GRID_DEGREES / 2, NATIVE_GRID_DEGREES)
_NATIVE_LONGITUDES = np.arange(0.0, 360.0, NATIVE_GRID_DEGREES)


def _validate_native_axes(data: xr.Dataset) -> None:
    if (
        "latitude" not in data.coords
        or "longitude" not in data.coords
        or not np.array_equal(data["latitude"].values, _NATIVE_LATITUDES)
        or not np.array_equal(data["longitude"].values, _NATIVE_LONGITUDES)
    ):
        raise ValueError("Expected the full global native 0.25-degree grid ([-90, 90] x [0, 360))")


def normalize_native(data: xr.Dataset) -> xr.Dataset:
    """Snap archive coordinates to the exact global 0.25-degree ERA5 grid."""
    coordinates = {}
    for name in ("latitude", "longitude"):
        values = data[name].values.astype(np.float64)
        snapped = NATIVE_GRID_DEGREES * np.round(values / NATIVE_GRID_DEGREES)
        # The archive's float32 coordinates can be a few millionths of a degree off-grid.
        if not np.allclose(values, snapped, rtol=0, atol=1e-4):
            raise ValueError(f"The archive {name} does not lie on the native 0.25-degree grid")
        coordinates[name] = (name, np.where(snapped == 0, 0.0, snapped))
    normalized = data.assign_coords(coordinates).sortby(["latitude", "longitude"])
    _validate_native_axes(normalized)
    return normalized


def global_grid(step: float, origin: tuple[float, float] = DEFAULT_GRID_ORIGIN) -> xr.Dataset:
    """Build a regular global latitude/longitude grid covering the full sphere."""
    if not np.isfinite(step) or step < NATIVE_GRID_DEGREES or step > 90.0:
        raise ValueError("Grid size must be between 0.25 and 90 degrees")
    if len(origin) != 2 or not np.isfinite(origin).all():
        raise ValueError("Grid origin must be two finite numbers (latitude, longitude)")

    lon_count = round(360.0 / step)
    if not np.isclose(lon_count * step, 360.0, rtol=0, atol=1e-9):
        raise ValueError("Grid size must evenly divide 360 degrees for a global longitude ring")
    lon_offset = float(origin[1]) % step
    if np.isclose(lon_offset, step, rtol=0, atol=1e-9):
        lon_offset = 0.0
    longitude = np.round(lon_offset + step * np.arange(lon_count, dtype=np.float64), 10)
    longitude = np.where(longitude == 0, 0.0, longitude)

    lat_offset = float(origin[0]) % step
    if np.isclose(lat_offset, step, rtol=0, atol=1e-9):
        lat_offset = 0.0
    first = int(np.ceil((-90.0 - lat_offset - 1e-11) / step))
    stop = int(np.floor((90.0 - lat_offset + 1e-11) / step)) + 1
    latitude = np.round(lat_offset + step * np.arange(first, stop, dtype=np.float64), 10)
    latitude = np.where(latitude == 0, 0.0, latitude)
    if latitude.size == 0 or latitude[0] - step / 2 > -90.0 + 1e-9 or latitude[-1] + step / 2 < 90.0 - 1e-9:
        raise ValueError(
            "The latitude grid must cover [-90, 90]; use a pole-centred or half-step latitude origin"
        )

    return xr.Dataset(coords={"latitude": latitude, "longitude": longitude})


def latitude_weights(latitude: xr.DataArray, step: float) -> xr.DataArray:
    """Spherical band-area weights for regular latitude cells clamped to [-90, 90]."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Grid size must be a positive finite number")
    values = latitude.values.astype(np.float64)
    south = np.maximum(values - step / 2, -90.0)
    north = np.minimum(values + step / 2, 90.0)
    weights = np.sin(np.deg2rad(north)) - np.sin(np.deg2rad(south))
    if np.any(weights <= 0):
        raise ValueError("Latitude cell weights must be strictly positive")
    axis = latitude.dims[0]
    return xr.DataArray(weights, dims=(axis,), coords={axis: latitude})


def _latitude_weights(source: xr.DataArray, targets: np.ndarray, step: float) -> xr.DataArray:
    native_half = NATIVE_GRID_DEGREES / 2
    centres = source.values[None, :]
    src_south = np.maximum(centres - native_half, -90.0)
    src_north = np.minimum(centres + native_half, 90.0)
    tgt_south = np.maximum(targets[:, None] - step / 2, -90.0)
    tgt_north = np.minimum(targets[:, None] + step / 2, 90.0)
    low = np.maximum(src_south, tgt_south)
    high = np.minimum(src_north, tgt_north)
    overlap = np.maximum(np.sin(np.deg2rad(high)) - np.sin(np.deg2rad(low)), 0.0)
    return xr.DataArray(
        overlap / overlap.sum(axis=1, keepdims=True),
        dims=("latitude", "source_latitude"),
        coords={"latitude": targets, "source_latitude": source.values},
    )


def _longitude_weights(source: xr.DataArray, targets: np.ndarray, step: float) -> xr.DataArray:
    native_half = NATIVE_GRID_DEGREES / 2
    delta = ((source.values[None, :] - targets[:, None] + 180.0) % 360.0) - 180.0
    low = np.maximum(delta - native_half, -step / 2)
    high = np.minimum(delta + native_half, step / 2)
    overlap = np.maximum(high - low, 0.0)
    return xr.DataArray(
        overlap / overlap.sum(axis=1, keepdims=True),
        dims=("longitude", "source_longitude"),
        coords={"longitude": targets, "source_longitude": source.values},
    )


def average_to_cells(data: xr.Dataset, grid: xr.Dataset, step: float) -> xr.Dataset:
    """Conservatively area-average native global fields onto a regular global grid."""
    if not np.isfinite(step) or step < NATIVE_GRID_DEGREES or step > 90.0:
        raise ValueError("Grid size must be between 0.25 and 90 degrees")
    _validate_native_axes(data)

    target_lat = grid["latitude"].values.astype(np.float64)
    target_lon = grid["longitude"].values.astype(np.float64)
    if (
        np.isclose(step, NATIVE_GRID_DEGREES, rtol=0, atol=1e-12)
        and np.array_equal(target_lat, data["latitude"].values)
        and np.array_equal(target_lon, data["longitude"].values)
    ):
        return data

    lat_w = _latitude_weights(data["latitude"], target_lat, step)
    lon_w = _longitude_weights(data["longitude"], target_lon, step)
    renamed = data.rename({"latitude": "source_latitude", "longitude": "source_longitude"})
    averaged = {}
    for name, field in renamed.data_vars.items():
        by_lon = xr.dot(field, lon_w, dim="source_longitude")
        averaged[name] = (
            xr.dot(by_lon, lat_w, dim="source_latitude")
            .transpose(..., "latitude", "longitude")
            .assign_attrs(field.attrs)
        )
    return xr.Dataset(averaged, attrs=data.attrs)
