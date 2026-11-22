import hashlib
import json
from urllib.request import urlopen

import dask
import numpy as np
import shapely
import xarray as xr
from zarr.storage import FsspecStore

from skillfront.grid import DEFAULT_GRID_ORIGIN, NATIVE_GRID_DEGREES, average_to_cells, crop_native


PROTOCOL = "australia-v0"
YEAR = 2022
INITIALIZATION_MONTHS = (1, 4, 7, 10)
DAYS_PER_BLOCK = 7
LEAD_HOURS = 24
ARCHIVE = "https://storage.googleapis.com/weatherbench2/datasets"
ERA5_URL = f"{ARCHIVE}/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
CLIMATOLOGY_URL = f"{ARCHIVE}/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"
BOUNDARY_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/"
    "geojson/ne_110m_admin_0_countries.geojson"
)
BOUNDARY_SHA256 = "6866c877d39cba9c357620878839b336d569f8c662d3cfab4cb1dbe2d39c977f"

# WeatherBench field name, output units, scale, and offset.
VARIABLES = {
    "temperature_2m": ("2m_temperature", "degC", 1.0, -273.15),
    "wind_speed_10m": ("10m_wind_speed", "m s-1", 1.0, 0.0),
    "mean_sea_level_pressure": ("mean_sea_level_pressure", "hPa", 0.01, 0.0),
    "precipitation_24h": ("total_precipitation_24hr", "mm", 1000.0, 0.0),
}


def initialization_times() -> np.ndarray:
    starts = np.array([f"{YEAR}-{month:02d}-01" for month in INITIALIZATION_MONTHS], dtype="datetime64[D]")
    return (starts[:, None] + np.arange(DAYS_PER_BLOCK)).ravel()


def _open_archive(url: str) -> xr.Dataset:
    store = FsspecStore.from_url(url, read_only=True)
    return xr.open_zarr(store, consolidated=True, chunks={})


def _download_boundary() -> str:
    with urlopen(BOUNDARY_URL, timeout=30) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != BOUNDARY_SHA256:
        raise ValueError("The Natural Earth boundary download failed its checksum")
    features = json.loads(payload)["features"]
    australia = next(feature for feature in features if feature["properties"]["ADM0_A3"] == "AUS")
    return json.dumps(australia["geometry"], separators=(",", ":"))


def prepare_benchmark_grid(
    data: xr.Dataset,
    step: float = NATIVE_GRID_DEGREES,
    origin: tuple[float, float] = DEFAULT_GRID_ORIGIN,
) -> xr.Dataset:
    """Select Australian land centres and prepare ERA5 on their chosen footprints."""
    if data.attrs.get("native_grid_degrees") != NATIVE_GRID_DEGREES:
        raise ValueError("Expected a native 0.25-degree snapshot. Use data/australia-v0-native.nc.")
    if not np.isfinite(step) or step < NATIVE_GRID_DEGREES or not np.isfinite(origin).all():
        raise ValueError("Grid size must be at least 0.25 degrees and the grid origin must be finite")

    geometry = shapely.from_geojson(data.attrs["australia_geometry"])
    west, south, east, north = geometry.bounds

    def centres(lower: float, upper: float, offset: float) -> np.ndarray:
        offset %= step
        first = int(np.ceil((lower - offset) / step))
        stop = int(np.floor((upper - offset) / step)) + 1
        return (offset + step * np.arange(first, stop)).round(10)

    longitude, latitude = np.meshgrid(
        centres(west, east, origin[1]), centres(south, north, origin[0])
    )
    on_land = shapely.covers(geometry, shapely.points(longitude, latitude))
    if not on_land.any():
        raise ValueError("The requested grid has no cell centres in the Australia boundary")
    grid = xr.Dataset(coords={
        "cell": np.arange(on_land.sum(), dtype=np.int32),
        "latitude": ("cell", latitude[on_land]),
        "longitude": ("cell", longitude[on_land]),
    })
    return average_to_cells(data, grid, step).assign_attrs(grid_degrees=step)


def _convert_fields(source: xr.Dataset) -> xr.Dataset:
    fields = {}
    for name, (archive_name, unit, scale, offset) in VARIABLES.items():
        field = source[archive_name] * scale + offset
        fields[name] = field.assign_attrs(units=unit)
    return xr.Dataset(fields)


def download_data() -> xr.Dataset:
    """Return daily ERA5 and fixed historical climatology fields on the native grid."""
    init_times = initialization_times()
    valid = init_times + np.timedelta64(LEAD_HOURS, "h")
    times = np.union1d(init_times, valid)
    valid_days = np.unique(xr.DataArray(valid).dt.dayofyear)
    fields = [spec[0] for spec in VARIABLES.values()]

    boundary = _download_boundary()
    with _open_archive(ERA5_URL) as era5, _open_archive(CLIMATOLOGY_URL) as climatology:
        analysis = _convert_fields(crop_native(era5[fields].sel(time=times)))
        daily_climate = _convert_fields(crop_native(
            climatology[fields].sel(hour=0, dayofyear=valid_days, drop=True)
        ))
        daily_climate = daily_climate.rename({name: f"climatology_{name}" for name in VARIABLES})
        data = xr.merge([analysis, daily_climate], join="exact", compat="equals")
        data = data.assign_attrs(
            protocol=PROTOCOL,
            native_grid_degrees=NATIVE_GRID_DEGREES,
            era5=ERA5_URL,
            climatology=CLIMATOLOGY_URL,
            boundary=BOUNDARY_URL,
            boundary_sha256=BOUNDARY_SHA256,
            australia_geometry=boundary,
        )
        with dask.config.set(scheduler="threads", num_workers=12):
            data.load()

    invalid = [name for name in data.data_vars if not np.isfinite(data[name]).all().item()]
    if invalid:
        raise ValueError(f"The archive slice contains missing or invalid data in {', '.join(invalid)}")
    return data
