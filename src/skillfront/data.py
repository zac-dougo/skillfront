import time

import numpy as np
import xarray as xr
from aiohttp import ClientError, ClientTimeout
from zarr.storage import FsspecStore

from skillfront.grid import (
    DEFAULT_GRID_ORIGIN,
    NATIVE_GRID_DEGREES,
    average_to_cells,
    global_grid,
    normalize_native,
)

PROTOCOL = "global-v0"
YEAR = 2022
INITIALIZATION_MONTHS = (1, 4, 7, 10)
DAYS_PER_BLOCK = 7
LEAD_HOURS = 24
ARCHIVE = "https://storage.googleapis.com/weatherbench2/datasets"
ERA5_URL = f"{ARCHIVE}/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
CLIMATOLOGY_URL = f"{ARCHIVE}/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"

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
    store = FsspecStore.from_url(
        url,
        read_only=True,
        storage_options={"client_kwargs": {"timeout": ClientTimeout(total=180, sock_read=60)}},
    )
    return xr.open_zarr(store, consolidated=True, chunks=None)


def prepare_benchmark_grid(
    data: xr.Dataset,
    step: float = NATIVE_GRID_DEGREES,
    origin: tuple[float, float] = DEFAULT_GRID_ORIGIN,
) -> xr.Dataset:
    """Prepare ERA5 and climatology on the chosen global scoring grid."""
    if data.attrs.get("native_grid_degrees") != NATIVE_GRID_DEGREES:
        raise ValueError(f"Expected a native 0.25-degree snapshot. Use data/{PROTOCOL}-native.nc.")
    grid = global_grid(step, origin)
    return average_to_cells(data, grid, step).assign_attrs(grid_degrees=float(step))


def _load_fields(source: xr.Dataset, label: str, prefix: str = "") -> xr.Dataset:
    fields = {}
    total = len(VARIABLES)
    for index, (name, (archive_name, unit, scale, offset)) in enumerate(VARIABLES.items(), start=1):
        print(f"  {label} {name} ({index}/{total})…", flush=True)
        for attempt in range(1, 4):
            try:
                raw = source[archive_name].load()
                break
            except (ClientError, OSError, TimeoutError) as error:
                if attempt == 3:
                    raise OSError(f"Failed to download {label} {name}: {error}") from error
                time.sleep(2 * attempt)
        field = raw * scale + offset
        fields[f"{prefix}{name}"] = field.assign_attrs(units=unit)
    return xr.Dataset(fields)


def download_data() -> xr.Dataset:
    """Return daily global ERA5 and fixed historical climatology fields on the native 0.25-degree grid."""
    init_times = initialization_times()
    valid = init_times + np.timedelta64(LEAD_HOURS, "h")
    times = np.union1d(init_times, valid)
    valid_days = np.unique(xr.DataArray(valid).dt.dayofyear)
    fields = [spec[0] for spec in VARIABLES.values()]

    with _open_archive(ERA5_URL) as era5, _open_archive(CLIMATOLOGY_URL) as climatology:
        analysis = _load_fields(normalize_native(era5[fields].sel(time=times)), "ERA5")
        daily_climate = _load_fields(
            normalize_native(climatology[fields].sel(hour=0, dayofyear=valid_days, drop=True)),
            "Climatology",
            prefix="climatology_",
        )
        data = xr.merge([analysis, daily_climate], join="exact", compat="equals")
        data = data.assign_attrs(
            protocol=PROTOCOL,
            native_grid_degrees=NATIVE_GRID_DEGREES,
            era5=ERA5_URL,
            climatology=CLIMATOLOGY_URL,
        )

    invalid = [str(name) for name in data.data_vars if not np.isfinite(data[name]).all().item()]
    if invalid:
        raise ValueError(f"The archive slice contains missing or invalid data in {', '.join(invalid)}")
    return data
