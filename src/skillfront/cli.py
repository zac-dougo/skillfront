import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import xarray as xr

from skillfront.benchmark import benchmark_cases, score_forecast
from skillfront.data import LEAD_HOURS, PROTOCOL, download_data, prepare_benchmark_grid
from skillfront.grid import DEFAULT_GRID_ORIGIN, NATIVE_GRID_DEGREES


@contextmanager
def _atomic_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_snapshot(path: Path) -> tuple[xr.Dataset, str]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        return xr.load_dataset(stream, engine="scipy", decode_timedelta=True), digest


def main() -> None:
    parser = argparse.ArgumentParser(prog="skillfront")
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("benchmark", help="Run the Global v0 benchmark")
    benchmark.add_argument("--data", type=Path, default=Path(f"data/{PROTOCOL}-native.nc"))
    benchmark.add_argument("--scorecard", type=Path)
    benchmark.add_argument(
        "--grid-degrees",
        type=float,
        default=NATIVE_GRID_DEGREES,
        help="Scoring cell width and height in degrees (default: 0.25)",
    )
    benchmark.add_argument(
        "--grid-origin",
        type=float,
        nargs=2,
        metavar=("LAT", "LON"),
        default=DEFAULT_GRID_ORIGIN,
        help="Anchor for cell centres (default: 0 0)",
    )
    benchmark.add_argument("--forecast", type=Path, help="Optional model forecast in NetCDF3 format")
    args = parser.parse_args()

    origin = tuple(args.grid_origin)
    grid_label = f"{args.grid_degrees}deg"
    if origin != DEFAULT_GRID_ORIGIN:
        grid_label += f"-origin-{origin[0]}_{origin[1]}"
    scorecard = args.scorecard or Path(f"runs/{PROTOCOL}-{grid_label}.json")
    try:
        run_benchmark(args.data, scorecard, args.forecast, args.grid_degrees, origin)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"skillfront: {error}\n")


def run_benchmark(
    data_path: Path,
    scorecard: Path,
    forecast_path: Path | None = None,
    grid_degrees: float = NATIVE_GRID_DEGREES,
    grid_origin: tuple[float, float] = DEFAULT_GRID_ORIGIN,
) -> None:
    paths = [data_path, scorecard]
    if forecast_path is not None:
        paths.append(forecast_path)
        if not forecast_path.is_file():
            raise ValueError(f"Forecast file does not exist: {forecast_path}")
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Data, scorecard, and forecast snapshot paths must be different")

    if not data_path.exists():
        print("Downloading the Global v0 ERA5 and climatology slices (~1 GB)…", flush=True)
        data = download_data()
        with _atomic_output(data_path) as temporary:
            data.to_netcdf(temporary, engine="scipy", format="NETCDF3_64BIT")

    data, data_hash = _read_snapshot(data_path)
    verification, baselines = benchmark_cases(prepare_benchmark_grid(data, grid_degrees, grid_origin))
    models: dict[str, dict[str, Any]] = {
        name: {"scores": score_forecast(model, verification)} for name, model in baselines.items()
    }
    if forecast_path is not None:
        forecast, forecast_hash = _read_snapshot(forecast_path)
        name = forecast_path.stem
        if name in models:
            raise ValueError(f"Forecast name {name!r} is reserved for a baseline")
        models[name] = {"sha256": forecast_hash, "scores": score_forecast(forecast, verification)}

    report = {
        "protocol": PROTOCOL,
        "skillfront_version": version("skillfront"),
        "region": "Global",
        "grid_degrees": grid_degrees,
        "grid_origin": {"latitude": grid_origin[0], "longitude": grid_origin[1]},
        "init_times_utc": verification.indexes["init_time"].strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
        "lead_hours": LEAD_HOURS,
        "initializations": verification.sizes["init_time"],
        "grid_shape": {
            "latitude": verification.sizes["latitude"],
            "longitude": verification.sizes["longitude"],
        },
        "grid_cells": verification.sizes["latitude"] * verification.sizes["longitude"],
        "data": {
            "sha256": data_hash,
            "sources": {name: data.attrs[name] for name in ("era5", "climatology")},
        },
        "models": models,
    }
    with _atomic_output(scorecard) as temporary:
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {scorecard}")
