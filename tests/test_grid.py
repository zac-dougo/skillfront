import unittest

import numpy as np
import xarray as xr

from skillfront.data import prepare_benchmark_grid
from skillfront.grid import (
    NATIVE_GRID_DEGREES,
    average_to_cells,
    global_grid,
    latitude_weights,
    normalize_native,
)


def native_global_grid() -> xr.Dataset:
    latitude = np.arange(-90.0, 90.0 + NATIVE_GRID_DEGREES / 2, NATIVE_GRID_DEGREES)
    longitude = np.arange(0.0, 360.0, NATIVE_GRID_DEGREES)
    data = xr.Dataset(
        coords={"latitude": latitude, "longitude": longitude},
        attrs={"native_grid_degrees": NATIVE_GRID_DEGREES},
    )
    data["rain"] = (("latitude", "longitude"), np.zeros((latitude.size, longitude.size), dtype=np.float32))
    data["rain"].attrs["units"] = "mm"
    return data


class GridTests(unittest.TestCase):
    def test_archive_coordinates_snap_to_quarter_degrees(self):
        data = native_global_grid()
        jittered = data.isel(latitude=slice(None, None, -1)).assign_coords(
            latitude=data.latitude.values[::-1] + 0.000004,
            longitude=data.longitude.values - 0.000003,
        )
        fixed = normalize_native(jittered)
        xr.testing.assert_identical(fixed, data)

    def test_native_values_are_used_directly(self):
        native = native_global_grid()
        native.rain.loc[{"latitude": -30.0, "longitude": 140.75}] = 12
        cells = prepare_benchmark_grid(native, 0.25, (0.0, 0.0))
        self.assertEqual(cells.rain.sel(latitude=-30.0, longitude=140.75).item(), 12)
        self.assertEqual(cells.rain.dtype, native.rain.dtype)

    def test_longitude_wraps_across_prime_meridian(self):
        native = native_global_grid()
        # A 1.5-degree cell centred at lon=0 spans [-0.75, 0.75], which overlaps lon=359.75 fully (0.25/1.5)
        # and lon=359.25 by half a native cell (0.125/1.5).
        native.rain.loc[{"longitude": 359.75}] = 6
        native.rain.loc[{"longitude": 359.25}] = 12
        coarse = average_to_cells(native, global_grid(1.5), 1.5)
        expected = 6 * (0.25 / 1.5) + 12 * (0.125 / 1.5)
        self.assertAlmostEqual(coarse.rain.sel(latitude=0.0, longitude=0.0).item(), expected)
        self.assertEqual(coarse.rain.attrs["units"], "mm")

    def test_polar_cap_clamping_and_area_weighting(self):
        native = native_global_grid()
        native.rain.loc[{"latitude": -90.0}] = 1
        coarse = prepare_benchmark_grid(native, 1.5, (0.0, 0.0))
        # Target pole cell at -90 covers [-90, -89.25]; native pole cell at -90 covers [-90, -89.875].
        south, native_north, target_north = np.sin(np.deg2rad([-90.0, -89.875, -89.25]))
        expected = (native_north - south) / (target_north - south)
        self.assertAlmostEqual(coarse.rain.sel(latitude=-90.0, longitude=0.0).item(), expected)

    def test_coarsening_conserves_global_area_weighted_mean(self):
        native = native_global_grid()
        lat = np.deg2rad(native.latitude.values)[:, None]
        lon = np.deg2rad(native.longitude.values)[None, :]
        field = (3.0 + np.cos(2 * lat) * np.sin(lon + 0.4) + 0.5 * np.sin(lat)).astype(np.float64)
        native["rain"] = (("latitude", "longitude"), field)

        native_weights = latitude_weights(native.latitude, 0.25)
        native_mean = native.rain.weighted(native_weights).mean().item()

        for step, origin, expected_shape in [
            (1.5, (0.0, 0.0), (121, 240)),
            (3.0, (-1.5, 1.5), (60, 120)),
            (5.625, (2.8125, 0.0), (32, 64)),
        ]:
            coarse = prepare_benchmark_grid(native, step, origin)
            self.assertEqual(coarse.rain.shape, expected_shape)
            self.assertEqual(coarse.rain.dims, ("latitude", "longitude"))
            coarse_weights = latitude_weights(coarse.latitude, step)
            coarse_mean = coarse.rain.weighted(coarse_weights).mean().item()
            self.assertAlmostEqual(coarse_mean, native_mean, places=10)

    def test_invalid_global_grids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "evenly divide 360"):
            global_grid(7.0)
        with self.assertRaisesRegex(ValueError, "cover \\[-90, 90\\]"):
            global_grid(3.0, (0.5, 0.0))
        cropped = native_global_grid().sel(latitude=slice(-45, -10))
        with self.assertRaisesRegex(ValueError, "full global native"):
            average_to_cells(cropped, global_grid(1.5), 1.5)
