import unittest

import numpy as np
import xarray as xr

from skillfront.data import prepare_benchmark_grid
from skillfront.grid import average_to_cells, crop_native


def native_grid() -> xr.Dataset:
    latitude = np.arange(-31, -28.75, 0.25)
    longitude = np.arange(139, 141.25, 0.25)
    data = xr.Dataset(coords={"latitude": latitude, "longitude": longitude})
    data["rain"] = (("latitude", "longitude"), np.zeros((latitude.size, longitude.size)))
    data.rain.loc[{"longitude": 140.75}] = 12
    data.rain.attrs["units"] = "mm"
    return data


def target_grid(latitude: float, longitude: float) -> xr.Dataset:
    return xr.Dataset(coords={
        "latitude": ("cell", [latitude]),
        "longitude": ("cell", [longitude]),
        "cell": [0],
    })


class GridTests(unittest.TestCase):
    def test_archive_coordinates_snap_to_quarter_degrees(self):
        data = native_grid()
        jittered = data.assign_coords(latitude=data.latitude + 0.000004)
        fixed = crop_native(jittered)
        xr.testing.assert_identical(fixed, data)

    def test_native_values_are_used_directly(self):
        native = native_grid()
        cells = average_to_cells(native, target_grid(-30, 140.75), 0.25)
        self.assertEqual(cells.rain.item(), 12)
        self.assertEqual(cells.rain.dtype, native.rain.dtype)

    def test_coarse_grid_counts_partial_cell_overlap(self):
        cells = average_to_cells(native_grid(), target_grid(-30, 140), 1.5)
        self.assertAlmostEqual(cells.rain.item(), 1)
        self.assertEqual(cells.rain.attrs["units"], "mm")

    def test_area_weighting_uses_latitude_and_grid_can_shift(self):
        native = native_grid()
        native.rain[:] = 0
        native.rain.loc[{"latitude": -30.25}] = 1
        native.attrs.update(
            native_grid_degrees=0.25,
            australia_geometry=(
                '{"type":"Polygon","coordinates":[[[139.9,-30.3],[140.1,-30.3],'
                '[140.1,-30.2],[139.9,-30.2],[139.9,-30.3]]]}'
            ),
        )
        cells = prepare_benchmark_grid(native, 1.0, (-30.25, 140))
        south, north, target_south, target_north = np.sin(
            np.deg2rad([-30.375, -30.125, -30.75, -29.75])
        )
        expected = (north - south) / (target_north - target_south)
        self.assertAlmostEqual(cells.rain.item(), expected)
        self.assertEqual(cells.latitude.item(), -30.25)

    def test_partial_target_coverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not cover"):
            average_to_cells(native_grid(), target_grid(-30, 141), 1.5)
