import unittest

import numpy as np
import xarray as xr

from skillfront.benchmark import benchmark_cases, score_forecast
from skillfront.data import PROTOCOL, VARIABLES


def example_data() -> xr.Dataset:
    timestamps = np.arange("2022-01-01", "2023-01-01", dtype="datetime64[D]")
    days = np.arange(1, 367)
    data = xr.Dataset(
        coords={
            "time": timestamps,
            "dayofyear": days,
            "cell": [0, 1],
            "latitude": ("cell", [-42.0, -15.0]),
            "longitude": ("cell", [147.0, 135.0]),
        },
        attrs={"protocol": PROTOCOL},
    )
    for name, (_, unit, _, _) in VARIABLES.items():
        data[name] = (("time", "cell"), np.repeat(np.arange(len(timestamps))[:, None], 2, axis=1))
        data[f"climatology_{name}"] = (("dayofyear", "cell"), np.repeat(days[:, None], 2, axis=1))
        data[name].attrs["units"] = unit
        data[f"climatology_{name}"].attrs["units"] = unit
    return data


class AustraliaBenchmarkTests(unittest.TestCase):
    def test_references_align_by_initialization_and_valid_time(self):
        truth, baselines = benchmark_cases(example_data())
        temperature = "temperature_2m"
        self.assertEqual(truth.sizes["init_time"], 28)
        self.assertEqual(truth.valid_time.values[-1, 0], np.datetime64("2022-10-08"))
        self.assertEqual(truth[temperature].values[0, 0, 0], 1)
        self.assertEqual(baselines["persistence"][temperature].values[0, 0, 0], 0)
        climate = baselines["climatology"][temperature]
        self.assertEqual(climate.sel(init_time="2022-04-01").values[0, 0], 92)
        self.assertEqual(climate.values[-1, 0, 0], 281)

    def test_metrics_weight_grid_cells_by_latitude(self):
        truth, _ = benchmark_cases(example_data())
        forecast = truth.copy(deep=True)
        forecast.temperature_2m.values[:, :, 0] += 2
        scores = score_forecast(forecast, truth)["temperature_2m"]
        weights = np.cos(np.deg2rad(truth.latitude.values))
        share = weights[0] / weights.sum()
        self.assertAlmostEqual(scores["bias"], 2 * share)
        self.assertAlmostEqual(scores["mae"], 2 * share)
        self.assertAlmostEqual(scores["rmse"], 2 * np.sqrt(share))

    def test_misaligned_coordinates_and_wrong_units_are_rejected(self):
        truth, baselines = benchmark_cases(example_data())
        forecast = baselines["persistence"]
        with self.assertRaisesRegex(ValueError, "init_time"):
            score_forecast(forecast.isel(init_time=slice(1, None)), truth)
        with self.assertRaisesRegex(ValueError, "longitude"):
            score_forecast(forecast.assign_coords(longitude=forecast.longitude + 1), truth)
        forecast.temperature_2m.attrs["units"] = "K"
        with self.assertRaisesRegex(ValueError, "degC"):
            score_forecast(forecast, truth)
