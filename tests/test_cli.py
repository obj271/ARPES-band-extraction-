import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from arpes_map.cli import main
from arpes_map import comparison_intensities
from arpes_map.synthetic import simulate


class CliTests(unittest.TestCase):
    def test_demo_and_extraction_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            demo = root / "demo"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["demo", "--energy-points", "64", "--momentum-points", "24", "--output", str(demo)])
                main(["extract", "--input", str(demo / "result.npz"), "--output", str(root / "extract")])
            for name in ("result.npz", "summary.json", "extraction.png", "band.csv"):
                self.assertGreater((demo / name).stat().st_size, 0)
            with np.load(demo / "result.npz") as first, np.load(root / "extract" / "result.npz") as second:
                np.testing.assert_array_equal(first["band_energy"], second["band_energy"])
                for key in ("curvature_intensity", "gradient_intensity"):
                    self.assertEqual(first[key].shape, first["intensity"].shape)
                    self.assertTrue(np.isfinite(first[key]).all())
                    self.assertEqual(first[key].max(), 1)
                    np.testing.assert_array_equal(first[key], second[key])
            summary = json.loads((demo / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("map_rmse_mev", summary["simulation_metrics"])
            self.assertFalse(summary["configuration"]["missing"])
            self.assertEqual(summary["featureless_columns"], 0)
            self.assertIn("intensity_comparison", summary)
            csv = np.loadtxt(demo / "band.csv", delimiter=",", skiprows=1)
            self.assertEqual(csv.shape, (24, 6))

    def test_demo_dropout_is_optional(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for flag, expected_missing in (("--dropout", 4), ("--no-dropout", 0)):
                with self.subTest(flag=flag), contextlib.redirect_stdout(io.StringIO()):
                    output = root / flag.removeprefix("--")
                    main(["demo", flag, "--energy-points", "64", "--momentum-points", "24",
                          "--output", str(output)])
                    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(summary["configuration"]["missing"], bool(expected_missing))
                    self.assertEqual(summary["featureless_columns"], expected_missing)
                    with np.load(output / "result.npz") as result:
                        missing = np.ptp(result["intensity"], axis=0) == 0
                        self.assertEqual(np.sum(missing), expected_missing)
                        np.testing.assert_array_equal(~result["observed"], missing)
                        for key in ("curvature_intensity", "gradient_intensity"):
                            np.testing.assert_array_equal(result[key][:, missing], 0)

    def test_window_reference_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = simulate(ne=64, nk=24)
            np.savez(root / "input.npz", **data)
            np.savez(root / "reference.npz", momentum=data["momentum"], band_energy=data["truth"])
            with contextlib.redirect_stdout(io.StringIO()):
                main(["extract", "--input", str(root / "input.npz"), "--reference", str(root / "reference.npz"),
                      "--energy-window", "-0.4", "-0.1", "--anchor", "12:-0.29",
                      "--smooth-energy", "1.2", "--smooth-momentum", "0.5", "--output", str(root / "out")])
            with np.load(root / "out" / "result.npz") as result:
                self.assertGreaterEqual(result["energy"].min(), -0.4)
                self.assertLessEqual(result["energy"].max(), -0.1)
                expected = comparison_intensities(result["intensity"], result["energy"], result["momentum"],
                                                   smoothing=(1.2, 0.5))
                for key, values in expected.items():
                    np.testing.assert_array_equal(result[key], values)
            np.savez(root / "bad.npz", energy=data["energy"])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["extract", "--input", str(root / "bad.npz")])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
