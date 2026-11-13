import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from arpes_map.cli import _parser, main
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
            expected_files = {"result.npz", "summary.json", "map_result.png", "band_comparison.png",
                              "band.csv", "metrics.csv"}
            self.assertEqual({path.name for path in demo.iterdir()}, expected_files)
            for name in expected_files:
                self.assertGreater((demo / name).stat().st_size, 0)
            with np.load(demo / "result.npz") as first, np.load(root / "extract" / "result.npz") as second:
                np.testing.assert_array_equal(first["band_energy"], second["band_energy"])
                np.testing.assert_array_equal(first["marginals"], second["marginals"])
                self.assertEqual(first["band_energy"].shape, (3, 24))
                self.assertEqual(first["marginals"].shape, (3, 64, 24))
                np.testing.assert_allclose(first["marginals"].sum(axis=1), 1)
                self.assertTrue(np.all(first["marginals"][~first["search_masks"]] == 0))
                self.assertGreater(np.sqrt(np.mean((first["initial_guess"] - first["truth"])**2)), 0.01)
                for key in ("curvature_intensity", "gradient_intensity"):
                    self.assertEqual(first[key].shape, first["intensity"].shape)
                    self.assertTrue(np.isfinite(first[key]).all())
                    self.assertEqual(first[key].max(), 1)
                    np.testing.assert_array_equal(first[key], second[key])
                for name in ("raw", "curvature", "gradient"):
                    values = first["intensity"] if name == "raw" else first[name + "_intensity"]
                    expected = first["energy"][np.argmax(
                        np.where(first["search_masks"], values[None, :, :], -np.inf), axis=1)]
                    np.testing.assert_array_equal(first[name + "_band_energy"], expected)
            summary = json.loads((demo / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("map_rmse_mev", summary["simulation_metrics"])
            self.assertFalse(summary["configuration"]["missing"])
            self.assertEqual(summary["featureless_columns"], 0)
            self.assertIn("intensity_comparison", summary)
            self.assertEqual(summary["band_count"], 3)
            self.assertEqual(len(summary["per_band_metrics"]), 3)
            self.assertGreater(summary["simulation_metrics"]["initial_guess_rmse_mev"], 10)
            csv = np.loadtxt(demo / "band.csv", delimiter=",", skiprows=1)
            self.assertEqual(csv.shape, (72, 7))
            np.testing.assert_array_equal(csv[:, 0], np.repeat([1, 2, 3], 24))
            self.assertEqual(_parser().parse_args(["demo"]).output, Path("outputs"))
            self.assertEqual(_parser().parse_args(["extract", "--input", "x.npz"]).output, Path("outputs"))

    def test_single_and_two_band_shapes(self):
        with tempfile.TemporaryDirectory() as folder, patch("arpes_map.cli._plot"):
            for bands in (1, 2):
                output = Path(folder) / str(bands)
                with self.subTest(bands=bands), contextlib.redirect_stdout(io.StringIO()):
                    main(["demo", "--bands", str(bands), "--energy-points", "64", "--momentum-points", "24",
                          "--output", str(output)])
                    with np.load(output / "result.npz") as arrays:
                        shape = (24,) if bands == 1 else (2, 24)
                        self.assertEqual(arrays["band_energy"].shape, shape)
                        self.assertEqual(arrays["truth"].shape, shape)
                        self.assertEqual(arrays["initial_guess"].shape, shape)
                    csv = np.loadtxt(output / "band.csv", delimiter=",", skiprows=1)
                    self.assertEqual(csv.shape, (24 * bands, 6 if bands == 1 else 7))

    def test_truth_never_changes_inference_and_external_guesses_work(self):
        with tempfile.TemporaryDirectory() as folder, patch("arpes_map.cli._plot"):
            root = Path(folder)
            data = simulate(ne=80, nk=30, bands=3)
            guess = data.pop("initial_guess")
            np.savez(root / "guess.npz", momentum=data["momentum"], initial_guess=guess)
            inferred = []
            for change_truth in (False, True):
                data["truth"] = data["truth"] + (0.05 if change_truth else 0)
                np.savez(root / "input.npz", **data)
                with contextlib.redirect_stdout(io.StringIO()):
                    main(["extract", "--input", str(root / "input.npz"), "--initial-guess", str(root / "guess.npz"),
                          "--anchor", "0:15:-0.405", "--output", str(root)])
                with np.load(root / "result.npz") as arrays:
                    inferred.append((arrays["band_energy"], arrays["marginals"], arrays["search_masks"]))
            for before, after in zip(*inferred):
                np.testing.assert_array_equal(before, after)
            del data["truth"]
            np.savez(root / "input.npz", **data)
            with contextlib.redirect_stdout(io.StringIO()):
                main(["extract", "--input", str(root / "input.npz"), "--reference", str(root / "guess.npz"),
                      "--output", str(root)])
            self.assertFalse((root / "metrics.csv").exists())
            summary = json.loads((root / "summary.json").read_text())
            self.assertNotIn("simulation_metrics", summary)

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
                        for key in ("raw_band_energy", "curvature_band_energy", "gradient_band_energy"):
                            self.assertTrue(np.isnan(result[key][:, missing]).all())
                    if expected_missing:
                        self.assertIsNone(summary["simulation_metrics"]["raw_argmax_rmse_mev"])

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
