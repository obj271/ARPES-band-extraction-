import unittest

import numpy as np

from arpes_map import comparison_intensities
from arpes_map.synthetic import simulate


class EnhancementTests(unittest.TestCase):
    def test_gaussian_ridge_is_bright_at_peak(self):
        e = np.linspace(-0.5, 0.5, 101)
        k = np.linspace(-1, 1, 9)
        intensity = np.broadcast_to(np.exp(-0.5 * (e[:, None] / 0.05)**2), (101, 9)).copy()
        original = intensity.copy()
        maps = comparison_intensities(intensity, e, k, smoothing=(0, 0))
        for values in maps.values():
            self.assertEqual(values.shape, intensity.shape)
            self.assertTrue(np.isfinite(values).all())
            self.assertGreaterEqual(values.min(), 0)
            self.assertEqual(values.max(), 1)
            np.testing.assert_array_equal(np.argmax(values, axis=0), 50)
        self.assertEqual(maps["curvature_intensity"][20, 4], 0)
        np.testing.assert_array_equal(intensity, original)

    def test_curvature_matches_analytic_surface_interior(self):
        y = np.arange(-4., 5.)
        x = np.arange(-4., 5.)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        f = (100 - yy**2 - 2 * xx**2 + 0.4 * yy * xx) / 100
        fx, fy = (-4 * xx + 0.4 * yy) / 100, (-2 * yy + 0.4 * xx) / 100
        expected = -((1 + fx**2) * -0.02 - 2 * fx * fy * 0.004 + (1 + fy**2) * -0.04)
        expected /= (1 + fx**2 + fy**2)**1.5
        expected /= expected.max()
        actual = comparison_intensities(f, y, x, smoothing=(0, 0))["curvature_intensity"]
        np.testing.assert_allclose(actual[2:-2, 2:-2], expected[2:-2, 2:-2], atol=1e-13)

    def test_eight_neighbor_gradient_on_irregular_grid(self):
        e = np.array([-0.5, -0.3, -0.2, 0.0])
        k = np.array([-1., -0.2, 0.3, 1.])
        f = np.array([[1., 2., 4., 2.], [3., 6., 8., 4.],
                      [2., 3., 5., 3.], [1., 1., 2., 1.]])
        expected = np.zeros_like(f)
        for row, col in np.ndindex(f.shape):
            slopes = []
            for nr in range(max(0, row - 1), min(4, row + 2)):
                for nc in range(max(0, col - 1), min(4, col + 2)):
                    if (nr, nc) != (row, col):
                        distance = np.hypot((e[nr] - e[row]) / np.median(np.diff(e)),
                                            (k[nc] - k[col]) / np.median(np.diff(k)))
                        slopes.append((f[nr, nc] - f[row, col]) / distance)
            expected[row, col] = f[row, col] / np.sqrt(np.mean(np.square(slopes)))
        expected /= expected.max()
        actual = comparison_intensities(f, e, k, smoothing=(0, 0))["gradient_intensity"]
        np.testing.assert_allclose(actual, expected)

    def test_scale_invariance_and_missing_columns_after_smoothing(self):
        data = simulate(ne=64, nk=24, missing=True)
        args = (data["energy"], data["momentum"])
        maps = comparison_intensities(data["intensity"], *args, smoothing=(1, 1))
        scaled = comparison_intensities(data["intensity"] * 1e9, *args, smoothing=(1, 1))
        missing = np.ptp(data["intensity"], axis=0) == 0
        for key, values in maps.items():
            np.testing.assert_allclose(values, scaled[key], atol=1e-12)
            np.testing.assert_array_equal(values[:, missing], 0)

    def test_small_flat_and_invalid_inputs(self):
        for f in (np.zeros((2, 2)), np.ones((2, 2)), np.array([[1., 0.], [2., 0.]])):
            maps = comparison_intensities(f, [0, 1], [0, 1])
            for values in maps.values():
                self.assertTrue(np.isfinite(values).all())
                if np.ptp(f) == 0:
                    np.testing.assert_array_equal(values, 0)
        for smoothing in ((-1, 0), (np.nan, 0), (0,)):
            with self.assertRaises(ValueError):
                comparison_intensities(np.ones((2, 2)), [0, 1], [0, 1], smoothing=smoothing)
        with self.assertRaises(ValueError):
            comparison_intensities(np.ones((2, 2)), [1, 0], [0, 1])


if __name__ == "__main__":
    unittest.main()
