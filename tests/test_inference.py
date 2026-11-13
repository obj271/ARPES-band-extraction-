import itertools
import unittest

import numpy as np
from scipy.special import logsumexp

from arpes_map.inference import _chain_inference, band_search_masks, extract_band, extract_bands
from arpes_map.synthetic import approximate_dispersion, simulate


class ChainTests(unittest.TestCase):
    def test_map_and_marginals_match_exhaustive_enumeration(self):
        rng = np.random.default_rng(8)
        energy = np.array([-0.3, -0.15, -0.02])
        unary = rng.uniform(0, 3, (3, 4))
        sigmas = np.array([0.07, 0.18, 0.11])
        paths = np.array(list(itertools.product(range(3), repeat=4)))
        costs = np.array([
            sum(unary[path[j], j] for j in range(4))
            + np.sum(0.5 * (np.diff(energy[path]) / sigmas) ** 2)
            for path in paths
        ])
        expected = paths[np.argmin(costs)]
        indices, optimum, marginals = _chain_inference(unary, energy, sigmas)
        np.testing.assert_array_equal(indices, expected)
        self.assertAlmostEqual(optimum, costs.min(), places=12)
        probability = np.exp(-costs - logsumexp(-costs))
        for j in range(4):
            exact = np.array([probability[paths[:, j] == i].sum() for i in range(3)])
            np.testing.assert_allclose(marginals[:, j], exact, atol=1e-13)

    def test_weak_and_strong_continuity_limits(self):
        energy = np.array([-0.3, -0.2, -0.1])
        unary = np.array([[0., 0., 2.], [2., 2., 0.], [3., 3., 3.]])
        indices, _, _ = _chain_inference(unary, energy, np.full(2, 100.0))
        np.testing.assert_array_equal(indices, [0, 0, 1])
        indices, _, _ = _chain_inference(unary, energy, np.full(2, 0.001))
        np.testing.assert_array_equal(indices, [0, 0, 0])

    def test_masked_states_match_exhaustive_enumeration(self):
        energy = np.array([-0.3, -0.15, -0.02])
        unary = np.array([[0., np.inf, 2.], [1., 0., np.inf], [np.inf, 2., 0.]])
        sigmas = np.array([0.12, 0.08])
        paths = np.array(list(itertools.product(range(3), repeat=3)))
        costs = np.array([
            sum(unary[path[j], j] for j in range(3))
            + np.sum(0.5 * (np.diff(energy[path]) / sigmas) ** 2) for path in paths
        ])
        indices, optimum, marginals = _chain_inference(unary, energy, sigmas)
        np.testing.assert_array_equal(indices, paths[np.argmin(costs)])
        self.assertAlmostEqual(optimum, costs.min())
        weights = np.exp(-costs - logsumexp(-costs))
        for j in range(3):
            exact = [weights[paths[:, j] == i].sum() for i in range(3)]
            np.testing.assert_allclose(marginals[:, j], exact, atol=1e-13)
        np.testing.assert_array_equal(marginals[~np.isfinite(unary)], 0)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.data = simulate(ne=96, nk=50, missing=True)

    def extract(self, **kwargs):
        return extract_band(self.data["intensity"], self.data["energy"], self.data["momentum"], **kwargs)

    def test_synthetic_data_is_complete_by_default(self):
        complete = simulate(ne=96, nk=50)
        self.assertTrue(np.all(np.ptp(complete["intensity"], axis=0) > 0))
        observed = np.ptp(self.data["intensity"], axis=0) > 0
        np.testing.assert_array_equal(complete["intensity"][:, observed],
                                      self.data["intensity"][:, observed])
        for name in ("clean", "energy", "momentum", "truth"):
            np.testing.assert_array_equal(complete[name], self.data[name])

    def test_synthetic_recovery_and_missing_evidence(self):
        result = self.extract()
        rmse = np.sqrt(np.mean((result.band_energy - self.data["truth"]) ** 2))
        self.assertLess(rmse, 0.008)
        self.assertEqual(np.sum(~result.observed), 4)
        np.testing.assert_allclose(result.likelihood[:, ~result.observed], 1 / 96)
        np.testing.assert_allclose(result.marginals.sum(axis=0), 1)
        self.assertTrue(np.all(result.lower_energy <= result.upper_energy))

    def test_intensity_unit_invariance_and_input_unchanged(self):
        original = self.data["intensity"].copy()
        result = self.extract()
        scaled = extract_band(original * 1e6, self.data["energy"], self.data["momentum"])
        np.testing.assert_array_equal(result.band_indices, scaled.band_indices)
        np.testing.assert_allclose(result.marginals, scaled.marginals, atol=1e-12)
        np.testing.assert_array_equal(self.data["intensity"], original)

    def test_reference_and_soft_anchor_select_target(self):
        e = np.linspace(-0.4, -0.05, 100)
        k = np.linspace(-1, 1, 20)
        intensity = np.broadcast_to(
            (np.exp(-0.5 * ((e + 0.30) / 0.009) ** 2)
             + 1.5 * np.exp(-0.5 * ((e + 0.15) / 0.009) ** 2))[:, None], (100, 20)
        )
        baseline = extract_band(intensity, e, k)
        self.assertLess(abs(baseline.band_energy.mean() + 0.15), 0.006)
        reference = extract_band(intensity, e, k, reference=np.full(20, -0.30), reference_sigma_ev=0.02)
        anchor = extract_band(intensity, e, k, anchors=[(10, -0.30)], anchor_sigma_ev=0.004)
        self.assertLess(abs(reference.band_energy.mean() + 0.30), 0.006)
        self.assertLess(abs(anchor.band_energy[10] + 0.30), 0.006)

    def test_nonuniform_axes(self):
        k = self.data["momentum"].copy()
        k[1:-1] += np.sin(np.arange(k.size - 2)) * 0.005
        result = extract_band(self.data["intensity"], self.data["energy"], k)
        self.assertTrue(np.isfinite(result.negative_log_posterior))

    def test_bad_data_and_parameters(self):
        for name, value in (("eta_ev", 0), ("eta_ev", np.nan),
                            ("likelihood_floor", 0), ("likelihood_floor", 1),
                            ("smoothing", (-1, 0)), ("smoothing", (0,)),
                            ("background_percentile", 100),
                            ("anchors", [(0.5, -0.2)]), ("anchors", [(0, -5)]),
                            ("reference", np.zeros(3))):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.extract(**{name: value})
        for invalid in (np.zeros((96, 50)), np.full((96, 50), 10.),
                        -np.ones((96, 50)), np.full((96, 50), np.nan)):
            with self.assertRaises(ValueError):
                extract_band(invalid, self.data["energy"], self.data["momentum"])
        with self.assertRaises(ValueError):
            extract_band(self.data["intensity"], self.data["energy"][::-1], self.data["momentum"])
        with self.assertRaises(ValueError):
            extract_band(self.data["intensity"].T, self.data["energy"], self.data["momentum"])


class MultiBandTests(unittest.TestCase):
    def test_biased_guesses_recover_distinct_bands_with_missing_columns(self):
        for count in (2, 3):
            with self.subTest(bands=count):
                data = simulate(ne=144, nk=70, bands=count, missing=True)
                guess = data["initial_guess"]
                np.testing.assert_array_equal(guess, approximate_dispersion(data["momentum"], count))
                initial_rmse = np.sqrt(np.mean((guess - data["truth"])**2, axis=1))
                self.assertTrue(np.all(initial_rmse > 0.01))
                results = extract_bands(data["intensity"], data["energy"], data["momentum"], initial_guess=guess)
                masks = band_search_masks(data["energy"], guess)
                np.testing.assert_array_equal(masks.sum(axis=0), 1)
                recovered = np.stack([r.band_energy for r in results])
                self.assertTrue(np.all(np.diff(recovered, axis=0) > 0))
                errors = np.sqrt(np.mean((recovered - data["truth"])**2, axis=1))
                self.assertTrue(np.all(errors < 0.008), errors)
                self.assertTrue(np.all(errors < initial_rmse / 2), (errors, initial_rmse))
                for result, mask in zip(results, masks):
                    self.assertEqual(np.sum(~result.observed), 4)
                    np.testing.assert_allclose(result.marginals.sum(axis=0), 1)
                    np.testing.assert_array_equal(result.marginals[~mask], 0)
                    np.testing.assert_allclose(result.likelihood[:, ~result.observed], 1 / len(data["energy"]))

    def test_invalid_band_guesses_and_anchor_indices(self):
        data = simulate(ne=80, nk=30, bands=3)
        guess = data["initial_guess"]
        invalid = (guess[::-1], guess[:, :-1], guess[0], np.full_like(guess, np.nan),
                   np.full_like(guess, -10), np.repeat(guess[:1], 3, axis=0), guess[:0])
        for value in invalid:
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                extract_bands(data["intensity"], data["energy"], data["momentum"], initial_guess=value)
        for band in (-1, 3, 0.5, True):
            with self.subTest(band=band), self.assertRaises(ValueError):
                extract_bands(data["intensity"], data["energy"], data["momentum"], initial_guess=guess,
                              anchors=[(band, 10, -0.3)])
        with self.assertRaises(ValueError):
            band_search_masks(np.array([-1., 1.]), np.array([[-0.5], [0.], [0.5]]))


if __name__ == "__main__":
    unittest.main()
